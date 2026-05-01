#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


DEFAULT_MAS_ORCHESTRA_ROOT = Path("/mnt/gpfs/xuexiangyuan/workspace/mas_orchestra")
if str(DEFAULT_MAS_ORCHESTRA_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_MAS_ORCHESTRA_ROOT))

from mas_orchestra_mcts.config import MCTSConfig
from mas_orchestra_mcts.runner import MCTSReasoningRunner, ResumedTreeState, SearchNode


@dataclass
class StagePlan:
    stage_name: str
    task_ids: list[str]
    fallback_random_pool_task_ids: list[str]
    reopen_nodes: dict[str, str]
    leaf_acc_before: dict[str, float]


class FixedPoolExpansionRunner(MCTSReasoningRunner):
    def __init__(
        self,
        config: MCTSConfig,
        *,
        pools_by_task: dict[str, list[str]],
        leaf_acc_threshold: float,
        max_final_leaf_count: int | None = None,
        progress_desc: str | None = None,
    ) -> None:
        super().__init__(config)
        self.pools_by_task = pools_by_task
        self.leaf_acc_threshold = float(leaf_acc_threshold)
        self.max_final_leaf_count = (
            int(max_final_leaf_count)
            if max_final_leaf_count is not None
            else None
        )
        self.progress_desc = progress_desc or "Expand"
        self._active_task_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            "active_task_id",
            default=None,
        )

    async def _run_sample(self, *, run_dir: Path, sample: Any, global_events: Path) -> dict[str, Any]:
        token = self._active_task_id.set(sample.task_id)
        try:
            return await super()._run_sample(run_dir=run_dir, sample=sample, global_events=global_events)
        finally:
            self._active_task_id.reset(token)

    def _build_child_model_pools(
        self,
        rng: random.Random,
        *,
        child_count: int,
        requires_image_inputs: bool = False,
    ) -> list[list[str]]:
        task_id = self._active_task_id.get()
        task_pool = list(self.pools_by_task.get(task_id, [])) if task_id else []
        if requires_image_inputs:
            task_pool = [model for model in task_pool if self._model_supports_images(model)]
        if not task_pool:
            task_pool = self._available_delegate_models(requires_image_inputs=requires_image_inputs)
        if not task_pool:
            raise RuntimeError("No delegate models available for child expansion.")
        return [[rng.choice(task_pool)] for _ in range(max(0, child_count))]

    @staticmethod
    def _model_supports_images(model_name: str) -> bool:
        from mas_orchestra.llm.model_capabilities import supports_image_inputs
        return supports_image_inputs(model_name)

    def _leaf_acc(self, final_leaves: list[SearchNode]) -> float:
        if not final_leaves:
            return 0.0
        correct = sum(1 for node in final_leaves if node.is_correct)
        return correct / len(final_leaves)

    def _choose_reopen_node_from_tree(
        self,
        *,
        all_nodes: dict[str, SearchNode],
        rng: random.Random,
    ) -> SearchNode | None:
        candidates = [
            node
            for node in all_nodes.values()
            if self._is_expandable_node(node)
        ]
        if not candidates:
            return None
        return rng.choice(sorted(candidates, key=lambda item: item.node_id))

    async def run_with_leaf_acc_stop(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        run_dir = self._build_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(run_dir / "config.snapshot.json", self.config.to_json())

        samples = self._select_samples(self._load_dataset_samples())
        selected_task_ids = [sample.task_id for sample in samples]

        self._write_json(
            run_dir / "selected_tasks.json",
            {
                "dataset_name": self.config.dataset_name,
                "dataset_split": self.config.dataset_split,
                "discipline": self.config.discipline,
                "sample_count": len(samples),
                "sample_seed": self.config.sample_seed,
                "selected_task_ids": selected_task_ids,
                "selected_samples": [
                    {"task_id": sample.task_id, "discipline": sample.discipline}
                    for sample in samples
                ],
            },
        )

        semaphore = asyncio.Semaphore(max(1, self.config.max_concurrency))
        progress = None
        if self.config.show_progress and tqdm is not None:
            progress = tqdm(
                total=len(samples),
                desc=self.progress_desc,
                unit="tree",
                dynamic_ncols=True,
            )

        async def run_limited(sample: Any) -> dict[str, Any]:
            async with semaphore:
                return await self._run_sample_with_leaf_acc_stop(
                    run_dir=run_dir,
                    sample=sample,
                    global_events=run_dir,
                )

        tasks = [asyncio.create_task(run_limited(sample)) for sample in samples]
        try:
            for future in asyncio.as_completed(tasks):
                await future
                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        records = self._collect_sample_results(run_dir)
        all_raw_calls = self._collect_sample_raw_calls(run_dir)
        summary = self._build_summary(records=records, run_dir=run_dir)
        self._write_json(run_dir / "scored.json", records)
        self._write_json(run_dir / "summary.json", summary)
        with (run_dir / "raw_calls.jsonl").open("w", encoding="utf-8") as handle:
            for row in all_raw_calls:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
        return records, summary

    async def _run_sample_with_leaf_acc_stop(
        self,
        *,
        run_dir: Path,
        sample: Any,
        global_events: Path,
    ) -> dict[str, Any]:
        sample_dir = run_dir / "samples" / sample.task_id
        resume_state: ResumedTreeState | None = None
        if sample_dir.exists():
            result_path = sample_dir / "result.json"
            latest_path = sample_dir / "latest.json"
            if self.config.resume and result_path.exists():
                payload = self._read_json(result_path)
                return {"summary": payload if isinstance(payload, dict) else {"task_id": sample.task_id}, "raw_calls": []}
            if self.config.resume and latest_path.exists():
                resume_state = self._load_resume_state(sample_dir)
                if resume_state is None:
                    raise RuntimeError(f"Non-resumable latest snapshot for {sample.task_id}: {latest_path}")
        sample_dir.mkdir(parents=True, exist_ok=True)

        task_token = self._active_task_id.set(sample.task_id)
        try:
            sample_start_tree = await self._run_tree_until_leaf_acc(
                sample=sample,
                sample_dir=sample_dir,
                global_events=global_events,
                rng=random.Random(self._stable_seed(sample.task_id, self.config.tree_seed)),
                resume_state=resume_state,
            )
            result = {
                "task_id": sample.task_id,
                "discipline": sample.discipline,
                "status": "completed",
                "question": sample.question,
                "options": sample.options,
                "gold_answer_letter": chr(ord("A") + sample.answer_index),
                "orchestra_model": self.config.orchestra_model,
                "success": sample_start_tree["success"],
                "any_correct_leaf": sample_start_tree["any_correct_leaf"],
                "best_leaf_correct": sample_start_tree["best_leaf_correct"],
                "majority_correct": sample_start_tree["majority_correct"],
                "correct_leaf_count": sample_start_tree["correct_leaf_count"],
                "final_leaf_count": sample_start_tree["final_leaf_count"],
                "open_leaf_count": sample_start_tree["open_leaf_count"],
                "target_leaf_trajectories": self.config.target_leaf_trajectories,
                "branching_factor": self.config.branching_factor,
                "leaf_expand_ratio": self.config.leaf_expand_ratio,
                "frontier_limit": self.config.frontier_limit,
                "sibling_pool_strategy": self.config.sibling_pool_strategy,
                "path_max_steps": self.config.node_max_steps,
                "budget_limit": self.config.tree_budget_usd,
                "budget_spent": sample_start_tree["budget_spent"],
                "budget_exhausted": sample_start_tree["budget_exhausted"],
                "stop_reason": sample_start_tree["stop_reason"],
                "expansion_rounds_ran": len(sample_start_tree["rounds"]),
                "best_leaf_node_id": sample_start_tree["best_leaf_node_id"],
                "best_leaf_boxed_letter": sample_start_tree["best_leaf_boxed_letter"],
                "best_leaf_latest_delegate_confidence": sample_start_tree["best_leaf_latest_delegate_confidence"],
                "majority_boxed_letter": sample_start_tree["majority_boxed_letter"],
                "latency_seconds": 0.0,
                "total_cost": sample_start_tree["cost"],
                "total_tokens": sample_start_tree["total_tokens"],
                "total_model_calls": sample_start_tree["model_calls"],
                "failed_terminal_count": sample_start_tree["failed_terminal_count"],
                "leaf_acc": self._leaf_acc_from_counts(
                    sample_start_tree["correct_leaf_count"],
                    sample_start_tree["final_leaf_count"],
                ),
            }
            self._write_json(sample_dir / "result.json", result)
            self._write_jsonl(sample_dir / "calls.jsonl", sample_start_tree["raw_calls"])
            self._write_jsonl(sample_dir / "nodes.jsonl", sample_start_tree["nodes"])
            self._write_json(
                sample_dir / "view.json",
                {
                    "task_id": sample.task_id,
                    "status": "completed",
                    "rounds": sample_start_tree["rounds"],
                    "trajectories": sample_start_tree["trajectories"],
                    "final_leaf_node_ids": sample_start_tree["final_leaf_node_ids"],
                    "open_frontier_node_ids": sample_start_tree["open_frontier_node_ids"],
                    "failed_terminal_node_ids": sample_start_tree["failed_terminal_node_ids"],
                    "final_summary": {
                        "stop_reason": sample_start_tree["stop_reason"],
                        "leaf_acc": result["leaf_acc"],
                        "success": result["success"],
                    },
                    "metrics": {
                        "budget_spent": result["budget_spent"],
                        "final_leaf_count": result["final_leaf_count"],
                        "correct_leaf_count": result["correct_leaf_count"],
                        "leaf_acc": result["leaf_acc"],
                    },
                },
            )
            return {"summary": result, "raw_calls": sample_start_tree["raw_calls"]}
        finally:
            self._active_task_id.reset(task_token)

    @staticmethod
    def _leaf_acc_from_counts(correct_leaf_count: int, final_leaf_count: int) -> float:
        if final_leaf_count <= 0:
            return 0.0
        return correct_leaf_count / final_leaf_count

    async def _run_tree_until_leaf_acc(
        self,
        *,
        sample: Any,
        sample_dir: Path,
        global_events: Path,
        rng: random.Random,
        resume_state: ResumedTreeState | None = None,
    ) -> dict[str, Any]:
        if resume_state is None:
            raise RuntimeError("This script only supports resume-from-copied-tree mode.")

        all_nodes = resume_state.all_nodes
        frontier = resume_state.frontier
        final_leaves = resume_state.final_leaves
        failed_terminal_nodes = resume_state.failed_terminal_nodes
        sample_raw_calls = resume_state.sample_raw_calls
        raw_call_counter = {"value": resume_state.raw_call_counter_value}
        budget_spent = resume_state.budget_spent
        total_tokens_spent = resume_state.total_tokens
        model_calls = max(len(sample_raw_calls), raw_call_counter["value"])
        node_counter = resume_state.node_counter
        rounds = resume_state.rounds

        stop_reason: str | None = None
        round_index = len(rounds)
        reopen_count = 0
        while stop_reason is None:
            if self._leaf_acc(final_leaves) >= self.leaf_acc_threshold:
                stop_reason = "leaf_acc_threshold_reached"
                break
            if self.max_final_leaf_count is not None and len(final_leaves) >= self.max_final_leaf_count:
                stop_reason = "max_final_leaf_count_reached"
                break
            if not frontier:
                reopen_node = self._choose_reopen_node_from_tree(all_nodes=all_nodes, rng=rng)
                if reopen_node is None:
                    stop_reason = (
                        "budget_exhausted_no_reopen_candidates"
                        if budget_spent >= self.config.tree_budget_usd
                        else "no_reopen_candidates"
                    )
                    break
                frontier = [reopen_node]
                reopen_count += 1

            round_index += 1
            selected_nodes, selected_count_requested = self._select_nodes_for_expansion(
                frontier=frontier,
                all_nodes=all_nodes,
            )
            if not selected_nodes:
                stop_reason = "no_frontier_selected"
                break
            selected_node_ids = {node.node_id for node in selected_nodes}
            pending_frontier = [node for node in frontier if node.node_id not in selected_node_ids]

            (
                round_summary,
                budget_spent,
                total_tokens_spent,
                model_calls,
                node_counter,
            ) = await self._expand_round(
                sample=sample,
                sample_dir=sample_dir,
                global_events=global_events,
                all_nodes=all_nodes,
                completed_rounds=rounds,
                selected_parents=selected_nodes,
                pending_frontier=pending_frontier,
                active_frontier_count=len(selected_nodes),
                round_index=round_index,
                selection_strategy="frontier_rank_expand",
                budget_spent=budget_spent,
                model_calls=model_calls,
                node_counter=node_counter,
                rng=rng,
                selected_count_requested=selected_count_requested,
                final_leaves=final_leaves,
                failed_terminal_nodes=failed_terminal_nodes,
                sample_raw_calls=sample_raw_calls,
                raw_call_counter=raw_call_counter,
                total_tokens_spent=total_tokens_spent,
            )
            rounds.append(round_summary)
            frontier = [all_nodes[node_id] for node_id in round_summary["next_frontier_node_ids"]]
            self._write_tree_snapshot(
                sample_dir=sample_dir,
                task_id=sample.task_id,
                all_nodes=all_nodes,
                rounds=rounds,
                budget_spent=budget_spent,
                final_leaves=final_leaves,
                failed_terminal_nodes=failed_terminal_nodes,
                open_frontier=frontier,
                sample_raw_calls=sample_raw_calls,
                total_tokens=total_tokens_spent,
                raw_call_counter_value=raw_call_counter["value"],
                node_counter=node_counter,
            )
            if self._leaf_acc(final_leaves) >= self.leaf_acc_threshold:
                stop_reason = "leaf_acc_threshold_reached"
                break
            if self.max_final_leaf_count is not None and len(final_leaves) >= self.max_final_leaf_count:
                stop_reason = "max_final_leaf_count_reached"
                break
            if budget_spent >= self.config.tree_budget_usd and not any(self._is_expandable_node(node) for node in all_nodes.values()):
                stop_reason = "budget_exhausted_no_reopen_candidates"
                break

        trajectories = self._build_leaf_trajectories(final_leaves=final_leaves, all_nodes=all_nodes)
        best_leaf = self._best_leaf(final_leaves, all_nodes=all_nodes)
        majority_boxed_letter = self._majority_letter(trajectories)
        gold_answer_letter = chr(ord("A") + sample.answer_index)
        any_correct_leaf = any(node.is_correct for node in final_leaves)
        majority_correct = majority_boxed_letter == gold_answer_letter if majority_boxed_letter else False
        return {
            "task_id": sample.task_id,
            "budget_limit": self.config.tree_budget_usd,
            "budget_spent": budget_spent,
            "budget_exhausted": budget_spent >= self.config.tree_budget_usd,
            "stop_reason": stop_reason or "completed",
            "target_leaf_trajectories": self.config.target_leaf_trajectories,
            "branching_factor": self.config.branching_factor,
            "leaf_expand_ratio": self.config.leaf_expand_ratio,
            "frontier_limit": self.config.frontier_limit,
            "sibling_pool_strategy": self.config.sibling_pool_strategy,
            "path_max_steps": self.config.node_max_steps,
            "final_leaf_count": len(final_leaves),
            "open_leaf_count": len(frontier),
            "failed_terminal_count": len(failed_terminal_nodes),
            "correct_leaf_count": sum(1 for node in final_leaves if node.is_correct),
            "success": any_correct_leaf,
            "any_correct_leaf": any_correct_leaf,
            "best_leaf_correct": bool(best_leaf and best_leaf.is_correct),
            "majority_correct": majority_correct,
            "best_leaf_node_id": best_leaf.node_id if best_leaf is not None else None,
            "best_leaf_boxed_letter": best_leaf.boxed_letter if best_leaf is not None else None,
            "best_leaf_latest_delegate_confidence": (
                self._latest_delegate_confidence(self._reconstruct_path(best_leaf, all_nodes))
                if best_leaf is not None
                else None
            ),
            "majority_boxed_letter": majority_boxed_letter,
            "final_leaf_node_ids": [node.node_id for node in final_leaves],
            "failed_terminal_node_ids": [node.node_id for node in failed_terminal_nodes],
            "model_calls": max(len(sample_raw_calls), raw_call_counter["value"]),
            "total_tokens": total_tokens_spent,
            "cost": budget_spent,
            "reopen_count": reopen_count,
            "rounds": rounds,
            "trajectories": trajectories,
            "open_frontier_node_ids": [node.node_id for node in frontier],
            "nodes": [node.to_json() for node in all_nodes.values()],
            "raw_calls": sample_raw_calls,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue an existing MCTS run in place with fixed per-task model pools."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--start-stage",
        choices=["all_wrong", "below_threshold"],
        default=None,
        help="Which stage to start from. Use below_threshold to skip stage1 and run stage2 directly.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only rewrite selected sample snapshots to prepare reopen frontier, do not make model calls.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_model_pools(summary_path: Path) -> dict[str, list[str]]:
    payload = load_json(summary_path)
    tasks = payload.get("tasks", [])
    pools: dict[str, list[str]] = {}
    for item in tasks:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id") or "").strip()
        models = [str(model) for model in item.get("correct_models", []) if str(model).strip()]
        if task_id:
            pools[task_id] = models
    return pools


def load_results(run_dir: Path) -> list[dict[str, Any]]:
    rows = load_json(run_dir / "scored.json")
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {run_dir / 'scored.json'}")
    return [row for row in rows if isinstance(row, dict)]


def load_latest(sample_dir: Path) -> dict[str, Any]:
    payload = load_json(sample_dir / "latest.json")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {sample_dir / 'latest.json'}")
    return payload


def leaf_acc_from_result(row: dict[str, Any]) -> float:
    final_leaf_count = int(row.get("final_leaf_count", 0) or 0)
    correct_leaf_count = int(row.get("correct_leaf_count", 0) or 0)
    if final_leaf_count <= 0:
        return 0.0
    return correct_leaf_count / final_leaf_count


def expandable_node_ids(latest: dict[str, Any], *, node_max_steps: int) -> list[str]:
    node_ids: list[str] = []
    for item in latest.get("nodes", []):
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "")
        is_terminal = bool(item.get("is_terminal", False))
        depth = int(item.get("depth", 0) or 0)
        node_id = str(item.get("node_id") or "").strip()
        if action in {"root", "delegate"} and not is_terminal and depth < node_max_steps and node_id:
            node_ids.append(node_id)
    return node_ids


def choose_reopen_node(
    latest: dict[str, Any],
    *,
    task_id: str,
    node_max_steps: int,
    random_seed: int,
) -> str | None:
    candidates = expandable_node_ids(latest, node_max_steps=node_max_steps)
    if not candidates:
        return None
    rng = random.Random(f"{random_seed}:{task_id}")
    return rng.choice(sorted(candidates))


def prepare_sample_dir_for_stage(
    sample_dir: Path,
    *,
    task_id: str,
    reopen_node_id: str,
    stage_name: str,
) -> None:
    latest = load_latest(sample_dir)
    latest["open_frontier_node_ids"] = [reopen_node_id]
    latest["open_leaf_count"] = 1
    latest["reopen_stage"] = stage_name
    latest["reopen_node_id"] = reopen_node_id
    write_json(sample_dir / "latest.json", latest)

    calls_partial = sample_dir / "calls.partial.jsonl"
    calls_jsonl = sample_dir / "calls.jsonl"
    if not calls_partial.exists() and calls_jsonl.exists():
        calls_partial.write_text(calls_jsonl.read_text(encoding="utf-8"), encoding="utf-8")

    for name in ("result.json", "view.json", "nodes.jsonl", "calls.jsonl"):
        path = sample_dir / name
        if path.exists():
            path.unlink()

    stage_marker = {
        "task_id": task_id,
        "stage_name": stage_name,
        "reopen_node_id": reopen_node_id,
    }
    write_json(sample_dir / f"reopen_{stage_name}.json", stage_marker)


def build_stage_plan(
    *,
    stage_name: str,
    rows: list[dict[str, Any]],
    run_dir: Path,
    pools_by_task: dict[str, list[str]],
    node_max_steps: int,
    leaf_acc_threshold: float,
    random_seed: int,
) -> StagePlan:
    if stage_name == "all_wrong":
        selected_rows = [row for row in rows if int(row.get("correct_leaf_count", 0) or 0) == 0]
    elif stage_name == "below_threshold":
        selected_rows = [row for row in rows if leaf_acc_from_result(row) < leaf_acc_threshold]
    else:
        raise ValueError(f"Unsupported stage name: {stage_name}")

    task_ids: list[str] = []
    fallback_random_pool_task_ids: list[str] = []
    reopen_nodes: dict[str, str] = {}
    leaf_acc_before: dict[str, float] = {}

    for row in selected_rows:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id:
            continue
        leaf_acc_before[task_id] = leaf_acc_from_result(row)
        pool = pools_by_task.get(task_id, [])
        if not pool:
            fallback_random_pool_task_ids.append(task_id)
        sample_dir = run_dir / "samples" / task_id
        latest = load_latest(sample_dir)
        reopen_node_id = choose_reopen_node(
            latest,
            task_id=task_id,
            node_max_steps=node_max_steps,
            random_seed=random_seed,
        )
        if reopen_node_id is None:
            continue
        task_ids.append(task_id)
        reopen_nodes[task_id] = reopen_node_id

    return StagePlan(
        stage_name=stage_name,
        task_ids=task_ids,
        fallback_random_pool_task_ids=fallback_random_pool_task_ids,
        reopen_nodes=reopen_nodes,
        leaf_acc_before=leaf_acc_before,
    )


def build_stage_config_payload(
    *,
    base_snapshot: dict[str, Any],
    output_parent_dir: Path,
    task_ids: list[str],
    all_candidate_models: list[str],
    tree_budget_usd: float,
    frontier_limit: int,
    max_concurrency: int,
) -> dict[str, Any]:
    payload = dict(base_snapshot)
    payload["output_dir"] = str(output_parent_dir)
    payload["task_ids"] = task_ids
    payload["sample_count"] = len(task_ids)
    payload["resume"] = True
    payload["tree_budget_usd"] = tree_budget_usd
    payload["frontier_limit"] = frontier_limit
    payload["max_concurrency"] = max_concurrency
    payload["candidate_models"] = list(all_candidate_models)
    payload.pop("exclude_task_ids", None)
    payload.pop("exclude_task_ids_path", None)
    return payload


async def run_stage(
    *,
    stage_plan: StagePlan,
    run_dir: Path,
    output_parent_dir: Path,
    base_snapshot: dict[str, Any],
    all_candidate_models: list[str],
    tree_budget_usd: float,
    frontier_limit: int,
    max_concurrency: int,
    leaf_acc_threshold: float,
    max_final_leaf_count: int | None,
    pools_by_task: dict[str, list[str]],
    meta_dir: Path,
) -> dict[str, Any]:
    if not stage_plan.task_ids:
        summary = {
            "stage_name": stage_plan.stage_name,
            "selected_task_count": 0,
            "selected_task_ids": [],
            "fallback_random_pool_task_ids": stage_plan.fallback_random_pool_task_ids,
        }
        write_json(meta_dir / f"{stage_plan.stage_name}_summary.json", summary)
        return summary

    for task_id in stage_plan.task_ids:
        prepare_sample_dir_for_stage(
            run_dir / "samples" / task_id,
            task_id=task_id,
            reopen_node_id=stage_plan.reopen_nodes[task_id],
            stage_name=stage_plan.stage_name,
        )

    config_payload = build_stage_config_payload(
        base_snapshot=base_snapshot,
        output_parent_dir=output_parent_dir,
        task_ids=stage_plan.task_ids,
        all_candidate_models=all_candidate_models,
        tree_budget_usd=tree_budget_usd,
        frontier_limit=frontier_limit,
        max_concurrency=max_concurrency,
    )
    stage_config_path = meta_dir / f"{stage_plan.stage_name}.runner.yaml"
    stage_config_path.parent.mkdir(parents=True, exist_ok=True)
    stage_config_path.write_text(
        yaml.safe_dump(config_payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    config = MCTSConfig.load(stage_config_path)
    runner = FixedPoolExpansionRunner(
        config,
        pools_by_task=pools_by_task,
        leaf_acc_threshold=leaf_acc_threshold,
        max_final_leaf_count=max_final_leaf_count,
        progress_desc=f"Expand-{stage_plan.stage_name}",
    )
    records, summary = await runner.run_with_leaf_acc_stop()
    stage_summary = {
        "stage_name": stage_plan.stage_name,
        "selected_task_count": len(stage_plan.task_ids),
        "selected_task_ids": stage_plan.task_ids,
        "reopen_nodes": stage_plan.reopen_nodes,
        "fallback_random_pool_task_ids": stage_plan.fallback_random_pool_task_ids,
        "runner_summary": summary,
        "post_stage_leaf_acc": {
            str(row.get("task_id")): leaf_acc_from_result(row)
            for row in records
            if str(row.get("task_id")) in set(stage_plan.task_ids)
        },
    }
    write_json(meta_dir / f"{stage_plan.stage_name}_summary.json", stage_summary)
    return stage_summary


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    start_stage = args.start_stage or str(config.get("start_stage", "all_wrong"))
    if start_stage not in {"all_wrong", "below_threshold"}:
        raise ValueError("start_stage must be one of: all_wrong, below_threshold")

    run_dir = Path(config.get("run_dir", config["base_run_dir"])).expanduser().resolve()
    pool_summary_path = Path(config["pool_summary_path"]).expanduser().resolve()
    leaf_acc_threshold = float(config.get("leaf_acc_threshold", 0.3))
    tree_budget_usd = float(config.get("tree_budget_usd", 5.0))
    max_final_leaf_count = (
        int(config["max_final_leaf_count"])
        if config.get("max_final_leaf_count") is not None
        else 128
    )
    frontier_limit = int(config.get("frontier_limit", 1))
    max_concurrency = int(config.get("max_concurrency", 1))
    random_seed = int(config.get("random_seed", 42))

    pools_by_task = normalize_model_pools(pool_summary_path)
    base_snapshot = load_json(run_dir / "config.snapshot.json")
    output_parent_dir = run_dir.parent
    all_candidate_models = sorted(
        {
            *[str(item) for item in base_snapshot.get("candidate_models", [])],
            *[model for pool in pools_by_task.values() for model in pool],
        }
    )
    if not all_candidate_models:
        raise ValueError("No candidate models available after merging base snapshot and task pools.")

    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    meta_dir = run_dir / "expansion_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    initial_rows = load_results(run_dir)
    initial_summary = {
        "run_dir": str(run_dir),
        "task_count": len(initial_rows),
        "leaf_acc_threshold": leaf_acc_threshold,
        "tree_budget_usd": tree_budget_usd,
        "max_final_leaf_count": max_final_leaf_count,
        "frontier_limit": frontier_limit,
        "max_concurrency": max_concurrency,
    }
    write_json(meta_dir / "initial_summary.json", initial_summary)

    stage_summaries: list[dict[str, Any]] = []
    stage_plans: list[StagePlan] = []
    if start_stage == "all_wrong":
        stage1 = build_stage_plan(
            stage_name="all_wrong",
            rows=initial_rows,
            run_dir=run_dir,
            pools_by_task=pools_by_task,
            node_max_steps=int(base_snapshot.get("node_max_steps", 8)),
            leaf_acc_threshold=leaf_acc_threshold,
            random_seed=random_seed,
        )
        write_json(meta_dir / "all_wrong_plan.json", stage1.__dict__)
        stage_plans.append(stage1)

        if not args.prepare_only:
            stage_summaries.append(
                asyncio.run(
                    run_stage(
                        stage_plan=stage1,
                        run_dir=run_dir,
                        output_parent_dir=output_parent_dir,
                        base_snapshot=base_snapshot,
                        all_candidate_models=all_candidate_models,
                        tree_budget_usd=tree_budget_usd,
                        frontier_limit=frontier_limit,
                        max_concurrency=max_concurrency,
                        leaf_acc_threshold=leaf_acc_threshold,
                        max_final_leaf_count=max_final_leaf_count,
                        pools_by_task=pools_by_task,
                        meta_dir=meta_dir,
                    )
                )
            )

    rows_for_stage2 = load_results(run_dir) if stage_summaries else initial_rows
    stage2 = build_stage_plan(
        stage_name="below_threshold",
        rows=rows_for_stage2,
        run_dir=run_dir,
        pools_by_task=pools_by_task,
        node_max_steps=int(base_snapshot.get("node_max_steps", 8)),
        leaf_acc_threshold=leaf_acc_threshold,
        random_seed=random_seed + 1,
    )
    write_json(meta_dir / "below_threshold_plan.json", stage2.__dict__)
    stage_plans.append(stage2)

    if args.prepare_only:
        prepared_stage = stage_plans[0]
        for task_id in prepared_stage.task_ids:
            prepare_sample_dir_for_stage(
                run_dir / "samples" / task_id,
                task_id=task_id,
                reopen_node_id=prepared_stage.reopen_nodes[task_id],
                stage_name=prepared_stage.stage_name,
            )
        write_json(
            meta_dir / "prepare_only_summary.json",
            {
                "run_dir": str(run_dir),
                "start_stage": start_stage,
                "prepared_stage": prepared_stage.stage_name,
                "prepared_task_ids": prepared_stage.task_ids,
                "fallback_random_pool_task_ids": prepared_stage.fallback_random_pool_task_ids,
            },
        )
        print(f"prepare_only: true")
        print(f"start_stage: {start_stage}")
        print(f"run_dir: {run_dir}")
        print(f"prepared_task_count: {len(prepared_stage.task_ids)}")
        print(f"meta_dir: {meta_dir}")
        return 0

    stage_summaries.append(
        asyncio.run(
            run_stage(
                stage_plan=stage2,
                run_dir=run_dir,
                output_parent_dir=output_parent_dir,
                base_snapshot=base_snapshot,
                all_candidate_models=all_candidate_models,
                tree_budget_usd=tree_budget_usd,
                frontier_limit=frontier_limit,
                max_concurrency=max_concurrency,
                leaf_acc_threshold=leaf_acc_threshold,
                max_final_leaf_count=max_final_leaf_count,
                pools_by_task=pools_by_task,
                meta_dir=meta_dir,
            )
        )
    )

    final_rows = load_results(run_dir)
    final_summary = {
        "run_dir": str(run_dir),
        "start_stage": start_stage,
        "leaf_acc_threshold": leaf_acc_threshold,
        "tree_budget_usd": tree_budget_usd,
        "max_final_leaf_count": max_final_leaf_count,
        "frontier_limit": frontier_limit,
        "stage_summaries": stage_summaries,
        "final_below_threshold_task_ids": [
            str(row.get("task_id"))
            for row in final_rows
            if leaf_acc_from_result(row) < leaf_acc_threshold
        ],
        "final_leaf_acc": {
            str(row.get("task_id")): leaf_acc_from_result(row)
            for row in final_rows
        },
    }
    write_json(meta_dir / "expansion_summary.json", final_summary)

    stage1_task_count = 0
    if start_stage == "all_wrong":
        stage1_task_count = len(stage_plans[0].task_ids) if stage_plans else 0
    print(f"run_dir: {run_dir}")
    print(f"start_stage: {start_stage}")
    print(f"stage1_task_count: {stage1_task_count}")
    print(f"stage2_task_count: {len(stage2.task_ids)}")
    print(f"expansion_summary_path: {meta_dir / 'expansion_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
