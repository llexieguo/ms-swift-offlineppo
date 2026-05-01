#!/usr/bin/env python3

# python3 merge_judge_into_msswift_jsonl.py \
#   /path/to/input/msswift_xxx.jsonl \
#   /path/to/judge_details.jsonl \
#   --output-jsonl /path/to/output/merged.jsonl

# python3 merge_judge_into_msswift_jsonl.py \
#   /mnt/gpfs/xuexiangyuan/workspace/ms-swift-offlineppo/data/mcts_v3_gpt_llm/msswift_ppo.jsonl \
#   /mnt/gpfs/xuexiangyuan/workspace/ms-swift-offlineppo/data/mcts_v3_gpt_llm/judge_details.jsonl \
#   --output-jsonl /mnt/gpfs/xuexiangyuan/workspace/ms-swift-offlineppo/data/mcts_v3_gpt_llm/acc_llm.jsonl
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


JsonDict = dict[str, Any]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list[JsonDict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def resolve_judge_details_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser().resolve()
    if path.is_dir():
        candidate = path / "judge_details.jsonl"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Missing judge_details.jsonl under {path}")
    if not path.exists():
        raise FileNotFoundError(f"Judge details path not found: {path}")
    return path


def insert_after_key(record: JsonDict, after_key: str, new_items: list[tuple[str, Any]]) -> JsonDict:
    if after_key not in record:
        updated = dict(record)
        for key, value in new_items:
            updated[key] = value
        return updated

    updated: JsonDict = {}
    inserted = False
    for key, value in record.items():
        updated[key] = value
        if key == after_key:
            for new_key, new_value in new_items:
                updated[new_key] = new_value
            inserted = True
    if not inserted:
        for new_key, new_value in new_items:
            updated[new_key] = new_value
    return updated


def judge_row_key(row: JsonDict) -> tuple[str, str, str]:
    return (
        str(row.get("judge_kind") or ""),
        str(row.get("task_id") or ""),
        str(row.get("node_id") or ""),
    )


def load_latest_judge_rows(path: Path) -> dict[tuple[str, str, str], JsonDict]:
    latest_by_job: dict[tuple[str, str, str], JsonDict] = {}
    for row in load_jsonl(path):
        latest_by_job[judge_row_key(row)] = row
    return latest_by_job


def merge_judge_into_rows(rows: list[JsonDict], latest_judge_rows: dict[tuple[str, str, str], JsonDict]) -> list[JsonDict]:
    merged_rows: list[JsonDict] = []
    for row in rows:
        task_id = str(row.get("task_id") or "")
        node_id = str(row.get("node_id") or "")
        main = latest_judge_rows.get(("main", task_id, node_id))
        delegate = latest_judge_rows.get(("delegate", task_id, node_id))
        base_reward = row.get("expected_acc_reward")
        reward_value: float | None = None
        try:
            if base_reward is not None:
                reward_value = float(base_reward)
        except Exception:
            reward_value = None

        main_score = main.get("score") if main is not None else None
        if reward_value is not None and main_score is not None:
            reward_value = reward_value - 0.1 * float(main_score)

        updated = insert_after_key(
            dict(row),
            "expected_acc_reward",
            [
                ("reward", reward_value),
                ("llm_judge_main_score", main_score),
                ("llm_judge_delegate_score", delegate.get("score") if delegate is not None else None),
            ],
        )
        merged_rows.append(updated)
    return merged_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge judge_details.jsonl into an existing msswift_*.jsonl using task_id/node_id keys."
    )
    parser.add_argument("input_jsonl", type=Path, help="Path to an existing msswift_*.jsonl file.")
    parser.add_argument(
        "judge_details",
        type=Path,
        help="Path to judge_details.jsonl, or a directory that contains judge_details.jsonl.",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Path to write the merged JSONL.")
    parser.add_argument(
        "--copy-summary-from",
        type=Path,
        help="Optional summary.json to copy next to the output JSONL with merge metadata.",
    )
    parser.add_argument(
        "--copy-judge-details",
        action="store_true",
        help="Also copy judge_details.jsonl next to the output JSONL.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_jsonl = args.input_jsonl.expanduser().resolve()
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")
    judge_details_path = resolve_judge_details_path(args.judge_details)
    output_jsonl = args.output_jsonl.expanduser().resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_jsonl)
    latest_judge_rows = load_latest_judge_rows(judge_details_path)
    merged_rows = merge_judge_into_rows(rows, latest_judge_rows)
    dump_jsonl(output_jsonl, merged_rows)

    main_nonnull = sum(1 for row in merged_rows if row.get("llm_judge_main_score") is not None)
    delegate_nonnull = sum(1 for row in merged_rows if row.get("llm_judge_delegate_score") is not None)
    summary_payload: JsonDict = {
        "source_input_jsonl": str(input_jsonl),
        "judge_details_path": str(judge_details_path),
        "output_jsonl": str(output_jsonl),
        "input_row_count": len(rows),
        "output_row_count": len(merged_rows),
        "main_score_nonnull_count": main_nonnull,
        "delegate_score_nonnull_count": delegate_nonnull,
        "merge_rule": "For each (judge_kind, task_id, node_id), the latest row in judge_details.jsonl wins.",
    }

    if args.copy_summary_from is not None:
        source_summary_path = args.copy_summary_from.expanduser().resolve()
        source_summary = load_json(source_summary_path)
        summary_payload["copied_from_summary_json"] = str(source_summary_path)
        summary_payload["source_summary"] = source_summary

    summary_path = output_jsonl.with_name("summary.json")
    dump_json(summary_path, summary_payload)

    if args.copy_judge_details:
        copied_judge_details_path = output_jsonl.with_name("judge_details.jsonl")
        if copied_judge_details_path != judge_details_path:
            shutil.copy2(judge_details_path, copied_judge_details_path)
        summary_payload["copied_judge_details_path"] = str(copied_judge_details_path)
        dump_json(summary_path, summary_payload)

    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
