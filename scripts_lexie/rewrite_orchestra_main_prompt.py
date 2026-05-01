#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rewrite old MCTS-exported MainAgent prompts to the current mas_orchestra prompt template.

This is intended for offline RL datasets whose `messages[-1].content` still uses the
older "3-phase delegation" prompt, while evaluation now uses the newer
`mas_orchestra.prompts.main_reasoning.build_main_prompt` template.

The script preserves:
  - question
  - options
  - prior delegate steps
  - step index
  - leading <image> placeholders

and rewrites only the prompt wrapper / policy text so the training prompt distribution
matches the current reasoning runtime more closely.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable


CURRENT_CONFIG = Path("/mnt/shared-storage-user/xuexiangyuan/workspace/mas_orchestra/configs/reasoning_gpt.yaml")
MAS_ORCHESTRA_ROOT = Path("/mnt/shared-storage-user/xuexiangyuan/workspace/mas_orchestra")


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _import_prompt_builder(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    from mas_orchestra.prompts.main_reasoning import build_main_prompt  # type: ignore

    return build_main_prompt


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        raise ValueError(f"Missing marker: {start_marker!r}")
    start += len(start_marker)
    end = text.find(end_marker, start)
    if end < 0:
        raise ValueError(f"Missing marker after {start_marker!r}: {end_marker!r}")
    return text[start:end].strip()


def _parse_lettered_options(options_text: str) -> list[str]:
    options: list[str] = []
    for raw_line in options_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^[A-Z]\.\s*(.*)$", line)
        if not match:
            raise ValueError(f"Invalid option line: {raw_line!r}")
        options.append(match.group(1).strip())
    if not options:
        raise ValueError("No options parsed from OPTIONS block")
    return options


def _parse_old_prompt(text: str) -> dict:
    image_prefix_match = re.match(r"^(?P<prefix>(?:<image>)+\n*)", text)
    image_prefix = image_prefix_match.group("prefix") if image_prefix_match else ""
    body = text[len(image_prefix):] if image_prefix else text

    step_match = re.search(r"Step\s+(\d+)\s+of\s+(\d+)\s+\((\d+)\s+remaining\)", body)
    if not step_match:
        raise ValueError("Failed to parse step index / max steps")
    step_index = int(step_match.group(1))

    question = _extract_section(body, "QUESTION:\n", "\n\nOPTIONS:\n")
    options_text = _extract_section(body, "OPTIONS:\n", "\n\nPRIOR DELEGATE STEPS:\n")

    end_marker = "\n\nOutput JSON only. Exactly one of:"
    if end_marker not in body:
        end_marker = "\nOutput JSON only. Exactly one of:"
    prior_steps_text = _extract_section(body, "PRIOR DELEGATE STEPS:\n", end_marker)

    return {
        "image_prefix": image_prefix,
        "step_index": step_index,
        "question": question,
        "options": _parse_lettered_options(options_text),
        "prior_steps_text": prior_steps_text,
    }


def _iter_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input jsonl with old MainAgent prompts.")
    ap.add_argument("--output", required=True, help="Output jsonl with rewritten prompts.")
    ap.add_argument(
        "--config",
        default=str(CURRENT_CONFIG),
        help="mas_orchestra reasoning config used to source current sub_models / max_steps.",
    )
    ap.add_argument(
        "--mas_orchestra_root",
        default=str(MAS_ORCHESTRA_ROOT),
        help="Path to mas_orchestra repo for importing the current prompt builder.",
    )
    ap.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max_steps in rewritten prompt. Defaults to config.max_steps.",
    )
    ap.add_argument(
        "--keep_old_max_steps",
        action="store_true",
        help="Preserve the old prompt's max_steps instead of using current config.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    config = _load_yaml(Path(args.config))
    build_main_prompt = _import_prompt_builder(Path(args.mas_orchestra_root))

    sub_models = [str(x) for x in config.get("sub_models", []) or []]
    default_max_steps = int(args.max_steps or config.get("max_steps") or 8)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    rewritten = 0
    unchanged = 0
    failures = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for row in _iter_rows(in_path):
            total += 1
            try:
                messages = row.get("messages") or []
                if not messages or messages[-1].get("role") != "user":
                    raise ValueError("Expected last message to be a user prompt")

                user_text = str(messages[-1].get("content") or "")
                parsed = _parse_old_prompt(user_text)

                old_max_steps_match = re.search(r"Step\s+\d+\s+of\s+(\d+)\s+\(\d+\s+remaining\)", user_text)
                old_max_steps = int(old_max_steps_match.group(1)) if old_max_steps_match else default_max_steps
                max_steps = old_max_steps if args.keep_old_max_steps else default_max_steps
                step_index = min(parsed["step_index"], max_steps)

                sample = SimpleNamespace(question=parsed["question"], options=parsed["options"])
                new_body = build_main_prompt(
                    sample=sample,
                    step_history_text=parsed["prior_steps_text"],
                    step_index=step_index,
                    max_steps=max_steps,
                    sub_models=sub_models,
                    force_submit=False,
                )

                new_text = f"{parsed['image_prefix']}{new_body}" if parsed["image_prefix"] else new_body
                if new_text == user_text:
                    unchanged += 1
                else:
                    rewritten += 1

                messages = list(messages)
                messages[-1] = dict(messages[-1])
                messages[-1]["content"] = new_text
                row["messages"] = messages
                row["_prompt_rewrite_meta"] = {
                    "source": "rewrite_orchestra_main_prompt.py",
                    "config": str(Path(args.config)),
                    "sub_models": sub_models,
                    "max_steps": max_steps,
                    "step_index": step_index,
                }
            except Exception as exc:
                failures += 1
                row["_prompt_rewrite_error"] = str(exc)

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[done] wrote {total} rows to {out_path}")
    print(f"[stats] rewritten={rewritten} unchanged={unchanged} failures={failures}")
    print(f"[config] sub_models={len(sub_models)} max_steps={default_max_steps}")


if __name__ == "__main__":
    main()
