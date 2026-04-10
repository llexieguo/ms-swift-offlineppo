#!/usr/bin/env python3
"""Find rows in a JSONL file that share the same prompt (conversation prefix).

Prompt = `messages` with trailing assistant turns removed (standard for chat data
with a separate `answer` field). Falls back to query/instruction/text if present.

Usage:
  python data_analysis/find_duplicate_prompts.py /path/to/data.jsonl
  python data_analysis/find_duplicate_prompts.py /path/to/data.jsonl --min-count 2 --top 50
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    # multimodal list of {type,text/image_url}
    return json.dumps(content, ensure_ascii=True, sort_keys=True)


def messages_to_prompt_prefix(messages: List[Dict[str, Any]]) -> Tuple[str, ...]:
    """Turn messages into a hashable prompt: strip trailing assistant role turns."""
    m = [dict(x) for x in messages]
    while m and m[-1].get("role") == "assistant":
        m.pop()
    return tuple(
        (str(x.get("role", "")), normalize_content(x.get("content"))) for x in m)


def row_prompt_key(obj: Dict[str, Any]) -> str:
    msgs = obj.get("messages")
    if msgs and isinstance(msgs, list):
        return json.dumps(messages_to_prompt_prefix(msgs), ensure_ascii=True)

    for k in ("query", "instruction", "prompt", "input", "text"):
        if k in obj and obj[k] is not None:
            return f"plain:{k}:" + normalize_content(obj[k])

    return json.dumps(obj, ensure_ascii=True, sort_keys=True)


def main() -> None:
    p = argparse.ArgumentParser(description="List duplicate prompts in JSONL")
    p.add_argument("jsonl", type=Path, help="Path to .jsonl")
    p.add_argument("--min-count", type=int, default=2, help="Minimum group size to print (default: 2)")
    p.add_argument("--top", type=int, default=200, help="Max duplicate groups to print (default: 200)")
    p.add_argument("--show-lines", action="store_true", help="Print 1-based line numbers for each group")
    args = p.parse_args()

    groups: Dict[str, List[int]] = defaultdict(list)
    n = 0
    with args.jsonl.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = row_prompt_key(obj)
            groups[key].append(line_no)
            n += 1

    dupes = [(k, lines) for k, lines in groups.items() if len(lines) >= args.min_count]
    dupes.sort(key=lambda x: len(x[1]), reverse=True)

    total_dup_lines = sum(len(lines) for _, lines in dupes)
    print(f"rows={n} unique_prompts={len(groups)} duplicate_groups(>={args.min_count})={len(dupes)}")
    print(f"lines_in_duplicate_groups={total_dup_lines}")
    print()

    def preview_key(key: str, max_len: int = 240) -> str:
        if len(key) <= max_len:
            return key
        return key[: max_len - 3] + "..."

    for i, (key, lines) in enumerate(dupes[: args.top]):
        print(f"--- group {i + 1} count={len(lines)} ---")
        print(preview_key(key))
        if args.show_lines:
            head = lines[:20]
            tail = "" if len(lines) <= 20 else f" ... (+{len(lines) - 20} more)"
            print(f"lines: {head}{tail}")
        print()


if __name__ == "__main__":
    main()
