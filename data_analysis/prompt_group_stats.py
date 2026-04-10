#!/usr/bin/env python3
"""Statistics: how many samples share the same prompt in a JSONL dataset.

Prompt identity matches offline REINFORCE++/find_duplicate_prompts: ``messages`` with
trailing assistant turns stripped, or plain fields query/instruction/prompt/input/text.

Usage:
  python data_analysis/prompt_group_stats.py /path/to/data.jsonl
  python data_analysis/prompt_group_stats.py /path/to/data.jsonl --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_DA = Path(__file__).resolve().parent
if str(_DA) not in sys.path:
    sys.path.insert(0, str(_DA))

from find_duplicate_prompts import row_prompt_key  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Count rows per unique prompt (group size distribution).")
    p.add_argument("jsonl", type=Path, help="Path to .jsonl")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON summary")
    args = p.parse_args()

    groups: dict[str, list[int]] = defaultdict(list)
    total = 0
    with args.jsonl.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = row_prompt_key(obj)
            groups[key].append(line_no)
            total += 1

    unique = len(groups)
    sizes = [len(lines) for lines in groups.values()]
    size_hist = Counter(sizes)
    max_size = max(sizes) if sizes else 0

    n_singleton_groups = size_hist.get(1, 0)
    n_multi_groups = unique - n_singleton_groups
    rows_singleton = size_hist.get(1, 0) * 1  # each size-1 group contributes 1 row
    rows_multi = total - rows_singleton

    summary = {
        "file": str(args.jsonl.resolve()),
        "total_rows": total,
        "unique_prompts": unique,
        "groups_with_single_row": n_singleton_groups,
        "groups_with_multiple_rows": n_multi_groups,
        "rows_in_singleton_groups": rows_singleton,
        "rows_in_multi_row_groups": rows_multi,
        "max_group_size": max_size,
        "group_size_histogram": {str(k): size_hist[k] for k in sorted(size_hist.keys())},
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print(f"file: {args.jsonl.resolve()}")
    print(f"total_rows:           {total}")
    print(f"unique_prompts:       {unique}")
    print(f"groups (size == 1):   {n_singleton_groups}  (only one solution / row for that prompt)")
    print(f"groups (size >= 2):   {n_multi_groups}  (same prompt, multiple rows)")
    print(f"rows in singleton groups: {rows_singleton}")
    print(f"rows in multi-row groups:   {rows_multi}")
    print(f"max_group_size:       {max_size}")
    print()
    print("group_size -> num_groups (how many prompts have exactly that many rows)")
    for k in sorted(size_hist.keys()):
        print(f"  size {k:4d}: {size_hist[k]:6d} groups")


if __name__ == "__main__":
    main()
