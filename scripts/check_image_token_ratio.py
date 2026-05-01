#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from transformers import AutoProcessor


JsonDict = dict[str, Any]


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


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q * 100.0))


def summarize(values: list[float]) -> JsonDict:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how many image tokens are used by a multimodal training JSONL, "
            "using the same processor that training uses."
        )
    )
    parser.add_argument("data", type=Path, help="Path to a JSONL dataset with messages/images fields.")
    parser.add_argument(
        "--model",
        required=True,
        help="Model name or local model path for AutoProcessor, e.g. /path/to/qwen3-vl-8b-instruct",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of rows to inspect.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load the processor from local cache/files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the summary JSON.",
    )
    return parser


def load_images(image_paths: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def format_messages(processor: Any, messages: list[JsonDict]) -> str:
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def compute_image_tokens(processor: Any, pil_images: list[Image.Image]) -> int:
    image_inputs = processor.image_processor(images=pil_images, return_tensors="pt")
    image_grid_thw = image_inputs.get("image_grid_thw")
    if image_grid_thw is None:
        return 0
    merge_length = int(processor.image_processor.merge_size) ** 2
    return int((image_grid_thw.prod(dim=-1) // merge_length).sum().item())


def main() -> int:
    args = build_arg_parser().parse_args()

    rows = load_jsonl(args.data.expanduser().resolve())
    if args.max_rows is not None:
        rows = rows[: int(args.max_rows)]

    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
    )

    image_token_counts: list[int] = []
    total_token_counts: list[int] = []
    text_token_counts: list[int] = []
    image_ratios: list[float] = []
    per_row_preview: list[JsonDict] = []
    error_count = 0

    for row in rows:
        task_id = str(row.get("task_id") or "")
        node_id = str(row.get("node_id") or "")
        messages = list(row.get("messages") or [])
        image_paths = [str(item) for item in list(row.get("images") or []) if str(item)]
        try:
            chat_text = format_messages(processor, messages)
            if image_paths:
                pil_images = load_images(image_paths)
                image_tokens = compute_image_tokens(processor, pil_images)
                encoded = processor(text=[chat_text], images=pil_images, return_tensors="pt")
            else:
                image_tokens = 0
                encoded = processor(text=[chat_text], return_tensors="pt")

            total_tokens = int(encoded["input_ids"].shape[-1])
            text_tokens = max(0, total_tokens - image_tokens)
            ratio = (image_tokens / total_tokens) if total_tokens else 0.0

            image_token_counts.append(image_tokens)
            total_token_counts.append(total_tokens)
            text_token_counts.append(text_tokens)
            image_ratios.append(ratio)

            if len(per_row_preview) < 10:
                per_row_preview.append(
                    {
                        "task_id": task_id,
                        "node_id": node_id,
                        "image_count": len(image_paths),
                        "image_tokens": image_tokens,
                        "text_tokens": text_tokens,
                        "total_tokens": total_tokens,
                        "image_ratio": ratio,
                    }
                )
        except Exception as exc:
            error_count += 1
            if len(per_row_preview) < 10:
                per_row_preview.append(
                    {
                        "task_id": task_id,
                        "node_id": node_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    sum_image_tokens = int(sum(image_token_counts))
    sum_total_tokens = int(sum(total_token_counts))
    weighted_image_ratio = (sum_image_tokens / sum_total_tokens) if sum_total_tokens else 0.0

    summary: JsonDict = {
        "data": str(args.data.expanduser().resolve()),
        "model": str(args.model),
        "rows_checked": len(rows),
        "rows_succeeded": len(image_token_counts),
        "rows_failed": error_count,
        "image_tokens": summarize([float(v) for v in image_token_counts]),
        "text_tokens": summarize([float(v) for v in text_token_counts]),
        "total_tokens": summarize([float(v) for v in total_token_counts]),
        "image_ratio_per_sample": summarize([float(v) for v in image_ratios]),
        "image_ratio_weighted_by_tokens": weighted_image_ratio,
        "sum_image_tokens": sum_image_tokens,
        "sum_total_tokens": sum_total_tokens,
        "preview": per_row_preview,
    }

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)

    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
