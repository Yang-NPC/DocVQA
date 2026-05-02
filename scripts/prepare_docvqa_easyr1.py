#!/usr/bin/env python3
"""Export DocVQA into an EasyR1-compatible local JSONL image-text dataset."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm


PROMPT_TEMPLATE = """<image>Answer the document question.

You must output exactly:
<think>brief visual evidence or OCR evidence you used</think>
<answer>the shortest exact answer text</answer>

Question: {question}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DocVQA for EasyR1 GRPO.")
    parser.add_argument("--dataset-name", default="lmms-lab/DocVQA")
    parser.add_argument("--dataset-config", default="DocVQA")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="data/docvqa_easyr1")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--skip-existing-images", action="store_true")
    return parser.parse_args()


def safe_id(value: Any, fallback: int) -> str:
    text = str(value if value is not None else fallback)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or str(fallback)


def coerce_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def save_image(image: Any, path: Path, image_format: str, skip_existing: bool) -> None:
    if skip_existing and path.exists() and path.stat().st_size > 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    source_path = Path(getattr(image, "filename", "") or "")
    if source_path.exists() and source_path.is_file():
        source_suffix = source_path.suffix.lower().lstrip(".")
        if source_suffix in {"jpg", "jpeg"} and image_format == "jpg":
            shutil.copyfile(source_path, path)
            return
        if source_suffix == image_format:
            shutil.copyfile(source_path, path)
            return

    if image_format == "jpg":
        image.convert("RGB").save(path, format="JPEG", quality=90, optimize=False)
    else:
        image.save(path, format="PNG")


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images" / args.split
    output_name = args.output_name or f"{args.split}.jsonl"
    output_path = output_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    rows: list[dict[str, Any]] = []
    futures: list[concurrent.futures.Future[None]] = []
    max_workers = max(1, args.num_workers)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in enumerate(tqdm(dataset, desc="Preparing EasyR1 rows")):
            question_id = safe_id(row.get("questionId"), index)
            image_path = image_dir / f"{question_id}.{args.image_format}"
            relative_image_path = image_path.relative_to(output_dir)
            futures.append(
                executor.submit(
                    save_image,
                    row["image"],
                    image_path,
                    args.image_format,
                    args.skip_existing_images,
                )
            )

            rows.append(
                {
                    "question_id": question_id,
                    "prompt": PROMPT_TEMPLATE.format(question=str(row["question"])),
                    "answer": coerce_answers(row.get("answers")),
                    "images": [str(relative_image_path)],
                }
            )

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Saving images ({max_workers} workers)",
        ):
            future.result()

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Use EasyR1 data.image_dir={output_dir.resolve()}")


if __name__ == "__main__":
    main()
