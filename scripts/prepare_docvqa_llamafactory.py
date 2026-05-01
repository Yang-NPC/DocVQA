#!/usr/bin/env python3
"""Export DocVQA into a local LLaMA-Factory multimodal ShareGPT dataset."""

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
    parser = argparse.ArgumentParser(description="Prepare DocVQA for LLaMA-Factory GRPO.")
    parser.add_argument("--dataset-name", default="lmms-lab/DocVQA")
    parser.add_argument("--dataset-config", default="DocVQA")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="data/docvqa_grpo")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel image-save workers. Use 1 for fully serial export.",
    )
    parser.add_argument(
        "--skip-existing-images",
        action="store_true",
        help="Do not rewrite image files that already exist.",
    )
    return parser.parse_args()


def safe_id(value: Any, fallback: int) -> str:
    text = str(value if value is not None else fallback)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or str(fallback)


def coerce_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
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
    output_name = args.output_name or f"{args.split}.json"
    output_path = output_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    rows: list[dict[str, Any]] = []
    futures: list[concurrent.futures.Future[None]] = []
    executor_cls = concurrent.futures.ThreadPoolExecutor
    max_workers = max(1, args.num_workers)

    with executor_cls(max_workers=max_workers) as executor:
        for index, row in enumerate(tqdm(dataset, desc="Preparing rows")):
            question_id = safe_id(row.get("questionId"), index)
            image_path = image_dir / f"{question_id}.{args.image_format}"
            futures.append(
                executor.submit(
                    save_image,
                    row["image"],
                    image_path,
                    args.image_format,
                    args.skip_existing_images,
                )
            )

            question = str(row["question"])
            answers = coerce_answers(row.get("answers"))
            rows.append(
                {
                    "question_id": question_id,
                    "answers": answers,
                    "messages": [
                        {
                            "role": "user",
                            "content": PROMPT_TEMPLATE.format(question=question),
                        }
                    ],
                    "images": [str(image_path.relative_to(output_dir.parent))],
                }
            )

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Saving images ({max_workers} workers)",
        ):
            future.result()

    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset_info = {
        "docvqa_grpo_train": {
            "file_name": str(output_path.relative_to(output_dir.parent)),
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    }
    (output_dir.parent / "dataset_info.docvqa_grpo.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Wrote dataset info snippet to {output_dir.parent / 'dataset_info.docvqa_grpo.json'}")


if __name__ == "__main__":
    main()
