#!/usr/bin/env python3
"""DocVQA evaluation through a vLLM OpenAI-compatible Qwen3-VL server."""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from eval_docvqa_qwen3vl import (
    anls_score,
    coerce_answers,
    exact_match,
    hardness_score,
    select_dataset,
    strip_thinking,
    thinking_instruction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DocVQA through a vLLM OpenAI-compatible server.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--dataset-name", default="lmms-lab/DocVQA")
    parser.add_argument("--dataset-config", default="DocVQA")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", default="outputs/baseline_qwen3vl8b_docvqa_vllm")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--selection", choices=["first", "random", "hard"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--answers-column", default="answers")
    parser.add_argument("--question-id-column", default="questionId")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--thinking-mode", choices=["default", "on", "off"], default="off")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    return parser.parse_args()


def resize_if_needed(image: Any, max_pixels: int | None) -> Any:
    from PIL import Image

    image = image.convert("RGB")
    if max_pixels is None or image.width * image.height <= max_pixels:
        return image

    scale = (max_pixels / (image.width * image.height)) ** 0.5
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resized = image.copy()
    resized.thumbnail(new_size, Image.Resampling.LANCZOS)
    return resized


def image_to_data_url(image: Any, max_pixels: int | None, jpeg_quality: int) -> str:
    image = resize_if_needed(image, max_pixels)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_prompt(question: str, thinking_mode: str) -> str:
    return (
        "Answer the question using only the document image. "
        "Return the shortest exact answer text, without explanation.\n"
        f"Question: {question}"
        f"{thinking_instruction(thinking_mode)}"
    )


def build_messages(row: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    question = str(row[args.question_column])
    data_url = image_to_data_url(row[args.image_column], args.max_pixels, args.jpeg_quality)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": build_prompt(question, args.thinking_mode)},
            ],
        }
    ]


def extra_body_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    extra_body: dict[str, Any] = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if args.min_p is not None:
        extra_body["min_p"] = args.min_p
    return extra_body or None


def request_prediction(client: Any, row: dict[str, Any], args: argparse.Namespace) -> str:
    request_kwargs: dict[str, Any] = {
        "model": args.model_id,
        "messages": build_messages(row, args),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    extra_body = extra_body_from_args(args)
    if extra_body is not None:
        request_kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**request_kwargs)
    return response.choices[0].message.content.strip()


def build_record(row: dict[str, Any], prediction: str, args: argparse.Namespace) -> tuple[dict[str, Any], bool, float]:
    answers = coerce_answers(row.get(args.answers_column))
    scored_prediction = strip_thinking(prediction)
    em = exact_match(scored_prediction, answers)
    anls = anls_score(scored_prediction, answers)
    record = {
        "question_id": row.get(args.question_id_column),
        "question": str(row[args.question_column]),
        "prediction": prediction,
        "scored_prediction": scored_prediction,
        "answers": answers,
        "hardness_score": hardness_score(row, args.question_column, args.answers_column),
        "exact_match": em,
        "anls": anls,
    }
    return record, em, anls


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")

    from datasets import load_dataset
    from openai import OpenAI
    from tqdm import tqdm

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    dataset = select_dataset(dataset, args)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    predictions_path = output_dir / "predictions.jsonl"

    total_exact = 0
    total_anls = 0.0
    count = 0

    with predictions_path.open("w", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            row_iter = iter(dataset)
            futures = {}

            def submit_next() -> bool:
                try:
                    row = next(row_iter)
                except StopIteration:
                    return False
                futures[executor.submit(request_prediction, client, row, args)] = row
                return True

            for _ in range(args.concurrency * 2):
                if not submit_next():
                    break

            progress = tqdm(total=len(dataset), desc="Evaluating")
            while futures:
                for future in as_completed(futures):
                    row = futures.pop(future)
                    prediction = re.sub(r"\s+", " ", future.result()).strip()
                    record, em, anls = build_record(row, prediction, args)
                    total_exact += int(em)
                    total_anls += anls
                    count += 1
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    progress.update(1)
                    submit_next()
                    break
            progress.close()

    metrics = {
        "backend": "vllm_openai_server",
        "model_id": args.model_id,
        "base_url": args.base_url,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "selection": args.selection,
        "seed": args.seed,
        "thinking_mode": args.thinking_mode,
        "concurrency": args.concurrency,
        "max_pixels": args.max_pixels,
        "num_examples": count,
        "exact_match": total_exact / count if count else 0.0,
        "anls": total_anls / count if count else 0.0,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
