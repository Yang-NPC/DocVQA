#!/usr/bin/env python3
"""Baseline DocVQA evaluation for Qwen3-VL."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL on DocVQA.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dataset-name", default="lmms-lab/DocVQA")
    parser.add_argument("--dataset-config", default="DocVQA")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", default="outputs/baseline_qwen3vl8b_docvqa")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--selection", choices=["first", "random", "hard"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--answers-column", default="answers")
    parser.add_argument("--question-id-column", default="questionId")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--thinking-mode", choices=["default", "on", "off"], default="default")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> str | "torch.dtype":
    import torch

    if name == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def normalize_answer(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"[\t\n\r]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (char_a != char_b)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def anls_score(prediction: str, answers: list[str]) -> float:
    pred = normalize_answer(prediction)
    best = 0.0
    for answer in answers:
        gold = normalize_answer(answer)
        max_len = max(len(pred), len(gold))
        if max_len == 0:
            score = 1.0
        else:
            score = 1.0 - (levenshtein_distance(pred, gold) / max_len)
            score = score if score >= 0.5 else 0.0
        best = max(best, score)
    return best


def exact_match(prediction: str, answers: list[str]) -> bool:
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(answer) for answer in answers)


def coerce_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]


def answer_token_count(answer: str) -> int:
    return len(re.findall(r"\w+", normalize_answer(answer)))


def hardness_score(row: dict[str, Any], question_column: str, answers_column: str) -> float:
    question = str(row.get(question_column, ""))
    answers = coerce_answers(row.get(answers_column))
    normalized_answers = [normalize_answer(answer) for answer in answers]
    longest_answer = max((len(answer) for answer in normalized_answers), default=0)
    max_answer_tokens = max((answer_token_count(answer) for answer in normalized_answers), default=0)
    question_tokens = len(re.findall(r"\w+", normalize_answer(question)))
    has_number = any(re.search(r"\d", answer) for answer in normalized_answers)
    has_date_like = any(re.search(r"\b\d{1,4}[-/]\d{1,2}([-/]\d{1,4})?\b", answer) for answer in normalized_answers)
    answer_disagreement = len(set(normalized_answers)) > 1

    score = 0.0
    score += min(question_tokens, 30) * 0.7
    score += min(longest_answer, 60) * 0.5
    score += min(max_answer_tokens, 10) * 3.0
    score += 8.0 if has_number else 0.0
    score += 6.0 if has_date_like else 0.0
    score += 4.0 if answer_disagreement else 0.0
    return score


def select_dataset(dataset: Any, args: argparse.Namespace) -> Any:
    if args.limit is None:
        return dataset

    limit = min(args.limit, len(dataset))
    if args.selection == "first":
        return dataset.select(range(limit))

    indices = list(range(len(dataset)))
    if args.selection == "random":
        random.Random(args.seed).shuffle(indices)
        return dataset.select(indices[:limit])

    scoring_dataset = dataset
    if args.image_column in scoring_dataset.column_names:
        scoring_dataset = scoring_dataset.remove_columns(args.image_column)

    scored_indices = [
        (hardness_score(scoring_dataset[index], args.question_column, args.answers_column), index)
        for index in indices
    ]
    scored_indices.sort(reverse=True)
    return dataset.select([index for _, index in scored_indices[:limit]])


def thinking_instruction(thinking_mode: str) -> str:
    if thinking_mode == "on":
        return "\n/think"
    if thinking_mode == "off":
        return "\n/no_think"
    return ""


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def has_thinking_tag(text: str) -> bool:
    return re.search(r"<think>.*?</think>", text, flags=re.DOTALL | re.IGNORECASE) is not None


def build_messages(
    image: Any,
    question: str,
    thinking_mode: str,
    min_pixels: int | None,
    max_pixels: int | None,
) -> list[dict[str, Any]]:
    if thinking_mode == "on":
        prompt = (
            "Use thinking mode to reason from the document image, then provide the shortest exact answer text.\n"
            f"Question: {question}"
            f"{thinking_instruction(thinking_mode)}"
        )
    else:
        prompt = (
            "Answer the question using only the document image. "
            "Return the shortest exact answer text, without explanation.\n"
            f"Question: {question}"
            f"{thinking_instruction(thinking_mode)}"
        )
    image_content: dict[str, Any] = {"type": "image", "image": image}
    if min_pixels is not None:
        image_content["min_pixels"] = min_pixels
    if max_pixels is not None:
        image_content["max_pixels"] = max_pixels

    return [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": prompt},
            ],
        }
    ]


def apply_chat_template(processor: Any, messages: list[dict[str, Any]], thinking_mode: str) -> str:
    chat_template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if thinking_mode != "default":
        chat_template_kwargs["enable_thinking"] = thinking_mode == "on"
    try:
        return processor.apply_chat_template(messages, **chat_template_kwargs)
    except TypeError:
        chat_template_kwargs.pop("enable_thinking", None)
        return processor.apply_chat_template(messages, **chat_template_kwargs)


def generation_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    generation_kwargs: dict[str, Any] = {"max_new_tokens": args.max_new_tokens}
    if args.temperature > 0.0:
        generation_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
    else:
        generation_kwargs["do_sample"] = False
    if args.top_k is not None:
        generation_kwargs["top_k"] = args.top_k
    if args.min_p is not None:
        generation_kwargs["min_p"] = args.min_p
    return generation_kwargs


def process_batch_vision_info(batch_messages: list[list[dict[str, Any]]]) -> tuple[Any, Any]:
    try:
        return process_vision_info(batch_messages)
    except Exception:
        image_inputs = []
        video_inputs = []
        for messages in batch_messages:
            row_images, row_videos = process_vision_info(messages)
            if row_images:
                image_inputs.extend(row_images)
            if row_videos:
                video_inputs.extend(row_videos)
        return image_inputs or None, video_inputs or None


def generate_answers(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[str]:
    import torch

    batch_messages = [
        build_messages(
            image=row[args.image_column],
            question=str(row[args.question_column]),
            thinking_mode=args.thinking_mode,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        for row in rows
    ]
    texts = [apply_chat_template(processor, messages, args.thinking_mode) for messages in batch_messages]
    image_inputs, video_inputs = process_batch_vision_info(batch_messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(select_input_device(model))

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs_from_args(args))
    generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [prediction.strip() for prediction in decoded]


def build_record(row: dict[str, Any], prediction: str, args: argparse.Namespace) -> tuple[dict[str, Any], bool, float]:
    question = str(row[args.question_column])
    answers = coerce_answers(row.get(args.answers_column))
    scored_prediction = strip_thinking(prediction)
    em = exact_match(scored_prediction, answers)
    anls = anls_score(scored_prediction, answers)

    record = {
        "question_id": row.get(args.question_id_column),
        "question": question,
        "prediction": prediction,
        "scored_prediction": scored_prediction,
        "answers": answers,
        "hardness_score": hardness_score(row, args.question_column, args.answers_column),
        "exact_match": em,
        "anls": anls,
    }
    return record, em, anls


def select_input_device(model: Any) -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    import torch
    from datasets import load_dataset
    from qwen_vl_utils import process_vision_info as imported_process_vision_info
    from tqdm import tqdm
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    globals()["process_vision_info"] = imported_process_vision_info

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs: dict[str, Any] = {
        "dtype": dtype_from_name(args.torch_dtype),
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    dataset = select_dataset(dataset, args)

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    model.eval()

    predictions_path = output_dir / "predictions.jsonl"
    total_exact = 0
    total_anls = 0.0
    thinking_tag_count = 0
    count = 0

    with predictions_path.open("w", encoding="utf-8") as handle:
        batch: list[dict[str, Any]] = []
        progress = tqdm(total=len(dataset), desc="Evaluating")
        for row in dataset:
            batch.append(row)
            if len(batch) < args.batch_size:
                continue

            predictions = generate_answers(model=model, processor=processor, rows=batch, args=args)
            for batch_row, prediction in zip(batch, predictions):
                record, em, anls = build_record(batch_row, prediction, args)
                thinking_tag_count += int(has_thinking_tag(prediction))
                total_exact += int(em)
                total_anls += anls
                count += 1
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress.update(len(batch))
            batch = []

        if batch:
            predictions = generate_answers(model=model, processor=processor, rows=batch, args=args)
            for batch_row, prediction in zip(batch, predictions):
                record, em, anls = build_record(batch_row, prediction, args)
                thinking_tag_count += int(has_thinking_tag(prediction))
                total_exact += int(em)
                total_anls += anls
                count += 1
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress.update(len(batch))
        progress.close()

    metrics = {
        "model_id": args.model_id,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "selection": args.selection,
        "seed": args.seed,
        "thinking_mode": args.thinking_mode,
        "batch_size": args.batch_size,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "num_examples": count,
        "thinking_tag_count": thinking_tag_count,
        "thinking_tag_rate": thinking_tag_count / count if count else 0.0,
        "thinking_tag_warning": (
            "thinking_mode=on but no complete <think>...</think> tags were found in predictions"
            if args.thinking_mode == "on" and count and thinking_tag_count == 0
            else (
                "thinking_mode=on but some predictions are missing complete <think>...</think> tags"
                if args.thinking_mode == "on" and count and thinking_tag_count < count
                else None
            )
        ),
        "missing_think_tag_count": (
            count - thinking_tag_count
            if args.thinking_mode == "on"
            else None
        ),
        "exact_match": total_exact / count if count else 0.0,
        "anls": total_anls / count if count else 0.0,
    }
    if metrics["thinking_tag_warning"]:
        print(f"WARNING: {metrics['thinking_tag_warning']}")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
