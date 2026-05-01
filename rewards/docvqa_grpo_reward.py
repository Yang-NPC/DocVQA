#!/usr/bin/env python3
"""Rule-based DocVQA rewards for GRPO.

The reward is intentionally strict:
- +1.5 when the text inside exactly one <answer>...</answer> block exact-matches
  any ground-truth answer after lowercase/whitespace normalization.
- +0.5 when a sufficiently long ground-truth answer appears inside a complete
  <think>...</think> block.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from typing import Any


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class RewardBreakdown:
    reward: float
    accuracy_reward: float
    grounding_reward: float
    extracted_answer: str | None
    matched_answer: str | None
    answer_in_think: bool
    has_single_answer_block: bool
    has_think_block: bool


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"[\t\n\r]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def coerce_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def extract_answer(completion: str) -> tuple[str | None, bool]:
    matches = ANSWER_RE.findall(completion or "")
    if len(matches) != 1:
        return None, False
    return matches[0].strip(), True


def extract_think(completion: str) -> tuple[str, bool]:
    matches = THINK_RE.findall(completion or "")
    if not matches:
        return "", False
    return "\n".join(match.strip() for match in matches if match.strip()), True


def exact_match_answer(prediction: str | None, answers: list[str]) -> tuple[bool, str | None]:
    if prediction is None:
        return False, None
    normalized_prediction = normalize_text(prediction)
    for answer in answers:
        if normalized_prediction == normalize_text(answer):
            return True, answer
    return False, None


def answer_appears_in_think(think: str, answers: list[str], min_chars: int) -> tuple[bool, str | None]:
    normalized_think = normalize_text(think)
    if not normalized_think:
        return False, None

    for answer in answers:
        normalized_answer = normalize_text(answer)
        if len(normalized_answer) < min_chars:
            continue
        if normalized_answer in normalized_think:
            return True, answer
    return False, None


def compute_reward(
    completion: str,
    answers: Any,
    accuracy_weight: float = 1.5,
    grounding_weight: float = 0.5,
    min_grounding_chars: int = 2,
) -> RewardBreakdown:
    gold_answers = coerce_answers(answers)
    extracted_answer, has_single_answer_block = extract_answer(completion)
    think, has_think_block = extract_think(completion)

    is_exact, matched_answer = exact_match_answer(extracted_answer, gold_answers)
    accuracy_reward = accuracy_weight if is_exact else 0.0

    grounded, grounded_answer = answer_appears_in_think(think, gold_answers, min_grounding_chars)
    grounding_reward = grounding_weight if grounded else 0.0

    return RewardBreakdown(
        reward=accuracy_reward + grounding_reward,
        accuracy_reward=accuracy_reward,
        grounding_reward=grounding_reward,
        extracted_answer=extracted_answer,
        matched_answer=matched_answer or grounded_answer,
        answer_in_think=grounded,
        has_single_answer_block=has_single_answer_block,
        has_think_block=has_think_block,
    )


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        if completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content", ""))
        return "\n".join(str(item) for item in completion)
    return str(completion)


def get_answer_list_for_index(index: int, kwargs: dict[str, Any]) -> list[str]:
    for key in ("answers", "answer", "ground_truth", "ground_truths", "labels"):
        if key not in kwargs:
            continue
        value = kwargs[key]
        if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
            return coerce_answers(value[index])
        if isinstance(value, list) and len(value) == len(kwargs.get("completions", value)):
            item = value[index]
            if isinstance(item, (list, tuple, str)):
                return coerce_answers(item)
        return coerce_answers(value)
    return []


def reward_func(completions: list[Any], **kwargs: Any) -> list[float]:
    """TRL/LLaMA-Factory-style reward function.

    Expected kwargs should include one of: answers, answer, ground_truth,
    ground_truths, labels. Each item can be a string or a list of valid answers.
    """
    kwargs = dict(kwargs)
    kwargs["completions"] = completions
    rewards: list[float] = []
    for index, completion in enumerate(completions):
        answers = get_answer_list_for_index(index, kwargs)
        breakdown = compute_reward(completion_to_text(completion), answers)
        rewards.append(breakdown.reward)
    return rewards


def compute_score(completion: str, answers: Any, **kwargs: Any) -> float:
    """Small adapter for frameworks that expect a scalar score function."""
    return compute_reward(completion, answers, **kwargs).reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one DocVQA completion.")
    parser.add_argument("--completion", required=True)
    parser.add_argument("--answers", required=True, help="JSON list or a single answer string.")
    args = parser.parse_args()

    try:
        answers = json.loads(args.answers)
    except json.JSONDecodeError:
        answers = [args.answers]

    print(json.dumps(asdict(compute_reward(args.completion, answers)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

