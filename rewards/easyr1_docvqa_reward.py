#!/usr/bin/env python3
"""EasyR1 reward adapter for DocVQA GRPO."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from docvqa_grpo_reward import compute_reward


REWARD_NAME = "docvqa"
REWARD_TYPE = "batch"


def _coerce_ground_truth(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return _coerce_ground_truth(parsed)
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def compute_score(
    reward_inputs: list[dict[str, Any]],
    accuracy_weight: float = 1.5,
    grounding_weight: float = 0.5,
    min_grounding_chars: int = 2,
) -> list[dict[str, float]]:
    scores: list[dict[str, float]] = []
    for reward_input in reward_inputs:
        breakdown = compute_reward(
            reward_input.get("response", ""),
            _coerce_ground_truth(reward_input.get("ground_truth")),
            accuracy_weight=accuracy_weight,
            grounding_weight=grounding_weight,
            min_grounding_chars=min_grounding_chars,
        )
        scores.append(
            {
                "overall": breakdown.reward,
                "accuracy": breakdown.accuracy_reward,
                "grounding": breakdown.grounding_reward,
                "format": 1.0 if breakdown.has_single_answer_block and breakdown.has_think_block else 0.0,
            }
        )

    return scores
