#!/usr/bin/env python3
"""Fail-fast check for LLaMA-Factory builds that can run the DocVQA GRPO config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover - only hit in broken environments.
    raise SystemExit("PyYAML is required for this check. Install with: pip install pyyaml") from exc


GRPO_CONFIG_KEYS = {
    "rlhf_algo",
    "num_generations",
    "kl_coef",
}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def has_grpo_support(llamafactory_dir: Path) -> bool:
    """Return true when the checked LLaMA-Factory tree appears to expose GRPO."""
    src_root = llamafactory_dir / "src" / "llamafactory"
    train_root = src_root / "train"

    if (train_root / "grpo").is_dir():
        return True

    search_roots = [
        src_root / "hparams",
        train_root,
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for py_file in root.rglob("*.py"):
            text = _read_text(py_file).lower()
            if "grpo" in text and ("num_generations" in text or "rlhf_algo" in text):
                return True

    return False


def load_config_keys(config_path: Path) -> set[str]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return set(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llamafactory-dir",
        default="/content/LLaMA-Factory",
        help="Path to the LLaMA-Factory checkout.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Training YAML that will be passed to llamafactory-cli train.",
    )
    args = parser.parse_args()

    llamafactory_dir = Path(args.llamafactory_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()

    if not llamafactory_dir.exists():
        print(f"ERROR: LLaMA-Factory checkout was not found: {llamafactory_dir}", file=sys.stderr)
        return 2
    if not config_path.exists():
        print(f"ERROR: Training config was not found: {config_path}", file=sys.stderr)
        return 2

    config_keys = load_config_keys(config_path)
    grpo_keys_in_config = sorted(config_keys & GRPO_CONFIG_KEYS)
    grpo_supported = has_grpo_support(llamafactory_dir)

    if grpo_keys_in_config and not grpo_supported:
        print(
            "ERROR: This LLaMA-Factory checkout does not appear to include GRPO support, "
            f"but the config uses GRPO-only keys: {grpo_keys_in_config}.",
            file=sys.stderr,
        )
        print(
            "Use a GRPO-capable LLaMA-Factory fork/branch and wire "
            "`rewards.docvqa_grpo_reward:reward_func` as the rule-based reward. "
            "Removing these keys would make the run not be GRPO.",
            file=sys.stderr,
        )
        return 1

    if grpo_supported:
        print("OK: GRPO support was detected in this LLaMA-Factory checkout.")
    else:
        print("OK: no GRPO-only config keys were found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
