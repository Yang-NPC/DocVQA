#!/usr/bin/env python3
"""Patch EasyR1's hardcoded StatefulDataLoader num_workers value."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--easyr1-dir", default="/content/EasyR1")
    parser.add_argument("--num-workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_loader_path = Path(args.easyr1_dir) / "verl" / "trainer" / "data_loader.py"
    if not data_loader_path.exists():
        raise FileNotFoundError(f"EasyR1 data loader not found: {data_loader_path}")

    text = data_loader_path.read_text(encoding="utf-8")
    patched, count = re.subn(r"num_workers=\d+", f"num_workers={args.num_workers}", text)
    if count == 0:
        raise RuntimeError(f"No num_workers assignment found in {data_loader_path}")

    data_loader_path.write_text(patched, encoding="utf-8")
    print(f"Patched {count} DataLoader num_workers assignment(s) to {args.num_workers}: {data_loader_path}")


if __name__ == "__main__":
    main()
