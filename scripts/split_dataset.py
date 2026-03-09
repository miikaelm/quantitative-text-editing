"""
split_dataset.py — Split a validated JSONL into train / test files.

Usage (from repo root, with venv active):
    python scripts/split_dataset.py --jsonl data/testing123/pairs.jsonl \\
                                    --valid-ids data/testing123/pairs_valid.jsonl

    # Or pass valid IDs directly from a prior validation run:
    python scripts/split_dataset.py --jsonl data/color/test/pairs.jsonl \\
                                    --valid-ids data/color/test/pairs_valid.jsonl \\
                                    --output-dir data/color/test \\
                                    --train-ratio 0.85
"""

import argparse
import json
import random
import sys
from pathlib import Path


def split_valid_pairs(
    jsonl_path: Path,
    valid_pair_ids: set[str],
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Read the original JSONL, keep only valid pairs, and split into train/test.

    Returns paths to the written train and test JSONL files.
    """
    valid_lines: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            if record.get("pair_id") in valid_pair_ids:
                valid_lines.append(record)

    random.seed(seed)
    random.shuffle(valid_lines)

    split_idx = int(len(valid_lines) * train_ratio)
    train_lines = valid_lines[:split_idx]
    test_lines = valid_lines[split_idx:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"

    for path, lines in [(train_path, train_lines), (test_path, test_lines)]:
        with open(path, "w") as f:
            for record in lines:
                f.write(json.dumps(record) + "\n")

    print(f"\nSplit {len(valid_lines)} valid pairs -> "
          f"{len(train_lines)} train, {len(test_lines)} test")
    print(f"  train: {train_path}")
    print(f"  test:  {test_path}")

    return train_path, test_path


def _load_valid_ids_from_jsonl(path: Path) -> set[str]:
    ids = set()
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            ids.add(record["pair_id"])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a validated JSONL into train/test.")
    parser.add_argument("--jsonl", type=Path, required=True,
                        help="Source pairs.jsonl (all pairs, pre-split).")
    parser.add_argument("--valid-ids", type=Path, required=True,
                        help="JSONL file containing only the valid pairs "
                             "(e.g. pairs_valid.jsonl produced by validate.py).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write train.jsonl and test.jsonl "
                             "(defaults to the same directory as --jsonl).")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Fraction of valid pairs to put in train (default 0.8).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42).")
    args = parser.parse_args()

    if not args.jsonl.exists():
        print(f"ERROR: {args.jsonl} not found.")
        sys.exit(1)
    if not args.valid_ids.exists():
        print(f"ERROR: {args.valid_ids} not found.")
        sys.exit(1)

    valid_ids = _load_valid_ids_from_jsonl(args.valid_ids)
    out_dir = args.output_dir or args.jsonl.parent
    split_valid_pairs(args.jsonl, valid_ids, out_dir, args.train_ratio, args.seed)


if __name__ == "__main__":
    main()
