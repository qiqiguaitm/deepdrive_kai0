#!/usr/bin/env python3
"""Deterministic train/val episode split for stage classifier.

Reads advantage/ episodes.jsonl, outputs stage_classifier_split.json with
train/val episode indices (90/10). Seed=42 for reproducibility.

Usage:
    python train_scripts/data/split_advantage_stage.py
"""
import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/advantage",
        help="Advantage dataset path",
    )
    parser.add_argument(
        "--output",
        default="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/stage_classifier_split.json",
        help="Output split JSON path",
    )
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source)
    episodes_file = source / "meta" / "episodes.jsonl"
    if not episodes_file.exists():
        raise SystemExit(f"episodes.jsonl not found at {episodes_file}")

    # Read all episode indices
    episode_indices = []
    with open(episodes_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            episode_indices.append(int(ep["episode_index"]))
    episode_indices = sorted(set(episode_indices))
    n_total = len(episode_indices)

    # Deterministic shuffle
    rng = random.Random(args.seed)
    shuffled = episode_indices.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(round(n_total * args.val_ratio)))
    val_eps = sorted(shuffled[:n_val])
    train_eps = sorted(shuffled[n_val:])

    # Sanity
    assert len(train_eps) + len(val_eps) == n_total
    assert not (set(train_eps) & set(val_eps))

    out = {
        "source": str(source),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "total_episodes": n_total,
        "n_train": len(train_eps),
        "n_val": len(val_eps),
        "train_episodes": train_eps,
        "val_episodes": val_eps,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[split] total={n_total}, train={len(train_eps)}, val={len(val_eps)}")
    print(f"[split] saved to {output_path}")
    print(f"[split] val sample: {val_eps[:10]}...")


if __name__ == "__main__":
    main()
