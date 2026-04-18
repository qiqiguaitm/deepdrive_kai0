#!/usr/bin/env python3
"""
Generate a LeRobot v2.1 compliant episodes_stats.jsonl for a Task_E split.

kai0 training uses its own norm_stats.json (from compute_norm_states_fast.py),
so per-episode stats here only need to satisfy LeRobot's schema. We compute
real stats for scalar/vector features from each episode's parquet, and use
placeholder stats (gray image) for video features.

Required shapes: count=(1,); image stats=(3,1,1); 1-D features → (1,); N-D → (N,).

Usage:
    python scripts/generate_episodes_stats.py /data1/.../Task_E/base
    python scripts/generate_episodes_stats.py /data1/.../Task_E/val
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def feature_stats_1d_or_nd(arr: np.ndarray, keepdims_when_1d: bool = True) -> dict:
    """Mirrors LeRobot compute_episode_stats for non-image features."""
    keepdims = arr.ndim == 1 and keepdims_when_1d
    return {
        "min": np.min(arr, axis=0, keepdims=keepdims).astype(np.float64),
        "max": np.max(arr, axis=0, keepdims=keepdims).astype(np.float64),
        "mean": np.mean(arr, axis=0, keepdims=keepdims).astype(np.float64),
        "std": np.std(arr, axis=0, keepdims=keepdims).astype(np.float64),
        "count": np.array([len(arr)], dtype=np.int64),
    }


def image_placeholder_stats(count: int) -> dict:
    """Minimal valid placeholder for video features: gray image, shape (3,1,1)."""
    gray = np.full((3, 1, 1), 0.5, dtype=np.float64)
    return {
        "min": np.zeros((3, 1, 1), dtype=np.float64),
        "max": np.ones((3, 1, 1), dtype=np.float64),
        "mean": gray,
        "std": np.full((3, 1, 1), 0.25, dtype=np.float64),
        "count": np.array([count], dtype=np.int64),
    }


def jsonify(stats: dict) -> dict:
    """Convert numpy arrays to JSON-serializable lists."""
    out = {}
    for k, v in stats.items():
        out[k] = v.tolist()
    return out


def compute_for_episode(parquet_path: Path, info: dict, ep_length: int) -> dict:
    t = pq.read_table(parquet_path).to_pandas()
    stats = {}
    for fkey, fspec in info["features"].items():
        dtype = fspec["dtype"]
        if dtype in ("image", "video"):
            stats[fkey] = jsonify(image_placeholder_stats(ep_length))
            continue
        # scalar/vector features: take column name matching feature key
        col_name = fkey  # columns are named by feature key
        if col_name not in t.columns:
            # e.g. observation.state ↔ column 'observation.state' — should exist
            # Fallback: skip silently (would break schema but shouldn't happen)
            raise KeyError(f"parquet missing column {col_name}")
        col = t[col_name]
        # pandas may store list-typed values for vector features
        first = col.iloc[0]
        if isinstance(first, (list, np.ndarray)):
            arr = np.stack([np.asarray(v) for v in col.values])
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
        else:
            arr = np.asarray(col.values)
            # For 1-D (e.g. timestamp [T]), get_feature_stats keeps dim via keepdims=True
        stats[fkey] = jsonify(feature_stats_1d_or_nd(arr))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir", help="path to e.g. Task_E/base (contains data/ meta/)")
    args = ap.parse_args()
    split_dir = Path(args.split_dir)
    info = json.loads((split_dir / "meta" / "info.json").read_text())
    episodes = [json.loads(l) for l in (split_dir / "meta" / "episodes.jsonl").read_text().splitlines()]
    out_path = split_dir / "meta" / "episodes_stats.jsonl"
    with out_path.open("w") as f:
        for ep in episodes:
            idx = ep["episode_index"]
            chunk = idx // info["chunks_size"]
            parquet = split_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{idx:06d}.parquet"
            stats = compute_for_episode(parquet, info, ep["length"])
            f.write(json.dumps({"episode_index": idx, "stats": stats}) + "\n")
            print(f"  ep {idx}  len={ep['length']}  features={len(stats)}")
    print(f"✅ wrote {out_path} ({len(episodes)} episodes)")


if __name__ == "__main__":
    main()
