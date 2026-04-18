#!/usr/bin/env python3
"""Merge N JAX orbax ckpts by simple equal-weight averaging and save as a new ckpt.

Usage:
    uv run python train_scripts/eval/ma_average.py \
        --ckpts <ckpt1> <ckpt2> [<ckpt3> ...] \
        --out <output_ckpt_dir> \
        [--weights 0.5 0.5]
"""
import argparse
import os
# Force CPU-only for the merge itself — weights are 6.6 GB × N, GPU can't fit
# when other runs are active, and averaging is I/O-bound anyway.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp


def load_params(ckpt_dir: Path):
    """Load the /params subdir of an orbax ckpt as a pytree of numpy arrays."""
    p = ckpt_dir / "params"
    if not p.exists():
        p = ckpt_dir
    ckpter = ocp.PyTreeCheckpointer()
    restored = ckpter.restore(str(p.resolve()))
    return restored


def save_params(params, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # ocp expects the target path not to exist
    p = out_dir / "params"
    if p.exists():
        shutil.rmtree(p)
    ckpter = ocp.PyTreeCheckpointer()
    ckpter.save(str(p.resolve()), params)


def merge(params_list, weights):
    def weighted_sum(*xs):
        return sum(w * jnp.asarray(x) for w, x in zip(weights, xs))
    return jax.tree.map(weighted_sum, *params_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=None)
    ap.add_argument("--norm-stats-from", default=None,
                    help="path to norm_stats.json; default: <first-ckpt>/../norm_stats.json")
    args = ap.parse_args()

    n = len(args.ckpts)
    weights = args.weights or [1.0 / n] * n
    assert abs(sum(weights) - 1.0) < 1e-3, f"weights must sum to 1, got {sum(weights)}"

    print(f"merging {n} ckpts with weights {weights}")
    for c, w in zip(args.ckpts, weights):
        print(f"  {w:.3f} × {c}")

    all_params = [load_params(Path(c)) for c in args.ckpts]
    merged = merge(all_params, weights)

    out = Path(args.out).resolve()
    save_params(merged, out)

    # copy norm_stats
    src_ns = Path(args.norm_stats_from) if args.norm_stats_from else (Path(args.ckpts[0]).parent / "norm_stats.json")
    if src_ns.exists():
        shutil.copy(src_ns, out / "norm_stats.json")
        print(f"copied norm_stats from {src_ns}")

    print(f"\n✅ merged ckpt → {out}")


if __name__ == "__main__":
    main()
