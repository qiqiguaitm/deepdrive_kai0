#!/usr/bin/env python3
"""Evaluate a trained policy on a deterministic held-out episode subset.

Unlike eval_val_action_mse.py which needs a pre-split val-root directory,
this script takes the full LeRobot repo and picks held-out episodes via
fixed seed+ratio, so we can compare multiple models on the *same* held-out set.

Reports mae_joint_{1,10,50} (rad) and mae_grip_{1,10,50} (m) matching the
definition in scripts/train.py:320 (joint dims = 0-5,7-12; grip dims = 6,13).

Usage:
    cd kai0/
    uv run python /vePFS/tim/workspace/deepdive_kai0/train_scripts/eval/eval_heldout_action_mse.py \
        --config pi05_flatten_fold_awbc \
        --ckpt   checkpoints/pi05_flatten_fold_awbc/awbc_v1/99999 \
        --val-repo data/Task_A/advantage \
        --prompt-mode awbc \
        --n-eval-ep 30 --n-sample-frames 50
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HORIZONS = (1, 10, 50)
JOINT_DIMS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIP_DIMS = np.array([6, 13])


def read_video_frames(path: Path, n_frames: int) -> np.ndarray:
    import av
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    out = []
    for frame in container.decode(stream):
        out.append(frame.to_ndarray(format="rgb24"))
        if len(out) >= n_frames:
            break
    container.close()
    arr = np.stack(out[:n_frames], axis=0)
    if arr.shape[0] < n_frames:
        pad = np.repeat(arr[-1:], n_frames - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    return arr


def select_heldout(repo_root: Path, seed: int, ratio: float, n_eval_ep: int | None) -> list[int]:
    lines = (repo_root / "meta" / "episodes.jsonl").read_text().splitlines()
    ep_indices = [json.loads(l)["episode_index"] for l in lines]
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ep_indices).tolist()
    n_heldout = max(1, int(len(ep_indices) * ratio))
    heldout = sorted(shuffled[:n_heldout])
    if n_eval_ep is not None:
        heldout = heldout[:n_eval_ep]
    return heldout


def load_tasks_map(repo_root: Path) -> dict[int, str]:
    m: dict[int, str] = {}
    for l in (repo_root / "meta" / "tasks.jsonl").read_text().splitlines():
        d = json.loads(l)
        m[int(d["task_index"])] = d["task"]
    return m


def eval_ckpt(
    config_name: str,
    ckpt: Path,
    repo_root: Path,
    heldout_eps: list[int],
    n_sample_frames: int,
    prompt_mode: str,
    fixed_prompt: str,
    norm_stats_json: Path | None = None,
) -> dict:
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _train_config

    print(f"[load] config={config_name}  ckpt={ckpt}  val_repo={repo_root.name}  prompt_mode={prompt_mode}")
    train_cfg = _train_config.get_config(config_name)
    t0 = time.time()
    kw = {}
    if norm_stats_json is not None:
        from openpi.shared import normalize as _norm
        kw["norm_stats"] = _norm.deserialize_json(Path(norm_stats_json).read_text())
        print(f"[norm] overriding with {norm_stats_json}")
    policy = _policy_config.create_trained_policy(train_cfg, ckpt, **kw)
    print(f"[load] policy ready in {time.time() - t0:.1f}s")

    tasks_map = load_tasks_map(repo_root)
    print(f"[tasks] {tasks_map}")

    acc = {h: {"joint": [], "grip": []} for h in HORIZONS}
    per_ep = []

    for i, ep_idx in enumerate(heldout_eps):
        pq_path = repo_root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        if not pq_path.exists():
            print(f"  [skip] missing parquet: {pq_path}")
            continue
        df = pq.read_table(pq_path).to_pandas()
        L = len(df)
        state = np.stack([np.asarray(x) for x in df["observation.state"]])
        action = np.stack([np.asarray(x) for x in df["action"]])
        task_indices = df["task_index"].to_numpy() if "task_index" in df.columns else None

        cams: dict[str, np.ndarray] = {}
        for cam in ("top_head", "hand_left", "hand_right"):
            key = f"observation.images.{cam}"
            vp = repo_root / "videos" / "chunk-000" / key / f"episode_{ep_idx:06d}.mp4"
            cams[cam] = read_video_frames(vp, L)

        q_indices = np.linspace(0, L - max(HORIZONS) - 1, n_sample_frames).astype(int)
        ep_acc = {h: {"joint": [], "grip": []} for h in HORIZONS}
        t_ep = time.time()

        for k in q_indices:
            if prompt_mode == "awbc" and task_indices is not None:
                prompt = tasks_map[int(task_indices[k])]
            else:
                prompt = fixed_prompt
            obs = {
                "images": {cam: cams[cam][k] for cam in cams},
                "state": state[k],
                "prompt": prompt,
            }
            result = policy.infer(obs)
            pred = np.asarray(result["actions"])  # (chunk_len, 14)
            chunk_len = pred.shape[0]
            for h in HORIZONS:
                if h > chunk_len:
                    continue
                gt = action[k + 1 : k + 1 + h]
                if gt.shape[0] < h:
                    continue
                err = np.abs(gt - pred[:h])  # (h, 14)
                ep_acc[h]["joint"].append(err[:, JOINT_DIMS].mean())
                ep_acc[h]["grip"].append(err[:, GRIP_DIMS].mean())

        ep_summary = {
            "episode_index": ep_idx,
            "length": L,
            "n_queries": len(q_indices),
            "sec": round(time.time() - t_ep, 1),
        }
        for h in HORIZONS:
            for kind in ("joint", "grip"):
                vals = ep_acc[h][kind]
                ep_summary[f"mae_{kind}_{h}"] = float(np.mean(vals)) if vals else None
                acc[h][kind].extend(vals)
        per_ep.append(ep_summary)
        print(
            f"  [{i+1}/{len(heldout_eps)}] ep{ep_idx:04d}  "
            f"j1={ep_summary['mae_joint_1']:.4f}  j10={ep_summary['mae_joint_10']:.4f}  "
            f"j50={ep_summary['mae_joint_50']:.4f}  g50={ep_summary['mae_grip_50']:.4f}  "
            f"({ep_summary['sec']}s)"
        )

    summary: dict = {
        "config": config_name,
        "ckpt": str(ckpt),
        "val_repo": str(repo_root),
        "prompt_mode": prompt_mode,
        "fixed_prompt": fixed_prompt,
        "n_heldout_episodes": len(heldout_eps),
        "n_sample_frames": n_sample_frames,
    }
    for h in HORIZONS:
        for kind in ("joint", "grip"):
            vals = acc[h][kind]
            summary[f"mae_{kind}_{h}"] = float(np.mean(vals)) if vals else None
    summary["per_episode"] = per_ep
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-repo", required=True, help="LeRobot repo dir (with meta/, data/, videos/)")
    ap.add_argument("--heldout-seed", type=int, default=0)
    ap.add_argument("--heldout-ratio", type=float, default=0.1)
    ap.add_argument("--n-eval-ep", type=int, default=30, help="subset of held-out for fast eval (None = all)")
    ap.add_argument("--n-sample-frames", type=int, default=50)
    ap.add_argument("--prompt-mode", choices=["default", "awbc"], default="default")
    ap.add_argument("--fixed-prompt", default="Flatten and fold the cloth.")
    ap.add_argument("--norm-stats-json", default=None,
                    help="path to norm_stats.json; overrides the loader's default assets/<asset_id>/norm_stats.json lookup. "
                         "Needed for official HF Kai0 checkpoints where norm_stats lives at ckpt root rather than assets/.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = Path(args.ckpt).resolve()
    repo_root = Path(args.val_repo).resolve()
    heldout_eps = select_heldout(repo_root, args.heldout_seed, args.heldout_ratio, args.n_eval_ep)
    print(f"[heldout] repo={repo_root.name} seed={args.heldout_seed} ratio={args.heldout_ratio} "
          f"n_eval={len(heldout_eps)} first5={heldout_eps[:5]} last5={heldout_eps[-5:]}")

    summary = eval_ckpt(
        args.config, ckpt, repo_root, heldout_eps, args.n_sample_frames,
        args.prompt_mode, args.fixed_prompt,
        norm_stats_json=Path(args.norm_stats_json) if args.norm_stats_json else None,
    )

    print("\n=== summary ===")
    for h in HORIZONS:
        print(f"  MAE@{h}:  joint={summary[f'mae_joint_{h}']:.4f}  grip={summary[f'mae_grip_{h}']:.4f}")

    if args.out is None:
        tag = f"heldout_{repo_root.name}_seed{args.heldout_seed}_ep{len(heldout_eps)}_f{args.n_sample_frames}_{args.prompt_mode}"
        out = ckpt / f"eval_{tag}.json"
    else:
        out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
