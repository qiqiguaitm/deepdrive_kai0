#!/usr/bin/env python3
"""
Evaluate an action policy on the held-out Task_E val split (9 episodes).

Computes action MAE at horizons 1 / 10 / 25 / 50 across every frame in val.
Writes summary + per-episode breakdown to <ckpt>/eval_val.jsonl.

Usage:
    cd kai0/
    uv run python ../scripts/eval_val_action_mse.py \
        --config pi05_stand_box_normal \
        --ckpt checkpoints/pi05_stand_box_normal/stand_box_v1/20000 \
        --val /data1/tim/workspace/deepdive_kai0/kai0/data/Task_E/val

The script loads videos by ffmpeg (via pyav) for speed. It calls the JAX/PyTorch
policy directly (no server) so there is no IPC / HTTP overhead.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _lazy_imports():
    """Defer heavy imports until after argparse (keeps --help fast)."""
    import av  # type: ignore
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _train_config
    return av, _policy_config, _train_config


def read_video_frames(path: Path, n_frames: int) -> np.ndarray:
    """Decode mp4 into np.uint8 (n,H,W,3). Crops/pads to exactly n_frames."""
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


HORIZONS = (1, 10, 25, 50)


def eval_ckpt(config_name: str, ckpt: Path, val_root: Path, n_sample_frames: int | None,
              flow_samples: int = 1) -> dict:
    """Run val eval. If flow_samples > 1, run the policy that many times per
    observation (each call uses fresh flow-matching noise) and take the
    median action — classic 'N-sample ensemble' at inference time."""
    av, _policy_config, _train_config = _lazy_imports()

    print(f"[load] config={config_name}  ckpt={ckpt}  flow_samples={flow_samples}")
    train_cfg = _train_config.get_config(config_name)
    t0 = time.time()
    policy = _policy_config.create_trained_policy(train_cfg, ckpt)
    print(f"[load] policy ready in {time.time() - t0:.1f}s")

    info = json.loads((val_root / "meta" / "info.json").read_text())
    episodes = [json.loads(l) for l in (val_root / "meta" / "episodes.jsonl").read_text().splitlines()]
    chunk = 0  # single chunk in our split

    per_ep = []
    acc_mae: dict[int, list[np.ndarray]] = {h: [] for h in HORIZONS}

    for ep in episodes:
        ep_idx = ep["episode_index"]
        L = ep["length"]
        pq_path = val_root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        df = pq.read_table(pq_path).to_pandas()
        state = np.stack([np.asarray(x) for x in df["observation.state"]])     # (L, 14)
        action = np.stack([np.asarray(x) for x in df["action"]])               # (L, 14)
        cams: dict[str, np.ndarray] = {}
        for cam in ("top_head", "hand_left", "hand_right"):
            key = f"observation.images.{cam}"
            vp = val_root / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{ep_idx:06d}.mp4"
            cams[cam] = read_video_frames(vp, L)

        # For efficiency, subsample query frames if n_sample_frames is set.
        if n_sample_frames is not None and n_sample_frames < L:
            q_indices = np.linspace(0, L - max(HORIZONS) - 1, n_sample_frames).astype(int)
        else:
            q_indices = np.arange(0, L - max(HORIZONS))

        ep_mae: dict[int, list[float]] = {h: [] for h in HORIZONS}
        t_ep = time.time()
        for k in q_indices:
            obs = {
                "images": {cam: cams[cam][k] for cam in cams},
                "state": state[k],
                "prompt": "stand up the fallen box",
            }
            if flow_samples <= 1:
                result = policy.infer(obs)
                pred = np.asarray(result["actions"])  # (chunk_len, 14)
            else:
                samples = []
                for _ in range(flow_samples):
                    r = policy.infer(obs)
                    samples.append(np.asarray(r["actions"]))
                stacked = np.stack(samples, axis=0)  # (N, chunk, 14)
                pred = np.median(stacked, axis=0)
            chunk_len = min(pred.shape[0], max(HORIZONS))
            for h in HORIZONS:
                if h > chunk_len:
                    continue
                gt = action[k + 1 : k + 1 + h]              # next h ground-truth
                ph = pred[:h]
                mae = np.mean(np.abs(gt - ph))
                ep_mae[h].append(mae)

        per_ep.append({
            "episode_index": ep_idx,
            "length": L,
            "n_queries": len(q_indices),
            "mae": {h: float(np.mean(ep_mae[h])) for h in HORIZONS},
            "mae_gripper": None,  # placeholder
            "sec": round(time.time() - t_ep, 1),
        })
        for h in HORIZONS:
            acc_mae[h].extend(ep_mae[h])
        print(f"  ep{ep_idx:02d}  MAE@1={per_ep[-1]['mae'][1]:.4f}  @10={per_ep[-1]['mae'][10]:.4f}  @50={per_ep[-1]['mae'][50]:.4f}  ({per_ep[-1]['sec']}s)")

    summary = {
        "config": config_name,
        "ckpt": str(ckpt),
        "val_root": str(val_root),
        "n_episodes": len(episodes),
        "mae": {h: float(np.mean(acc_mae[h])) for h in HORIZONS},
        "per_episode": per_ep,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="e.g. pi05_stand_box_normal")
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (absolute or relative to kai0/)")
    ap.add_argument("--val", default="/data1/tim/workspace/deepdive_kai0/kai0/data/Task_E/val")
    ap.add_argument("--n-sample-frames", type=int, default=200,
                    help="subsample this many query frames per episode (None = all; default 200)")
    ap.add_argument("--flow-samples", type=int, default=1,
                    help="N-sample ensemble: run policy.infer N times per obs, take median (default 1)")
    ap.add_argument("--out", default=None, help="output jsonl; default <ckpt>/eval_val.json")
    args = ap.parse_args()

    ckpt = Path(args.ckpt).resolve()
    val = Path(args.val).resolve()
    suffix = f"_N{args.flow_samples}" if args.flow_samples > 1 else ""
    out = Path(args.out) if args.out else ckpt / f"eval_val{suffix}.json"

    summary = eval_ckpt(args.config, ckpt, val, args.n_sample_frames, args.flow_samples)

    print("\n=== summary ===")
    for h in HORIZONS:
        print(f"  MAE@{h}: {summary['mae'][h]:.4f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
