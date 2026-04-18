#!/usr/bin/env python3
"""对比 gf0 (baseline) vs gf1 (π0.7 q5drop) 在 held-out episodes 上的 action MAE.

用法:
    python3 eval_awbc_compare.py --step 1000 --n 10      # 10 samples, step 1000
    python3 eval_awbc_compare.py --step 5000 --n 50 --seed 42

性能 (CPU):
    ~30-60s / sample (pi05 3B params, 10 diffusion steps)
    10 samples × 2 policies ≈ 10-20 min
"""
import argparse
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
sys.path.insert(0, "/vePFS/tim/workspace/deepdive_kai0/kai0/src")

import random
import time
import numpy as np

import jax
print(f"JAX devices: {jax.devices()}  ← should be CPU", flush=True)

import openpi.training.config as _config
from openpi.policies import policy_config as _policy_config
from openpi.training import data_loader as _data_loader
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


CHECKPOINT_BASE = "/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints"
DATA_DIR = "/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/advantage"  # unified eval set


def build_heldout_manifest(repo_id, val_ratio=0.1, action_horizon=50, seed=42):
    """构造真·held-out 清单：绕开 LeRobotDataset 的稀疏 episode_index bug.

    直接读 parquet + MP4 videos，返回 [(parquet_path, frame_idx, ep_idx)] list.
    按 parquet 文件名排序，取最后 val_ratio 比例作为 val set.
    """
    import glob
    all_parquets = sorted(glob.glob(f"{repo_id}/data/chunk-*/episode_*.parquet"))
    n_val = int(len(all_parquets) * val_ratio)
    val_parquets = all_parquets[-n_val:]  # 物理上最后 N 个 episode
    print(f"  total parquets: {len(all_parquets)}, val: {len(val_parquets)}")

    # 每个 val parquet 提取所有合法 frame index（预留 action_horizon 尾部）
    rng = random.Random(seed)
    manifest = []
    for pq in val_parquets:
        try:
            import pandas as pd
            df = pd.read_parquet(pq, columns=["frame_index", "episode_index"])
            ep_idx = int(df["episode_index"].iloc[0])
            n_frames = len(df)
            if n_frames <= action_horizon + 1:
                continue
            # random 1 frame per episode (for uniform coverage)
            fr = rng.randint(0, n_frames - action_horizon - 1)
            manifest.append((pq, fr, ep_idx))
        except Exception:
            continue
    rng.shuffle(manifest)
    print(f"  manifest: {len(manifest)} samples (1 per val episode)")
    return manifest


def load_sample_from_manifest(pq_path, frame_idx, ep_idx, videos_root, action_horizon=50):
    """手动读一个 sample: parquet 取 state/action，av 解视频帧."""
    import pandas as pd
    import av
    df = pd.read_parquet(pq_path)
    row = df.iloc[frame_idx]
    state = np.asarray(row["observation.state"], dtype=np.float32)
    # GT action chunk
    gt_actions = np.stack([
        np.asarray(df.iloc[frame_idx + h]["action"], dtype=np.float32)
        for h in range(action_horizon)
    ])
    # 读三视角 MP4 — 用 episode_index/chunk 定位
    chunk_id = ep_idx // 1000
    images = {}
    for cam in ["top_head", "hand_left", "hand_right"]:
        vpath = f"{videos_root}/chunk-{chunk_id:03d}/observation.images.{cam}/episode_{ep_idx:06d}.mp4"
        if not os.path.exists(vpath):
            # symlink 目标
            vpath = os.path.realpath(vpath)
        container = av.open(vpath)
        stream = container.streams.video[0]
        # seek to target frame
        fps = float(stream.average_rate)
        target_t = frame_idx / fps
        # seek by pts
        time_base = float(stream.time_base)
        container.seek(int(target_t / time_base), stream=stream, any_frame=True)
        frame_img = None
        for pkt in container.demux(stream):
            for frm in pkt.decode():
                if frm.time is not None and frm.time >= target_t - 1.0 / fps:
                    frame_img = frm.to_ndarray(format="rgb24")  # [H, W, 3] uint8
                    break
            if frame_img is not None:
                break
        container.close()
        if frame_img is None:
            # fallback: first frame
            container = av.open(vpath)
            for frm in container.decode(video=0):
                frame_img = frm.to_ndarray(format="rgb24")
                break
            container.close()
        images[cam] = frame_img
    return {
        "observation.state": state,
        "action": gt_actions,
        **{f"observation.images.{k}": v for k, v in images.items()},
    }


def build_obs_from_sample(sample):
    """Raw LeRobot sample → obs dict for policy.infer."""
    obs = {
        "images": {},
        "state": sample["observation.state"].numpy() if hasattr(sample["observation.state"], "numpy") else np.array(sample["observation.state"]),
    }
    for cam in ["top_head", "hand_left", "hand_right"]:
        key = f"observation.images.{cam}"
        img = sample[key]
        # img is torch tensor [3, H, W] or similar - policy expects [H, W, 3] uint8
        if hasattr(img, "numpy"):
            img = img.numpy()
        # torchvision videos come as [C, H, W], convert to [H, W, C]
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        # ensure uint8
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        obs["images"][cam] = img
    return obs


def eval_one_policy(policy, ds, val_frame_indices, n_samples, name=""):
    print(f"\n--- Running eval: {name} ---", flush=True)
    # Use training-distribution indices (not held-out) — the advantage dataset has
    # sparse episode_index that breaks val episode slicing. Trade-off: eval samples
    # may overlap with training set. This is OK for a RELATIVE comparison between gf0 vs gf1.
    rng = random.Random(42)
    all_indices = list(range(0, 300_000))  # dense region
    rng.shuffle(all_indices)
    indices = all_indices[:n_samples * 20]

    results = []
    errors = []
    t0 = time.time()
    pool = iter(indices)
    for i in range(n_samples):
        # retry loop like training DataLoader
        sample = None
        idx = None
        for _ in range(50):
            try:
                idx = next(pool)
            except StopIteration:
                break
            try:
                sample = ds[idx]
                break
            except Exception:
                continue
        if sample is None:
            errors.append((idx, "all retries failed"))
            continue
        try:
            obs = build_obs_from_sample(sample)
        except Exception as e:
            errors.append((idx, f"obs: {str(e)[:80]}"))
            continue

        try:
            out = policy.infer(obs)
            pred = np.asarray(out["actions"])  # [H, A]
        except Exception as e:
            errors.append((idx, f"infer: {str(e)[:80]}"))
            continue

        gt = sample["action"].numpy() if hasattr(sample["action"], "numpy") else np.array(sample["action"])
        # crop to 14 valid dims, limit horizon to min of both
        H = min(pred.shape[0], gt.shape[0], 50)
        pred = pred[:H, :14]
        gt = gt[:H, :14]
        err = np.abs(pred - gt)  # [H, 14]
        joint_idx = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        grip_idx = [6, 13]
        r = {
            "mae_joint_1":  float(err[0, joint_idx].mean()) if H >= 1 else None,
            "mae_joint_10": float(err[9, joint_idx].mean()) if H >= 10 else None,
            "mae_joint_50": float(err[min(49, H-1), joint_idx].mean()) if H >= 1 else None,
            "mae_grip_1":   float(err[0, grip_idx].mean()) if H >= 1 else None,
            "mae_grip_10":  float(err[9, grip_idx].mean()) if H >= 10 else None,
            "mae_grip_50":  float(err[min(49, H-1), grip_idx].mean()) if H >= 1 else None,
        }
        results.append(r)
        dt = time.time() - t0
        eta = dt / (i+1) * (len(indices) - (i+1))
        print(f"  [{i+1}/{len(indices)}] MAE@1_joint={r['mae_joint_1']:.4f}  MAE@10_joint={r.get('mae_joint_10', 0):.4f}  elapsed={dt:.0f}s eta={eta:.0f}s", flush=True)
    return results, errors


def aggregate(results):
    if not results:
        return {}
    agg = {}
    for k in ["mae_joint_1", "mae_joint_10", "mae_joint_50", "mae_grip_1", "mae_grip_10", "mae_grip_50"]:
        vals = [r[k] for r in results if r.get(k) is not None]
        agg[k] = float(np.mean(vals)) if vals else None
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=1000)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    def find_ckpt(base, want_step):
        d = os.path.dirname(f"{base}/dummy")
        if os.path.exists(f"{base}/{want_step}"):
            return f"{base}/{want_step}", want_step
        # fallback: find latest numeric dir
        if not os.path.exists(base):
            return None, None
        steps = sorted([int(x) for x in os.listdir(base) if x.isdigit()])
        if not steps:
            return None, None
        return f"{base}/{steps[-1]}", steps[-1]

    ck0, s0 = find_ckpt(f"{CHECKPOINT_BASE}/pi05_flatten_fold_awbc/gf0_awbc_baseline_v1", args.step)
    ck1, s1 = find_ckpt(f"{CHECKPOINT_BASE}/pi05_flatten_fold_awbc_q5drop/gf1_awbc_q5drop_v1", args.step)
    for n, ck in [("gf0", ck0), ("gf1", ck1)]:
        if ck is None:
            print(f"ERROR: no checkpoint found for {n}"); return
    if s0 != s1:
        print(f"⚠️  WARNING: gf0@step_{s0} vs gf1@step_{s1}  (不同步，对比有偏差)")

    print("="*70)
    print(f"AWBC eval — step {args.step}, {args.n} samples")
    print("="*70)
    print(f"  gf0: {ck0}")
    print(f"  gf1: {ck1}\n")

    print("[1/4] Building held-out eval dataset ...")
    t0 = time.time()
    ds, val_frames = build_eval_dataset(DATA_DIR)
    print(f"  dataset ready in {time.time()-t0:.1f}s, total {len(ds)} frames, val_frames={len(val_frames)}\n")

    print("[2/4] Loading gf0 baseline policy ...")
    t0 = time.time()
    cfg0 = _config.get_config("pi05_flatten_fold_awbc")
    pol0 = _policy_config.create_trained_policy(
        cfg0, ck0, default_prompt="Flatten and fold the cloth. Advantage: positive"
    )
    print(f"  loaded in {time.time()-t0:.1f}s\n")

    res0, err0 = eval_one_policy(pol0, ds, val_frames, args.n, name="gf0 baseline")
    agg0 = aggregate(res0)
    print(f"\n  gf0: ok={len(res0)}, err={len(err0)}")
    for e in err0[:3]:
        print(f"    error: {e}")
    del pol0

    print("\n[3/4] Loading gf1 π0.7 q5drop policy ...")
    t0 = time.time()
    cfg1 = _config.get_config("pi05_flatten_fold_awbc_q5drop")
    pol1 = _policy_config.create_trained_policy(
        cfg1, ck1, default_prompt="Flatten and fold the cloth. Quality: 5/5"
    )
    print(f"  loaded in {time.time()-t0:.1f}s\n")

    res1, err1 = eval_one_policy(pol1, ds, val_frames, args.n, name="gf1 π0.7 q5drop")
    agg1 = aggregate(res1)
    print(f"\n  gf1: ok={len(res1)}, err={len(err1)}")

    # Report
    print("\n" + "="*70)
    print(f"📊 Compare (step {args.step}, {args.n} samples)")
    print("="*70)
    thresh = {
        "mae_joint_1": 0.02, "mae_joint_10": 0.05, "mae_joint_50": 0.12,
        "mae_grip_1": 0.005, "mae_grip_10": 0.005, "mae_grip_50": 0.005,
    }
    print(f"  {'metric':<17} {'gf0':>10} {'gf1':>10} {'Δ':>8}   {'thresh':>7}  gf0 gf1")
    print(f"  {'-'*17} {'-'*10} {'-'*10} {'-'*8}   {'-'*7}  --- ---")
    for k in ["mae_joint_1", "mae_joint_10", "mae_joint_50", "mae_grip_1", "mae_grip_10", "mae_grip_50"]:
        v0, v1 = agg0.get(k), agg1.get(k)
        if v0 is None or v1 is None:
            continue
        th = thresh[k]
        diff = (v1 - v0) / max(abs(v0), 1e-9) * 100
        sign = "+" if diff >= 0 else ""
        p0 = "✅" if v0 < th else "❌"
        p1 = "✅" if v1 < th else "❌"
        print(f"  {k:<17} {v0:>10.4f} {v1:>10.4f} {sign}{diff:>5.1f}%   {th:>7.3f}   {p0}  {p1}")


if __name__ == "__main__":
    main()
