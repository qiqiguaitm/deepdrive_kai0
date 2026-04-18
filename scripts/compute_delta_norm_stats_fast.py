#!/usr/bin/env python3
"""Fast norm_stats for delta-action configs (no video loading).

For each frame t in each episode, computes a full action chunk:
  delta_chunk[h] = action[t+h] - state[t]   for h in [0, action_horizon)
then accumulates RunningStats over all (t,h) pairs. Gripper dims (6, 13) are not
shifted. Writes to <config.assets_dirs>/<asset_id>/norm_stats.json.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "kai0" / "src"))
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
from openpi.transforms import make_bool_mask


def pad_to_dim(a: np.ndarray, dim: int) -> np.ndarray:
    if a.shape[-1] >= dim:
        return a[..., :dim]
    pad = np.zeros((*a.shape[:-1], dim - a.shape[-1]), dtype=a.dtype)
    return np.concatenate([a, pad], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--action-horizon", type=int, default=50)
    args = ap.parse_args()

    cfg = _config.get_config(args.config_name)
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    repo_id = data_cfg.repo_id
    asset_id = data_cfg.asset_id or repo_id
    action_dim = cfg.model.action_dim
    H = args.action_horizon
    print(f"config={args.config_name}  repo_id={repo_id}  asset_id={asset_id}  horizon={H}")

    mask_14 = np.array(make_bool_mask(6, -1, 6, -1), dtype=bool)
    mask = np.zeros(action_dim, dtype=bool)
    mask[:14] = mask_14

    parquet_files = sorted((Path(repo_id) / "data").rglob("*.parquet"))
    print(f"parquet files: {len(parquet_files)}")

    state_stats = _normalize.RunningStats()
    action_stats = _normalize.RunningStats()

    for pq in tqdm(parquet_files, desc="files"):
        df = pd.read_parquet(pq)
        state = np.stack([np.asarray(x, dtype=np.float32) for x in df["observation.state"]])
        action = np.stack([np.asarray(x, dtype=np.float32) for x in df["action"]])
        state = pad_to_dim(state, action_dim)
        action = pad_to_dim(action, action_dim)
        state = np.where(np.abs(state) > np.pi, 0, state)
        action = np.where(np.abs(action) > np.pi, 0, action)
        T = len(state)

        # state stats (per frame)
        state_stats.update(state)

        # build delta action chunks: for each t, delta[h] = action[t+h] - state[t] * mask
        # Use sliding windows; pad end by clamping.
        if T < H:
            # episode shorter than horizon — pad action at end
            pad = np.repeat(action[-1:], H - T, axis=0)
            action_padded_t = np.concatenate([action, pad], axis=0)
        else:
            action_padded_t = action
        # produce chunks shape (T, H, action_dim) without heavy memory for 77k * 50 * 32 = 12M floats = 48 MB — ok
        for t in range(T):
            end = min(t + H, len(action_padded_t))
            chunk = action_padded_t[t:end]  # (≤H, D)
            if chunk.shape[0] < H:
                pad = np.repeat(chunk[-1:], H - chunk.shape[0], axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)  # (H, D)
            chunk = chunk.copy()
            chunk[..., mask] -= state[t][mask]
            action_stats.update(chunk)

    norm_stats = {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }

    from etils import epath
    out_dir = cfg.assets_dirs / data_cfg.asset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _normalize.save(epath.Path(str(out_dir)), norm_stats)
    print(f"\n✅ wrote → {out_dir}")
    print(f"action(delta) mean[:7]: {norm_stats['actions'].mean[:7]}")
    print(f"action(delta) std[:7]:  {norm_stats['actions'].std[:7]}")
    print(f"state mean[:7]:         {norm_stats['state'].mean[:7]}")


if __name__ == "__main__":
    main()
