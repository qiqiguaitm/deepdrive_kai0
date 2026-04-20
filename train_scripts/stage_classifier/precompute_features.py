#!/usr/bin/env python3
"""Pre-compute tube features for stage classifier training.

For every episode in `advantage/` (or other LeRobot dataset), extract
V-JEPA 2 / VideoMAE v2 tube features for all overlapping T-frame clips.
Save per-episode to disk for fast training.

Usage:
    python train_scripts/stage_classifier/precompute_features.py \
        --source /vePFS/.../Task_A/advantage \
        --split /vePFS/.../Task_A/stage_classifier_split.json \
        --split-key train \
        --cache-root /vePFS/.../cache/stage_classifier/advantage \
        --backbone vjepa2_ssv2 \
        --gpu 0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

# Add kai0 src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kai0" / "src"))
from openpi.models.video.stage_classifier import (
    BackboneInfo, extract_tube_features, load_backbone
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_video_frames(video_path: Path, target_size: int = 256) -> np.ndarray:
    """Load all frames from mp4, resize. Returns: (N, H, W, 3) uint8 RGB."""
    import av
    import cv2
    container = av.open(str(video_path))
    container.streams.video[0].thread_type = "AUTO"
    frames = []
    for frame in container.decode(video=0):
        arr = frame.to_ndarray(format="rgb24")
        if arr.shape[0] != target_size or arr.shape[1] != target_size:
            arr = cv2.resize(arr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        frames.append(arr)
    container.close()
    return np.stack(frames, axis=0)


def normalize_frames(frames: np.ndarray, mean: list[float], std: list[float]) -> torch.Tensor:
    x = torch.from_numpy(frames).float() / 255.0
    x = x.permute(0, 3, 1, 2)  # (N, 3, H, W)
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


def episode_stage_labels(parquet_dir: Path, ep_idx: int) -> np.ndarray:
    candidates = list(parquet_dir.rglob(f"episode_{ep_idx:06d}.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet for ep {ep_idx}")
    table = pq.read_table(candidates[0])
    if "stage_progress_gt" not in table.column_names:
        raise ValueError(f"Ep {ep_idx}: missing stage_progress_gt column")
    sp_gt = table["stage_progress_gt"].to_numpy()
    return (sp_gt >= 0.5).astype(np.uint8)


def find_video_path(source: Path, ep_idx: int, camera_key: str) -> Path:
    video_name = f"episode_{ep_idx:06d}.mp4"
    for chunk_dir in sorted((source / "videos").iterdir()):
        if not chunk_dir.is_dir():
            continue
        cand = chunk_dir / f"observation.images.{camera_key}" / video_name
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No video for ep {ep_idx} cam {camera_key}")


def precompute_episode(
    ep_idx: int,
    source: Path,
    backbone: torch.nn.Module,
    info: BackboneInfo,
    device: str,
    cache_path: Path,
    camera_key: str,
    stride: int,
    mean: list[float],
    std: list[float],
    batch_size: int,
) -> dict:
    video_path = find_video_path(source, ep_idx, camera_key)
    frames = load_video_frames(video_path, target_size=info.image_size)
    N_video = len(frames)

    labels_full = episode_stage_labels(source / "data", ep_idx)
    N = min(len(labels_full), N_video)
    frames = frames[:N]
    labels_full = labels_full[:N]

    x = normalize_frames(frames, mean, std)

    T = info.num_frames
    starts = list(range(0, max(N - T + 1, 1), stride))
    if starts[-1] + T < N:
        starts.append(N - T)
    n_clips = len(starts)

    # Build clips with tail padding for short episodes
    clip_list = []
    label_list = []
    last_frame = x[-1:]
    for s in starts:
        e = s + T
        if e <= N:
            clip = x[s:e]
            lbl = labels_full[s:e]
        else:
            pad = e - N
            clip = torch.cat([x[s:], last_frame.expand(pad, -1, -1, -1)], dim=0)
            lbl = np.concatenate([labels_full[s:], np.full(pad, labels_full[-1], dtype=np.uint8)])
        clip_list.append(clip)
        label_list.append(lbl)
    clips = torch.stack(clip_list, dim=0)  # (n_clips, T, 3, H, W)
    clip_labels = np.stack(label_list, axis=0)  # (n_clips, T)

    # Backbone forward in batches
    all_tube_feats = []
    for i in range(0, n_clips, batch_size):
        batch = clips[i : i + batch_size].to(device)
        tube_feats = extract_tube_features(backbone, info, batch)
        all_tube_feats.append(tube_feats.cpu().half())
    tube_features = torch.cat(all_tube_feats, dim=0)

    out = {
        "tube_features": tube_features,
        "labels": torch.from_numpy(clip_labels),
        "clip_starts": torch.tensor(starts, dtype=torch.int32),
        "ep_idx": ep_idx,
        "n_frames": N,
        "backbone_repo": info.hf_repo,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache_path)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--split-key", default="train", choices=["train", "val", "all"])
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--backbone", default="vjepa2_ssv2")
    parser.add_argument("--camera-key", default="top_head")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=None, help="If set, overrides CUDA_VISIBLE_DEVICES. Omit to respect shell env.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    with open(args.split) as f:
        split = json.load(f)
    if args.split_key == "all":
        episodes = split["train_episodes"] + split["val_episodes"]
    else:
        episodes = split[f"{args.split_key}_episodes"]
    if args.num_workers > 1:
        episodes = [e for i, e in enumerate(episodes) if i % args.num_workers == args.worker_id]
    logger.info(f"Processing {len(episodes)} episodes (split={args.split_key}, worker {args.worker_id}/{args.num_workers})")

    backbone, info = load_backbone(args.backbone, device=device, dtype=dtype)
    logger.info(f"Backbone info: {info}")

    try:
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(info.hf_repo, trust_remote_code=True)
        mean = proc.image_mean if hasattr(proc, "image_mean") else [0.485, 0.456, 0.406]
        std = proc.image_std if hasattr(proc, "image_std") else [0.229, 0.224, 0.225]
    except Exception as e:
        logger.warning(f"Preprocessor load failed: {e}, using ImageNet defaults")
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    logger.info(f"Preprocessor: mean={mean}, std={std}")

    source = Path(args.source)
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(episodes, desc=f"w{args.worker_id}")
    ok, fail = 0, 0
    for ep_idx in pbar:
        cache_path = cache_root / f"ep_{ep_idx:06d}.pt"
        if cache_path.exists() and not args.force:
            ok += 1
            continue
        try:
            out = precompute_episode(
                ep_idx=ep_idx, source=source, backbone=backbone, info=info, device=device,
                cache_path=cache_path, camera_key=args.camera_key, stride=args.stride,
                mean=mean, std=std, batch_size=args.batch_size,
            )
            pbar.set_postfix({"clips": out["tube_features"].shape[0], "N": out["n_frames"], "ok": ok, "fail": fail})
            ok += 1
        except Exception as e:
            logger.warning(f"Ep {ep_idx} failed: {e}")
            fail += 1

    logger.info(f"✅ Done. OK={ok}, Fail={fail}. Cache: {cache_root}")


if __name__ == "__main__":
    main()
