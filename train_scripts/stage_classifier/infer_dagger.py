#!/usr/bin/env python3
"""Run trained StageClassifier on dagger/ to produce pseudo stage_progress_gt labels.

Pipeline:
  1. Precompute tube features for dagger episodes (backbone forward, skip if cached)
  2. Load trained cross-attn+MLP head
  3. Per-episode sliding-window inference → DP boundary
  4. Write pseudo_stage_progress_gt back to dagger parquet files (new dataset folder)

Usage:
    python train_scripts/stage_classifier/infer_dagger.py \
        --dagger-source /vePFS/.../Task_A/dagger \
        --dagger-output /vePFS/.../Task_A/dagger_with_stage \
        --cache-root   /vePFS/.../cache/stage_classifier_dagger_vjepa2 \
        --ckpt /vePFS/.../checkpoints/stage_classifier_vjepa2/run1/best.pt \
        --backbone vjepa2_1_large
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kai0" / "src"))
from openpi.models.video.stage_classifier import (
    VideoStageClassifier, best_boundary_dp, load_backbone,
    extract_tube_features, BACKBONE_CHOICES
)

# Reuse precompute helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
from precompute_features import (
    load_video_frames, normalize_frames, find_video_path
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def infer_one_episode(
    ep_idx: int,
    dagger_source: Path,
    model: VideoStageClassifier,
    backbone: torch.nn.Module,
    info,
    device: str,
    camera_key: str,
    stride: int,
    mean, std, batch_size: int,
) -> tuple[torch.Tensor, int, float]:
    """Run backbone + head + DP boundary on one dagger episode."""
    video_path = find_video_path(dagger_source, ep_idx, camera_key)
    frames = load_video_frames(video_path, target_size=info.image_size)
    N = len(frames)

    # Normalize
    x = normalize_frames(frames, mean, std)  # (N, 3, H, W)

    T = info.num_frames
    # Sliding window with full coverage
    starts = list(range(0, max(N - T + 1, 1), stride))
    if starts[-1] + T < N:
        starts.append(N - T)

    # Per-clip forward
    logits_sum = torch.zeros(N, 2, device=device)
    count = torch.zeros(N, device=device)

    last_frame = x[-1:]
    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            batch_starts = starts[i:i + batch_size]
            clips = []
            for s in batch_starts:
                e = s + T
                if e <= N:
                    clip = x[s:e]
                else:
                    pad = e - N
                    clip = torch.cat([x[s:], last_frame.expand(pad, -1, -1, -1)], dim=0)
                clips.append(clip)
            clips = torch.stack(clips, dim=0).to(device)

            # Backbone
            tube_feats = extract_tube_features(backbone, info, clips)  # (B, num_tubes, hidden)
            # Head
            clip_logits = model.forward_from_tubes(tube_feats)  # (B, T, 2)

            # Accumulate
            for bi, s in enumerate(batch_starts):
                e = min(s + T, N)
                logits_sum[s:e] += clip_logits[bi, :e - s]
                count[s:e] += 1

    count = count.clamp(min=1)
    per_frame_logits = logits_sum / count.unsqueeze(-1)

    # DP boundary
    labels, t_star, confidence = best_boundary_dp(per_frame_logits)
    return labels.cpu(), t_star, confidence


def write_pseudo_labels_parquet(
    src_parquet_dir: Path,
    dst_parquet_dir: Path,
    ep_idx: int,
    pseudo_labels: torch.Tensor,
):
    """Find parquet for ep_idx, add stage_progress_gt column, write to dst."""
    src = list(src_parquet_dir.rglob(f"episode_{ep_idx:06d}.parquet"))
    if not src:
        raise FileNotFoundError(f"No parquet for ep {ep_idx}")
    src_path = src[0]
    rel = src_path.relative_to(src_parquet_dir)
    dst_path = dst_parquet_dir / rel
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(src_path)
    n_parquet = table.num_rows

    # Pseudo stage_progress_gt: 0 (flat) → 0.25, 1 (fold) → 0.75
    n_labels = min(len(pseudo_labels), n_parquet)
    pseudo_sp_gt = np.zeros(n_parquet, dtype=np.float32)
    pseudo_sp_gt[:n_labels] = np.where(pseudo_labels[:n_labels].numpy() == 0, 0.25, 0.75).astype(np.float32)
    if n_labels < n_parquet:
        # pad with last label
        last_val = pseudo_sp_gt[n_labels - 1] if n_labels > 0 else 0.25
        pseudo_sp_gt[n_labels:] = last_val

    # Add column (replace if exists)
    col = pa.array(pseudo_sp_gt, type=pa.float32())
    if "stage_progress_gt" in table.column_names:
        idx = table.column_names.index("stage_progress_gt")
        table = table.set_column(idx, "stage_progress_gt", col)
    else:
        table = table.append_column("stage_progress_gt", col)

    pq.write_table(table, dst_path)


def prepare_output_dataset(src: Path, dst: Path):
    """Copy meta/, symlink videos/ (they're unchanged)."""
    if dst.exists():
        logger.warning(f"Output dir exists, will write into: {dst}")
    dst.mkdir(parents=True, exist_ok=True)

    # meta: copy (we'll overwrite tasks.jsonl shortly to match advantage format)
    meta_src = src / "meta"
    meta_dst = dst / "meta"
    if not meta_dst.exists():
        shutil.copytree(meta_src, meta_dst)
        logger.info(f"Copied meta → {meta_dst}")

    # videos: symlink (saves 40GB disk)
    videos_dst = dst / "videos"
    if not videos_dst.exists():
        videos_dst.symlink_to((src / "videos").resolve())
        logger.info(f"Symlinked videos → {videos_dst}")

    # norm_stats: symlink
    nss = src / "norm_stats.json"
    nsd = dst / "norm_stats.json"
    if nss.exists() and not nsd.exists():
        nsd.symlink_to(nss.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dagger-source", required=True)
    ap.add_argument("--dagger-output", required=True, help="Output dataset (with stage_progress_gt added)")
    ap.add_argument("--ckpt", required=True, help="Trained StageClassifier checkpoint (best.pt)")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--camera-key", default="top_head")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=None, help="If set overrides CUDA_VISIBLE_DEVICES; omit to respect shell env")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--min-ep", type=int, default=0)
    ap.add_argument("--max-ep", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=1, help="Multi-GPU sharding")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--metrics-out", default=None, help="Dir to write per-worker per-episode metrics JSON")
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    # Load backbone
    backbone, info = load_backbone(args.backbone, device=device, dtype=dtype)

    # Preprocessor mean/std
    if BACKBONE_CHOICES[args.backbone]["loader"] == "torch_hub":
        # V-JEPA 2 uses ImageNet defaults
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        try:
            from transformers import AutoImageProcessor
            proc = AutoImageProcessor.from_pretrained(info.hf_repo, trust_remote_code=True)
            mean = list(proc.image_mean)
            std = list(proc.image_std)
        except Exception:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    logger.info(f"Preprocessor: mean={mean}, std={std}")

    # Load trained head
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_args = ckpt.get("args", {})
    model = VideoStageClassifier(
        backbone_hidden=model_args.get("backbone_hidden", info.hidden_size),
        num_tubes=model_args.get("num_tubes", info.num_tubes),
        num_frames=model_args.get("num_frames", info.num_frames),
        hidden_dim=model_args.get("hidden_dim", 384),
        n_cross_attn_layers=model_args.get("n_layers", 2),
        n_heads=model_args.get("n_heads", 8),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Prepare output dataset
    dagger_source = Path(args.dagger_source)
    dagger_output = Path(args.dagger_output)
    prepare_output_dataset(dagger_source, dagger_output)

    # List episodes
    episodes_file = dagger_source / "meta" / "episodes.jsonl"
    episodes = []
    with open(episodes_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episodes.append(int(json.loads(line)["episode_index"]))
    episodes = sorted(set(episodes))

    # Filter by min/max
    if args.min_ep is not None:
        episodes = [e for e in episodes if e >= args.min_ep]
    if args.max_ep is not None:
        episodes = [e for e in episodes if e < args.max_ep]

    # Shard for multi-GPU
    if args.num_workers > 1:
        episodes = [e for i, e in enumerate(episodes) if i % args.num_workers == args.worker_id]

    logger.info(f"Processing {len(episodes)} dagger episodes (worker {args.worker_id}/{args.num_workers})")

    # Resume: skip episodes whose output parquet already exists (from previous run)
    def _output_parquet_exists(ep_idx: int) -> bool:
        cands = list((dagger_output / "data").rglob(f"episode_{ep_idx:06d}.parquet"))
        return bool(cands)

    metrics_dir = Path(args.metrics_out) if args.metrics_out else None
    if metrics_dir:
        metrics_dir.mkdir(parents=True, exist_ok=True)
    per_ep_metrics = {}

    ok, fail, skipped = 0, 0, 0
    t_stars = []
    confidences = []

    for ep_idx in tqdm(episodes, desc=f"w{args.worker_id}"):
        if _output_parquet_exists(ep_idx):
            skipped += 1
            continue
        try:
            labels, t_star, conf = infer_one_episode(
                ep_idx=ep_idx, dagger_source=dagger_source,
                model=model, backbone=backbone, info=info, device=device,
                camera_key=args.camera_key, stride=args.stride,
                mean=mean, std=std, batch_size=args.batch_size,
            )
            write_pseudo_labels_parquet(
                dagger_source / "data", dagger_output / "data", ep_idx, labels
            )
            ok += 1
            t_stars.append(t_star)
            confidences.append(conf)
            n_flat = int((labels == 0).sum().item())
            n_fold = int((labels == 1).sum().item())
            per_ep_metrics[str(ep_idx)] = {
                "t_star": int(t_star),
                "confidence": float(conf),
                "n_frames": int(len(labels)),
                "n_flat": n_flat,
                "n_fold": n_fold,
                "t_star_ratio": float(t_star) / max(len(labels), 1),
            }
            # Incremental flush every episode (small file, ~20 KB at 1000 eps)
            if metrics_dir:
                tmp_path = metrics_dir / f"w{args.worker_id}.json.tmp"
                final_path = metrics_dir / f"w{args.worker_id}.json"
                with open(tmp_path, "w") as f:
                    json.dump(per_ep_metrics, f, indent=2)
                tmp_path.replace(final_path)  # atomic
        except Exception as e:
            logger.warning(f"ep {ep_idx} failed: {e}")
            fail += 1

    logger.info(f"Done: ok={ok}, fail={fail}, skipped={skipped}")
    if t_stars:
        logger.info(f"Boundary stats: mean t_star = {np.mean(t_stars):.1f}, confidence mean = {np.mean(confidences):.2f}")
    if metrics_dir:
        logger.info(f"Metrics: {metrics_dir}/w{args.worker_id}.json")


if __name__ == "__main__":
    main()
