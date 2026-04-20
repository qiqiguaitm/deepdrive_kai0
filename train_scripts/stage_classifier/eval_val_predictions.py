#!/usr/bin/env python3
"""Run trained head on cached val features to produce per-episode GT vs predicted boundary metrics.

For each val episode:
  - Load precomputed tube features
  - Run head → per-frame logits → DP boundary
  - Record: gt_boundary, pred_boundary_dp, pred_boundary_raw, confidence, n_frames, offset

Output: JSON keyed by ep_idx (str).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kai0" / "src"))
from openpi.models.video.stage_classifier import VideoStageClassifier, best_boundary_dp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--split-key", default="val", choices=["val", "train", "all"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.split) as f:
        split = json.load(f)
    if args.split_key == "all":
        ep_list = split["train_episodes"] + split["val_episodes"]
    else:
        ep_list = split[f"{args.split_key}_episodes"]
    logger.info(f"Processing {len(ep_list)} episodes (split={args.split_key})")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    margs = ckpt.get("args", {})
    model = VideoStageClassifier(
        backbone_hidden=margs.get("backbone_hidden", 1024),
        num_tubes=margs.get("num_tubes", 8),
        num_frames=margs.get("num_frames", 16),
        hidden_dim=margs.get("hidden_dim", 384),
        n_cross_attn_layers=margs.get("n_layers", 2),
        n_heads=margs.get("n_heads", 8),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    metrics = {}
    cache_root = Path(args.cache_root)
    missing = 0

    for ep_idx in ep_list:
        path = cache_root / f"ep_{ep_idx:06d}.pt"
        if not path.exists():
            missing += 1
            continue
        meta = torch.load(path, map_location=device, weights_only=False)
        tube_feats = meta["tube_features"].float().to(device)
        labels_clip = meta["labels"].long()
        clip_starts = meta["clip_starts"]
        n_frames = meta["n_frames"]
        T = labels_clip.shape[1]

        with torch.no_grad():
            logits_clip = model.forward_from_tubes(tube_feats)

        # Aggregate overlapping clip logits → per-frame logits
        frame_logits = torch.zeros(n_frames, 2, device=device)
        count = torch.zeros(n_frames, device=device)
        for ci, s in enumerate(clip_starts.tolist()):
            e = min(s + T, n_frames)
            frame_logits[s:e] += logits_clip[ci, :e - s]
            count[s:e] += 1
        frame_logits = frame_logits / count.clamp(min=1).unsqueeze(-1)

        # GT: reconstruct per-frame from clip labels
        frame_gt = torch.zeros(n_frames, dtype=torch.long)
        for ci, s in enumerate(clip_starts.tolist()):
            e = min(s + T, n_frames)
            frame_gt[s:e] = labels_clip[ci, :e - s]
        gt_above = (frame_gt == 1).nonzero().flatten()
        gt_t = int(gt_above[0]) if len(gt_above) else n_frames

        # Raw argmax boundary
        raw_labels = frame_logits.argmax(-1)
        raw_above = (raw_labels == 1).nonzero().flatten()
        raw_t = int(raw_above[0].item()) if len(raw_above) else n_frames

        # DP boundary
        _dp_labels, dp_t, dp_conf = best_boundary_dp(frame_logits)
        dp_t = int(dp_t)

        metrics[str(ep_idx)] = {
            "gt_boundary": gt_t,
            "pred_boundary_dp": dp_t,
            "pred_boundary_raw": raw_t,
            "confidence": float(dp_conf),
            "n_frames": int(n_frames),
            "offset_dp": dp_t - gt_t,          # signed
            "abs_offset_dp": abs(dp_t - gt_t),
            "offset_raw": raw_t - gt_t,
        }

    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    if metrics:
        import numpy as np
        abs_offsets = [m["abs_offset_dp"] for m in metrics.values()]
        logger.info(f"✅ {len(metrics)} episodes processed, {missing} missing cache")
        logger.info(f"  boundary_mae (mean abs offset) = {np.mean(abs_offsets):.2f} frames")
        logger.info(f"  median offset = {np.median(abs_offsets):.1f}  max = {max(abs_offsets)}")


if __name__ == "__main__":
    main()
