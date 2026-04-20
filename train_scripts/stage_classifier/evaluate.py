#!/usr/bin/env python3
"""Standalone val evaluation for trained StageClassifier.

Loads a checkpoint and reports metrics on val episodes.

Usage:
    python train_scripts/stage_classifier/evaluate.py \
        --ckpt /vePFS/.../checkpoints/stage_classifier_vjepa2/run1/best.pt \
        --cache-root /vePFS/.../cache/stage_classifier_vjepa2_ssv2 \
        --split /vePFS/.../Task_A/stage_classifier_split.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kai0" / "src"))
from openpi.models.video.stage_classifier import VideoStageClassifier

# Reuse evaluate function from training script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--output", default=None, help="Output JSON path (default: ckpt dir)")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = ckpt.get("args", {})
    model = VideoStageClassifier(
        backbone_hidden=model_args.get("backbone_hidden", 1024),
        num_tubes=model_args.get("num_tubes", 8),
        num_frames=model_args.get("num_frames", 16),
        hidden_dim=model_args.get("hidden_dim", 384),
        n_cross_attn_layers=model_args.get("n_layers", 2),
        n_heads=model_args.get("n_heads", 8),
        dropout=model_args.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load split
    with open(args.split) as f:
        split = json.load(f)
    val_episodes = split["val_episodes"]

    print(f"Evaluating {ckpt_path} on {len(val_episodes)} val episodes...")
    metrics = evaluate(model, args.cache_root, val_episodes, device)

    # Check pass/fail against targets
    targets = {
        "frame_accuracy": 0.97,
        "boundary_mae_frames": 15.0,  # LOWER is better
        "monotonic_rate_raw": 0.90,
        "dp_agreement": 0.95,
        "confidence_mean": 1.0,
    }
    print("\n=== Metrics vs Targets ===")
    pass_count = 0
    for k, target in targets.items():
        v = metrics[k]
        # For boundary_mae, lower is better; others higher is better
        if k == "boundary_mae_frames":
            ok = v <= target
        else:
            ok = v >= target
        mark = "✅" if ok else "❌"
        print(f"  {mark} {k}: {v:.4f} (target {target})")
        if ok:
            pass_count += 1
    print(f"\nPassed {pass_count}/{len(targets)} targets")

    # Save
    out_path = Path(args.output) if args.output else ckpt_path.parent / f"{ckpt_path.stem}_val_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {out_path}")


if __name__ == "__main__":
    main()
