#!/usr/bin/env python3
"""Train VideoStageClassifier on precomputed V-JEPA 2 / VideoMAE tube features.

Reads cached features from precompute_features.py output. Much faster than
end-to-end training (backbone frozen, no forward/backward).

Usage:
    python train_scripts/stage_classifier/train.py \
        --cache-root /vePFS/.../cache/stage_classifier_vjepa2_ssv2 \
        --split /vePFS/.../Task_A/stage_classifier_split.json \
        --out-dir /vePFS/.../checkpoints/stage_classifier_vjepa2/run1 \
        --num-steps 20000 --batch-size 128 --lr 5e-4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kai0" / "src"))
from openpi.models.video.stage_classifier import (
    VideoStageClassifier, best_boundary_dp, compute_stage_loss, count_trainable_params
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ─── Dataset: flat list of (episode, clip) pairs from cache ──────────────

class ClipDataset(Dataset):
    """All clips preloaded into memory (few GB total, fp16 tubes).

    Each item returns one clip's (tube_feats, labels).
    Boundary-focused sampling: ~70% of clips will be selected near the boundary
    frame (if found), 30% uniform.
    """

    def __init__(
        self,
        cache_root: Path,
        episode_indices: list[int],
        boundary_focus_ratio: float = 0.7,
    ):
        self.cache_root = Path(cache_root)
        self.boundary_focus_ratio = boundary_focus_ratio

        # Preload everything into big tensors
        tube_parts, label_parts = [], []
        boundary_clip_idxs = []
        cursor = 0
        for ep_idx in tqdm(episode_indices, desc="Load cache"):
            path = self.cache_root / f"ep_{ep_idx:06d}.pt"
            if not path.exists():
                continue
            meta = torch.load(path, map_location="cpu", weights_only=False)
            tubes = meta["tube_features"]  # (n_clips, num_tubes, hidden) fp16
            labels = meta["labels"]        # (n_clips, T) uint8
            tube_parts.append(tubes)
            label_parts.append(labels)
            for ci in range(labels.shape[0]):
                if labels[ci, 0] == 0 and labels[ci, -1] == 1:
                    boundary_clip_idxs.append(cursor + ci)
            cursor += labels.shape[0]
        self.tubes = torch.cat(tube_parts, dim=0)    # (N, num_tubes, hidden) fp16
        self.labels = torch.cat(label_parts, dim=0)  # (N, T) uint8
        self.boundary_idxs = torch.tensor(boundary_clip_idxs, dtype=torch.long)

        mem_gb = self.tubes.element_size() * self.tubes.nelement() / 1e9
        logger.info(
            f"Dataset: {self.tubes.shape[0]} clips ({mem_gb:.2f} GB fp16), "
            f"{len(self.boundary_idxs)} boundary clips "
            f"({100 * len(self.boundary_idxs) / max(self.tubes.shape[0], 1):.1f}%)"
        )

    def __len__(self):
        return self.tubes.shape[0]

    def __getitem__(self, idx):
        if random.random() < self.boundary_focus_ratio and len(self.boundary_idxs) > 0:
            idx = int(self.boundary_idxs[random.randrange(len(self.boundary_idxs))])
        tube = self.tubes[idx].float()
        lbl = self.labels[idx].long()
        return tube, lbl


def collate(batch):
    tubes = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.stack([b[1] for b in batch], dim=0)
    return tubes, labels


# ─── Val evaluation ──────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, cache_root, val_episodes, device, fps: float = 30.0):
    """Run full-episode inference + DP boundary on val set."""
    model.eval()

    frame_accs, boundary_maes_frames, monotonic_rates = [], [], []
    dp_agreements, confidences = [], []

    for ep_idx in val_episodes:
        path = Path(cache_root) / f"ep_{ep_idx:06d}.pt"
        if not path.exists():
            continue
        meta = torch.load(path, map_location="cpu", weights_only=False)
        tube_feats = meta["tube_features"].float().to(device)  # (n_clips, num_tubes, hidden)
        labels_clip = meta["labels"].long()                    # (n_clips, T)
        clip_starts = meta["clip_starts"]                      # (n_clips,)
        n_frames = meta["n_frames"]

        T = labels_clip.shape[1]

        # Forward all clips
        logits_clip = model.forward_from_tubes(tube_feats)     # (n_clips, T, 2)

        # Aggregate per-frame logits across overlapping clips
        frame_logits = torch.zeros(n_frames, 2, device=device)
        count = torch.zeros(n_frames, device=device)
        for ci, s in enumerate(clip_starts.tolist()):
            e = min(s + T, n_frames)
            real_len = e - s
            frame_logits[s:e] += logits_clip[ci, :real_len]
            count[s:e] += 1
        count = count.clamp(min=1)
        frame_logits = frame_logits / count.unsqueeze(-1)

        # Ground truth per-frame labels (reconstruct from clip labels)
        frame_gt = torch.zeros(n_frames, dtype=torch.long)
        for ci, s in enumerate(clip_starts.tolist()):
            e = min(s + T, n_frames)
            frame_gt[s:e] = labels_clip[ci, :e - s]

        # Raw monotonic?
        raw_labels = frame_logits.argmax(-1).cpu()
        raw_is_mono = (raw_labels.diff() >= 0).all().item()
        monotonic_rates.append(float(raw_is_mono))

        # DP boundary
        dp_labels, dp_t, conf = best_boundary_dp(frame_logits)
        dp_labels_cpu = dp_labels.cpu()

        # Metrics
        frame_acc = (dp_labels_cpu == frame_gt).float().mean().item()
        gt_labels_diff = frame_gt.diff()
        gt_transitions = (gt_labels_diff != 0).nonzero().flatten()
        if len(gt_transitions) > 0:
            gt_boundary = gt_transitions[0].item()
        else:
            gt_boundary = n_frames - 1 if frame_gt[-1] == 0 else -1
        dp_agreement = (dp_labels_cpu == raw_labels).float().mean().item()

        frame_accs.append(frame_acc)
        boundary_maes_frames.append(abs(dp_t - gt_boundary))
        dp_agreements.append(dp_agreement)
        confidences.append(conf)

    model.train()
    return {
        "frame_accuracy": float(np.mean(frame_accs)) if frame_accs else 0.0,
        "boundary_mae_frames": float(np.mean(boundary_maes_frames)) if boundary_maes_frames else 0.0,
        "boundary_mae_sec": float(np.mean(boundary_maes_frames)) / fps if boundary_maes_frames else 0.0,
        "monotonic_rate_raw": float(np.mean(monotonic_rates)) if monotonic_rates else 0.0,
        "dp_agreement": float(np.mean(dp_agreements)) if dp_agreements else 0.0,
        "confidence_mean": float(np.mean(confidences)) if confidences else 0.0,
        "n_episodes": len(frame_accs),
    }


# ─── Main training loop ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out-dir", required=True)

    # Model
    parser.add_argument("--backbone-hidden", type=int, default=1024)
    parser.add_argument("--num-tubes", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--num-steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--num-workers-dl", type=int, default=8)
    parser.add_argument("--boundary-focus-ratio", type=float, default=0.7)

    # Loss
    parser.add_argument("--class-weight-flat", type=float, default=1.0)
    parser.add_argument("--class-weight-fold", type=float, default=3.0)
    parser.add_argument("--loss-smooth", type=float, default=0.1)
    parser.add_argument("--loss-mono", type=float, default=0.2)

    # Eval / save
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load split
    with open(args.split) as f:
        split = json.load(f)

    # Datasets
    train_ds = ClipDataset(
        args.cache_root, split["train_episodes"],
        boundary_focus_ratio=args.boundary_focus_ratio,
    )
    val_episodes = split["val_episodes"]
    logger.info(f"Train clips: {len(train_ds)}, Val episodes: {len(val_episodes)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers_dl,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers_dl > 0 else False,
    )

    # Model
    model = VideoStageClassifier(
        backbone_hidden=args.backbone_hidden,
        num_tubes=args.num_tubes,
        num_frames=args.num_frames,
        hidden_dim=args.hidden_dim,
        n_cross_attn_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)
    logger.info(f"Trainable params: {count_trainable_params(model):,}")

    # Optimizer + scheduler
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.num_steps
    warmup = args.warmup_steps

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # Training loop
    step = 0
    train_iter = iter(train_loader)
    loss_tracker = []
    t0 = time.time()
    best_frame_acc = 0.0
    best_boundary_mae = float("inf")

    while step < total_steps:
        try:
            tubes, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            tubes, labels = next(train_iter)

        tubes = tubes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model.forward_from_tubes(tubes)
        loss, info = compute_stage_loss(
            logits, labels,
            class_weights=(args.class_weight_flat, args.class_weight_fold),
            smooth_weight=args.loss_smooth,
            mono_weight=args.loss_mono,
        )

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()

        loss_tracker.append(info["loss_total"].item())

        if step % args.log_every == 0 and step > 0:
            avg = np.mean(loss_tracker[-args.log_every:])
            lr = optim.param_groups[0]["lr"]
            elapsed = time.time() - t0
            logger.info(
                f"Step {step}/{total_steps}: loss={avg:.4f}  "
                f"ce={info['loss_ce'].item():.4f}  "
                f"smooth={info['loss_smooth'].item():.4f}  "
                f"mono={info['loss_mono'].item():.4f}  "
                f"lr={lr:.2e}  elapsed={elapsed/60:.1f}min"
            )

        if step % args.val_every == 0 and step > 0:
            logger.info("Running val evaluation...")
            metrics = evaluate(model, args.cache_root, val_episodes, device)
            logger.info(
                f"Val@{step}: frame_acc={metrics['frame_accuracy']:.4f} "
                f"boundary_mae={metrics['boundary_mae_frames']:.1f} frames ({metrics['boundary_mae_sec']:.3f}s) "
                f"mono={metrics['monotonic_rate_raw']:.4f} "
                f"dp_agr={metrics['dp_agreement']:.4f} "
                f"conf={metrics['confidence_mean']:.2f}"
            )
            # Save best by frame_accuracy
            if metrics["frame_accuracy"] > best_frame_acc:
                best_frame_acc = metrics["frame_accuracy"]
                best_path = out_dir / "best.pt"
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "metrics": metrics,
                    "args": vars(args),
                }, best_path)
                logger.info(f"💾 New best frame_acc={best_frame_acc:.4f} → {best_path}")

            # Save best by boundary_mae (lower is better)
            if metrics["boundary_mae_frames"] < best_boundary_mae:
                best_boundary_mae = metrics["boundary_mae_frames"]
                best_mae_path = out_dir / "best_boundary_mae.pt"
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "metrics": metrics,
                    "args": vars(args),
                }, best_mae_path)
                logger.info(
                    f"💾 New best boundary_mae={best_boundary_mae:.1f} frames "
                    f"({best_boundary_mae/30:.3f}s) → {best_mae_path}"
                )

        if step % args.save_every == 0 and step > 0:
            ckpt_path = out_dir / f"step_{step}.pt"
            torch.save({"step": step, "model": model.state_dict()}, ckpt_path)
            logger.info(f"💾 Checkpoint saved to {ckpt_path}")

        step += 1

    # Final save
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "args": vars(args),
    }, out_dir / "final.pt")

    # Final val
    metrics = evaluate(model, args.cache_root, val_episodes, device)
    logger.info(f"Final val: {metrics}")
    logger.info(
        f"Summary: best_frame_acc={best_frame_acc:.4f}, "
        f"best_boundary_mae={best_boundary_mae:.1f} frames ({best_boundary_mae/30:.3f}s)"
    )
    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump({
            **metrics,
            "best_frame_acc": best_frame_acc,
            "best_boundary_mae_frames": best_boundary_mae,
        }, f, indent=2)

    logger.info(f"✅ Training complete. Results in {out_dir}")


if __name__ == "__main__":
    main()
