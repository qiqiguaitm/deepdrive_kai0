#!/usr/bin/env python3
"""Sanity-check human `stage_progress_gt` label precision.

For each sampled val episode:
  1. Read parquet → get stage_progress_gt per frame (continuous, not binarized)
  2. Find GT boundary = first frame where sp >= 0.5
  3. Render 15 frames densely around boundary (±7), annotate GT value + label
  4. Attach full progress curve for wider context

Stratified sampling: 10 uniform + 5 short (N<500) + 5 long (N>1200) val episodes.

Usage:
  python sanity_check_labels.py \
    --source /vePFS/.../Task_A/advantage \
    --split  /vePFS/.../stage_classifier_split.json \
    --out-dir /vePFS/.../label_sanity_viz \
    --n-uniform 10 --n-short 5 --n-long 5 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_video_path(source: Path, ep_idx: int, camera_key: str) -> Path:
    video_name = f"episode_{ep_idx:06d}.mp4"
    for chunk_dir in sorted((source / "videos").iterdir()):
        if not chunk_dir.is_dir():
            continue
        cand = chunk_dir / f"observation.images.{camera_key}" / video_name
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No video for ep {ep_idx} cam {camera_key}")


def load_frames_at(video_path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    """Returns {frame_idx: (H,W,3) uint8 RGB}."""
    import av
    container = av.open(str(video_path))
    container.streams.video[0].thread_type = "AUTO"
    wanted = set(indices)
    frames = {}
    for i, frame in enumerate(container.decode(video=0)):
        if i in wanted:
            frames[i] = frame.to_ndarray(format="rgb24")
        if len(frames) == len(wanted) or i > max(indices):
            break
    container.close()
    return frames


def load_gt_labels(source: Path, ep_idx: int) -> tuple[np.ndarray, int]:
    """Returns (stage_progress_gt [N] float32, N_parquet)."""
    cands = list((source / "data").rglob(f"episode_{ep_idx:06d}.parquet"))
    if not cands:
        raise FileNotFoundError(f"No parquet for ep {ep_idx}")
    table = pq.read_table(cands[0])
    if "stage_progress_gt" not in table.column_names:
        raise ValueError(f"Ep {ep_idx}: missing stage_progress_gt")
    sp = table["stage_progress_gt"].to_numpy().astype(np.float32)
    return sp, len(sp)


def render_episode(
    ep_idx: int,
    source: Path,
    out_dir: Path,
    model_pred: dict | None = None,
    camera_key: str = "top_head",
    n_frames_around: int = 7,
) -> dict | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    sp, N = load_gt_labels(source, ep_idx)
    # Training label = binary (sp >= 0.5). Human annotation = the ONE boundary frame.
    train_labels = (sp >= 0.5).astype(int)
    above = np.where(train_labels >= 1)[0]
    if len(above) == 0:
        logger.warning(f"ep {ep_idx}: no fold frames? skip")
        return None
    boundary = int(above[0])  # this IS the human-annotated stage boundary

    pred_boundary = model_pred.get("pred_boundary_dp") if model_pred else None
    pred_conf = model_pred.get("confidence") if model_pred else None
    offset_dp = (pred_boundary - boundary) if pred_boundary is not None else None

    # Frames to show: cover both GT and pred with buffer
    if pred_boundary is not None:
        span_lo = min(boundary, pred_boundary) - n_frames_around
        span_hi = max(boundary, pred_boundary) + n_frames_around
    else:
        span_lo = boundary - n_frames_around
        span_hi = boundary + n_frames_around
    start = max(0, span_lo)
    end = min(N - 1, span_hi)
    sample_idxs = list(range(start, end + 1))

    video_path = find_video_path(source, ep_idx, camera_key)
    frames_map = load_frames_at(video_path, sample_idxs)
    frames = [frames_map.get(i) for i in sample_idxs]

    # Stage lengths
    n_flat = int((train_labels == 0).sum())
    n_fold = int((train_labels == 1).sum())

    # Layout: top row = dense zoom frames; bottom row = binary label timeline
    n_cols = len(sample_idxs)
    fig = plt.figure(figsize=(2.0 * n_cols, 6))
    # Top frames
    for i, (f, idx) in enumerate(zip(frames, sample_idxs)):
        ax = fig.add_subplot(2, n_cols, i + 1)
        if f is not None:
            ax.imshow(f)
        is_gt = idx == boundary
        is_pred = pred_boundary is not None and idx == pred_boundary
        offset = idx - boundary
        lbl_int = int(train_labels[idx])
        lbl_str = "FOLD" if lbl_int == 1 else "flat"
        color = "#1f77b4" if lbl_int == 1 else "#2ca02c"
        # Titles
        markers = []
        if is_gt: markers.append("◆GT◆")
        if is_pred: markers.append("◇PRED◇")
        prefix = (" ".join(markers) + "\n") if markers else ""
        title = f"{prefix}f{idx}  (Δ{offset:+d})\nlabel={lbl_int}  [{lbl_str}]"
        fontweight = "bold" if (is_gt or is_pred) else "normal"
        ax.set_title(title, color=color, fontsize=9, fontweight=fontweight)
        # Border: red=GT, purple=pred, overlap = both (gradient not supported; use yellow)
        if is_gt and is_pred:
            border_color, border_w = "gold", 4
        elif is_gt:
            border_color, border_w = "red", 4
        elif is_pred:
            border_color, border_w = "#9467bd", 4  # purple
        else:
            border_color, border_w = "lightgray", 0.5
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(border_w)
        ax.set_xticks([]); ax.set_yticks([])

    # Bottom: binary training label as step function
    ax_tl = fig.add_subplot(2, 1, 2)
    t = np.arange(N)
    # Step function: filled green under label=0, filled blue under label=1
    ax_tl.fill_between(t, 0, 1 - train_labels, step="post", color="#2ca02c", alpha=0.5, label="flat (label=0)")
    ax_tl.fill_between(t, 0, train_labels, step="post", color="#1f77b4", alpha=0.5, label="fold (label=1)")
    ax_tl.axvline(boundary, color="red", linestyle="--", linewidth=2,
                  label=f"GT (human): frame {boundary}")
    if pred_boundary is not None:
        ax_tl.axvline(pred_boundary, color="#9467bd", linestyle="-.", linewidth=2,
                      label=f"MODEL pred: frame {pred_boundary}  (Δ={offset_dp:+d})")
    # Highlight zoom zone
    ax_tl.add_patch(Rectangle(
        (start - 0.5, -0.05), (end - start + 1), 1.15,
        linewidth=0, facecolor="orange", alpha=0.2, zorder=0,
    ))
    ax_tl.set_xlim(0, N)
    ax_tl.set_ylim(-0.05, 1.1)
    ax_tl.set_xlabel("frame index")
    ax_tl.set_ylabel("training label (binary)")
    ax_tl.legend(loc="center left", fontsize=10)
    ax_tl.grid(True, alpha=0.3)
    ax_tl.set_yticks([0, 1])

    pred_text = (f"  |  MODEL pred=frame {pred_boundary}  (Δ={offset_dp:+d} frames)  conf={pred_conf:.1f}"
                 if pred_boundary is not None else "")
    fig.suptitle(
        f"ep {ep_idx:06d}   N={N}   GT=frame {boundary} ({100*boundary/N:.0f}%){pred_text}   "
        f"|   flat/fold={n_flat}/{n_fold}",
        fontsize=12, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = out_dir / f"ep_{ep_idx:06d}.png"
    fig.savefig(out_path, dpi=75, bbox_inches="tight")
    plt.close(fig)

    return {
        "ep_idx": ep_idx,
        "N": int(N),
        "gt_boundary": int(boundary),
        "gt_boundary_ratio": float(boundary) / N,
        "pred_boundary": int(pred_boundary) if pred_boundary is not None else None,
        "offset_dp": int(offset_dp) if offset_dp is not None else None,
        "abs_offset": int(abs(offset_dp)) if offset_dp is not None else None,
        "confidence": float(pred_conf) if pred_conf is not None else None,
        "n_flat": n_flat,
        "n_fold": n_fold,
        "png": out_path.name,
    }


def sample_episodes(
    val_ep: list[int],
    source: Path,
    n_uniform: int = 10,
    n_short: int = 5,
    n_long: int = 5,
    short_N: int = 500,
    long_N: int = 1200,
    seed: int = 42,
) -> list[int]:
    """Stratified sampling."""
    rng = random.Random(seed)

    # Get N per episode
    ep_lengths = {}
    for e in val_ep:
        try:
            _, N = load_gt_labels(source, e)
            ep_lengths[e] = N
        except Exception:
            continue

    short_pool = [e for e, n in ep_lengths.items() if n < short_N]
    long_pool = [e for e, n in ep_lengths.items() if n > long_N]
    rest_pool = [e for e in val_ep if e in ep_lengths and e not in short_pool and e not in long_pool]

    selected = []
    selected += rng.sample(rest_pool, min(n_uniform, len(rest_pool)))
    selected += rng.sample(short_pool, min(n_short, len(short_pool)))
    selected += rng.sample(long_pool, min(n_long, len(long_pool)))

    return sorted(selected)


def build_html(summary: list[dict], out_dir: Path, sort_by: str = "abs_offset"):
    if sort_by == "abs_offset":
        # Worst predictions first
        rows = sorted(summary, key=lambda x: -(x.get("abs_offset") or 0))
    elif sort_by == "confidence":
        rows = sorted(summary, key=lambda x: (x.get("confidence") or 0))
    else:
        rows = sorted(summary, key=lambda x: x["ep_idx"])

    # Aggregate stats
    has_pred = [r for r in summary if r.get("abs_offset") is not None]
    stats_block = ""
    if has_pred:
        offsets = [r["abs_offset"] for r in has_pred]
        signed = [r["offset_dp"] for r in has_pred]
        bins = {"=0": 0, "1-2": 0, "3-5": 0, "6-10": 0, "11-20": 0, ">20": 0}
        for o in offsets:
            if o == 0: bins["=0"] += 1
            elif o <= 2: bins["1-2"] += 1
            elif o <= 5: bins["3-5"] += 1
            elif o <= 10: bins["6-10"] += 1
            elif o <= 20: bins["11-20"] += 1
            else: bins[">20"] += 1
        import numpy as np
        stats_block = (
            f"<div class='stats'>"
            f"<b>Model vs GT 统计</b> (N={len(has_pred)}):<br>"
            f"boundary_mae = {np.mean(offsets):.2f} frames ({np.mean(offsets)/30:.3f}s) | "
            f"median = {np.median(offsets):.1f} | max = {max(offsets)}<br>"
            f"signed mean (+ = model 偏晚) = {np.mean(signed):+.2f}<br>"
            f"分布: " + " | ".join(f"{k}={v}" for k, v in bins.items()) +
            f"</div>"
        )

    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Label Sanity Check — GT vs Model</title>",
        "<style>",
        "body { font-family: -apple-system, system-ui, sans-serif; margin: 20px; background: #f5f5f5; }",
        "h1 { color: #333; }",
        ".ep { background: white; padding: 10px; border-radius: 6px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
        ".ep-big-offset { border-left: 6px solid #d62728; }",
        ".ep-perfect { border-left: 6px solid #2ca02c; }",
        "img { width: 100%; max-width: 1800px; display: block; }",
        ".meta { font-size: 13px; color: #555; padding: 4px 0; font-family: monospace; }",
        ".instr { background: #fff3cd; padding: 12px; border-radius: 6px; margin-bottom: 10px; line-height: 1.6; }",
        ".stats { background: #d1ecf1; padding: 12px; border-radius: 6px; margin-bottom: 15px; font-family: monospace; line-height: 1.8; }",
        ".controls { position: sticky; top: 0; background: white; padding: 10px; border-bottom: 2px solid #ccc; z-index: 100; }",
        "</style></head><body>",
        f"<h1>Sanity Check — GT (human) vs MODEL prediction — {len(summary)} val episodes</h1>",
        "<div class='controls'><b>排序</b>: " + sort_by + "</div>",
        "<div class='instr'>",
        "<b>图例</b>：<br>",
        "&nbsp;&nbsp;<span style='color:red'>◆GT◆</span> 红色粗框 = 人工标注的 boundary frame (训练用 GT)<br>",
        "&nbsp;&nbsp;<span style='color:#9467bd'>◇PRED◇</span> 紫色粗框 = 模型 DP 预测的 boundary<br>",
        "&nbsp;&nbsp;<span style='color:#b8860b'>金色框</span> = GT 和 PRED 重合 (完美)<br><br>",
        "<b>请判断</b>：两者哪个更接近真实 flat→fold 切换？<br>",
        "&nbsp;&nbsp;- GT 正确 + PRED 偏 N 帧 → 模型误差<br>",
        "&nbsp;&nbsp;- GT 偏 + PRED 准 → <b>人工标签有噪声</b> (模型更聪明)<br>",
        "&nbsp;&nbsp;- GT 和 PRED 都偏 → 都不准 (罕见，通常发生在难 ep)<br><br>",
        "下半部分：step function = 训练 label (绿 flat / 蓝 fold)，红虚=GT，紫点划=模型预测",
        "</div>",
        stats_block,
    ]
    for r in rows:
        cls = "ep"
        abs_off = r.get("abs_offset")
        if abs_off is not None:
            if abs_off == 0:
                cls += " ep-perfect"
            elif abs_off > 10:
                cls += " ep-big-offset"

        pred_str = ""
        if r.get("pred_boundary") is not None:
            pred_str = f"  |  PRED=f{r['pred_boundary']}  Δ={r['offset_dp']:+d}  conf={r['confidence']:.1f}"
        meta = (
            f"ep_{r['ep_idx']:06d}  N={r['N']}  GT=f{r['gt_boundary']} "
            f"({100*r['gt_boundary_ratio']:.0f}%){pred_str}  |  flat/fold={r['n_flat']}/{r['n_fold']}"
        )
        html.append(
            f"<div class='{cls}'><div class='meta'>{meta}</div>"
            f"<img src='episodes/{r['png']}' loading='lazy'></div>"
        )
    html.append("</body></html>")
    (out_dir / "overview.html").write_text("\n".join(html))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="advantage/ dataset root (has GT)")
    ap.add_argument("--split", required=True, help="stage_classifier_split.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--camera-key", default="top_head")
    ap.add_argument("--n-frames-around", type=int, default=7, help="Frames ±k around GT boundary")
    ap.add_argument("--model-preds", default=None, help="JSON from eval_val_predictions.py")
    ap.add_argument("--sort-by", default="abs_offset", choices=["abs_offset", "ep_idx", "confidence"],
                    help="HTML order (default: worst prediction first)")
    ap.add_argument("--n-uniform", type=int, default=10)
    ap.add_argument("--n-short", type=int, default=5)
    ap.add_argument("--n-long", type=int, default=5)
    ap.add_argument("--short-N", type=int, default=500)
    ap.add_argument("--long-N", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out_dir)
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)

    with open(args.split) as f:
        split = json.load(f)
    val_ep = split["val_episodes"]
    logger.info(f"Val pool: {len(val_ep)} episodes")

    # Load model predictions if available
    model_preds = {}
    if args.model_preds and Path(args.model_preds).exists():
        with open(args.model_preds) as f:
            model_preds = json.load(f)
        logger.info(f"Loaded {len(model_preds)} model predictions")

    # Sampling: if --n-uniform >= total val, just use all val episodes
    if args.n_uniform + args.n_short + args.n_long >= len(val_ep):
        selected = sorted(val_ep)
        logger.info(f"Requested >= val pool size; using all {len(selected)} val episodes")
    else:
        selected = sample_episodes(
            val_ep, source,
            n_uniform=args.n_uniform, n_short=args.n_short, n_long=args.n_long,
            short_N=args.short_N, long_N=args.long_N, seed=args.seed,
        )
    logger.info(f"Rendering {len(selected)} episodes")

    summary = []
    for i, ep_idx in enumerate(selected):
        try:
            s = render_episode(
                ep_idx=ep_idx, source=source, out_dir=out_dir / "episodes",
                model_pred=model_preds.get(str(ep_idx)),
                camera_key=args.camera_key, n_frames_around=args.n_frames_around,
            )
            if s:
                summary.append(s)
            if (i + 1) % 20 == 0:
                logger.info(f"  {i+1}/{len(selected)} rendered")
        except Exception as e:
            logger.warning(f"  ep {ep_idx} failed: {e}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    build_html(summary, out_dir, sort_by=args.sort_by)
    logger.info(f"✅ Done. {len(summary)} episodes → {out_dir}/overview.html")


if __name__ == "__main__":
    main()
