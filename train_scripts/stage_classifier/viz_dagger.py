#!/usr/bin/env python3
"""Visualize dagger stage-classifier predictions for manual verification.

Reads dagger_with_stage/ (parquet with pseudo stage_progress_gt) + per-episode
metrics JSON. For each episode, renders a contact-sheet PNG:
    - 10 sample frames (5 before boundary, 5 after)
    - Colored label under each frame (green=flat, blue=fold)
    - Timeline bar showing t* position

Also generates overview.html for browsing.

Usage:
    python viz_dagger.py \
        --source /vePFS/.../Task_A/dagger \
        --labeled /vePFS/.../Task_A/dagger_with_stage \
        --metrics-dir /vePFS/.../kai0/cache/dagger_infer_metrics \
        --out-dir /vePFS/.../kai0/dagger_stage_viz
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
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


def load_video_frames_at(video_path: Path, indices: list[int]) -> list[np.ndarray]:
    """Load specific frame indices from mp4. Returns list of (H, W, 3) uint8 RGB."""
    import av
    container = av.open(str(video_path))
    container.streams.video[0].thread_type = "AUTO"
    wanted = set(indices)
    frames = {}
    for i, frame in enumerate(container.decode(video=0)):
        if i in wanted:
            frames[i] = frame.to_ndarray(format="rgb24")
        if len(frames) == len(wanted):
            break
    container.close()
    return [frames[i] for i in indices if i in frames]


def load_pseudo_labels(labeled: Path, ep_idx: int) -> np.ndarray | None:
    """Read stage_progress_gt column. Returns (N,) or None if not found."""
    cands = list((labeled / "data").rglob(f"episode_{ep_idx:06d}.parquet"))
    if not cands:
        return None
    table = pq.read_table(cands[0])
    if "stage_progress_gt" not in table.column_names:
        return None
    return table["stage_progress_gt"].to_numpy()


def render_episode(
    ep_idx: int,
    source: Path,
    labeled: Path,
    metrics: dict,
    out_dir: Path,
    n_samples: int = 10,
    camera_key: str = "top_head",
) -> dict | None:
    """Render single-episode contact sheet. Returns per-ep summary dict or None on error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Read pseudo labels
    sp = load_pseudo_labels(labeled, ep_idx)
    if sp is None:
        logger.warning(f"ep {ep_idx}: no stage_progress_gt in parquet")
        return None
    n = len(sp)
    labels_bin = (sp >= 0.5).astype(int)
    # Boundary = first frame where label == 1 (fold)
    above = np.where(labels_bin >= 1)[0]
    boundary = int(above[0]) if len(above) else n
    t_star = metrics.get("t_star", boundary)
    conf = metrics.get("confidence", 0.0)

    # Sample 10 frames: 5 before boundary, 5 after
    half = n_samples // 2
    if boundary > 0:
        pre = np.linspace(0, max(boundary - 1, 0), half, dtype=int)
    else:
        pre = np.zeros(half, dtype=int)
    if boundary < n:
        post = np.linspace(boundary, n - 1, half, dtype=int)
    else:
        post = np.full(half, n - 1, dtype=int)
    sample_idxs = np.unique(np.concatenate([pre, post]))  # dedup in case boundary=0/N

    # Load frames
    video_path = find_video_path(source, ep_idx, camera_key)
    try:
        frames = load_video_frames_at(video_path, sample_idxs.tolist())
    except Exception as e:
        logger.warning(f"ep {ep_idx}: video load failed: {e}")
        return None

    # Render contact sheet
    cols = len(sample_idxs)
    fig, axes = plt.subplots(
        2, cols, figsize=(2.2 * cols, 5.2), gridspec_kw={"height_ratios": [5, 1]}
    )
    for i, (f, idx) in enumerate(zip(frames, sample_idxs)):
        ax = axes[0, i]
        ax.imshow(f)
        lbl = "fold" if labels_bin[idx] else "flat"
        color = "#1f77b4" if lbl == "fold" else "#2ca02c"
        border_w = 4 if idx == t_star else 0
        ax.set_title(f"f{idx}\n[{lbl}]", color=color, fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("red" if idx == t_star else "gray")
            spine.set_linewidth(border_w if idx == t_star else 0.5)
        ax.set_xticks([]); ax.set_yticks([])

    # Timeline at bottom (spanning all columns)
    gs = axes[1, 0].get_gridspec()
    for ax in axes[1, :]:
        ax.remove()
    ax_tl = fig.add_subplot(gs[1, :])
    time_idx = np.arange(n)
    ax_tl.fill_between(time_idx, 0, labels_bin, step="post", color="#1f77b4", alpha=0.4, label="fold")
    ax_tl.fill_between(time_idx, 0, 1 - labels_bin, step="post", color="#2ca02c", alpha=0.4, label="flat")
    ax_tl.axvline(t_star, color="red", linestyle="--", linewidth=2, label=f"t*={t_star}")
    ax_tl.set_xlim(0, n)
    ax_tl.set_ylim(0, 1)
    ax_tl.set_yticks([])
    ax_tl.set_xlabel("frame index")
    ax_tl.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"episode_{ep_idx:06d}  N={n}  t*={t_star} ({100*t_star/max(n,1):.0f}%)  "
        f"conf={conf:.2f}  n_flat={int((labels_bin==0).sum())}  n_fold={int((labels_bin==1).sum())}",
        fontsize=11,
    )
    out_path = out_dir / f"ep_{ep_idx:06d}.png"
    fig.savefig(out_path, dpi=70, bbox_inches="tight")
    plt.close(fig)

    return {
        "ep_idx": ep_idx,
        "n_frames": int(n),
        "t_star": int(t_star),
        "t_star_ratio": float(t_star) / max(n, 1),
        "confidence": float(conf),
        "n_flat": int((labels_bin == 0).sum()),
        "n_fold": int((labels_bin == 1).sum()),
        "png": out_path.name,
    }


def build_html(summary: list[dict], out_dir: Path):
    """Sortable HTML index of episode thumbnails."""
    rows = sorted(summary, key=lambda x: x["confidence"])  # lowest conf first
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Dagger Stage Predictions — Manual Check</title>",
        "<style>",
        "body { font-family: -apple-system, system-ui, sans-serif; margin: 20px; background: #f5f5f5; }",
        "h1 { color: #333; }",
        "table { border-collapse: collapse; margin-top: 10px; }",
        "td { padding: 10px; vertical-align: top; }",
        ".ep { background: white; padding: 8px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 6px; }",
        ".low-conf { background: #fff3cd; }",
        ".extreme-t { background: #f8d7da; }",
        "img { width: 640px; display: block; }",
        ".meta { font-size: 12px; color: #555; padding: 4px 0; }",
        ".grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }",
        ".controls { position: sticky; top: 0; background: white; padding: 10px; border-bottom: 2px solid #ccc; }",
        "</style></head><body>",
        f"<h1>Dagger 视觉检查 ({len(summary)} episodes)</h1>",
        "<div class='controls'>",
        "<p>排序: <b>按 confidence 升序</b> (low confidence 最先显示，方便定位错误)</p>",
        "<p>高亮: 🟡 低置信度 (conf &lt; 500)  |  🔴 boundary 极端位置 (t*/N &lt; 0.1 或 &gt; 0.9)</p>",
        "</div>",
        "<div class='grid'>",
    ]
    for r in rows:
        cls = "ep"
        if r["confidence"] < 500:
            cls += " low-conf"
        if r["t_star_ratio"] < 0.1 or r["t_star_ratio"] > 0.9:
            cls += " extreme-t"
        meta = (
            f"ep {r['ep_idx']}  N={r['n_frames']}  t*={r['t_star']} ({100*r['t_star_ratio']:.0f}%)  "
            f"conf={r['confidence']:.1f}  flat/fold={r['n_flat']}/{r['n_fold']}"
        )
        html.append(
            f"<div class='{cls}'><div class='meta'>{meta}</div>"
            f"<img src='episodes/{r['png']}' loading='lazy'></div>"
        )
    html.append("</div></body></html>")
    (out_dir / "overview.html").write_text("\n".join(html))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Original dagger/ dataset (for videos)")
    ap.add_argument("--labeled", required=True, help="dagger_with_stage/ (has stage_progress_gt)")
    ap.add_argument("--metrics-dir", required=True, help="Dir with w*.json from infer_dagger")
    ap.add_argument("--out-dir", required=True, help="Output viz dir")
    ap.add_argument("--camera-key", default="top_head")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--max-ep", type=int, default=None, help="Limit episodes for testing (first N)")
    ap.add_argument("--sample-n", type=int, default=None, help="Evenly sample N episodes spanning full range (deterministic)")
    ap.add_argument("--num-workers", type=int, default=1, help="Multi-proc sharding (by ep_idx % num_workers)")
    ap.add_argument("--worker-id", type=int, default=0)
    args = ap.parse_args()

    source = Path(args.source)
    labeled = Path(args.labeled)
    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)

    # Merge all per-worker metrics (may be partial if inference still running)
    all_metrics = {}
    for mf in sorted(metrics_dir.glob("w*.json")):
        try:
            with open(mf) as f:
                all_metrics.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            logger.warning(f"Skip malformed metrics: {mf}")
    logger.info(f"Loaded {len(all_metrics)} episode metrics from {metrics_dir}")

    # Union with parquet-available episodes (handles resume case without metrics)
    parquet_eps = set()
    for p in (labeled / "data").rglob("episode_*.parquet"):
        try:
            parquet_eps.add(int(p.stem.split("_")[-1]))
        except ValueError:
            pass
    logger.info(f"Found {len(parquet_eps)} labeled parquet files in {labeled}/data")
    all_eps = parquet_eps | {int(k) for k in all_metrics.keys()}
    ep_list = sorted(all_eps)
    if args.sample_n is not None and args.sample_n < len(ep_list):
        # Evenly spaced across full range (deterministic, covers early/mid/late)
        idxs = np.linspace(0, len(ep_list) - 1, args.sample_n, dtype=int)
        ep_list = [ep_list[i] for i in idxs]
    if args.max_ep is not None:
        ep_list = ep_list[: args.max_ep]
    if args.num_workers > 1:
        ep_list = [e for i, e in enumerate(ep_list) if i % args.num_workers == args.worker_id]

    summary = []
    for ep_idx in ep_list:
        try:
            s = render_episode(
                ep_idx=ep_idx, source=source, labeled=labeled,
                metrics=all_metrics.get(str(ep_idx), {}), out_dir=out_dir / "episodes",
                n_samples=args.n_samples, camera_key=args.camera_key,
            )
            if s is not None:
                summary.append(s)
        except Exception as e:
            logger.warning(f"ep {ep_idx} render failed: {e}")

    # Save summary JSON
    if args.worker_id == 0 and args.num_workers == 1:
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        build_html(summary, out_dir)
        logger.info(f"Summary + HTML saved to {out_dir}/overview.html")
    else:
        # multi-worker: save shard
        with open(out_dir / f"summary_w{args.worker_id}.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Shard saved: {out_dir}/summary_w{args.worker_id}.json")

    logger.info(f"Rendered {len(summary)} episodes → {out_dir}/episodes/")


if __name__ == "__main__":
    main()
