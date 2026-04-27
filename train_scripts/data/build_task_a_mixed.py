#!/usr/bin/env python3
"""Build a mixed Task_A dataset on gf1.

Sources
-------
Supports two source layouts for ``--vis-root``:

1. Dated dynamic layout:
   /vePFS/visrobot01/KAI0/Task_A/<date>/<base|dagger>
2. Flat official Kai0 layout:
   /path/to/Task_A/{base,dagger,advantage}

Optionally, ``--old-root`` may add extra Task_A/{base,dagger} buckets.
All source buckets are balanced to the minimum available episode count.

Key normalizations
------------------
- video cam dir: `<cam>/` (visrobot01) or `observation.images.<cam>/` → unified as
  `observation.images.<cam>/` (what the training loader expects)
- meta/episodes.jsonl: {episode_id,length,...} or {episode_index,tasks,length} →
  unified as v2.1: {episode_index, tasks:[prompt], length}
- prompt: unified to "Flatten and fold the cloth."
- depth videos: skipped (training doesn't use them)
- parquet: re-indexed episode_index + global index + uniform timestamp

Usage
-----
    python build_task_a_mixed.py                 # run
    python build_task_a_mixed.py --dry-run       # plan only (prints counts + sources)
    python build_task_a_mixed.py --force         # overwrite existing dest
    python build_task_a_mixed.py --seed 7        # reproducibility

After building, run (not part of this script):
    python train_scripts/data/generate_episodes_stats.py <dst>/base
    cd kai0 && .venv/bin/python scripts/compute_norm_states_fast.py \\
        --config-name pi05_flatten_fold_mixed_visrobot01
"""
from __future__ import annotations
import argparse, json, random, shutil, sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VIS_ROOT = "/vePFS/visrobot01/KAI0/Task_A"
OLD_ROOT = "/home/tim/workspace/deepdive_kai0/kai0/data/Task_A"
DST_ROOT = "/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A_mixed_gf1"
SEED = 42
FPS = 30
CAMERAS = ("top_head", "hand_left", "hand_right")
PROMPT = "Flatten and fold the cloth."
CHUNK = 0
KEEP_COLUMNS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


def _read_ep(meta_path: Path) -> list[dict]:
    """Parse episodes.jsonl, return list with common keys."""
    out = []
    for line in meta_path.open():
        d = json.loads(line)
        ep_id = d.get("episode_index", d.get("episode_id"))
        if ep_id is None:
            raise ValueError(f"no episode index in {meta_path}: {d}")
        out.append({"src_ep": ep_id, "length": d["length"], "raw": d})
    return out


def _ep_has_all_cams(kind_dir: Path, ep_id: int, cam_naming: str, src_chunk: int | None = None) -> bool:
    """Check that parquet + all 3 RGB videos exist for this episode."""
    chunk = CHUNK if src_chunk is None else src_chunk
    pq = kind_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_id:06d}.parquet"
    if not pq.exists():
        return False
    for cam in CAMERAS:
        cam_dir = cam if cam_naming == "bare" else f"observation.images.{cam}"
        mp4 = kind_dir / "videos" / f"chunk-{chunk:03d}" / cam_dir / f"episode_{ep_id:06d}.mp4"
        if not mp4.exists():
            return False
    return True


def collect_visrobot01_date_first(root: Path) -> list[dict]:
    """Walk date-subdirs (2026-MM-DD/{base,dagger}), gather episodes with COMPLETE data (parquet + 3 cams)."""
    items = []
    skipped_counts = {}
    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir() or not date_dir.name.startswith("2026"):
            continue
        for kind in ("base", "dagger"):
            kind_dir = date_dir / kind
            ep_file = kind_dir / "meta" / "episodes.jsonl"
            if not ep_file.exists():
                continue
            all_eps = _read_ep(ep_file)
            kept = 0
            for e in all_eps:
                if _ep_has_all_cams(kind_dir, e["src_ep"], "bare"):
                    items.append({
                        "src_dir": str(kind_dir),
                        "src_ep": e["src_ep"],
                        "src_chunk": 0,   # visrobot01 single-chunk
                        "length": e["length"],
                        "source": f"visrobot01/{date_dir.name}/{kind}",
                        "cam_naming": "bare",
                    })
                    kept += 1
            skipped = len(all_eps) - kept
            if skipped > 0:
                skipped_counts[f"{date_dir.name}/{kind}"] = (kept, len(all_eps), skipped)
    if skipped_counts:
        print("  [warning] incomplete episodes skipped (missing parquet or 3-cam videos):")
        for src, (kept, total, miss) in skipped_counts.items():
            print(f"    {src}: kept {kept}/{total}, skipped {miss}")
    return items


def collect_visrobot01_kind_first(root: Path) -> list[dict]:
    """Walk kind/date-subdirs ({base,dagger}/2026-MM-DD), gather episodes with COMPLETE data."""
    items = []
    skipped_counts = {}
    for kind in ("base", "dagger"):
        kind_root = root / kind
        if not kind_root.is_dir():
            continue
        for date_dir in sorted(kind_root.iterdir()):
            if not date_dir.is_dir() or not date_dir.name.startswith("2026"):
                continue
            ep_file = date_dir / "meta" / "episodes.jsonl"
            if not ep_file.exists():
                continue
            all_eps = _read_ep(ep_file)
            kept = 0
            for e in all_eps:
                if _ep_has_all_cams(date_dir, e["src_ep"], "bare"):
                    items.append({
                        "src_dir": str(date_dir),
                        "src_ep": e["src_ep"],
                        "src_chunk": 0,
                        "length": e["length"],
                        "source": f"visrobot01/{kind}/{date_dir.name}",
                        "cam_naming": "bare",
                    })
                    kept += 1
            skipped = len(all_eps) - kept
            if skipped > 0:
                skipped_counts[f"{kind}/{date_dir.name}"] = (kept, len(all_eps), skipped)
    if skipped_counts:
        print("  [warning] incomplete episodes skipped (missing parquet or 3-cam videos):")
        for src, (kept, total, miss) in skipped_counts.items():
            print(f"    {src}: kept {kept}/{total}, skipped {miss}")
    return items


def collect_task_a_direct(root: Path) -> dict[str, list[dict]]:
    """Collect direct child datasets under Task_A/{base,dagger,advantage}."""
    buckets: dict[str, list[dict]] = {}
    skipped_counts = {}
    for kind in ("base", "dagger", "advantage"):
        kind_dir = root / kind
        ep_file = kind_dir / "meta" / "episodes.jsonl"
        if not ep_file.exists():
            continue
        all_eps = _read_ep(ep_file)
        kept_items = []
        for e in all_eps:
            src_chunk = e["src_ep"] // 1000
            if _ep_has_all_cams(kind_dir, e["src_ep"], "observation.images", src_chunk=src_chunk):
                kept_items.append({
                    "src_dir": str(kind_dir),
                    "src_ep": e["src_ep"],
                    "src_chunk": src_chunk,
                    "length": e["length"],
                    "source": f"task_a_root/{kind}",
                    "cam_naming": "observation.images",
                })
        if kept_items:
            buckets[f"task_a_root/{kind}"] = kept_items
        skipped = len(all_eps) - len(kept_items)
        if skipped > 0:
            skipped_counts[kind] = (len(kept_items), len(all_eps), skipped)
    if skipped_counts:
        print("  [warning] incomplete direct-root episodes skipped (missing parquet or 3-cam videos):")
        for src, (kept, total, miss) in skipped_counts.items():
            print(f"    {src}: kept {kept}/{total}, skipped {miss}")
    return buckets


def collect_source_buckets(root: Path) -> dict[str, list[dict]]:
    """Collect source buckets from either dated or flat Task_A layout."""
    dated_items = collect_visrobot01_date_first(root)
    if dated_items:
        return {"visroot/all": dated_items}

    kind_first_items = collect_visrobot01_kind_first(root)
    if kind_first_items:
        return {"visroot/all": kind_first_items}

    direct_buckets = collect_task_a_direct(root)
    if direct_buckets:
        return direct_buckets

    raise ValueError(
        f"Could not discover supported source layout under {root}. "
        "Expected either <date>/<base|dagger> or direct {base,dagger,advantage} subdatasets."
    )


def collect_existing(root: Path, subdir: str) -> list[dict]:
    """Existing Task_A/{base,dagger} uses multi-chunk layout: ep N lives in chunk-<N//1000>."""
    kind_dir = root / subdir
    ep_file = kind_dir / "meta" / "episodes.jsonl"
    if not ep_file.exists():
        return []
    return [
        {
            "src_dir": str(kind_dir),
            "src_ep": e["src_ep"],
            "src_chunk": e["src_ep"] // 1000,   # multi-chunk (1000 eps per chunk)
            "length": e["length"],
            "source": f"existing/{subdir}",
            "cam_naming": "observation.images",
        }
        for e in _read_ep(ep_file)
    ]


def copy_parquet(src: Path, dst: Path, new_ep: int, global_offset: int) -> tuple[int, set[int]]:
    t = pq.read_table(src)
    keep_cols = [c for c in KEEP_COLUMNS if c in t.column_names]
    missing = [c for c in KEEP_COLUMNS if c not in t.column_names]
    if missing:
        raise ValueError(f"missing required parquet columns in {src}: {missing}")
    t = t.select(keep_cols)
    n = t.num_rows
    observed_task_indices = {int(v) for v in t.column("task_index").to_pylist()}
    t = t.set_column(t.schema.get_field_index("episode_index"),
                     "episode_index", pa.array([new_ep] * n, type=pa.int64()))
    t = t.set_column(t.schema.get_field_index("index"),
                     "index", pa.array(list(range(global_offset, global_offset + n)), type=pa.int64()))
    t = t.set_column(t.schema.get_field_index("timestamp"),
                     "timestamp", pa.array((np.arange(n, dtype=np.float32) / FPS), type=pa.float32()))
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, dst)
    return n, observed_task_indices


def symlink_video(info: dict, new_ep: int, dst_root: Path):
    src_chunk = info.get("src_chunk", 0)
    src_vid_root = Path(info["src_dir"]) / "videos" / f"chunk-{src_chunk:03d}"
    for cam in CAMERAS:
        src_cam_dir = cam if info["cam_naming"] == "bare" else f"observation.images.{cam}"
        src = src_vid_root / src_cam_dir / f"episode_{info['src_ep']:06d}.mp4"
        if not src.exists():
            raise FileNotFoundError(f"missing video {src}")
        dst_cam = f"observation.images.{cam}"
        dst = dst_root / "videos" / f"chunk-{CHUNK:03d}" / dst_cam / f"episode_{new_ep:06d}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())


def _write_split(dst_split: Path, picks: list[dict], old_root: Path):
    (dst_split / "meta").mkdir(parents=True)
    new_episodes = []
    total_frames = 0
    observed_task_indices: set[int] = set()
    for new_ep, info in enumerate(picks):
        src_chunk = info.get("src_chunk", 0)
        src_pq = Path(info["src_dir"]) / "data" / f"chunk-{src_chunk:03d}" / f"episode_{info['src_ep']:06d}.parquet"
        dst_pq = dst_split / "data" / f"chunk-{CHUNK:03d}" / f"episode_{new_ep:06d}.parquet"
        n, task_indices = copy_parquet(src_pq, dst_pq, new_ep, total_frames)
        symlink_video(info, new_ep, dst_split)
        observed_task_indices.update(task_indices)
        new_episodes.append({
            "episode_index": new_ep,
            "tasks": [PROMPT],
            "length": n,
            "orig_source": info["source"],
            "orig_ep": info["src_ep"],
        })
        total_frames += n
    # info.json
    info_template = json.loads((old_root / "base" / "meta" / "info.json").read_text())
    info_template["total_episodes"] = len(picks)
    info_template["total_frames"] = total_frames
    info_template["total_videos"] = len(picks) * len(CAMERAS)
    info_template["total_chunks"] = 1
    info_template["chunks_size"] = max(1, len(picks))
    info_template["total_tasks"] = max(1, len(observed_task_indices))
    info_template["splits"] = {dst_split.name: f"0:{len(picks)}"}
    info_template["features"] = {k: v for k, v in info_template["features"].items()
                                  if not k.startswith("observation.depth.")}
    info_template.pop("depth_path", None)
    (dst_split / "meta" / "info.json").write_text(json.dumps(info_template, indent=2))
    with (dst_split / "meta" / "episodes.jsonl").open("w") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep) + "\n")
    # Keep the mixed dataset prompt text unified, but emit one tasks.jsonl row per
    # task_index actually present in the rewritten parquet files. This avoids
    # runtime KeyError when source buckets contribute labeled frames (e.g. 0/1).
    task_rows = "\n".join(
        json.dumps({"task_index": task_index, "task": PROMPT})
        for task_index in sorted(observed_task_indices or {0})
    )
    (dst_split / "meta" / "tasks.jsonl").write_text(task_rows + "\n")
    return total_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis-root", default=VIS_ROOT)
    ap.add_argument(
        "--old-root",
        default=OLD_ROOT,
        help="Optional existing Task_A root with {base,dagger}. If same realpath as vis-root, skipped.",
    )
    ap.add_argument("--out-root", default=DST_ROOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--val-size", type=int, default=0,
                    help="total held-out eps budget; split evenly across discovered source buckets (0=no val split)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    source_buckets = collect_source_buckets(Path(args.vis_root))
    print("source buckets:")
    for src, items in source_buckets.items():
        print(f"    {src}: {len(items)}")

    vis_root_resolved = Path(args.vis_root).resolve()
    old_root_resolved = Path(args.old_root).resolve() if args.old_root else None
    if old_root_resolved is not None and old_root_resolved == vis_root_resolved:
        print("old_root resolves to vis_root; skipping extra existing/{base,dagger} buckets to avoid duplication")
    elif args.old_root:
        base_eps = collect_existing(Path(args.old_root), "base")
        dagger_eps = collect_existing(Path(args.old_root), "dagger")
        advantage_eps = collect_existing(Path(args.old_root), "advantage")
        if base_eps:
            source_buckets["existing/base"] = base_eps
            print(f"    existing/base: {len(base_eps)}")
        if dagger_eps:
            source_buckets["existing/dagger"] = dagger_eps
            print(f"    existing/dagger: {len(dagger_eps)}")
        if advantage_eps:
            source_buckets["existing/advantage"] = advantage_eps
            print(f"    existing/advantage: {len(advantage_eps)}")

    if not source_buckets:
        print("ERROR: no source buckets found.", file=sys.stderr); sys.exit(3)

    bucket_sizes = {name: len(items) for name, items in source_buckets.items()}
    N_total = min(bucket_sizes.values())
    if N_total == 0:
        print("ERROR: at least one source bucket is empty.", file=sys.stderr); sys.exit(3)

    num_sources = len(source_buckets)
    val_per_src = args.val_size // num_sources if args.val_size > 0 else 0
    N_train = N_total - val_per_src
    N_val = val_per_src
    if N_train <= 0:
        print(f"ERROR: val-size {args.val_size} leaves no train data.", file=sys.stderr); sys.exit(3)

    rng = random.Random(args.seed)
    sampled_by_bucket: dict[str, list[dict]] = {}
    train_picks = []
    val_picks = []
    for name, items in source_buckets.items():
        sampled = rng.sample(items, N_train + N_val)
        sampled_by_bucket[name] = sampled
        train_picks.extend(sampled[:N_train])
        val_picks.extend(sampled[N_train:])

    print(f"\nsplit: {num_sources} source buckets, per-source N_train={N_train}, N_val={N_val}")
    print(f"  train total: {len(train_picks)} eps")
    print(f"  val total:   {len(val_picks)} eps")

    if args.dry_run:
        print("\n--- dry-run: train first 5 / val all ---")
        for ep in train_picks[:5]:
            print(f"  TRAIN {ep['source']:40s}  ep={ep['src_ep']}")
        for ep in val_picks:
            print(f"  VAL   {ep['source']:40s}  ep={ep['src_ep']}")
        return

    dst = Path(args.out_root)
    if dst.exists():
        if args.force:
            print(f"[force] removing {dst}")
            shutil.rmtree(dst)
        else:
            print(f"ERROR: {dst} exists. Use --force.", file=sys.stderr); sys.exit(2)

    print(f"\nwriting train to {dst}/base ...")
    train_frames = _write_split(dst / "base", train_picks, Path(args.old_root))
    print(f"  train: {len(train_picks)} eps, {train_frames} frames")

    if N_val > 0:
        print(f"writing val to {dst}/val ...")
        val_frames = _write_split(dst / "val", val_picks, Path(args.old_root))
        print(f"  val:   {len(val_picks)} eps, {val_frames} frames")
    else:
        val_frames = 0

    manifest = {
        "seed": args.seed,
        "prompt": PROMPT,
        "train_episodes": len(train_picks),
        "train_frames": train_frames,
        "val_episodes": len(val_picks),
        "val_frames": val_frames,
        "N_train_per_source": N_train,
        "N_val_per_source": N_val,
        "source_bucket_sizes": bucket_sizes,
        "selected_orig_ids_by_bucket": {
            name: sorted(e["src_ep"] for e in sampled)
            for name, sampled in sampled_by_bucket.items()
        },
        "source_train_val_counts": {
            name: {"train": N_train, "val": N_val}
            for name in source_buckets.keys()
        },
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n✅ built: {dst}")
    print(f"   train: {len(train_picks)} eps / {train_frames} frames")
    print(f"   val:   {len(val_picks)} eps / {val_frames} frames")
    print(f"\nNext:")
    print(f"   python train_scripts/data/generate_episodes_stats.py {dst}/base")
    if N_val > 0:
        print(f"   python train_scripts/data/generate_episodes_stats.py {dst}/val")
    print(f"   compute_norm_states_fast.py --config-name pi05_flatten_fold_mixed_visrobot01")


if __name__ == "__main__":
    main()
