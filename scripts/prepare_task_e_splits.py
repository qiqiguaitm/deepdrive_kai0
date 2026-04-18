#!/usr/bin/env python3
"""
Prepare Task E train/val split with a seeded random shuffle.

- Source (read-only):  /data1/DATA_IMP/KAI0/Task_E_2026-04-17/base
- Target work copy:    /data1/tim/workspace/deepdive_kai0/kai0/data/Task_E/{base,val}

64 train + 9 val (seed=42). RGB videos are symlinked (large), parquet
files are copied with rewritten episode_index/index columns and renamed
to be consecutive. Depth zarr is skipped — agilex policy never reads it.

Usage:
    python scripts/prepare_task_e_splits.py                 # do it
    python scripts/prepare_task_e_splits.py --dry-run       # plan only
    python scripts/prepare_task_e_splits.py --force         # overwrite existing target
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC_ROOT = Path("/data1/DATA_IMP/KAI0/Task_E_2026-04-17/base")
DST_ROOT = Path("/data1/tim/workspace/deepdive_kai0/kai0/data/Task_E")
SEED = 42
N_VAL = 9
FPS = 30  # training fps (also declared in info.json)
CAMERAS_RGB = ("top_head", "hand_left", "hand_right")
CHUNK = 0  # single chunk


def load_source_meta() -> tuple[dict, list[dict], list[dict]]:
    info = json.loads((SRC_ROOT / "meta" / "info.json").read_text())
    episodes = [json.loads(l) for l in (SRC_ROOT / "meta" / "episodes.jsonl").read_text().splitlines()]
    tasks = [json.loads(l) for l in (SRC_ROOT / "meta" / "tasks.jsonl").read_text().splitlines()]
    return info, episodes, tasks


def make_split(n_total: int, n_val: int, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.RandomState(seed)
    perm = np.arange(n_total)
    rng.shuffle(perm)
    val_ids = sorted(perm[:n_val].tolist())
    train_ids = sorted(perm[n_val:].tolist())
    return train_ids, val_ids


def strip_depth_features(features: dict) -> dict:
    return {k: v for k, v in features.items() if not k.startswith("observation.depth.")}


def copy_parquet_reindexed(
    src_file: Path,
    dst_file: Path,
    new_ep_idx: int,
    global_offset: int,
) -> int:
    """Copy parquet with rewritten episode_index column and global index.

    Returns number of frames (to advance global_offset)."""
    import numpy as np  # local to keep top-level imports minimal

    t = pq.read_table(src_file)
    n = t.num_rows
    # Rewrite episode_index (all rows = new_ep_idx)
    ep_col = pa.array([new_ep_idx] * n, type=pa.int64())
    t = t.set_column(t.schema.get_field_index("episode_index"), "episode_index", ep_col)
    # Rewrite global `index` to be global_offset .. global_offset+n-1
    idx_col = pa.array(list(range(global_offset, global_offset + n)), type=pa.int64())
    t = t.set_column(t.schema.get_field_index("index"), "index", idx_col)
    # Rewrite timestamp to exact 1/fps * frame_index. Source data has jittery
    # wall-clock timestamps that break LeRobot's uniform-timestamp check.
    ts_col = pa.array((np.arange(n, dtype=np.float32) / FPS), type=pa.float32())
    t = t.set_column(t.schema.get_field_index("timestamp"), "timestamp", ts_col)
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, dst_file)
    return n


def symlink_videos(src_dir: Path, dst_dir: Path, old_ep: int, new_ep: int):
    """Symlink 3 RGB mp4 files from old episode name to new episode name.

    Source dirs are bare names (top_head) but LeRobot's get_video_file_path
    substitutes the full feature key (observation.images.top_head) into the
    video_path template, so destination must use the full key.
    """
    for cam in CAMERAS_RGB:
        src = src_dir / "videos" / f"chunk-{CHUNK:03d}" / cam / f"episode_{old_ep:06d}.mp4"
        if not src.exists():
            raise FileNotFoundError(f"missing video: {src}")
        dst_key = f"observation.images.{cam}"
        dst = dst_dir / "videos" / f"chunk-{CHUNK:03d}" / dst_key / f"episode_{new_ep:06d}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())


def build_split(
    split_name: str,
    old_ids: list[int],
    src_info: dict,
    src_episodes: list[dict],
    src_tasks: list[dict],
    dry_run: bool,
):
    dst = DST_ROOT / split_name
    print(f"\n[{split_name}] {len(old_ids)} episodes → {dst}")
    print(f"  original episode_ids: {old_ids}")

    if dry_run:
        total = sum(src_episodes[i]["length"] for i in old_ids)
        print(f"  dry-run: {total} frames would be written")
        return

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(parents=True, exist_ok=True)

    new_episodes = []
    total_frames = 0
    for new_ep, old_ep in enumerate(old_ids):
        src_pq = SRC_ROOT / "data" / f"chunk-{CHUNK:03d}" / f"episode_{old_ep:06d}.parquet"
        dst_pq = dst / "data" / f"chunk-{CHUNK:03d}" / f"episode_{new_ep:06d}.parquet"
        n = copy_parquet_reindexed(src_pq, dst_pq, new_ep, total_frames)
        symlink_videos(SRC_ROOT, dst, old_ep, new_ep)

        src_ep_meta = src_episodes[old_ep]
        # LeRobot v2.1 expects keys: episode_index, tasks (list[str]), length.
        # We keep custom fields as bookkeeping extras.
        prompt = src_ep_meta.get("prompt", "stand up the fallen box")
        new_ep_meta = {
            "episode_index": new_ep,
            "tasks": [prompt],
            "length": src_ep_meta["length"],
            "duration_s": src_ep_meta.get("duration_s"),
            "operator": src_ep_meta.get("operator"),
            "orig_episode_id": old_ep,
            "success": src_ep_meta.get("success", True),
        }
        new_episodes.append(new_ep_meta)

        total_frames += n
        if (new_ep + 1) % 10 == 0 or new_ep == len(old_ids) - 1:
            print(f"  wrote {new_ep + 1}/{len(old_ids)} (frames so far: {total_frames})")

    # Meta
    new_info = dict(src_info)
    new_info["total_episodes"] = len(old_ids)
    new_info["total_frames"] = total_frames
    new_info["total_videos"] = len(old_ids) * len(CAMERAS_RGB)
    new_info["total_chunks"] = 1
    new_info["splits"] = {split_name: f"0:{len(old_ids)}"}
    new_info["features"] = strip_depth_features(new_info["features"])
    new_info.pop("depth_path", None)
    (dst / "meta" / "info.json").write_text(json.dumps(new_info, indent=2))

    with (dst / "meta" / "episodes.jsonl").open("w") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep) + "\n")
    with (dst / "meta" / "tasks.jsonl").open("w") as f:
        for t in src_tasks:
            f.write(json.dumps(t) + "\n")

    print(f"  [{split_name}] done: {len(old_ids)} ep, {total_frames} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite existing base/ val/")
    args = ap.parse_args()

    info, episodes, tasks = load_source_meta()
    assert info["total_episodes"] == len(episodes) == 73, f"unexpected src size: {info['total_episodes']}"
    train_ids, val_ids = make_split(len(episodes), N_VAL, SEED)
    assert len(train_ids) + len(val_ids) == len(episodes)
    assert set(train_ids).isdisjoint(val_ids)

    # Sanity: val ids should span collection range (not last-9 clumped)
    first_idx_of_val = min(val_ids)
    last_idx_of_val = max(val_ids)
    span = (last_idx_of_val - first_idx_of_val) / (len(episodes) - 1)
    print(f"val_ids = {val_ids}")
    print(f"val span over collection order = {span:.2%} (should be >= 60%)")
    assert span >= 0.6, "val split clumped; change seed"

    # Clean target
    for split in ("base", "val"):
        p = DST_ROOT / split
        if p.exists():
            if p.is_symlink():
                p.unlink()
            elif args.force:
                print(f"[force] removing existing {p}")
                shutil.rmtree(p)
            else:
                print(f"ERROR: {p} already exists. Pass --force to overwrite.", file=sys.stderr)
                sys.exit(2)

    # Splits manifest
    manifest = {
        "seed": SEED,
        "src": str(SRC_ROOT),
        "n_total": len(episodes),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "train_ids": train_ids,
        "val_ids": val_ids,
    }
    if not args.dry_run:
        DST_ROOT.mkdir(parents=True, exist_ok=True)
        (DST_ROOT / "splits.json").write_text(json.dumps(manifest, indent=2))
        print(f"wrote {DST_ROOT / 'splits.json'}")

    build_split("base", train_ids, info, episodes, tasks, args.dry_run)
    build_split("val", val_ids, info, episodes, tasks, args.dry_run)

    if not args.dry_run:
        print("\n✅ DONE. Train/val ready at:")
        print(f"  {DST_ROOT}/base   ({len(train_ids)} ep)")
        print(f"  {DST_ROOT}/val    ({len(val_ids)} ep)")
        print(f"  {DST_ROOT}/splits.json")


if __name__ == "__main__":
    main()
