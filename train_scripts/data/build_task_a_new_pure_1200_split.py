#!/usr/bin/env python3
"""Build pure_vis600 with train/val split into pure_vis600/{base,val}/{data,videos,meta}.

Reads existing flat pure_vis600 (309 originals + 291 mirrors = 600).

Val strategy (avoid hflip leakage):
- Pick 30 source vis_base eps, stratified by date
- For each picked source ep: ALL its derived eps (1 original + N mirrors of it) go to val together
- Train = remaining (originals + mirrors)
- Target: ~60 val (30 originals + ~30 mirrors), ~540 train

Reuses existing parquet (already has correct schema) and re-symlinks/re-copies videos.
"""
from __future__ import annotations
import argparse, json, random, shutil, sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict

SRC = Path("/home/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_pure_1200")
DST = SRC  # restructure in-place
CHUNK = 0
FPS = 30
CAMERAS = ("top_head", "hand_left", "hand_right")
PROMPT = "Flatten and fold the cloth."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-pairs", type=int, default=50,
                    help="Number of (source vis_base ep) pairs held out. Each pair = 1 original + its mirrors.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    eps = [json.loads(l) for l in (SRC / "meta" / "episodes.jsonl").open()]
    print(f"loaded {len(eps)} eps from pure_vis600 flat")

    originals = [e for e in eps if e["kind"] == "original"]
    mirrors = [e for e in eps if e["kind"] == "mirror"]
    print(f"  originals: {len(originals)}, mirrors: {len(mirrors)}")

    # Group by (source, source_ep) - "pair" = 1 original + N mirrors of same source_ep
    pair_key = lambda src, ep: (src, ep)
    pairs = defaultdict(list)  # (source, source_ep) -> [eps]
    for o in originals:
        pairs[pair_key(o["orig_source"], o["orig_ep"])].append(o)
    for m in mirrors:
        # mirror's `mirror_of_orig_ep` is the source vis_base ep id
        pairs[pair_key(m["orig_source"], m["mirror_of_orig_ep"])].append(m)

    valid_pairs = list(pairs.keys())
    print(f"  unique (source, source_ep) pairs: {len(valid_pairs)}")

    # Stratify by source date
    by_date = defaultdict(list)
    for k in valid_pairs:
        by_date[k[0]].append(k)
    sizes = {d: len(v) for d, v in by_date.items()}
    print(f"  pairs per date: {sizes}")

    # Allocate val pairs proportionally
    rng = random.Random(args.seed)
    val_target = {}
    total = sum(sizes.values())
    for d, n in sizes.items():
        val_target[d] = max(1, round(args.val_pairs * n / total))
    while sum(val_target.values()) > args.val_pairs:
        big = max(val_target, key=val_target.get)
        val_target[big] -= 1
    while sum(val_target.values()) < args.val_pairs:
        small = min(val_target, key=val_target.get)
        val_target[small] += 1
    print(f"  val pair allocation: {val_target}")

    # Pick val pair keys
    val_keys = []
    for d, n_pick in val_target.items():
        candidates = sorted(by_date[d])
        rng.shuffle(candidates)
        val_keys.extend(candidates[:n_pick])
    val_keys_set = set(val_keys)

    # Split eps
    val_eps_orig = []  # all eps belonging to val pairs
    train_eps_orig = []
    for k, eps_in_pair in pairs.items():
        if k in val_keys_set:
            val_eps_orig.extend(eps_in_pair)
        else:
            train_eps_orig.extend(eps_in_pair)

    # Sort by original episode_index for stable reindex
    val_eps_orig.sort(key=lambda x: x["episode_index"])
    train_eps_orig.sort(key=lambda x: x["episode_index"])

    val_orig_cnt = sum(1 for e in val_eps_orig if e["kind"] == "original")
    val_mir_cnt = sum(1 for e in val_eps_orig if e["kind"] == "mirror")
    train_orig_cnt = sum(1 for e in train_eps_orig if e["kind"] == "original")
    train_mir_cnt = sum(1 for e in train_eps_orig if e["kind"] == "mirror")
    print(f"\nsplit: train={len(train_eps_orig)} ({train_orig_cnt} orig + {train_mir_cnt} mir)")
    print(f"       val=  {len(val_eps_orig)} ({val_orig_cnt} orig + {val_mir_cnt} mir)")
    print(f"       val pairs: {len(val_keys)}")

    if args.dry_run:
        return

    # Backup existing flat data dirs
    backup = SRC.parent / (SRC.name + "_flat_backup")
    needs_restructure = (SRC / "data").exists() and not (SRC / "base").exists()
    if needs_restructure:
        if backup.exists():
            if args.force:
                shutil.rmtree(backup)
            else:
                print(f"ERROR: backup {backup} exists. Use --force.", file=sys.stderr); sys.exit(2)
        backup.mkdir()
        for sub in ("data", "videos", "meta", "manifest.json", "README.md"):
            src_p = SRC / sub
            if src_p.exists():
                shutil.move(str(src_p), str(backup / sub))
        print(f"backed up flat layout -> {backup}")
        flat_data = backup / "data"
        flat_videos = backup / "videos"
        flat_meta = backup / "meta"
    else:
        flat_data = SRC / "data"
        flat_videos = SRC / "videos"
        flat_meta = SRC / "meta"

    # Read template info from backup or current
    info_template = json.loads((flat_meta / "info.json").read_text())

    # Write splits
    for split_name, split_eps in [("base", train_eps_orig), ("val", val_eps_orig)]:
        dst_split = SRC / split_name
        if dst_split.exists():
            if args.force:
                shutil.rmtree(dst_split)
            else:
                print(f"ERROR: {dst_split} exists. Use --force.", file=sys.stderr); sys.exit(2)
        (dst_split / "meta").mkdir(parents=True)

        new_eps_meta = []
        total_frames = 0
        for new_ep, e_orig in enumerate(split_eps):
            old_ep = e_orig["episode_index"]
            src_pq = flat_data / f"chunk-{CHUNK:03d}" / f"episode_{old_ep:06d}.parquet"
            dst_pq = dst_split / "data" / f"chunk-{CHUNK:03d}" / f"episode_{new_ep:06d}.parquet"
            t = pq.read_table(src_pq)
            n = t.num_rows
            t = t.set_column(t.schema.get_field_index("episode_index"),
                             "episode_index", pa.array([new_ep] * n, type=pa.int64()))
            t = t.set_column(t.schema.get_field_index("index"),
                             "index", pa.array(list(range(total_frames, total_frames + n)), type=pa.int64()))
            t = t.set_column(t.schema.get_field_index("timestamp"),
                             "timestamp", pa.array((np.arange(n, dtype=np.float32) / FPS), type=pa.float32()))
            if "task_index" in t.column_names:
                t = t.set_column(t.schema.get_field_index("task_index"),
                                 "task_index", pa.array([0] * n, type=pa.int64()))
            dst_pq.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(t, dst_pq)

            # videos: re-symlink or re-copy. For symlinks (originals), follow to source.
            # For real files (mirrors), make a symlink to the file in backup.
            for cam in CAMERAS:
                src_vid = flat_videos / f"chunk-{CHUNK:03d}" / f"observation.images.{cam}" / f"episode_{old_ep:06d}.mp4"
                if not src_vid.exists():
                    raise FileNotFoundError(f"missing {src_vid}")
                dst_vid = dst_split / "videos" / f"chunk-{CHUNK:03d}" / f"observation.images.{cam}" / f"episode_{new_ep:06d}.mp4"
                dst_vid.parent.mkdir(parents=True, exist_ok=True)
                if dst_vid.exists() or dst_vid.is_symlink():
                    dst_vid.unlink()
                # If src_vid is symlink, follow to original source. Else, link to the real file.
                if src_vid.is_symlink():
                    target = src_vid.resolve()
                else:
                    target = src_vid.resolve()  # absolute path to real file in backup
                dst_vid.symlink_to(target)

            new_eps_meta.append({
                "episode_index": new_ep,
                "tasks": [PROMPT],
                "length": n,
                "kind": e_orig["kind"],
                "orig_source": e_orig["orig_source"],
                "orig_ep": e_orig["orig_ep"],
                **({"mirror_of_orig_ep": e_orig["mirror_of_orig_ep"]} if "mirror_of_orig_ep" in e_orig else {}),
            })
            total_frames += n

        # Write info.json
        info_out = dict(info_template)
        info_out["total_episodes"] = len(split_eps)
        info_out["total_frames"] = total_frames
        info_out["total_videos"] = len(split_eps) * len(CAMERAS)
        info_out["total_chunks"] = 1
        info_out["chunks_size"] = max(1000, len(split_eps))
        info_out["splits"] = {split_name: f"0:{len(split_eps)}"}
        (dst_split / "meta" / "info.json").write_text(json.dumps(info_out, indent=2))

        with (dst_split / "meta" / "episodes.jsonl").open("w") as f:
            for em in new_eps_meta:
                f.write(json.dumps(em) + "\n")
        (dst_split / "meta" / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": PROMPT}) + "\n")

        print(f"  wrote {split_name}: {len(split_eps)} eps / {total_frames} frames -> {dst_split}")

    # New manifest at top-level
    (SRC / "manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "val_pairs": args.val_pairs,
        "train_total": len(train_eps_orig),
        "train_originals": train_orig_cnt,
        "train_mirrors": train_mir_cnt,
        "val_total": len(val_eps_orig),
        "val_originals": val_orig_cnt,
        "val_mirrors": val_mir_cnt,
        "val_pair_per_date": val_target,
        "prompt": PROMPT,
        "note": "val keeps source-vis_base-ep PAIRS together (orig + mirrors) to avoid hflip leakage.",
    }, indent=2))

    print(f"\nbuilt: {SRC}/{{base,val}}")


if __name__ == "__main__":
    main()
