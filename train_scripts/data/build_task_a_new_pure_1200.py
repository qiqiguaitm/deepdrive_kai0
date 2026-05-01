#!/usr/bin/env python3
"""Build A_new_pure_1200 = ALL *-new subdirs of vis_base + mirror to fill 1200 eps.

"Mirror" 数据增强:
  - state / action: 前 7 维 (左臂) ↔ 后 7 维 (右臂) 互换 (swap_arms_in_array)
  - top_head 视频: 水平翻转
  - hand_left 视频 → 翻转后落到 hand_right 通道
  - hand_right 视频 → 翻转后落到 hand_left 通道
镜像后整条 episode 在物理上是原 episode 的左右对称版.

Sources (gf0):
  /home/tim/workspace/deepdive_kai0/kai0/data/Task_A/vis_base/<date>/{data,meta,videos}/

Dest:
  /home/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/pure_vis600/

Layout:
  new_ep 000..309 = original (parquet 实拷, 视频 symlink, 0 额外磁盘)
  new_ep 310..599 = mirror   (parquet swap + 写新, 视频 ffmpeg hflip 真实重编码)

Usage:
  python build_pure_vis600.py [--seed 42] [--total 600] [--dry-run] [--force] [--workers 16]
"""
from __future__ import annotations
import argparse, json, os, random, shutil, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/home/tim/workspace/deepdive_kai0/kai0/data/Task_A")
VIS_BASE = Path("/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/vis_base")  # *-new subdirs only (filtered in collect_vis_base)
DST_DEFAULT = ROOT / "self_built" / "A_new_pure_1200"

CAMERAS = ("top_head", "hand_left", "hand_right")
PROMPT = "Flatten and fold the cloth."
FPS = 30
CHUNK = 0
LEFT_DIM = 7
RIGHT_DIM = 7
FFMPEG = "/home/tim/workspace/deepdive_kai0/kai0/.venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"


def _read_eps(meta_path: Path) -> list[dict]:
    out = []
    for line in meta_path.open():
        d = json.loads(line)
        ep_id = d.get("episode_index", d.get("episode_id"))
        if ep_id is None:
            raise ValueError(f"no episode index in {meta_path}: {d}")
        out.append({"src_ep": int(ep_id), "length": int(d["length"])})
    return out


def _ep_files_present(date_dir: Path, src_ep: int) -> list[Path] | None:
    """Return [parquet, top_head.mp4, hand_left.mp4, hand_right.mp4] or None if any missing."""
    paths = [date_dir / "data" / f"chunk-{CHUNK:03d}" / f"episode_{src_ep:06d}.parquet"]
    for cam in CAMERAS:
        paths.append(date_dir / "videos" / f"chunk-{CHUNK:03d}" / cam / f"episode_{src_ep:06d}.mp4")
    return paths if all(p.exists() for p in paths) else None


def _probe_mp4(p: Path) -> tuple[Path, bool]:
    """ffmpeg null-decode probe; True = readable header & no decode error."""
    try:
        r = subprocess.run([FFMPEG, "-v", "error", "-i", str(p), "-f", "null", "-"],
                           capture_output=True, text=True, timeout=60)
        return (p, r.returncode == 0 and not r.stderr.strip())
    except Exception:
        return (p, False)


def collect_vis_base(root: Path, workers: int = 32) -> list[dict]:
    """Walk vis_base dates, probe every mp4 in parallel, drop eps with any corrupt video.

    Production recording sometimes leaves mp4s without `moov` atom if the recorder
    was killed before flush — those decode-fail in ffmpeg/cv2 and would break training.
    """
    candidates = []  # (date_dir, src_ep, length, [4 paths])
    for date_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith("-new")):  # filter to *-new dirs only
        ep_file = date_dir / "meta" / "episodes.jsonl"
        if not ep_file.exists():
            continue
        for e in _read_eps(ep_file):
            paths = _ep_files_present(date_dir, e["src_ep"])
            if paths is not None:
                candidates.append((date_dir, e["src_ep"], e["length"], paths))
    print(f"  candidates with all 4 files present: {len(candidates)}; probing mp4 integrity ...")

    probe_jobs: list[tuple[int, Path]] = []
    for i, (_, _, _, paths) in enumerate(candidates):
        for mp4 in paths[1:]:
            probe_jobs.append((i, mp4))
    bad_idx: set[int] = set()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_probe_mp4, mp4): (idx, mp4) for idx, mp4 in probe_jobs}
        for fut in as_completed(futs):
            idx, mp4 = futs[fut]
            _, ok = fut.result()
            if not ok:
                bad_idx.add(idx)

    if bad_idx:
        print(f"  [warn] dropping {len(bad_idx)} episode(s) with corrupt mp4(s):")
        for i in sorted(bad_idx):
            d, ep, _, _ = candidates[i]
            print(f"    vis_base/{d.name}  src_ep={ep}")

    items = []
    for i, (date_dir, src_ep, length, _) in enumerate(candidates):
        if i in bad_idx:
            continue
        items.append({
            "src_dir": str(date_dir),
            "src_ep": src_ep,
            "length": length,
            "source": f"vis_base/{date_dir.name}",
        })
    return items


def swap_arms_flat(arr_list: list[float]) -> list[float]:
    """Swap first LEFT_DIM dims with last RIGHT_DIM dims (preserve any padding)."""
    total = LEFT_DIM + RIGHT_DIM
    if len(arr_list) < total:
        raise ValueError(f"array len {len(arr_list)} < {total}")
    head = arr_list[:LEFT_DIM]
    mid = arr_list[LEFT_DIM:total]
    tail = arr_list[total:]
    return mid + head + tail


def write_parquet(src: Path, dst: Path, new_ep: int, global_offset: int, mirror: bool) -> int:
    t = pq.read_table(src)
    n = t.num_rows
    if mirror:
        # swap state and action element-wise (each row is a list of 14 floats)
        for col in ("observation.state", "action"):
            if col in t.column_names:
                vals = t.column(col).to_pylist()
                swapped = [swap_arms_flat(v) for v in vals]
                t = t.set_column(t.schema.get_field_index(col), col,
                                 pa.array(swapped, type=t.column(col).type))
    t = t.set_column(t.schema.get_field_index("episode_index"),
                     "episode_index", pa.array([new_ep] * n, type=pa.int64()))
    t = t.set_column(t.schema.get_field_index("index"),
                     "index", pa.array(list(range(global_offset, global_offset + n)), type=pa.int64()))
    t = t.set_column(t.schema.get_field_index("timestamp"),
                     "timestamp", pa.array((np.arange(n, dtype=np.float32) / FPS), type=pa.float32()))
    if "task_index" in t.column_names:
        t = t.set_column(t.schema.get_field_index("task_index"),
                         "task_index", pa.array([0] * n, type=pa.int64()))
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, dst)
    return n


def symlink_video(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def hflip_video(src: Path, dst: Path) -> tuple[Path, bool, str]:
    """ffmpeg hflip → libx264. Output dir must already exist."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    # NOTE: -preset ultrafast -bf 0 -x264opts keyint=15:scenecut=0 produces fast random-seek
    # (matches vis_base orig 0.9ms/seek; preset=veryfast was 3.2ms/seek = 3.5x slower training).
    # Cost: file ~2.5x larger (1.4 MB -> 3.5 MB) but +1.7 GB total is trivial.
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
           "-vf", "hflip", "-c:v", "libx264",
           "-preset", "ultrafast", "-bf", "0",
           "-x264opts", "keyint=15:min-keyint=15:scenecut=0",
           "-crf", "23", "-pix_fmt", "yuv420p", "-an", str(dst)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        return (dst, False, p.stderr[-300:])
    return (dst, True, "")


def write_videos_for_ep(info: dict, new_ep: int, dst: Path, mirror: bool) -> list[tuple[Path, Path, bool]]:
    """Return list of (src, dst, needs_ffmpeg) tuples to be processed.
    For original: symlink immediately, return empty list.
    For mirror: caller will run ffmpeg in parallel."""
    src_vid_root = Path(info["src_dir"]) / "videos" / f"chunk-{CHUNK:03d}"
    work = []
    if not mirror:
        # original: symlink, bare cam name → observation.images.<cam>
        for cam in CAMERAS:
            src = src_vid_root / cam / f"episode_{info['src_ep']:06d}.mp4"
            dst_cam = f"observation.images.{cam}"
            d = dst / "videos" / f"chunk-{CHUNK:03d}" / dst_cam / f"episode_{new_ep:06d}.mp4"
            symlink_video(src, d)
        return work
    # mirror:
    #   top_head    -> observation.images.top_head    (hflip)
    #   hand_left   -> observation.images.hand_right  (hflip + cam-name-swap)
    #   hand_right  -> observation.images.hand_left   (hflip + cam-name-swap)
    src_top  = src_vid_root / "top_head"    / f"episode_{info['src_ep']:06d}.mp4"
    src_left = src_vid_root / "hand_left"   / f"episode_{info['src_ep']:06d}.mp4"
    src_right= src_vid_root / "hand_right"  / f"episode_{info['src_ep']:06d}.mp4"
    dst_top  = dst / "videos" / f"chunk-{CHUNK:03d}" / "observation.images.top_head"   / f"episode_{new_ep:06d}.mp4"
    dst_left = dst / "videos" / f"chunk-{CHUNK:03d}" / "observation.images.hand_left"  / f"episode_{new_ep:06d}.mp4"
    dst_right= dst / "videos" / f"chunk-{CHUNK:03d}" / "observation.images.hand_right" / f"episode_{new_ep:06d}.mp4"
    work.append((src_top,  dst_top,  True))
    work.append((src_left, dst_right, True))   # left → right
    work.append((src_right, dst_left, True))   # right → left
    return work


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=str(DST_DEFAULT))
    ap.add_argument("--workers", type=int, default=16, help="ffmpeg parallel workers (per video)")
    args = ap.parse_args()

    print("=== sources ===")
    vis = collect_vis_base(VIS_BASE, workers=args.workers)
    n_vis = len(vis)
    print(f"  vis_base: {n_vis} clean eps (use ALL as originals)")
    if n_vis == 0:
        print("ERROR: vis_base empty.", file=sys.stderr); sys.exit(2)

    n_mirror = args.total - n_vis
    if n_mirror < 0:
        print(f"ERROR: vis_base ({n_vis}) >= total ({args.total}); shrink total.", file=sys.stderr)
        sys.exit(2)
    if n_mirror > n_vis:
        print(f"WARN: requested {n_mirror} mirrors but only {n_vis} originals; capping to {n_vis}")
        n_mirror = n_vis

    rng = random.Random(args.seed)
    mirror_picks = rng.sample(vis, n_mirror)

    # final ordering: originals (sorted by date,src_ep for reproducibility) then mirrors
    originals = sorted(vis, key=lambda x: (x["source"], x["src_ep"]))
    final = (
        [{**e, "kind": "original", "src_info": e}                  for e in originals] +
        [{**e, "kind": "mirror",   "src_info": e, "mirror_of_orig": None} for e in mirror_picks]
    )
    # populate mirror_of_orig (new_ep of the same source episode in originals[])
    orig_lookup = {(e["source"], e["src_ep"]): i for i, e in enumerate(originals)}
    for i, e in enumerate(final):
        if e["kind"] == "mirror":
            e["mirror_of_orig"] = orig_lookup.get((e["source"], e["src_ep"]))

    print(f"\n=== plan ===")
    print(f"  originals: {len(originals)} eps  → new_ep 000..{len(originals)-1:03d}")
    print(f"  mirrors:   {n_mirror} eps  → new_ep {len(originals):03d}..{len(final)-1:03d} (sampled w/ seed={args.seed})")
    print(f"  TOTAL:     {len(final)} eps")
    print(f"  dest:      {args.out}")

    if args.dry_run:
        print("\n--- dry-run preview (first 5 / mirrors first 3) ---")
        for i, e in enumerate(final[:5]):
            print(f"  new_ep={i:03d}  {e['kind']:8s}  {e['source']:30s}  src_ep={e['src_ep']}  len={e['length']}")
        print("  ...")
        for i, e in enumerate(final[len(originals):len(originals)+3]):
            new_ep = len(originals) + i
            print(f"  new_ep={new_ep:03d}  {e['kind']:8s}  {e['source']:30s}  src_ep={e['src_ep']}  ↪ mirror_of_orig=ep{e['mirror_of_orig']:03d}")
        return

    dst = Path(args.out)
    if dst.exists():
        if args.force:
            print(f"[force] removing {dst}")
            shutil.rmtree(dst)
        else:
            print(f"ERROR: {dst} exists. Use --force.", file=sys.stderr); sys.exit(2)
    (dst / "meta").mkdir(parents=True, exist_ok=True)

    # ---------- write parquet + collect ffmpeg jobs ----------
    print(f"\n[1/3] writing parquet + queueing video work ...")
    new_eps_meta = []
    total_frames = 0
    ffmpeg_jobs: list[tuple[Path, Path]] = []
    for new_ep, e in enumerate(final):
        info = e["src_info"]
        mirror = e["kind"] == "mirror"
        src_pq = Path(info["src_dir"]) / "data" / f"chunk-{CHUNK:03d}" / f"episode_{info['src_ep']:06d}.parquet"
        dst_pq = dst / "data" / f"chunk-{CHUNK:03d}" / f"episode_{new_ep:06d}.parquet"
        n = write_parquet(src_pq, dst_pq, new_ep, total_frames, mirror)
        for src, d, needs_ff in write_videos_for_ep(info, new_ep, dst, mirror):
            if needs_ff:
                ffmpeg_jobs.append((src, d))
        rec = {
            "episode_index": new_ep,
            "tasks": [PROMPT],
            "length": n,
            "kind": e["kind"],
            "orig_source": info["source"],
            "orig_ep": info["src_ep"],
        }
        if mirror:
            rec["mirror_of_orig_ep"] = e["mirror_of_orig"]
        new_eps_meta.append(rec)
        total_frames += n
    print(f"   parquet done. {total_frames} frames, {len(ffmpeg_jobs)} videos to flip.")

    # ---------- ffmpeg in parallel ----------
    print(f"\n[2/3] flipping {len(ffmpeg_jobs)} videos with {args.workers} workers ...")
    failed = []
    if ffmpeg_jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(hflip_video, s, d): (s, d) for s, d in ffmpeg_jobs}
            done = 0
            for fut in as_completed(futs):
                s, d = futs[fut]
                _, ok, err = fut.result()
                done += 1
                if not ok:
                    failed.append((s, d, err))
                if done % 50 == 0 or done == len(ffmpeg_jobs):
                    print(f"   {done}/{len(ffmpeg_jobs)}  fails={len(failed)}", flush=True)
    if failed:
        print(f"\n!!! {len(failed)} video flips failed; first 3:", file=sys.stderr)
        for s, d, err in failed[:3]:
            print(f"   {s} → {d}\n     {err}", file=sys.stderr)
        sys.exit(3)

    # ---------- meta ----------
    print(f"\n[3/3] writing meta ...")
    # use kai0_base info.json as template (same agilex schema as vis_base)
    info_template = json.loads((ROOT / "kai0_base" / "meta" / "info.json").read_text())
    info_template["total_episodes"] = len(final)
    info_template["total_frames"] = total_frames
    info_template["total_videos"] = len(final) * len(CAMERAS)
    info_template["total_chunks"] = 1
    info_template["chunks_size"] = max(1000, len(final))
    info_template["splits"] = {"train": f"0:{len(final)}"}
    info_template["features"] = {k: v for k, v in info_template["features"].items()
                                  if not k.startswith("observation.depth.")}
    info_template.pop("depth_path", None)
    (dst / "meta" / "info.json").write_text(json.dumps(info_template, indent=2))
    with (dst / "meta" / "episodes.jsonl").open("w") as f:
        for ep in new_eps_meta:
            f.write(json.dumps(ep) + "\n")
    (dst / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": PROMPT}) + "\n")

    manifest = {
        "seed": args.seed,
        "total": len(final),
        "originals_count": len(originals),
        "mirrors_count": n_mirror,
        "vis_base_dates": sorted(set(e["source"] for e in originals)),
        "mirror_picked_orig_eps": sorted([e["src_ep"] for e in mirror_picks]),
        "left_dim": LEFT_DIM,
        "right_dim": RIGHT_DIM,
        "prompt": PROMPT,
        "video_codec": "h264 (libx264, crf=23, preset=ultrafast, bf=0, keyint=15 — decode-friendly)",
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n✅ built pure_vis600: {len(final)} eps, {total_frames} frames at {dst}")


if __name__ == "__main__":
    main()
