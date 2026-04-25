#!/usr/bin/env python3
"""Build mix_vis600 with train/val split into mix_vis600/{base,val}/{data,videos,meta}.

Identical source collection to build_task_a_mix_vis600.py but adds stratified val split.
"""
from __future__ import annotations
import argparse, json, random, shutil, sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/home/tim/workspace/deepdive_kai0/kai0/data/Task_A")
VIS_BASE = ROOT / "vis_base"
KAI0_BASE = ROOT / "kai0_base"
KAI0_DAGGER = ROOT / "kai0_dagger"
DST_DEFAULT = ROOT / "self_built" / "mix_vis600"
REFERENCE_INFO = ROOT / "kai0_base" / "meta" / "info.json"

CAMERAS = ("top_head", "hand_left", "hand_right")
PROMPT = "Flatten and fold the cloth."
FPS = 30
CHUNK = 0


def _read_eps(meta_path: Path) -> list[dict]:
    out = []
    for line in meta_path.open():
        d = json.loads(line)
        ep_id = d.get("episode_index", d.get("episode_id"))
        if ep_id is None:
            raise ValueError(f"no episode index in {meta_path}: {d}")
        out.append({"src_ep": int(ep_id), "length": int(d["length"])})
    return out


def _ep_complete(parquet_p: Path, vid_root: Path, cam_naming: str, src_ep: int) -> bool:
    if not parquet_p.exists():
        return False
    for cam in CAMERAS:
        cam_dir = cam if cam_naming == "bare" else f"observation.images.{cam}"
        mp4 = vid_root / cam_dir / f"episode_{src_ep:06d}.mp4"
        if not mp4.exists():
            return False
    return True


def collect_vis_base(root: Path) -> list[dict]:
    items = []
    for date_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        ep_file = date_dir / "meta" / "episodes.jsonl"
        if not ep_file.exists():
            continue
        eps = _read_eps(ep_file)
        for e in eps:
            pq_p = date_dir / "data" / f"chunk-{CHUNK:03d}" / f"episode_{e['src_ep']:06d}.parquet"
            vid_root = date_dir / "videos" / f"chunk-{CHUNK:03d}"
            if _ep_complete(pq_p, vid_root, "bare", e["src_ep"]):
                items.append({
                    "src_dir": str(date_dir),
                    "src_ep": e["src_ep"],
                    "src_chunk": CHUNK,
                    "length": e["length"],
                    "source": f"vis_base/{date_dir.name}",
                    "bucket": "vis_base",
                    "cam_naming": "bare",
                })
    return items


def collect_multichunk(root: Path, source_label: str) -> list[dict]:
    info = json.loads((root / "meta" / "info.json").read_text())
    chunks_size = info.get("chunks_size", 1000)
    eps = _read_eps(root / "meta" / "episodes.jsonl")
    out = []
    for e in eps:
        ch = e["src_ep"] // chunks_size
        pq_p = root / "data" / f"chunk-{ch:03d}" / f"episode_{e['src_ep']:06d}.parquet"
        vid_root = root / "videos" / f"chunk-{ch:03d}"
        if _ep_complete(pq_p, vid_root, "observation.images", e["src_ep"]):
            out.append({
                "src_dir": str(root),
                "src_ep": e["src_ep"],
                "src_chunk": ch,
                "length": e["length"],
                "source": source_label,
                "bucket": source_label,
                "cam_naming": "observation.images",
            })
    return out


def copy_parquet(src: Path, dst: Path, new_ep: int, global_offset: int) -> int:
    t = pq.read_table(src)
    n = t.num_rows
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


def symlink_videos(info: dict, new_ep: int, dst_root: Path) -> None:
    src_chunk = info["src_chunk"]
    src_vid_root = Path(info["src_dir"]) / "videos" / f"chunk-{src_chunk:03d}"
    for cam in CAMERAS:
        src_cam = cam if info["cam_naming"] == "bare" else f"observation.images.{cam}"
        src = src_vid_root / src_cam / f"episode_{info['src_ep']:06d}.mp4"
        if not src.exists():
            raise FileNotFoundError(f"missing {src}")
        dst_cam = f"observation.images.{cam}"
        dst = dst_root / "videos" / f"chunk-{CHUNK:03d}" / dst_cam / f"episode_{new_ep:06d}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())


def write_split(picks: list[dict], dst_split: Path, template_info: dict) -> tuple[int, int]:
    (dst_split / "meta").mkdir(parents=True, exist_ok=True)
    new_eps_meta = []
    total_frames = 0
    for new_ep, info in enumerate(picks):
        src_pq = Path(info["src_dir"]) / "data" / f"chunk-{info['src_chunk']:03d}" / f"episode_{info['src_ep']:06d}.parquet"
        dst_pq = dst_split / "data" / f"chunk-{CHUNK:03d}" / f"episode_{new_ep:06d}.parquet"
        n = copy_parquet(src_pq, dst_pq, new_ep, total_frames)
        symlink_videos(info, new_ep, dst_split)
        new_eps_meta.append({
            "episode_index": new_ep,
            "tasks": [PROMPT],
            "length": n,
            "orig_source": info["source"],
            "orig_ep": info["src_ep"],
        })
        total_frames += n

    info_out = dict(template_info)
    info_out["total_episodes"] = len(picks)
    info_out["total_frames"] = total_frames
    info_out["total_videos"] = len(picks) * len(CAMERAS)
    info_out["total_chunks"] = 1
    info_out["chunks_size"] = max(1000, len(picks))
    info_out["splits"] = {dst_split.name: f"0:{len(picks)}"}
    info_out["features"] = {k: v for k, v in info_out["features"].items()
                             if not k.startswith("observation.depth.")}
    info_out.pop("depth_path", None)
    (dst_split / "meta" / "info.json").write_text(json.dumps(info_out, indent=2))
    with (dst_split / "meta" / "episodes.jsonl").open("w") as f:
        for e in new_eps_meta:
            f.write(json.dumps(e) + "\n")
    (dst_split / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": PROMPT}) + "\n")
    return len(picks), total_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=600)
    ap.add_argument("--val-size", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=str(DST_DEFAULT))
    args = ap.parse_args()

    print("=== sources ===")
    vis = collect_vis_base(VIS_BASE)
    print(f"  vis_base:    {len(vis)} eps (use ALL)")
    base_pool = collect_multichunk(KAI0_BASE, "kai0_base")
    print(f"  kai0_base:   {len(base_pool)} eps (pool)")
    dagger_pool = collect_multichunk(KAI0_DAGGER, "kai0_dagger")
    print(f"  kai0_dagger: {len(dagger_pool)} eps (pool)")

    n_vis = len(vis)
    K = (args.total - n_vis) // 2
    rng = random.Random(args.seed)
    base_pick = rng.sample(base_pool, K)
    dagger_pick = rng.sample(dagger_pool, K)
    all_picks = list(vis) + base_pick + dagger_pick
    final_total = len(all_picks)

    # stratified val split by bucket
    val_target = {
        "vis_base": int(round(args.val_size * len(vis) / final_total)),
        "kai0_base": int(round(args.val_size * K / final_total)),
        "kai0_dagger": int(round(args.val_size * K / final_total)),
    }
    while sum(val_target.values()) < args.val_size:
        val_target["vis_base"] += 1
    while sum(val_target.values()) > args.val_size:
        val_target["kai0_dagger"] -= 1

    val_picks, train_picks = [], []
    for bucket in ("vis_base", "kai0_base", "kai0_dagger"):
        items = [x for x in all_picks if x["bucket"] == bucket]
        rng.shuffle(items)
        v = val_target[bucket]
        val_picks.extend(items[:v])
        train_picks.extend(items[v:])

    rng.shuffle(train_picks)
    rng.shuffle(val_picks)

    print(f"\n=== split (seed={args.seed}, val_size={args.val_size}) ===")
    print(f"  total picks: {final_total}")
    print(f"  train: {len(train_picks)}  val: {len(val_picks)}")
    print(f"  per-bucket val: {val_target}")

    if args.dry_run:
        return

    dst = Path(args.out)
    if dst.exists():
        if args.force:
            print(f"[force] removing {dst}")
            shutil.rmtree(dst)
        else:
            print(f"ERROR: {dst} exists. Use --force.", file=sys.stderr); sys.exit(2)

    template_info = json.loads(REFERENCE_INFO.read_text())

    print(f"\nwriting train -> {dst}/base ...")
    n_tr, f_tr = write_split(train_picks, dst / "base", template_info)
    print(f"  train: {n_tr} eps / {f_tr} frames")

    print(f"writing val -> {dst}/val ...")
    n_va, f_va = write_split(val_picks, dst / "val", template_info)
    print(f"  val:   {n_va} eps / {f_va} frames")

    (dst / "manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "total": final_total,
        "val_size": args.val_size,
        "val_target_per_bucket": val_target,
        "vis_base_count": len(vis),
        "kai0_base_count": K,
        "kai0_dagger_count": K,
        "prompt": PROMPT,
    }, indent=2))

    print(f"\nbuilt: {dst}")
    print(f"   train: {n_tr} eps / {f_tr} frames")
    print(f"   val:   {n_va} eps / {f_va} frames")


if __name__ == "__main__":
    main()
