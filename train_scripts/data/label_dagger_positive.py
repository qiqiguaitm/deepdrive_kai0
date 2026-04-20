#!/usr/bin/env python3
"""Copy Task_A/dagger to Task_A/dagger_labeled with all frames as `Advantage: positive`.

Heuristic (Option B in awbc_v2_training_plan): treat every DAgger episode as
positive since DAgger data is by construction policy-corrected trajectory. This
avoids 1.75h of KAI0 Advantage Estimator inference + discretization.

Output layout (identical to Task_A/advantage):
    dagger_labeled/
      data/chunk-NNN/episode_*.parquet   # task_index=1 rewritten
      videos/chunk-NNN/{cam}/episode_*.mp4 -> symlinked to original
      meta/
        info.json                        # total_tasks=2
        episodes.jsonl                   # copied from source
        episodes_stats.jsonl             # copied from source
        tasks.jsonl                      # 2 entries matching advantage format
      norm_stats.json                    # symlinked from source
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/dagger")
    ap.add_argument("--dst", default="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/dagger_labeled")
    ap.add_argument("--task-index", type=int, default=1, help="task_index for all frames (default 1 = positive)")
    ap.add_argument(
        "--prompts",
        default="Flatten and fold the cloth. Advantage: negative|Flatten and fold the cloth. Advantage: positive",
        help="Pipe-separated task_index=0,1,...,N-1 prompts",
    )
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.exists():
        raise SystemExit(f"src does not exist: {src}")
    if dst.exists():
        raise SystemExit(f"dst already exists: {dst}. Remove it first.")

    prompts = args.prompts.split("|")

    (dst / "meta").mkdir(parents=True)
    # ---- meta ----
    shutil.copy(src / "meta" / "episodes.jsonl", dst / "meta" / "episodes.jsonl")
    if (src / "meta" / "episodes_stats.jsonl").exists():
        shutil.copy(src / "meta" / "episodes_stats.jsonl", dst / "meta" / "episodes_stats.jsonl")
    with open(dst / "meta" / "tasks.jsonl", "w") as f:
        for i, p in enumerate(prompts):
            f.write(json.dumps({"task_index": i, "task": p}) + "\n")

    info = json.loads((src / "meta" / "info.json").read_text())
    info["total_tasks"] = len(prompts)
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    print(f"[meta] tasks.jsonl written with {len(prompts)} prompts, info.json total_tasks={len(prompts)}")

    # ---- norm_stats (same state/action distribution as source) ----
    if (src / "norm_stats.json").exists():
        (dst / "norm_stats.json").symlink_to(src / "norm_stats.json")
        print(f"[norm_stats] symlinked")

    # ---- videos: symlink whole tree (no content change needed) ----
    (dst / "videos").mkdir(exist_ok=True)
    for chunk_dir in sorted((src / "videos").iterdir()):
        if not chunk_dir.is_dir():
            continue
        (dst / "videos" / chunk_dir.name).symlink_to(chunk_dir.resolve())
    print(f"[videos] chunks symlinked")

    # ---- data parquets: rewrite task_index column ----
    data_src = src / "data"
    data_dst = dst / "data"
    parquets = sorted(data_src.rglob("episode_*.parquet"))
    print(f"[data] rewriting {len(parquets)} parquets (setting task_index={args.task_index})...")
    for p in tqdm(parquets, desc="parquets"):
        rel = p.relative_to(data_src)
        out = data_dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(p)
        if "task_index" not in table.column_names:
            # Dagger has task_index=0 (flat the cloth); add or overwrite
            pass
        new_col = pa.array([args.task_index] * table.num_rows, type=pa.int64())
        idx = table.column_names.index("task_index") if "task_index" in table.column_names else None
        if idx is not None:
            table = table.set_column(idx, "task_index", new_col)
        else:
            table = table.append_column("task_index", new_col)
        pq.write_table(table, out)
    print(f"[done] dst = {dst}")


if __name__ == "__main__":
    main()
