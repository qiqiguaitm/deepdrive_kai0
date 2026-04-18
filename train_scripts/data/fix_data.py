"""Fix kai0 Task_A dataset issues:
  B2: Symlink missing advantage videos from base
  B3: Remove bad (truncated) episodes from all datasets
"""

import json
import os
import shutil
from pathlib import Path

ROOT = Path("/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A")
CAMS = [
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
]


# ── B2: Symlink missing advantage videos from base ──────────────────────

def fix_advantage_videos():
    """Create symlinks for missing advantage videos from base."""
    base_videos = ROOT / "base" / "videos"
    adv_videos = ROOT / "advantage" / "videos"

    created = 0
    skipped = 0
    missing_src = 0

    # Read advantage episode count
    with open(ROOT / "advantage" / "meta" / "info.json") as f:
        info = json.load(f)
    total_eps = info["total_episodes"]
    chunks_size = info["chunks_size"]

    for ep in range(total_eps):
        chunk = ep // chunks_size
        chunk_dir = f"chunk-{chunk:03d}"
        fname = f"episode_{ep:06d}.mp4"

        for cam in CAMS:
            dst = adv_videos / chunk_dir / cam / fname
            if dst.exists() or dst.is_symlink():
                skipped += 1
                continue

            src = base_videos / chunk_dir / cam / fname
            if not src.exists():
                missing_src += 1
                continue

            dst.parent.mkdir(parents=True, exist_ok=True)
            # Use relative symlink for portability
            rel = os.path.relpath(src, dst.parent)
            os.symlink(rel, dst)
            created += 1

    print(f"[B2] Advantage videos: {created} symlinks created, {skipped} already existed, {missing_src} missing in base")


# ── B3: Remove bad episodes ─────────────────────────────────────────────

# Each entry: (dataset, episode_index)
BAD_EPISODES = {
    "base": [176, 2548, 2953],
    "advantage": [176],
    "dagger": [369, 635, 764, 1160, 1329, 1521, 1600, 2032, 2069, 2653, 3305],
}


def remove_bad_episodes():
    """Remove parquet + video files for bad episodes, rewrite meta to exclude them.

    Strategy: delete the bad video files (all 3 cams) and mark them.
    We do NOT re-index episodes (that would require rewriting all parquets),
    instead we record excluded episodes so training can skip them via --data.episodes.
    """
    excluded = {}

    for ds_name, eps in BAD_EPISODES.items():
        ds_root = ROOT / ds_name
        info_path = ds_root / "meta" / "info.json"

        with open(info_path) as f:
            info = json.load(f)
        chunks_size = info["chunks_size"]

        removed_videos = 0
        for ep in eps:
            chunk = ep // chunks_size
            chunk_dir = f"chunk-{chunk:03d}"
            fname_mp4 = f"episode_{ep:06d}.mp4"

            # Remove all 3 camera videos for this episode (even if only 1 is bad,
            # the episode is unusable since training needs all 3)
            for cam in CAMS:
                vid_path = ds_root / "videos" / chunk_dir / cam / fname_mp4
                if vid_path.exists() or vid_path.is_symlink():
                    vid_path.unlink()
                    removed_videos += 1

        excluded[ds_name] = eps
        print(f"[B3] {ds_name}: removed {removed_videos} video files for {len(eps)} bad episodes")

    # Write excluded episodes list for reference
    out_path = ROOT / "excluded_episodes.json"
    with open(out_path, "w") as f:
        json.dump(excluded, f, indent=2)
    print(f"[B3] Wrote {out_path}")

    # Generate valid episode lists for each dataset
    for ds_name, bad_eps in excluded.items():
        ds_root = ROOT / ds_name
        info_path = ds_root / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)
        total = info["total_episodes"]
        bad_set = set(bad_eps)
        valid = [i for i in range(total) if i not in bad_set]

        out = ROOT / f"valid_episodes_{ds_name}.json"
        with open(out, "w") as f:
            json.dump(valid, f)
        print(f"[B3] {ds_name}: {len(valid)}/{total} valid episodes -> {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("Fixing B2: advantage missing videos")
    print("=" * 60)
    fix_advantage_videos()

    print()
    print("=" * 60)
    print("Fixing B3: corrupted video episodes")
    print("=" * 60)
    remove_bad_episodes()
