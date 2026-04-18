"""Re-download corrupted video files for bad episodes from HuggingFace."""

import json
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "OpenDriveLab-org/Kai0"
DATA_ROOT = Path("/vePFS/tim/workspace/deepdive_kai0/kai0/data")
EXCLUDED = json.load(open(DATA_ROOT / "Task_A" / "excluded_episodes.json"))

CAMS = [
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
]

CHUNKS_SIZE = 1000  # from info.json


def main():
    files_to_download = []

    for ds_name, eps in EXCLUDED.items():
        for ep in eps:
            chunk = ep // CHUNKS_SIZE
            for cam in CAMS:
                repo_path = f"Task_A/{ds_name}/videos/chunk-{chunk:03d}/{cam}/episode_{ep:06d}.mp4"
                local_path = DATA_ROOT / "Task_A" / ds_name / "videos" / f"chunk-{chunk:03d}" / cam / f"episode_{ep:06d}.mp4"
                files_to_download.append((repo_path, local_path))

    print(f"Downloading {len(files_to_download)} video files...")

    success = 0
    failed = []
    for i, (repo_path, local_path) in enumerate(files_to_download, 1):
        print(f"  [{i}/{len(files_to_download)}] {repo_path}")
        try:
            # Remove existing bad/symlink file
            if local_path.exists() or local_path.is_symlink():
                local_path.unlink()

            local_path.parent.mkdir(parents=True, exist_ok=True)

            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                filename=repo_path,
                repo_type="dataset",
                local_dir=str(DATA_ROOT),
                local_dir_use_symlinks=False,
            )
            # hf_hub_download saves to DATA_ROOT/repo_path which matches local_path
            success += 1
        except Exception as e:
            print(f"    FAILED: {e}")
            failed.append(repo_path)

    print(f"\nDone: {success} downloaded, {len(failed)} failed")
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
