#!/usr/bin/env python
"""Fetch RoboPRO benchmark assets from HuggingFace into benchmark/assets/.

Usage:
    python scripts/install/download_assets.py [--dest <path>]

Pulls the full Hoshipu/RoboPRO_assets dataset (~15 GB) including objects,
embodiments, background textures, and scene-camera configs.
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Hoshipu/RoboPRO_assets"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEST = REPO_ROOT / "benchmark" / "assets"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help=f"target directory (default: {DEFAULT_DEST})")
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"[download] repo={REPO_ID} → {args.dest}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(args.dest),
        max_workers=8,
    )
    print(f"[download] done. Verify: ls {args.dest}/objects | wc -l")


if __name__ == "__main__":
    main()
