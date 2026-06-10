#!/usr/bin/env python3
"""
Randomly delete images until only TARGET_COUNT remain.

- Folders with fewer than TARGET_COUNT images are left untouched.
"""

import random
from pathlib import Path

# --- Configure these ---
ROOT_FOLDER = "/workspace/Desktop/LFS-GAN/all_datasets/BAGAN_dataset"  # Change this to your dataset folder
TARGET_COUNT = 2000
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
DRY_RUN = False
SEED = None
# -----------------------


def process_folder(folder: Path) -> tuple[int, int, int]:
    """Process one folder. Returns (found, to_delete, deleted)."""

    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    current = len(images)

    if current <= TARGET_COUNT:
        print(f"  [{folder.name}] {current} images — below target, skipping.")
        return current, 0, 0

    to_delete_count = current - TARGET_COUNT
    to_delete = random.sample(images, to_delete_count)

    print(f"  [{folder.name}] {current} images — will delete {to_delete_count} to reach {TARGET_COUNT}.")

    if DRY_RUN:
        return current, to_delete_count, 0

    deleted = 0

    for p in to_delete:
        try:
            p.unlink()
            deleted += 1
        except OSError as e:
            print(f"    Failed to delete {p.name}: {e}")

    return current, to_delete_count, deleted


def main():
    if SEED is not None:
        random.seed(SEED)

    root = Path(ROOT_FOLDER)

    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    subfolders = sorted([p for p in root.iterdir() if p.is_dir()])

    if subfolders:
        folders_to_process = subfolders
        print(f"Found {len(subfolders)} subfolders in {root}")
        print("Mode: subfolder mode")
    else:
        folders_to_process = [root]
        print(f"No subfolders found in {root}")
        print("Mode: single-folder mode")

    print(f"Target per folder: {TARGET_COUNT} images")
    print(f"DRY_RUN: {DRY_RUN}\n")

    total_found = 0
    total_to_delete = 0
    total_deleted = 0

    for folder in folders_to_process:
        found, planned, deleted = process_folder(folder)
        total_found += found
        total_to_delete += planned
        total_deleted += deleted

    print("\n--- Summary ---")
    print(f"Total images found:     {total_found}")
    print(f"Total marked to delete: {total_to_delete}")

    if DRY_RUN:
        print("DRY_RUN is True — nothing was deleted. Set DRY_RUN=False to actually delete.")
    else:
        print(f"Total deleted:          {total_deleted}")
        print(f"Total remaining:        {total_found - total_deleted}")


if __name__ == "__main__":
    main()

