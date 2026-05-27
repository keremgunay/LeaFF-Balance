#!/usr/bin/env python3
"""
Delete class subfolders that contain fewer than MIN_COUNT images.
Used to remove minority classes that fail to meet the minimum sample threshold.
"""

import shutil
from pathlib import Path

# --- Configure these ---
ROOT_FOLDER = "/workspace/Desktop/LFS-GAN/LFS_dataset_crops"
MIN_COUNT = 400
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
DRY_RUN = False  # Set to False to actually delete
# -----------------------


def count_images(folder: Path) -> int:
    """Count image files in a folder (non-recursive)."""
    return sum(1 for p in folder.iterdir()
               if p.is_file() and p.suffix.lower() in EXTENSIONS)


def main():
    root = Path(ROOT_FOLDER)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    class_folders = sorted([p for p in root.iterdir() if p.is_dir()])
    if not class_folders:
        raise SystemExit(f"No subfolders found in {root}")

    print(f"Found {len(class_folders)} class folders in {root}")
    print(f"Minimum required images per class: {MIN_COUNT}")
    print(f"DRY_RUN: {DRY_RUN}\n")

    to_delete = []
    kept = []

    for folder in class_folders:
        count = count_images(folder)
        if count < MIN_COUNT:
            to_delete.append((folder, count))
            print(f"  [{folder.name}] {count} images — BELOW threshold, will delete folder.")
        else:
            kept.append((folder, count))
            print(f"  [{folder.name}] {count} images — OK, keeping.")

    print("\n--- Summary ---")
    print(f"Folders to keep:   {len(kept)}")
    print(f"Folders to delete: {len(to_delete)}")

    if not to_delete:
        print("No folders below threshold. Nothing to do.")
        return

    if DRY_RUN:
        print("\nDRY_RUN is True — nothing was deleted.")
        print("Folders that would be deleted:")
        for folder, count in to_delete:
            print(f"  - {folder.name} ({count} images)")
        print("\nSet DRY_RUN=False to actually delete these folders.")
        return

    print("\nDeleting folders...")
    deleted = 0
    for folder, count in to_delete:
        try:
            shutil.rmtree(folder)
            print(f"  Deleted: {folder.name} ({count} images)")
            deleted += 1
        except OSError as e:
            print(f"  Failed to delete {folder.name}: {e}")

    print(f"\nDone. Deleted {deleted} of {len(to_delete)} folders.")


if __name__ == "__main__":
    main()