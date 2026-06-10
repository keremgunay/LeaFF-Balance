import os
from PIL import Image

# === CONFIGURATION ===
ROOT_FOLDER = r"/workspace/Desktop/LFS-GAN/LFS_all_datasets/TEMPORARY SHIT"   # root directory to scan for images
MIN_SIZE = 128                        # minimum allowed for BOTH height and width
DRY_RUN = False                       # True = only print what would be deleted


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def prune_small_images(root_folder, min_size, dry_run=True):
    checked = 0
    flagged = 0
    deleted = 0
    unreadable = []

    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VALID_EXTS:
                continue

            fpath = os.path.join(dirpath, fname)
            checked += 1

            try:
                with Image.open(fpath) as img:
                    w, h = img.size
            except Exception as e:
                unreadable.append((fpath, str(e)))
                continue

            if w < min_size or h < min_size:
                flagged += 1
                print(f"[{'DRY-RUN' if dry_run else 'DELETE'}] {fpath}  ({w}x{h})")
                if not dry_run:
                    try:
                        os.remove(fpath)
                        deleted += 1
                    except Exception as e:
                        print(f"  -> failed to delete: {e}")

    print("\n===== SUMMARY =====")
    print(f"Images checked:    {checked}")
    print(f"Below {min_size}x{min_size}:   {flagged}")
    if dry_run:
        print(f"(Dry run — nothing was actually deleted.)")
    else:
        print(f"Deleted:           {deleted}")
    if unreadable:
        print(f"Unreadable files:  {len(unreadable)}")
        for path, err in unreadable:
            print(f"  - {path}  ({err})")


if __name__ == "__main__":
    prune_small_images(ROOT_FOLDER, MIN_SIZE, DRY_RUN)

