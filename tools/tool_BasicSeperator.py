from pathlib import Path
import random
import shutil

# Point directly to ONE class folder that contains images
SOURCE_DIR = Path("/workspace/Desktop/LFS-GAN/LFS_dataset/trips_low_leaf")

# Output dataset root
OUTPUT_DIR = Path("/workspace/Desktop/LFS-GAN/LFS_synthetic_split_dataset")

TRAIN_RATIO = 0.4
VAL_RATIO = 0.3
TEST_RATIO = 0.3

SEED = 42
COPY_FILES = True   # True = copy, False = move

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-9:
    raise SystemExit("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")

if not SOURCE_DIR.is_dir():
    raise SystemExit(f"Source folder not found: {SOURCE_DIR}")

random.seed(SEED)

class_name = SOURCE_DIR.name
images = [p for p in SOURCE_DIR.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]

if not images:
    raise SystemExit(f"No images found in: {SOURCE_DIR}")

random.shuffle(images)

n = len(images)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)
n_test = n - n_train - n_val

train_files = images[:n_train]
val_files = images[n_train:n_train + n_val]
test_files = images[n_train + n_val:]

split_map = {
    "train": train_files,
    "val": val_files,
    "test": test_files,
}

for split, files in split_map.items():
    target_class_dir = OUTPUT_DIR / split / class_name
    target_class_dir.mkdir(parents=True, exist_ok=True)

    for src in files:
        dst = target_class_dir / src.name

        if dst.exists():
            print(f"SKIP (already exists): {dst.name}")
            continue

        if COPY_FILES:
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))

print(
    f"{class_name}: total={n}, "
    f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
)

print("\nDone.")
print(f"Output dataset root: {OUTPUT_DIR}")