from pathlib import Path
import random
import shutil

SOURCE_DIR = Path("/workspace/Desktop/LFS-GAN/LFS_dataset_severe_critical")
OUTPUT_DIR = Path("/workspace/Desktop/LFS-GAN/LFS_split_dataset_severe_critical")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
COPY_FILES = True  # True = copy, False = move

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)

for split in ["train", "val", "test"]:
    (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

class_dirs = [p for p in SOURCE_DIR.iterdir() if p.is_dir()]

for class_dir in class_dirs:
    images = [p for p in class_dir.iterdir() if p.suffix.lower() in EXTENSIONS]
    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_files = images[:n_train]
    val_files = images[n_train:n_train + n_val]
    test_files = images[n_train + n_val:]

    split_map = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    for split, files in split_map.items():
        target_class_dir = OUTPUT_DIR / split / class_dir.name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for src in files:
            dst = target_class_dir / src.name
            if COPY_FILES:
                shutil.copy2(src, dst)
            else:
                shutil.move(str(src), str(dst))

    print(
        f"{class_dir.name}: total={n}, "
        f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
    )

print("\nDone.")
print(f"Output dataset: {OUTPUT_DIR}")

