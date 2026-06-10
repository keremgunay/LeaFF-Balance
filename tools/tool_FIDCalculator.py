import os
from datetime import datetime
from cleanfid import fid
 
# ===== CONFIGURATION =====
REAL_DIR = "/workspace/Desktop/LFS-GAN/results/BAGAN_runs/BAGAN_REAL/trips_low_leaf"
FAKE_DIR = "/workspace/Desktop/LFS-GAN/results/BAGAN_runs/BAGAN_GENERATED/thrips_class8_test2800"
OUT_FILE = "/workspace/Desktop/LFS-GAN/results/BAGAN_runs/BAGAN_FID/fid_results_thrips.txt"
LABEL    = "thrips_700"   
 
MODE        = "clean"
NUM_WORKERS = 6
BATCH_SIZE  = 64
DEVICE      = "cuda"            
# =========================
 
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
 
 
def count_images(folder: str) -> int:
    """Count image files in a folder (non-recursive)."""
    if not os.path.isdir(folder):
        return -1
    return sum(
        1 for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )
 
 
def main():
    # --- sanity checks ---
    for d, name in [(REAL_DIR, "REAL_DIR"), (FAKE_DIR, "FAKE_DIR")]:
        if not os.path.isdir(d):
            raise SystemExit(f"{name} does not exist: {d}")
 
    n_real = count_images(REAL_DIR)
    n_fake = count_images(FAKE_DIR)
 
    if n_real == 0 or n_fake == 0:
        raise SystemExit("One of the folders contains no images. Aborting.")
 
    print(f"Real images: {n_real}")
    print(f"Fake images: {n_fake}")
    print("Computing FID...")
 
    # --- compute FID ---
    fid_score = fid.compute_fid(
        REAL_DIR,
        FAKE_DIR,
        mode=MODE,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )
 
    # --- build report ---
    lines = [
        "===== FID Evaluation =====",
        f"Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Label       : {LABEL}",
        f"Real dir    : {REAL_DIR}",
        f"Fake dir    : {FAKE_DIR}",
        f"Real images : {n_real}",
        f"Fake images : {n_fake}",
        f"FID         : {fid_score:.4f}",
        "==========================",
        "",
    ]
    report = "\n".join(lines)
    print("\n" + report)
 
    # --- append to log file ---
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "a") as f:
        f.write(report)
 
    print(f"Result appended to: {OUT_FILE}")
 
 
if __name__ == "__main__":
    main()
 
