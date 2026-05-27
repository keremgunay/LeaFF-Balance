import os
import glob

# ===== CONFIG =====
folder = "/workspace/Desktop/LFS-GAN/LeafGAN/results/LeafGAN_run_flies/test_165/images"  # Folder to scan for files
phrase = "real"
dry_run = False  # Set to False to actually delete
# ==================

# Find all matching files
matches = glob.glob(os.path.join(folder, f"*{phrase}*"))

print(f"Found {len(matches)} file(s) containing '{phrase}' in '{folder}'\n")

if not matches:
    print("Nothing to delete. Exiting.")
    exit()

# Process matches
for path in matches:
    if dry_run:
        print(f"[DRY RUN] Would delete: {os.path.basename(path)}")
    else:
        os.remove(path)
        print(f"Deleted: {os.path.basename(path)}")

# Summary
print()
if dry_run:
    print(f"Dry run complete. {len(matches)} file(s) matched.")
    print("Set dry_run = False at the top of the script to actually delete them.")
else:
    print(f"Done. Removed {len(matches)} file(s).")


'''

python3 tool_SpecificImageDeleter.py

'''