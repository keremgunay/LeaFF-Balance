from PIL import Image
import os

src = r"C:\Users\cagla\Desktop\Kerem Günay\hey"
dst = r"C:\Users\cagla\Desktop\Kerem Günay\StyleGAN2_project\datasets_raw\severe_nutdef_1000"
os.makedirs(dst, exist_ok=True)

# Pre-count how many image files we'll process so we can show "X / total"
all_files = [f for f in os.listdir(src)
             if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"))]
total = len(all_files)
print(f"Found {total} images to convert.")

count = 0
skipped = 0
for fname in all_files:
    try:
        img = Image.open(os.path.join(src, fname)).convert("RGB")
        out_name = os.path.splitext(fname)[0] + ".png"
        img.save(os.path.join(dst, out_name))
        count += 1

        # Notify every 50 successfully converted images
        if count % 50 == 0:
            print(f"  ...converted {count} / {total} images", flush=True)

    except Exception as e:
        skipped += 1
        print(f"Skipped {fname}: {e}")

print(f"Done. Converted {count} images, skipped {skipped}.")

