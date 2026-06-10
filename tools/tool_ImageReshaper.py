"""
squash_resize.py

Force-resizes every image in a folder (recursively) to a fixed square size,
ignoring aspect ratio. No content is cropped -- the image is squashed to fit.
Useful when you want to keep every leaf pattern at the cost of mild distortion.

Usage:
    python squash_resize.py --input ./datasets/tomato_raw --output ./datasets/tomato_256

Optional:
    --size 256       # output size (default 256)
    --ext jpg        # output extension: jpg or png (default jpg)
    --quality 95     # JPEG quality (default 95)
"""

import argparse
from pathlib import Path
from PIL import Image, ImageOps

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def process_image(src: Path, dst: Path, size: int, ext: str, quality: int) -> bool:
    try:
        img = Image.open(src)
        # Honor EXIF orientation (phone photos are often saved rotated with an EXIF tag)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except Exception as e:
        print(f"  SKIP (cannot open): {src} -- {e}")
        return False

    # Force-resize to square, ignoring aspect ratio.
    # LANCZOS is the highest-quality downscaling filter in PIL.
    img = img.resize((size, size), Image.LANCZOS)

    dst.parent.mkdir(parents=True, exist_ok=True)
    out_path = dst.with_suffix(f".{ext}")
    save_kwargs = {"quality": quality, "optimize": True} if ext.lower() in ("jpg", "jpeg") else {}
    img.save(out_path, **save_kwargs)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input folder (searched recursively)")
    parser.add_argument("--output", required=True, help="Output folder (subfolder structure preserved)")
    parser.add_argument("--size", type=int, default=256, help="Output side length (default 256)")
    parser.add_argument("--ext", choices=["jpg", "png"], default="jpg", help="Output extension")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality, ignored for PNG")
    args = parser.parse_args()

    in_root = Path(args.input).expanduser().resolve()
    out_root = Path(args.output).expanduser().resolve()

    if not in_root.is_dir():
        raise SystemExit(f"Input folder not found: {in_root}")

    if out_root.resolve() == in_root.resolve():
        raise SystemExit("Output folder must be different from input folder.")

    # Collect all valid image files recursively
    files = [p for p in in_root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
    if not files:
        raise SystemExit(f"No images found under {in_root}")

    print(f"Found {len(files)} images. Resizing to {args.size}x{args.size} -> {out_root}")
    ok = 0
    for i, src in enumerate(files, 1):
        rel = src.relative_to(in_root)
        dst = out_root / rel
        if process_image(src, dst, args.size, args.ext, args.quality):
            ok += 1
        if i % 50 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] processed, {ok} ok")

    print(f"Done. {ok}/{len(files)} images successfully written to {out_root}")


if __name__ == "__main__":
    main()

'''

python3 tool_ImageReshaper.py \
--input "/workspace/Desktop/LFS-GAN/all_datasets/CycleGAN_raw" \
--output "/workspace/Desktop/LFS-GAN/all_datasets/CycleGAN_dataroot" \
--size 256 \
--ext png

'''