import json
import math
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np


# DIRECTORIES & SETTINGS
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
COCO_JSON_PATH = "instances_default.json"
IMAGES_DIR = "images/default"
OUTPUT_DIR = "LFS_Kerems_crops"

MIN_CROP_SIZE = 16
WRITE_LOG = False
LOG_FILENAME = "crop_log.csv"
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# HELPER FUNCTIONS
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
def sanitize_name(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)

def rotated_rect_corners_from_bbox_and_rotation(x, y, w, h, rotation_deg):
    cx = x + w / 2.0
    cy = y + h / 2.0

    theta = math.radians(rotation_deg)
    c = math.cos(theta)
    s = math.sin(theta)

    pts = np.array(
        [
            [-w / 2.0, -h / 2.0],  # top-left
            [w / 2.0, -h / 2.0],   # top-right
            [w / 2.0, h / 2.0],    # bottom-right
            [-w / 2.0, h / 2.0],   # bottom-left
        ],
        dtype=np.float32,
    )

    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    rot_pts = pts @ R.T
    rot_pts[:, 0] += cx
    rot_pts[:, 1] += cy

    return rot_pts


def crop_rotated_object_preserve_orientation(image, bbox, rotation):

    img_h, img_w = image.shape[:2]
    x, y, w, h = bbox

    if w <= 1 or h <= 1:
        return None

    corners = rotated_rect_corners_from_bbox_and_rotation(x, y, w, h, rotation)

    min_x = max(0, int(math.floor(np.min(corners[:, 0]))))
    min_y = max(0, int(math.floor(np.min(corners[:, 1]))))
    max_x = min(img_w, int(math.ceil(np.max(corners[:, 0]))))
    max_y = min(img_h, int(math.ceil(np.max(corners[:, 1]))))

    if max_x <= min_x or max_y <= min_y:
        return None

    crop = image[min_y:max_y, min_x:max_x]
    if crop.size == 0:
        return None

    local_corners = corners.copy()
    local_corners[:, 0] -= min_x
    local_corners[:, 1] -= min_y

    mask = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
    poly = np.round(local_corners).astype(np.int32)
    cv2.fillConvexPoly(mask, poly, 255)

    crop_bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    crop_bgra[:, :, 3] = mask

    ch, cw = crop_bgra.shape[:2]
    if ch < MIN_CROP_SIZE or cw < MIN_CROP_SIZE:
        return None

    return crop_bgra
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# MAIN DEFINITIONS
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
def main():
    coco_json_path = Path(COCO_JSON_PATH)
    images_dir = Path(IMAGES_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    with coco_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    categories = data["categories"]
    images = data["images"]
    annotations = data["annotations"]

    category_id_to_name = {
        c["id"]: sanitize_name(c["name"]) for c in categories
    }
    image_id_to_filename = {
        img["id"]: img["file_name"] for img in images
    }

    class_counters = defaultdict(int)

    saved = 0
    skipped = 0
    missing_images = 0

    log_lines = [
        "output_path,source_image,annotation_id,category_name,rotation,bbox_x,bbox_y,bbox_w,bbox_h"
    ]

    for ann in annotations:
        ann_id = ann["id"]
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        bbox = ann["bbox"]

        attrs = ann.get("attributes", {})
        rotation = float(attrs.get("rotation", 0.0))

        class_name = category_id_to_name.get(category_id, f"class_{category_id}")
        file_name = image_id_to_filename.get(image_id)

        if file_name is None:
            skipped += 1
            continue

        image_path = images_dir / file_name
        if not image_path.exists():
            print(f"[WARN] Missing image: {image_path}")
            missing_images += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] Could not read image: {image_path}")
            skipped += 1
            continue

        crop = crop_rotated_object_preserve_orientation(
            image=image,
            bbox=bbox,
            rotation=rotation,
        )

        if crop is None:
            skipped += 1
            continue

        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        class_counters[class_name] += 1
        idx = class_counters[class_name]

        stem = Path(file_name).stem
        out_name = (
            f"{class_name}_{idx:06d}_img{image_id}_ann{ann_id}_rot{rotation}_{stem}.png"
        )
        out_path = class_dir / out_name

        ok = cv2.imwrite(str(out_path), crop)
        if not ok:
            print(f"[WARN] Failed to save crop: {out_path}")
            skipped += 1
            continue

        saved += 1

        x, y, w, h = bbox
        if WRITE_LOG:
            log_lines.append(
                f"{out_path},{file_name},{ann_id},{class_name},{rotation},{x},{y},{w},{h}"
            )

        if saved % 500 == 0:
            print(f"[INFO] Saved {saved} crops so far...")

    if WRITE_LOG:
        log_path = output_dir / LOG_FILENAME
        with log_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))

    print("\nDone.")
    print(f"Saved crops     : {saved}")
    print(f"Skipped         : {skipped}")
    print(f"Missing images  : {missing_images}")
    print(f"Output folder   : {output_dir.resolve()}")
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

if __name__ == "__main__":
    main()

# EOC