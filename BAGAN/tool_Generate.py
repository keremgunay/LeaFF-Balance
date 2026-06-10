import os
from pathlib import Path

import numpy as np
from PIL import Image

import balancing_gan as bagan


# =========================
# USER SETTINGS
# =========================

RES_DIR = "/workspace/Desktop/LFS-GAN/results/BAGAN_runs/BAGAN_trips/res_folder_dmode_uniform_gmode_uniform_unbalance_0.2_epochs_200_lr_0.000050_seed_0_bagan_trips_128_epoch200_700img"

CLASS_ID = 8
NUM_CLASSES = 9

IMAGE_SIZE = 128
CHANNELS = 3

NUM_IMAGES = 2800
OUTPUT_DIR = "/workspace/Desktop/LFS-GAN/results/BAGAN_generated/thrips_class8_test2800"

SEED = 42


# =========================
# GENERATION
# =========================

np.random.seed(SEED)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

classes = np.arange(NUM_CLASSES)
image_shape = [CHANNELS, IMAGE_SIZE, IMAGE_SIZE]

gan = bagan.BalancingGAN(
    classes=classes,
    target_class_id=CLASS_ID,
    res_dir=RES_DIR,
    image_shape=image_shape,
    min_latent_res=8
)

gan.load_models(
    os.path.join(RES_DIR, f"class_{CLASS_ID}_generator.h5"),
    os.path.join(RES_DIR, f"class_{CLASS_ID}_discriminator.h5"),
    os.path.join(RES_DIR, f"class_{CLASS_ID}_reconstructor.h5"),
    bg_train=None
)

images = gan.generate_samples(c=CLASS_ID, samples=NUM_IMAGES)


def convert_image(img):
    img = np.asarray(img)

    # BAGAN uses channels-first: C,H,W
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))

    # tanh output: [-1, 1] -> [0, 1]
    if img.min() < -0.1:
        img = (img + 1.0) / 2.0

    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)

    return img


for i, img in enumerate(images):
    img = convert_image(img)

    if img.ndim == 2:
        pil_img = Image.fromarray(img).convert("RGB")
    else:
        pil_img = Image.fromarray(img).convert("RGB")

    save_path = os.path.join(OUTPUT_DIR, f"bagan_thrips_class8_{i:05d}.png")
    pil_img.save(save_path)

print(f"Saved {NUM_IMAGES} images to:")
print(OUTPUT_DIR)


'''

cd /workspace/Desktop/LFS-GAN/BAGAN

export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

python3 tool_Generate.py


'''