"""
ae_overfit_check.py

Standalone script to test whether the BAGAN autoencoder is overfitting.

What it does:
  1. Loads your trained encoder + decoder from disk
  2. Loads a sample of TRAIN images and a sample of TEST images
  3. Runs both through encoder -> decoder
  4. Computes mean MSE reconstruction loss on each set
  5. Saves side-by-side reconstruction grids for visual inspection
  6. Prints a verdict

How to read the output:
  - If train_mse and test_mse are similar  -> AE generalizes well
  - If test_mse is much higher than train_mse -> AE is overfitting
  - "Much higher" rule of thumb: > 1.5x is concerning, > 2x is bad

Run from the same directory you run bagan_train.py from:

    cd /workspace/Desktop/LFS-GAN/BAGAN
    python3 ae_overfit_check.py \
        --data_root /workspace/Desktop/LFS-GAN/LFS_dataset_crops \
        --encoder ./res_..._target_class_1/1_encoder.h5 \
        --decoder ./res_..._target_class_1/1_decoder.h5 \
        --image_size 128 \
        --n_samples 200 \
        --output_dir ./ae_overfit_report
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

# Match BAGAN's data format
K.set_image_data_format('channels_first')

from tensorflow.keras.models import load_model

# Reuse the project's batch generator so we read images exactly the same way
# BAGAN does at training time. This is critical: any normalization mismatch
# would invalidate the comparison.
from folder_batch_generator import FolderBatchGenerator


def load_images_from_generator(bg, n_samples, seed=0):
    """
    Pull n_samples images out of a FolderBatchGenerator without exhausting
    the underlying class structure. Returns a numpy array shaped
    (n_samples, channels, H, W) using whatever normalization the generator
    applies (BAGAN typically uses tanh range [-1, 1]).
    """
    rng = np.random.RandomState(seed)
    collected = []
    needed = n_samples

    # Iterate batches and stack until we have enough
    for image_batch, _ in bg.next_batch():
        collected.append(image_batch)
        needed -= image_batch.shape[0]
        if needed <= 0:
            break

    images = np.concatenate(collected, axis=0)

    # Random subset of size n_samples
    if images.shape[0] > n_samples:
        idx = rng.choice(images.shape[0], size=n_samples, replace=False)
        images = images[idx]

    return images


def reconstruct(encoder, decoder, images, batch_size=32):
    """encoder -> decoder pass, batched to avoid OOM."""
    recons = []
    for start in range(0, images.shape[0], batch_size):
        chunk = images[start:start + batch_size]
        latent = encoder.predict(chunk, verbose=0)
        out = decoder.predict(latent, verbose=0)
        recons.append(out)
    return np.concatenate(recons, axis=0)


def per_image_mse(originals, recons):
    """Returns per-image MSE as a 1-D array of length N."""
    diff = (originals - recons) ** 2
    # Mean over channels and spatial dims
    return diff.reshape(diff.shape[0], -1).mean(axis=1)


def save_comparison_grid(originals, recons, out_path, n_show=10):
    """
    Save a side-by-side image: top row originals, bottom row reconstructions.
    Works whether images are in [-1, 1] or [0, 1].
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable, skipping comparison grid")
        return

    n_show = min(n_show, originals.shape[0])

    def to_display(img):
        # img is (C, H, W). Convert to (H, W, C) and rescale to [0, 1].
        img = np.transpose(img, (1, 2, 0))
        if img.min() < -0.1:  # likely tanh range
            img = (img + 1.0) / 2.0
        return np.clip(img, 0.0, 1.0)

    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 1.5, 3.2))
    for i in range(n_show):
        axes[0, i].imshow(to_display(originals[i]))
        axes[0, i].axis('off')
        axes[1, i].imshow(to_display(recons[i]))
        axes[1, i].axis('off')

    axes[0, 0].set_title("original", loc='left', fontsize=9)
    axes[1, 0].set_title("reconstructed", loc='left', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved comparison grid:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                        help="Same path you pass to bagan_train.py --data_root")
    parser.add_argument("--encoder", required=True,
                        help="Path to *_encoder.h5")
    parser.add_argument("--decoder", required=True,
                        help="Path to *_decoder.h5")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=200,
                        help="How many images to evaluate from each split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="./ae_overfit_report")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Loading encoder:", args.encoder)
    encoder = load_model(args.encoder, compile=False)
    print("Loading decoder:", args.decoder)
    decoder = load_model(args.decoder, compile=False)
    print("=" * 60)

    print("\nLoading TRAIN images...")
    bg_train = FolderBatchGenerator(
        FolderBatchGenerator.TRAIN,
        batch_size=args.batch_size,
        data_root=args.data_root,
        image_size=args.image_size,
        seed=args.seed,
    )
    train_imgs = load_images_from_generator(bg_train, args.n_samples, seed=args.seed)
    print("Train images shape:", train_imgs.shape)

    print("\nLoading TEST images...")
    bg_test = FolderBatchGenerator(
        FolderBatchGenerator.TEST,
        batch_size=args.batch_size,
        data_root=args.data_root,
        image_size=args.image_size,
        seed=args.seed,
    )
    test_imgs = load_images_from_generator(bg_test, args.n_samples, seed=args.seed)
    print("Test images shape:", test_imgs.shape)

    print("\nReconstructing TRAIN set...")
    train_recons = reconstruct(encoder, decoder, train_imgs, args.batch_size)
    print("Reconstructing TEST set...")
    test_recons = reconstruct(encoder, decoder, test_imgs, args.batch_size)

    train_mse_per_img = per_image_mse(train_imgs, train_recons)
    test_mse_per_img = per_image_mse(test_imgs, test_recons)

    train_mse = float(train_mse_per_img.mean())
    test_mse = float(test_mse_per_img.mean())
    train_std = float(train_mse_per_img.std())
    test_std = float(test_mse_per_img.std())

    ratio = test_mse / train_mse if train_mse > 0 else float('inf')

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("Train reconstruction MSE: {:.6f}  (+/- {:.6f})".format(train_mse, train_std))
    print("Test  reconstruction MSE: {:.6f}  (+/- {:.6f})".format(test_mse, test_std))
    print("Test/Train ratio:         {:.3f}".format(ratio))
    print()

    # Verdict
    if ratio < 1.2:
        verdict = "GOOD: AE generalizes well. Train and test loss are close."
    elif ratio < 1.5:
        verdict = "OK: Mild gap between train and test. Likely fine for GAN init."
    elif ratio < 2.0:
        verdict = "CONCERNING: Test loss is notably higher. AE may be partially overfitting."
    else:
        verdict = "OVERFIT: Test loss is much higher than train loss. AE memorized training data."

    print("Verdict:", verdict)
    print("=" * 60)

    # Save metrics to a text file
    report_path = os.path.join(args.output_dir, "metrics.txt")
    with open(report_path, "w") as f:
        f.write("AE Overfitting Check\n")
        f.write("=" * 40 + "\n")
        f.write("Encoder: {}\n".format(args.encoder))
        f.write("Decoder: {}\n".format(args.decoder))
        f.write("Data root: {}\n".format(args.data_root))
        f.write("N samples per split: {}\n\n".format(args.n_samples))
        f.write("Train MSE: {:.6f} (+/- {:.6f})\n".format(train_mse, train_std))
        f.write("Test  MSE: {:.6f} (+/- {:.6f})\n".format(test_mse, test_std))
        f.write("Test/Train ratio: {:.3f}\n\n".format(ratio))
        f.write("Verdict: {}\n".format(verdict))
    print("Saved metrics to:", report_path)

    # Save comparison grids
    save_comparison_grid(
        train_imgs, train_recons,
        os.path.join(args.output_dir, "train_reconstructions.png"),
        n_show=10,
    )
    save_comparison_grid(
        test_imgs, test_recons,
        os.path.join(args.output_dir, "test_reconstructions.png"),
        n_show=10,
    )

    # Save histogram of per-image MSE for both splits
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        bins = np.linspace(
            0,
            max(train_mse_per_img.max(), test_mse_per_img.max()) * 1.05,
            40
        )
        plt.hist(train_mse_per_img, bins=bins, alpha=0.6, label='train', color='tab:blue')
        plt.hist(test_mse_per_img, bins=bins, alpha=0.6, label='test', color='tab:orange')
        plt.axvline(train_mse, color='tab:blue', linestyle='--', linewidth=1)
        plt.axvline(test_mse, color='tab:orange', linestyle='--', linewidth=1)
        plt.xlabel("per-image reconstruction MSE")
        plt.ylabel("count")
        plt.title("AE reconstruction error distribution")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        hist_path = os.path.join(args.output_dir, "mse_histogram.png")
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print("Saved histogram:", hist_path)
    except Exception as exc:
        print("Could not save histogram:", exc)


if __name__ == "__main__":
    main()

    '''
    cd /workspace/Desktop/LFS-GAN/BAGAN

python3 ae_overfit_check.py \
    --data_root /workspace/Desktop/LFS-GAN/LFS_dataset_crops \
    --encoder ./res_..._target_class_1/1_encoder.h5 \
    --decoder ./res_..._target_class_1/1_decoder.h5 \
    --image_size 128 \
    --n_samples 200 \
    --output_dir ./ae_overfit_report
    '''