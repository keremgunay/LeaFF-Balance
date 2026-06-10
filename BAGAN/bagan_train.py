"""
(C) Copyright IBM Corporation 2018

Modified training launcher for LFS-GAN/BAGAN experiments.
Adds:
- run_config.json
- class_counts.csv
- class_status.csv
- final generated samples per class
- automatic curve plotting from BAGAN class_<id>_score.csv

Important:
This file wraps BalancingGAN.train(...). It can plot whatever the inner BAGAN
code writes through gan.save_history(...). For true every-N-epoch sample grids
or every-N-epoch weight checkpoints, we must also edit balancing_gan.py because
this launcher does not control the inside of the training loop.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from optparse import OptionParser

import numpy as np

import balancing_gan as bagan
from folder_batch_generator import FolderBatchGenerator

try:
    from rw.batch_generator import BatchGenerator as BatchGenerator
except Exception:
    BatchGenerator = None

from utils import save_image_array


# Tracking/helper functions
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)
    return path


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def json_safe(obj):
    """Converts common numpy objects to JSON-safe Python objects."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(json_safe(k)): json_safe(v) for k, v in obj.items()}
    return obj


def get_class_name(classes, class_id):
    """Handle list-like label tables and dict-style label tables."""
    try:
        if isinstance(classes, dict):
            if class_id in classes:
                return str(classes[class_id])
            # common form may be {class_name: class_id}
            for k, v in classes.items():
                if int(v) == int(class_id):
                    return str(k)
        if isinstance(classes, (list, tuple, np.ndarray)) and class_id < len(classes):
            return str(classes[class_id])
    except Exception:
        pass
    return "class_{}".format(class_id)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2, sort_keys=True)


def write_run_config(res_dir, options, classes, shape, min_latent_res, min_classes):
    config = {
        "created_at": now_str(),
        "dataset": options.dataset,
        "data_root": options.data_root,
        "image_size": options.image_size,
        "image_shape_from_loader": shape,
        "batch_size": options.batch_size,
        "epochs": options.epochs,
        "learning_rate": options.adam_lr,
        "seed": options.seed,
        "unbalance": options.unbalance,
        "dratio_mode": options.dratio_mode,
        "gratio_mode": options.gratio_mode,
        "target_class_option": options.target_class,
        "actual_classes_to_train": list(map(int, min_classes)),
        "min_latent_res": min_latent_res,
        "class_table": classes,
        "notes": [
            "Curves are generated from class_<id>_score.csv after gan.save_history().",
            "GAN losses are diagnostic signals, not final image-quality proof.",
            "Always inspect generated image grids together with curves."
        ]
    }
    write_json(os.path.join(res_dir, "run_config.json"), config)


def save_class_counts(path, per_class_count, classes):
    """Save class counts in a stable CSV format."""
    rows = []
    if isinstance(per_class_count, dict):
        iterator = per_class_count.items()
    else:
        try:
            iterator = enumerate(per_class_count)
        except TypeError:
            iterator = []

    for cid, count in iterator:
        try:
            cid_int = int(cid)
        except Exception:
            cid_int = cid
        rows.append({
            "class_id": cid_int,
            "class_name": get_class_name(classes, cid_int) if isinstance(cid_int, int) else str(cid),
            "count": int(count) if str(count).isdigit() else count,
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "count"])
        writer.writeheader()
        writer.writerows(rows)


def append_status(path, row):
    fieldnames = [
        "timestamp", "class_id", "class_name", "status", "epochs", "duration_sec",
        "score_csv", "curves_png", "sample_png", "notes"
    ]
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        clean = {k: row.get(k, "") for k in fieldnames}
        writer.writerow(clean)


def try_float(value):
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.lower() in {"nan", "none"}:
            return None
        return float(text)
    except Exception:
        return None


def read_numeric_csv(path):
    """
    Read a CSV and return (columns, data_dict).
    Works with header CSVs and simple headerless numeric CSVs.
    """
    if not os.path.exists(path):
        return [], {}

    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample.strip() else False

        if has_header:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            data = {col: [] for col in columns}
            for row in reader:
                for col in columns:
                    data[col].append(try_float(row.get(col)))
        else:
            reader = csv.reader(f)
            rows = [r for r in reader if r]
            if not rows:
                return [], {}
            ncols = max(len(r) for r in rows)
            columns = ["col_{}".format(i) for i in range(ncols)]
            data = {col: [] for col in columns}
            for row in rows:
                for i, col in enumerate(columns):
                    data[col].append(try_float(row[i]) if i < len(row) else None)

    # Keep only columns that have at least one numeric value.
    numeric_cols = [c for c in columns if any(v is not None for v in data[c])]
    return numeric_cols, {c: data[c] for c in numeric_cols}

def plot_training_curves(score_csv, out_dir, class_id, smooth_window=5):
    """
    Create exactly 3 BAGAN loss plots:

    1. train discriminator loss vs train generator loss
    2. train discriminator loss vs test discriminator loss
    3. train generator loss vs test generator loss

    X-axis is epoch number: 1, 2, 3, ..., N.
    """
    columns, data = read_numeric_csv(score_csv)

    if not columns:
        print("[tracking] No numeric columns found in {}".format(score_csv))
        return None

    required_cols = [
        "train_gen_loss",
        "train_disc_loss",
        "test_gen_loss",
        "test_disc_loss",
    ]

    missing = [c for c in required_cols if c not in data]
    if missing:
        print("[tracking] Missing required columns in {}: {}".format(score_csv, missing))
        print("[tracking] Available columns:", columns)
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("[tracking] matplotlib unavailable; cannot plot curves: {}".format(exc))
        return None

    ensure_dir(out_dir)

    def to_array(col):
        return np.array(
            [np.nan if v is None else float(v) for v in data[col]],
            dtype=float
        )

    def moving_average_local(values, window):
        arr = np.array(
            [np.nan if v is None else float(v) for v in values],
            dtype=float
        )

        if window is None or window <= 1 or len(arr) < window:
            return arr

        out = np.copy(arr)

        for i in range(len(arr)):
            start = max(0, i - window + 1)
            chunk = arr[start:i + 1]
            out[i] = np.nanmean(chunk) if not np.all(np.isnan(chunk)) else np.nan

        return out

    train_gen = to_array("train_gen_loss")
    train_disc = to_array("train_disc_loss")
    test_gen = to_array("test_gen_loss")
    test_disc = to_array("test_disc_loss")

    n_epochs = len(train_gen)
    x = np.arange(1, n_epochs + 1)

    def plot_pair(y1, y2, raw1, raw2, label1, label2, title, filename):
        out_path = os.path.join(out_dir, filename)

        plt.figure(figsize=(10, 5))

        plt.plot(x, y1, alpha=0.35, label="{} raw".format(label1))
        plt.plot(x, y2, alpha=0.35, label="{} raw".format(label2))

        if smooth_window and smooth_window > 1:
            plt.plot(
                x,
                moving_average_local(raw1, smooth_window),
                linewidth=2,
                label="{} MA{}".format(label1, smooth_window)
            )
            plt.plot(
                x,
                moving_average_local(raw2, smooth_window),
                linewidth=2,
                label="{} MA{}".format(label2, smooth_window)
            )

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

        return out_path

    p1 = plot_pair(
        train_disc,
        train_gen,
        data["train_disc_loss"],
        data["train_gen_loss"],
        "disc train",
        "gen train",
        "Class {} - Disc vs Gen".format(class_id),
        "class_{}_disc_vs_gen.png".format(class_id),
    )

    p2 = plot_pair(
        train_disc,
        test_disc,
        data["train_disc_loss"],
        data["test_disc_loss"],
        "disc train",
        "disc test",
        "Class {} - Disc Train vs Test".format(class_id),
        "class_{}_disc_train_vs_test.png".format(class_id),
    )

    p3 = plot_pair(
        train_gen,
        test_gen,
        data["train_gen_loss"],
        data["test_gen_loss"],
        "gen train",
        "gen test",
        "Class {} - Gen Train vs Test".format(class_id),
        "class_{}_gen_train_vs_test.png".format(class_id),
    )

    print("[tracking] Saved plots:")
    print("[tracking]", p1)
    print("[tracking]", p2)
    print("[tracking]", p3)

    return p1


def write_readme(res_dir):
    text = """# BAGAN Run Tracking

This folder contains automatic tracking files added by `bagan_train_with_tracking.py`.

## Key files

- `run_config.json`: exact run settings and class mapping.
- `class_counts.csv`: class sample counts seen by the batch generator.
- `class_status.csv`: train/load status per class and generated output paths.
- `plots/class_<id>_training_curves.png`: numeric curves parsed from `class_<id>_score.csv`.
- `samples/plot_class_<id>.png`: final generated samples from the selected class.
"""

    with open(os.path.join(res_dir, "TRACKING_README.md"), "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == '__main__':
    # Collect arguments
    argParser = OptionParser()

    argParser.add_option("-u", "--unbalance", default=0.2,
                  action="store", type="float", dest="unbalance",
                  help="Unbalance factor u. The minority class has at most u * otherClassSamples instances.")

    argParser.add_option("-s", "--random_seed", default=0,
                  action="store", type="int", dest="seed",
                  help="Random seed for repeatable subsampling.")

    argParser.add_option("-d", "--sampling_mode_for_discriminator", default="uniform",
                  action="store", type="string", dest="dratio_mode",
                  help="Dratio sampling mode (\"uniform\",\"rebalance\").")

    argParser.add_option("-g", "--sampling_mode_for_generator", default="uniform",
                  action="store", type="string", dest="gratio_mode",
                  help="Gratio sampling mode (\"uniform\",\"rebalance\").")

    argParser.add_option("-e", "--epochs", default=3,
                  action="store", type="int", dest="epochs",
                  help="Training epochs.")

    argParser.add_option("-l", "--learning_rate", default=0.00005,
                  action="store", type="float", dest="adam_lr",
                  help="Training learning rate.")

    argParser.add_option("-c", "--target_class", default=-1,
                  action="store", type="int", dest="target_class",
                  help="If greater or equal to 0, model trained only for the specified class.")

    argParser.add_option("-D", "--dataset", default='folder',
                  action="store", type="string", dest="dataset",
                  help="Dataset type. Use 'folder' for LFS_dataset_crops, or original 'MNIST'/'CIFAR10'.")

    argParser.add_option("--data_root", default="/workspace/Desktop/LFS-GAN/LFS_dataset_crops",
                  action="store", type="string", dest="data_root",
                  help="Path to folder dataset root.")

    argParser.add_option("--image_size", default=128,
                  action="store", type="int", dest="image_size",
                  help="Image size.")

    argParser.add_option("-b", "--batch_size", default=32,
                  action="store", type="int", dest="batch_size",
                  help="Batch size.")

    argParser.add_option("--run_name", default="",
                  action="store", type="string", dest="run_name",
                  help="Optional readable name appended to the result directory.")

    argParser.add_option("--diagnostic_samples", default=64,
                  action="store", type="int", dest="diagnostic_samples",
                  help="Number of generated samples saved at the end of each class run.")

    argParser.add_option("--smooth_window", default=5,
                  action="store", type="int", dest="smooth_window",
                  help="Moving-average window used in curve plots.")

    argParser.add_option("--force_retrain", default=False,
                  action="store_true", dest="force_retrain",
                  help="Retrain even if model files already exist.")

    (options, args) = argParser.parse_args()

    assert (options.unbalance <= 1.0 and options.unbalance > 0.0), "Data unbalance factor must be > 0 and <= 1"

    print("Executing BAGAN with tracking.")

    # Read command line parameters
    np.random.seed(options.seed)
    unbalance = options.unbalance
    gratio_mode = options.gratio_mode
    dratio_mode = options.dratio_mode
    gan_epochs = options.epochs
    adam_lr = options.adam_lr
    opt_class = options.target_class
    batch_size = options.batch_size
    dataset_name = options.dataset

    # Set channels for mnist.
    channels = 1 if dataset_name == 'MNIST' else 3
    print('Using dataset: ', dataset_name)
    print('Channels: ', channels)

    # Result directory
    run_suffix = "_{}".format(options.run_name) if options.run_name else ""
    res_dir = "./res_{}_dmode_{}_gmode_{}_unbalance_{}_epochs_{}_lr_{:f}_seed_{}{}".format(
        dataset_name, dratio_mode, gratio_mode, unbalance, options.epochs, adam_lr, options.seed, run_suffix
    )
    ensure_dir(res_dir)
    plots_dir = ensure_dir(os.path.join(res_dir, "plots"))
    samples_dir = ensure_dir(os.path.join(res_dir, "samples"))
    tracking_status_csv = os.path.join(res_dir, "class_status.csv")
    write_readme(res_dir)

    # Read initial data.
    if dataset_name.lower() == "folder":
        bg_train_full = FolderBatchGenerator(
            FolderBatchGenerator.TRAIN,
            batch_size=batch_size,
            data_root=options.data_root,
            image_size=options.image_size,
            seed=options.seed,
        )

        bg_test = FolderBatchGenerator(
            FolderBatchGenerator.TEST,
            batch_size=batch_size,
            data_root=options.data_root,
            image_size=options.image_size,
            seed=options.seed,
        )
    else:
        if BatchGenerator is None:
            raise RuntimeError(
                "Original IBM BatchGenerator is unavailable in this TensorFlow version. "
                "Use --dataset folder instead."
            )

        bg_train_full = BatchGenerator(
            BatchGenerator.TRAIN,
            batch_size,
            class_to_prune=None,
            unbalance=None,
            dataset=dataset_name,
        )

        bg_test = BatchGenerator(
            BatchGenerator.TEST,
            batch_size,
            class_to_prune=None,
            unbalance=None,
            dataset=dataset_name,
        )

    print("input data loaded...")

    shape = bg_train_full.get_image_shape()
    print("Image shape from loader:", shape)

    min_latent_res = shape[-1]
    while min_latent_res > 8:
        min_latent_res = min_latent_res / 2
    min_latent_res = int(min_latent_res)

    classes = bg_train_full.get_label_table()

    # Initialize statistics information
    gan_train_losses = defaultdict(list)  # kept from original script for compatibility/future use
    gan_test_losses = defaultdict(list)   # kept from original script for compatibility/future use
    img_samples = defaultdict(list)

    # For all possible minority classes.
    target_classes = np.array(range(len(classes)))
    if opt_class >= 0:
        min_classes = np.array([opt_class])
    else:
        min_classes = target_classes

    write_run_config(res_dir, options, classes, shape, min_latent_res, min_classes)
    save_class_counts(os.path.join(res_dir, "class_counts.csv"), bg_train_full.per_class_count, classes)

    for c in min_classes:
        c = int(c)
        class_name = get_class_name(classes, c)
        class_start = time.time()
        print("\n==============================")
        print("Class {} ({})".format(c, class_name))
        print("Started at:", now_str())
        print("==============================")

        # If unbalance is 1.0, then the same BAGAN model can be applied to every class because
        # we do not drop any instance at training time.
        if unbalance == 1.0 and c > 0 and (
            os.path.exists("{}/class_0_score.csv".format(res_dir)) and
            os.path.exists("{}/class_0_discriminator.h5".format(res_dir)) and
            os.path.exists("{}/class_0_generator.h5".format(res_dir)) and
            os.path.exists("{}/class_0_reconstructor.h5".format(res_dir))
        ):
            # Without additional imbalance, BAGAN does not need to be retrained, we symlink the pregenerated model.
            for src_name, dst_name in [
                ("class_0_score.csv", "class_{}_score.csv".format(c)),
                ("class_0_discriminator.h5", "class_{}_discriminator.h5".format(c)),
                ("class_0_generator.h5", "class_{}_generator.h5".format(c)),
                ("class_0_reconstructor.h5", "class_{}_reconstructor.h5".format(c)),
            ]:
                src = os.path.join(res_dir, src_name)
                dst = os.path.join(res_dir, dst_name)
                if not os.path.exists(dst):
                    os.symlink(src, dst)

        # Unbalance the training set.
        if dataset_name.lower() == "folder":
            bg_train_partial = bg_train_full
        else:
            bg_train_partial = BatchGenerator(
                BatchGenerator.TRAIN,
                batch_size,
                class_to_prune=c,
                unbalance=unbalance,
                dataset=dataset_name,
            )

        save_class_counts(
            os.path.join(res_dir, "class_{}_counts_seen_by_training.csv".format(c)),
            bg_train_partial.per_class_count,
            classes
        )

        score_csv = "{}/class_{}_score.csv".format(res_dir, c)
        generator_path = "{}/class_{}_generator.h5".format(res_dir, c)
        discriminator_path = "{}/class_{}_discriminator.h5".format(res_dir, c)
        reconstructor_path = "{}/class_{}_reconstructor.h5".format(res_dir, c)

        already_available = (
            os.path.exists(score_csv) and
            os.path.exists(discriminator_path) and
            os.path.exists(generator_path) and
            os.path.exists(reconstructor_path)
        )

        # Train the model or reload it if already available.
        if options.force_retrain or not already_available:
            print("Required GAN for class {} ({})".format(c, class_name))
            print('Class counters: ', bg_train_partial.per_class_count)

            gan = bagan.BalancingGAN(
                target_classes, c, dratio_mode=dratio_mode, gratio_mode=gratio_mode,
                adam_lr=adam_lr, res_dir=res_dir, image_shape=shape, min_latent_res=min_latent_res
            )

            print("[tracking] Training starts now. Curves will be plotted after save_history writes the CSV.")
            gan.train(bg_train_partial, bg_test, epochs=gan_epochs)
            gan.save_history(res_dir, c)
            status = "trained"
        else:
            print("Loading GAN for class {} ({})".format(c, class_name))
            gan = bagan.BalancingGAN(
                target_classes, c, dratio_mode=dratio_mode, gratio_mode=gratio_mode,
                adam_lr=adam_lr, res_dir=res_dir, image_shape=shape, min_latent_res=min_latent_res
            )
            gan.load_models(
                generator_path,
                discriminator_path,
                reconstructor_path,
                bg_train=bg_train_partial  # Required to initialize per-class mean/covariance matrix.
            )
            status = "loaded_existing"

        # Plot training curves from the score CSV.
        curves_png = plot_training_curves(
            score_csv=score_csv,
            out_dir=plots_dir,
            class_id=c,
            smooth_window=options.smooth_window,
        )

        # Sample and save images.
        n_samples = max(1, int(options.diagnostic_samples))
        print("[tracking] Generating {} diagnostic samples for class {}".format(n_samples, c))
        img_samples['class_{}'.format(c)] = gan.generate_samples(c=c, samples=n_samples)

        sample_path = os.path.join(samples_dir, 'plot_class_{}.png'.format(c))
        save_image_array(np.array([img_samples['class_{}'.format(c)]]), sample_path)

        # Keep original-style location too, so old scripts do not break.
        legacy_sample_path = '{}/plot_class_{}.png'.format(res_dir, c)
        save_image_array(np.array([img_samples['class_{}'.format(c)]]), legacy_sample_path)

        duration = time.time() - class_start
        append_status(tracking_status_csv, {
            "timestamp": now_str(),
            "class_id": c,
            "class_name": class_name,
            "status": status,
            "epochs": gan_epochs,
            "duration_sec": round(duration, 2),
            "score_csv": score_csv,
            "curves_png": curves_png or "",
            "sample_png": sample_path,
            "notes": "Inspect curves together with generated samples. Loss alone is not enough."
        })

        print("[tracking] Class {} complete in {:.2f} sec".format(c, duration))
        print("[tracking] Score CSV:", score_csv)
        print("[tracking] Curves PNG:", curves_png)
        print("[tracking] Samples PNG:", sample_path)

    print("\nAll requested classes complete.")
    print("Tracking files written to:", res_dir)


'''

cd /workspace/Desktop/LFS-GAN/BAGAN

export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

python3 bagan_train.py \
  --dataset folder \
  --data_root /workspace/Desktop/LFS-GAN/all_datasets/BAGAN_dataset \
  --image_size 128 \
  --epochs 200 \
  --batch_size 32 \
  --target_class 8 \
  --learning_rate 0.00005 \
  --run_name bagan_trips_128_epoch200_700img \
  --force_retrain

'''