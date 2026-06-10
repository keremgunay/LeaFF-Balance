import numpy as np
from pathlib import Path
from PIL import Image


class FolderBatchGenerator:
    TRAIN = "train"
    TEST = "test"

    def __init__(
        self,
        mode,
        batch_size,
        data_root,
        image_size=128,
        extensions=(".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"),
        shuffle=True,
        seed=42,
        train_split=0.90,
    ):
        self.mode = mode
        self.batch_size = batch_size
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.extensions = tuple(e.lower() for e in extensions)
        self.shuffle = shuffle
        self.seed = seed
        self.train_split = train_split
        self.rng = np.random.default_rng(seed)

        if not self.data_root.is_dir():
            raise ValueError(f"Data root does not exist: {self.data_root}")

        self.class_names = sorted([
            p.name for p in self.data_root.iterdir()
            if p.is_dir()
        ])

        if not self.class_names:
            raise ValueError(f"No class folders found in {self.data_root}")

        self.classes = np.arange(len(self.class_names))

        all_x = []
        all_y = []

        for label, class_name in enumerate(self.class_names):
            class_dir = self.data_root / class_name

            print(f"\n[{label + 1}/{len(self.class_names)}] Scanning class: {class_name}")

            files = sorted([
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in self.extensions
            ])

            if not files:
                print(f"Warning: no images in class folder: {class_name}")
                continue

            self.rng.shuffle(files)

            split_idx = int(len(files) * train_split)

            if mode == self.TRAIN:
                selected_files = files[:split_idx]
            elif mode == self.TEST:
                selected_files = files[split_idx:]
            else:
                raise ValueError("mode must be TRAIN or TEST")

            print(f"  Found {len(files)} images, loading {len(selected_files)} for {mode}...")

            for i, img_path in enumerate(selected_files, start=1):
                if i == 1 or i % 100 == 0 or i == len(selected_files):
                    print(f"  Loading {i}/{len(selected_files)} images from {class_name}")
                    
                img = Image.open(img_path).convert("RGB")
                img = img.resize((image_size, image_size), Image.BICUBIC)

                arr = np.asarray(img, dtype=np.float32)

                # Normalize to [-1, 1], because BAGAN generator uses tanh.
                arr = (arr / 127.5) - 1.0

                # IBM BAGAN expects channels-first format: C, H, W
                arr = np.transpose(arr, (2, 0, 1))

                all_x.append(arr)
                all_y.append(label)

        if not all_x:
            raise ValueError(f"No images loaded for mode={mode}")

        self.dataset_x = np.asarray(all_x, dtype=np.float32)
        self.dataset_y = np.asarray(all_y, dtype=np.int64)

        self.per_class_ids = []
        self.per_class_count = []

        for c in self.classes:
            ids = np.where(self.dataset_y == c)[0]
            self.per_class_ids.append(ids)
            self.per_class_count.append(len(ids))

        self.per_class_count = np.asarray(self.per_class_count, dtype=np.int64)

        print(f"\nLoaded {mode} dataset")
        print(f"Root: {self.data_root}")
        print(f"Image shape: {self.dataset_x.shape[1:]}")
        print(f"Total images: {len(self.dataset_x)}")
        print("Class counts:")

        for name, count in zip(self.class_names, self.per_class_count):
            print(f"  {name}: {count}")

    def get_image_shape(self):
        return self.dataset_x.shape[1:]

    def get_label_table(self):
        return self.classes

    def get_num_samples(self):
        return len(self.dataset_x)

    def get_class_probability(self):
        total = len(self.dataset_y)
        return self.per_class_count.astype(np.float32) / total

    def get_samples_for_class(self, c, samples):
        ids = self.per_class_ids[c]

        if len(ids) == 0:
            raise ValueError(f"No samples found for class {c}")

        replace = len(ids) < samples
        chosen = self.rng.choice(ids, size=samples, replace=replace)
        return self.dataset_x[chosen]

    def next_batch(self):
        indices = np.arange(len(self.dataset_x))

        if self.shuffle:
            self.rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            end = start + self.batch_size
            batch_ids = indices[start:end]

            if len(batch_ids) == 0:
                continue

            yield self.dataset_x[batch_ids], self.dataset_y[batch_ids]