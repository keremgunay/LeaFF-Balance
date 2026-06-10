# LeaFF-Balance
A Leaf–Fruit–Flower Synthetic Image Generation Pipeline for Balancing Plant Disease Classification Datasets

LeaFF-Balance is a capstone project repository for a tomato plant disease classification and synthetic data balancing pipeline.

The project investigates how generative models can be used to reduce class imbalance in image-based plant disease datasets. 

The main goal is to test whether GAN-generated synthetic images can improve classifier performance when some tomato plant disease classes are underrepresented.


## Project Overview

This repository contains modified and integrated implementations used to experiment with synthetic image generation and dataset balancing.

The project focuses on:

* Tomato plant image classification
* Leaf, fruit, and flower image categories
* Class imbalance simulation
* Synthetic data generation
* GAN-based data augmentation
* Classifier performance comparison before and after balancing

Synthetic images are used only for training-set augmentation. Validation and test sets should remain real-only to avoid artificially inflated results.


## Repository Structure

```text
LeaFF-Balance/
│
├── BAGAN/
│   └── Modified BAGAN-based implementation for minority-class image generation.
│
├── LeafGAN/
│   └── Modified LeafGAN/CycleGAN-style image-to-image translation implementation.
│
├── StyleGAN2-ADA/
│   └── Modified StyleGAN2-ADA PyTorch implementation and related generation files.
│
├── tools/
│   └── Utility scripts for dataset preparation, resizing, conversion, filtering, and folder handling.
│
├── main.py
│   └── Main project script / entry point.
│
├── README.md
│
└── LICENSE
```


## Methods Included

### BAGAN

BAGAN was used as a class-balancing GAN approach. It is designed for imbalanced image datasets and uses class-conditioning to generate minority-class samples.

In this project, the BAGAN implementation was modified and adapted for tomato plant image classes.

### LeafGAN / CycleGAN Translation

The LeafGAN folder contains a modified image-to-image translation workflow based on the LeafGAN/CycleGAN family of methods.

In this project, this type of method was used to study whether translating healthy plant images into diseased-looking images can be useful for synthetic data balancing.

### StyleGAN2-ADA

StyleGAN2-ADA was used for high-quality noise-to-image synthetic image generation under limited-data conditions.

ADA, or adaptive discriminator augmentation, is useful when training GANs with relatively small datasets.


## Dataset Notes

The full dataset is not included in this repository.

Large image datasets, generated samples, trained model weights, and experiment outputs may be excluded because of file size, privacy, licensing, or storage limitations.

If you want to reproduce the experiments, prepare your dataset in class-based folders such as:

```text
dataset/
├── healthy_low_leaf/
├── flies_low_leaf/
├── trips_low_leaf/
├── nutdef_low_leaf/
├── healthy_fruit/
├── healthy_flower/
└── ...
```

Recommended practice:

* Keep training, validation, and test splits separate.
* Add synthetic images only to the training split.
* Keep validation and test data real-only.
* Avoid committing large datasets or model checkpoints directly to GitHub.
* Use Git LFS for large files if needed.


## Tools

The `tools/` folder contains helper scripts used during the project workflow.

These may include utilities for:

* RGB image conversion
* Image resizing
* Folder restructuring
* Random sample deletion or class count reduction
* Dataset balancing preparation
* File cleanup and preprocessing

The exact usage depends on the script. Check the script names and internal comments before running them.


## Basic Usage

Clone the repository:

```bash
git clone https://github.com/keremgunay/LeaFF-Balance.git
cd LeaFF-Balance
```

Because this repository combines multiple GAN implementations, there is no single universal environment guaranteed to run every method. Each GAN folder may require its own dependencies.

General workflow:

```bash
# 1. Prepare dataset folders
# 2. Resize / clean images using tools
# 3. Train selected GAN model
# 4. Generate synthetic images
# 5. Add synthetic images only to training set
# 6. Train classifier on real-only baseline and GAN-balanced data
# 7. Compare test performance
```


## Third-Party Code and Attribution

This repository contains modified versions of existing open-source research implementations. The included third-party folders are preserved for academic experimentation and project integration.

The original projects and authors should be credited as follows:

| Folder           | Original Project                                                                     | Original Source                                 |
| ---------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------- |
| `BAGAN/`         | BAGAN: Data Augmentation with Balancing GAN                                          | https://github.com/IBM/BAGAN                    |
| `LeafGAN/`       | LeafGAN: An Effective Data Augmentation Method for Practical Plant Disease Diagnosis | https://github.com/IyatomiLab/LeafGAN           |
| `StyleGAN2-ADA/` | StyleGAN2-ADA PyTorch                                                                | https://github.com/NVlabs/stylegan2-ada-pytorch |

These folders may contain project-specific modifications made for the LeaFF-Balance capstone experiments. They should not be treated as untouched official releases of the original repositories.

Please refer to the license files inside each third-party folder for their original licensing terms.


## Project Status

This repository was developed as part of an undergraduate engineering capstone project.

The code is research-oriented and may require manual path editing, environment setup, and dataset preparation before running. Some scripts were written for specific local experiment folders and may need adjustment before reuse.


## Author

Kerem Günay
Department of Electrical and Electronics Engineering
Yeditepe University


## License

The root repository contains an MIT license for original project code unless otherwise stated.

Third-party code inside `BAGAN/`, `LeafGAN/`, and `StyleGAN2-ADA/` remains under the licenses of the original projects. Always check the license file inside each folder before reuse or redistribution.

