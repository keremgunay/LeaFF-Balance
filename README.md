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


## Methods Included

### BAGAN

BAGAN was used as a class-balancing GAN approach. It is designed for imbalanced image datasets and uses class-conditioning to generate minority-class samples.

In this project, the BAGAN implementation was modified and adapted for tomato plant image classes.

### LeafGAN / CycleGAN

The LeafGAN folder contains a modified image-to-image translation workflow based on the LeafGAN/CycleGAN family of methods.

In this project, this type of method was used to study whether translating healthy plant images into diseased-looking images can be useful for synthetic data balancing.

### StyleGAN2-ADA

StyleGAN2-ADA was used for high-quality noise-to-image synthetic image generation under limited-data conditions.

ADA, or adaptive discriminator augmentation, is useful when training GANs with relatively small datasets.


## Dataset Notes

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


## Tools

The `tools/` folder contains helper scripts used during the project workflow.

Tools include:

* RGB image conversion
* Image resizing
* Folder restructuring
* Random sample deletion or class count reduction
* Image cropping based on CVAT exports in COCO data format
* Dataset splitting to Train Validation and Test folders


## Basic Usage

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


## Author

Kerem Günay
Department of Electrical and Electronics Engineering
Yeditepe University

This repository was developed as part of an undergraduate engineering capstone project.

The code is research-oriented and may require manual path editing, environment setup, and dataset preparation before running. Some scripts were written for specific local experiment folders and may need adjustment before reuse.

## License

Third-party code inside `BAGAN/`, `LeafGAN/`, and `StyleGAN2-ADA/` remains under the licenses of the original projects. Always check the license file inside each folder before reuse or redistribution.

### How to Run StyleGAN2-ADA

StyleGAN2-ADA was used for single-class noise-to-image generation under limited-data conditions.

Example training command:

```bash
cd StyleGAN2-ADA

python train.py \
  --outdir=/path/to/results \
  --data=/path/to/dataset.zip \
  --gpus=1 \
  --cfg=paper256 \
  --batch=16 \
  --mirror=1 \
  --aug=ada \
  --target=0.6 \
  --kimg=1200 \
  --snap=20 \
  --metrics=fid50k_full
  
### How to Run CycleGAN
  
  python train.py \
  --dataroot /path/to/dataset \
  --name tomato_domain_translation \
  --model cycle_gan \
  --batch_size 1 \
  --n_epochs 100 \
  --n_epochs_decay 100
  
  
### How to Run BAGAN

BAGAN was used for minority-class image generation and dataset balancing.

Example workflow:

```bash
cd BAGAN

export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

python3 bagan_train.py \
  --dataset folder \
  --data_root /path/to/dataset \
  --image_size 128 \
  --epochs 200 \
  --batch_size 32 \
  --target_class 8 \
  --learning_rate 0.00005 \
  --run_name bagan_trips_128_epoch200_300img \

# Generate synthetic samples
python tool_Generate.py
