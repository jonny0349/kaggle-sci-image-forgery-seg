# Kaggle Scientific Image Forgery Segmentation

_A learning-focused, end-to-end segmentation pipeline_

---

# 1. Project Motivation

Scientific image forgery, particularly **copy-move manipulation**, is one of the most common forms of misconduct in biomedical publications. In these cases, regions of an image are duplicated to fabricate or exaggerate results.

This project is a **learning-driven implementation** of a computer vision pipeline designed to:

- Detect whether an image contains copy-move forgery
- Segment the forged regions at the pixel level

The work is inspired by a Kaggle competition built on real retracted scientific figures. The primary goal is **not leaderboard performance**, but to deeply understand:

- The data flow
- The modeling decisions
- The training and inference lifecycle and
- The practical challenges of segmentation tasks

---

# 2. High-Level Approach

This project follows a standard **semantic segmentation workflow**:

1. Load biomedical images and their corresponding pixel-level masks
2. Apply preprocessing and augmentation
3. Train convolutional neural network (U-Net)
4. Evaluate predictions using Dice and IoU
5. Run inference to generate predicted masks and overlays
6. Visualize results for qualitative validation
7. Scale training using Kaggle GPU resources

The project was intentionally developed **locally first**, using a very small dataset, to validate correctness before scaling.

---

# 3. Repository Structure

kaggle-sci-image-forgery-seg/
│
├── configs/
│ └── baseline.yaml # Central configuration file (paths, model, training)
│
├── src/
│ ├── data.py # Dataset, transforms, and dataloaders
│ ├── model.py # Model factory (U-Net via segmentation_models_pytorch)
│ ├── losses.py # BCE, Dice, and combined losses
│ ├── metrics.py # Dice and IoU metrics
│ ├── train.py # Training + validation loop
│ ├── infer.py # Inference on folders + mask/overlay export
│ └── utils.py # Config loading, logging, reproducibility helpers
│
├── notebooks/
│ └── 01_eda.ipynb # Data sanity checks and visualization
│
├── data/ # Local data storage (ignored by git)
│ └── raw/
│ ├── train/
│ │ ├── images/
│ │ └── masks/
│ └── val/
│ ├── images/
│ └── masks/
│
├── outputs/ # Generated artifacts (ignored by git)
│ ├── checkpoints/
│ └── preds/
│
├── requirements.txt
└── README.md

---

# 4. Configuration-Driven Design

All major decisions are controlled via `configs/baseline.yaml`:

- dataset paths
- file extensions (.png, .npy)
- model architecture (encoder, pretrained weights)
- optimizer and scheduler
- batch size, epochs, and device

This allows:

- reproducibility
- clean experimentation
- easy migration to Kaggle without changing code

---

# 5. Local Development Strategy (Why Small Data?)

Local training was intentionally performed on a **tiny dataset** (10 images) to:

- verify that images and masks load correctly
- confirm mask alignment (via overlays)
- validate the training loop and loss behavior
- test inference and output generation
- debug device handling (CPU/MPS)

Because the local dataset contained **very few positive (forged) samples**, the model learned:

- to correctly predict empty masks for non-forged images
- to over-predict large regions for forged images

This behavior is **expected** and confirms that the pipeline is working, even though segmentation quality is not yet strong.

---

# 6. Exploratory Data Analysis (EDA)

The notebook `notebooks/01_eda.ipynb` is a critical component of the project.

It is used to:

- visually inspect image-mask pairs
- overlay masks on images
- quantify dataset properties:
  - number of samples
  - fraction of empty masks
  - foreground pixel ratios
- compare **ground truth vs predicted masks**

This notebook acts as a sanity checkpoint before any large-scale training.

---

# 7. Training and Evaluation

## Training

`python -m src.train --cfg configs/baseline.yaml`

- Supports CPU, Apple MPS, and CUDA
- Saves best checkpoints based on Dice or IoU
- Logs training progress

## Inference

`python -m src.infer \
    --cfg configs/baseline.yaml \
    --checkpoint outputs/checkpoints/best_dice.pt \
    --imput_dir data/raw/val/images \
    --output_dir outputs/preds/val \
    --thr 0.5 \
    --save_overlay`

## Outputs

- binary mask predictions (.png)
- optional red overlays for visual inspection

---

# 8. Key Learning So Far

- Segmentation quality is **data-limited**, not code-limited
- Empty masks dominate small samples and bias learning
- Visual inspection is as important as metrics
- Correct pipelines matter more than early performance
- Local validation prevents costly Kaggle debugging

---

# 9. Nexst Steps (Kaggle Phase)

The next phase of this project will move training to **Kagle GPU** in order to:

1. Train on the **full dataset** with hundreds of positive samples
2. Use CUDA and mixed precision for efficient training
3. Improve localization quality of forged regions
4. Compare quantitative metrics before vs after scaling
5. Explore thresholding and post-processing strategies

The local setup will remain as a **unit test environment** to validate future changes.

---

# 10. Disclaimer

This project is educational in nature.
It is focused on understanding workflow, not claiming state-of-the-art results.
