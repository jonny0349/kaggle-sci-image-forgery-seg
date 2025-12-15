# Kaggle Scientific Image Forgery Segmentation (Learning Project)

This repo is a learning-focused implementation of a segmentation pipeline for detecting copy-move forgeries in biomedical images (kaggle-style workflow).

## Repository Structure

- `configs/`
  - `baseline.yaml`: single source of truth for paths, model, training params
- `src/`
  - `utils.py`: config loading, logging, reproducibility helpers
  - `data.py`: dataset + transforms + dataloaders (supports `.png` images and `.npy` or `.png` masks)
  - `model.py`: model factory (segmentation_models_pytorch)
  - `losses.py`: BCE/Dice losses
  - `metric.py`: Dice/IoU
  - `train.py`: training loop (logs + checkpoints)
  - `infer.py`: inference on a folder and save predicted masks (+ optional overlays)
- `notebooks/`
  - `01_eda.ipynb`: visualize image/mask pairs and sanity-check alignment
- `data/` (ignored by git since it contains images)
  - `local dataset storage
- `outputs/` (ignored by git)
  - logs, checkpoints, predictions

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```
