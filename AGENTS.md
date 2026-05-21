# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

LumbarSeg is a graduation research project for reproducing and improving a lumbar spine MRI segmentation model from Ahmed et al. (2025). The goal is to first replicate Dice ~0.97 results, then surpass them.

- **Dataset**: SPIDER (Zenodo) — 218 patients, 447 MRI series (T1/T2/T2 SPACE), MHA format
- **Model**: Modified U-Net with Leaky ReLU (α=0.1), Glorot init, Combined Loss (0.6×Focal + 0.4×Dice)
- **Execution**: Google Colab (GPU) with data on Google Drive at `/content/drive/MyDrive/SPIDER/DataSet/`

## Architecture

The baseline is available in two forms:

- `spine_segmentation.ipynb` — Colab-friendly exploratory notebook.
- Python CLI/package implementation:
  - `preprocess.py` — MHA to 2D slice preprocessing.
  - `train.py` — Modified U-Net baseline training.
  - `evaluate.py` — class-wise validation metrics.
  - `spine_baseline/` — preprocessing, dataset, model, losses, and metrics.
  - `arguments/` — grouped CLI parameters, following the FastGS-style script layout.

### SPIDER Mask Label Mapping (confirmed)

```
0       → 0 (Background)
1–99    → 1 (Vertebrae)
100     → 2 (Spinal Canal)
200+    → 3 (IVDs)
```

### Key Parameters

- Input: 2D slices (512×640) extracted from 3D MHA
- Loss: α=0.6 (Focal weight), γ=4.0 (Focal gamma)
- Training: batch_size=8, epochs=100, early stopping on val Mean IoU
- Filtering: exclude slices with <4 classes or class imbalance ratio >55%

## Conventions

- All communication with the user should be in Japanese
- Code comments and variable names in English
- Framework: TensorFlow/Keras (matching the paper's style)
- Documentation lives in `docs/` — see `docs/overview.md` for paper summary, `docs/project_goals.md` for goals
- Site assets that must be served by Astro live in `public/`; paper PDFs are under `public/papers/`
