# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Graduation project: reproducing and improving a lumbar spine MRI segmentation model from Ahmed et al. (2025). The goal is to first replicate Dice ~0.97 results, then surpass them.

- **Dataset**: SPIDER (Zenodo) — 218 patients, 447 MRI series (T1/T2/T2 SPACE), MHA format
- **Model**: Modified U-Net with Leaky ReLU (α=0.1), Glorot init, Combined Loss (0.6×Focal + 0.4×Dice)
- **Execution**: Google Colab (GPU) with data on Google Drive at `/content/drive/MyDrive/SPIDER/DataSet/`

## Architecture

All code lives in a single Colab notebook (`spine_segmentation.ipynb`). The plan is to extract `.py` modules only at paper submission time.

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
