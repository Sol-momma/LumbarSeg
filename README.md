# LumbarSeg: Four-Class Lumbar MRI Segmentation

## Graduation Research 2026

[Homepage](https://lumbarseg.github.io/) | [Paper](https://www.sciencedirect.com/science/article/pii/S2666827025000180) | [Dataset](https://doi.org/10.5281/zenodo.10159290) | [Detailed README](docs/archive/readme-snapshots/README.detailed.md)

![LumbarSeg overview](https://img.shields.io/badge/Task-Lumbar%20MRI%20Segmentation-blue)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange)
![Dataset](https://img.shields.io/badge/Dataset-SPIDER-green)

## What Makes LumbarSeg Special?

LumbarSeg is a research repository for reproducing and improving the lumbar spine MRI segmentation baseline from Ahmed et al. (2025). The project first targets the reported Dice score of about **0.97** on SPIDER T2 SPACE data, then explores improvements beyond the reproduced baseline.

- **Paper-aligned baseline**: Modified U-Net with Leaky ReLU, Glorot initialization, and combined Focal + Dice loss.
- **Four-class segmentation**: Background, vertebrae, spinal canal, and intervertebral discs.
- **SPIDER-ready preprocessing**: Converts 3D MHA volumes into sagittal 2D training slices.
- **Colab-first execution**: Designed for Google Colab GPU runs with data stored on Google Drive.
- **Separated project page**: Website source lives in a dedicated `lumbarseg.github.io` repository, following the FastGS-style code/page split.

## Latest Updates

#### **[2026.06]**

Repository split completed: research code stays in `Sol-momma/LumbarSeg`, while the Astro project website has been moved to `lumbarseg.github.io`.

#### **[2026.06]**

Baseline CLI added for preprocessing, training, and validation evaluation.

#### **[2026.06]**

SPIDER mask label mapping confirmed:

| Raw label | Class |
| --- | --- |
| `0` | Background |
| `1-99` | Vertebrae |
| `100` | Spinal Canal |
| `200+` | IVDs |

## Coming Soon

#### Released

- **Baseline preprocessing**: `preprocess.py`
- **Modified U-Net training**: `train.py`
- **Validation metrics**: `evaluate.py`
- **Reusable package modules**: `spine_baseline/`

#### To Be Released

- **Full T2 SPACE reproduction run**
- **Mask overlay visualization**
- **Architecture and augmentation improvements**
- **T1/T2/T2 SPACE comparison**

## Training Framework

The baseline uses **TensorFlow/Keras** and follows the implementation style of the target paper.

### Hardware Requirements

- **GPU**: Google Colab GPU is recommended.
- **Memory**: Batch size `8` is the default target; reduce it for smaller GPUs.
- **Storage**: SPIDER dataset and generated slices should be stored on Google Drive or local fast storage.

### Software Requirements

- Python 3.10+
- TensorFlow/Keras
- SimpleITK
- NumPy / pandas / scikit-learn

Install the baseline dependencies:

```bash
pip install -r requirements-baseline.txt
```

## Quick Start

### Clone the Repository

```bash
git clone https://github.com/Sol-momma/LumbarSeg.git
cd LumbarSeg
```

### Dataset Organization

Download SPIDER from Zenodo and organize it as:

```text
SPIDER/DataSet/
├── images/
│   ├── *.mha
│   └── ...
├── masks/
│   ├── *.mha
│   └── ...
└── SPIDER Lumbar Spine Segmentation Overview.csv
```

See [data/README.md](data/README.md) for dataset notes.

### Preprocess

```bash
python preprocess.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --sequences T2_SPACE
```

### Smoke Test

Run one epoch before a full experiment:

```bash
python train.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --sequences T2_SPACE \
  --epochs 1 \
  --batch_size 2
```

## Training & Evaluation

### Baseline Training

```bash
python train.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --sequences T2_SPACE \
  --batch_size 8 \
  --epochs 100
```

### Evaluation

```bash
python evaluate.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --model_path outputs/t2_space_baseline/checkpoints/best_model.keras
```

### Advanced: Key Command Line Arguments

#### `--sequences`

MRI sequences to use. Supports `T1`, `T2`, and `T2_SPACE`.

#### `--target_height` / `--target_width`

2D slice size. Defaults to `512 x 640`.

#### `--imbalance_threshold`

Maximum dominant foreground class ratio allowed during slice filtering. Default: `0.55`.

#### `--max_slices_per_sequence`

Maximum number of slices retained per sequence. Default: `1000`.

#### `--focal_weight` / `--focal_gamma`

Combined loss settings. Defaults: `0.6` and `4.0`.

## Quick Facts

| Feature | LumbarSeg |
| --- | --- |
| Task | Four-class lumbar MRI segmentation |
| Dataset | SPIDER |
| Input | 2D sagittal slices from 3D MHA |
| Model | Modified U-Net |
| Loss | `0.6 x Focal + 0.4 x Dice` |
| Target metric | Dice around `0.97` on T2 SPACE |
| Main runtime | Google Colab GPU |

## Repository Layout

```text
LumbarSeg/
├── preprocess.py
├── train.py
├── evaluate.py
├── spine_baseline/
├── arguments/
├── data/
├── requirements-baseline.txt
└── REPOSITORY_SPLIT.md
```

## Acknowledgements

This project is based on the SPIDER dataset and the lumbar MRI segmentation baseline reported by Ahmed et al. (2025). The repository structure is inspired by FastGS, especially the separation between research code and the public website.

## Citation

If this repository helps your research, please cite the original dataset and baseline papers:

```bibtex
@article{ahmed2025lumbar,
  title={Pioneering Precision in Lumbar Spine MRI Segmentation with Advanced Deep Learning and Data Enhancement},
  author={Ahmed, I. and others},
  journal={Machine Learning with Applications},
  volume={20},
  year={2025}
}

@article{vanderGraaf2024spider,
  title={Lumbar spine segmentation in MR images: a dataset and a public benchmark},
  author={van der Graaf, J. W. and others},
  journal={Scientific Data},
  volume={11},
  pages={264},
  year={2024}
}
```

---

**If LumbarSeg helps your research, please consider starring this repository.**
