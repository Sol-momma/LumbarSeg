# LumbarSeg: Four-Class Lumbar MRI Segmentation

## Graduation Research 2026

[Homepage](https://lumbarseg.github.io/) | [Paper](https://www.sciencedirect.com/science/article/pii/S2666827025000180) | [Dataset](https://doi.org/10.5281/zenodo.10159290) | [Detailed README](docs/archive/readme-snapshots/README.zh-TW.detailed.md)

LumbarSeg is a research repository for reproducing and improving the Ahmed et al. (2025) lumbar MRI segmentation baseline.

## What Makes LumbarSeg Special?

- **Paper-aligned baseline**: Modified U-Net with Focal + Dice loss.
- **Four-class segmentation**: Background, vertebrae, spinal canal, and IVDs.
- **SPIDER-ready preprocessing**: 3D MHA to sagittal 2D slices.
- **Colab-first execution**: Designed for Google Colab GPU.
- **Separated project page**: Website source lives in `lumbarseg.github.io`.

## Quick Start

```bash
git clone https://github.com/Sol-momma/LumbarSeg.git
cd LumbarSeg
pip install -r requirements-baseline.txt
```

```bash
python preprocess.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --sequences T2_SPACE
```

```bash
python train.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --sequences T2_SPACE \
  --epochs 1 \
  --batch_size 2
```

## Quick Facts

| Feature | LumbarSeg |
| --- | --- |
| Task | Four-class lumbar MRI segmentation |
| Dataset | SPIDER |
| Model | Modified U-Net |
| Loss | `0.6 x Focal + 0.4 x Dice` |
| Runtime | Google Colab GPU |

See the main [README.md](README.md) for the full current overview.
