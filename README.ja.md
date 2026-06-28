# LumbarSeg: Four-Class Lumbar MRI Segmentation

## Graduation Research 2026

[Homepage](https://lumbarseg.github.io/) | [Paper](https://www.sciencedirect.com/science/article/pii/S2666827025000180) | [Dataset](https://doi.org/10.5281/zenodo.10159290) | [Detailed README](docs/archive/readme-snapshots/README.ja.detailed.md)

LumbarSeg は、Ahmed et al. (2025) の腰椎 MRI セグメンテーション baseline を再現し、そこから改良するための研究用リポジトリです。

## What Makes LumbarSeg Special?

- **Paper-aligned baseline**: Modified U-Net、Leaky ReLU、Glorot initialization、Focal + Dice loss。
- **Four-class segmentation**: Background、vertebrae、spinal canal、IVDs。
- **SPIDER-ready preprocessing**: 3D MHA から sagittal 2D slice を生成。
- **Colab-first execution**: Google Colab GPU と Google Drive 上の SPIDER データを前提に実行。
- **Separated project page**: Web サイトは `lumbarseg.github.io` 側に分離。

## Latest Updates

#### **[2026.06]**

FastGS 風にコードリポジトリと Web リポジトリを分離しました。

#### **[2026.06]**

`preprocess.py`、`train.py`、`evaluate.py` による baseline CLI を追加しました。

## Quick Start

### Clone the Repository

```bash
git clone https://github.com/Sol-momma/LumbarSeg.git
cd LumbarSeg
pip install -r requirements-baseline.txt
```

### Preprocess

```bash
python preprocess.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --sequences T2_SPACE
```

### Smoke Test

```bash
python train.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --sequences T2_SPACE \
  --epochs 1 \
  --batch_size 2
```

### Windows GPU

Windows PC の NVIDIA GPU で実行する場合は、[docs/windows_gpu.md](docs/windows_gpu.md) を参照してください。native Windows GPU は TensorFlow `2.10` 系が前提です。

最短コマンド:

```powershell
git clone https://github.com/Sol-momma/LumbarSeg.git
cd LumbarSeg
powershell -ExecutionPolicy Bypass -File scripts\setup_windows_native_gpu.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_windows_smoke.ps1 -DataRoot "D:\SPIDER\DataSet"
powershell -ExecutionPolicy Bypass -File scripts\run_windows_train.ps1 -DataRoot "D:\SPIDER\DataSet"
```

## Training & Evaluation

```bash
python train.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --sequences T2_SPACE \
  --batch_size 8 \
  --epochs 100
```

```bash
python evaluate.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --model_path outputs/t2_space_baseline/checkpoints/best_model.keras
```

## Quick Facts

| Feature | LumbarSeg |
| --- | --- |
| Task | Four-class lumbar MRI segmentation |
| Dataset | SPIDER |
| Input | 2D sagittal slices from 3D MHA |
| Model | Modified U-Net |
| Loss | `0.6 x Focal + 0.4 x Dice` |
| Target metric | Dice around `0.97` on T2 SPACE |
| Runtime | Google Colab GPU |

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

## Detailed Notes

以前の詳しい README は [docs/archive/readme-snapshots/README.ja.detailed.md](docs/archive/readme-snapshots/README.ja.detailed.md) に退避しています。

## Experiment Records

ローカル GPU / Colab での軽量な実験記録は [docs/experiments/README.md](docs/experiments/README.md) に保存します。モデル重みや前処理済み slice は Git 管理対象外です。
