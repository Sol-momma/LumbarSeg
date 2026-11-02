# LumbarSeg: Four-Class Lumbar MRI Segmentation

[English](README.md) | [日本語](README.ja.md)

腰椎MRIを4クラス（背景・椎骨・脊柱管・椎間板）に分ける卒業研究のコードです。

## Graduation Research 2026

[Homepage](https://lumbarseg.sol-momma.com/) | [Paper](https://www.sciencedirect.com/science/article/pii/S2666827025000180) | [Dataset](https://doi.org/10.5281/zenodo.10159290) | [Detailed README](docs/archive/readme-snapshots/README.ja.detailed.md)

![LumbarSeg overview](https://img.shields.io/badge/Task-Lumbar%20MRI%20Segmentation-blue)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange)
![Dataset](https://img.shields.io/badge/Dataset-SPIDER-green)

## What Makes LumbarSeg Special?

LumbarSeg は、Ahmed et al. (2025) の腰椎 MRI セグメンテーション baseline を再現し、再現した baseline から先の改良を探る研究用リポジトリです。まず SPIDER の T2 SPACE で報告されている Dice 約 **0.97** を目標にし、その後に改良を調べます。

- **Paper-aligned baseline**: Leaky ReLU、Glorot initialization、Focal と Dice を組み合わせた損失の Modified U-Net。
- **Four-class segmentation**: 背景、椎骨、脊柱管、椎間板。
- **SPIDER-ready preprocessing**: 3D MHA ボリュームを sagittal の 2D 学習スライスに変換します。
- **Colab-first execution**: Google Colab の GPU で動かし、データは Google Drive に置く前提です。
- **Separated project page**: Web サイトのソースは専用リポジトリ `lumbarseg.github.io` にあり、FastGS のようにコードと公開ページを分けています。

## Latest Updates

#### **[2026.06]**

リポジトリを分割しました。研究コードは `Sol-momma/LumbarSeg` に残し、Astro のプロジェクトサイトは `lumbarseg.github.io` へ移しました。

#### **[2026.06]**

前処理、学習、検証評価のための baseline CLI を追加しました。

#### **[2026.06]**

SPIDER のマスクラベル対応を確認しました。

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

baseline は **TensorFlow/Keras** を使い、対象論文の実装方針に沿っています。

### Hardware Requirements

- **GPU**: Google Colab の GPU を推奨します。
- **Memory**: バッチサイズの目標は `8` です。GPU が小さい場合は下げてください。
- **Storage**: SPIDER データセットと生成したスライスは、Google Drive かローカルの速いストレージに置いてください。

### Software Requirements

- Python 3.10+
- TensorFlow/Keras
- SimpleITK
- NumPy / pandas / scikit-learn

baseline の依存関係を入れます。

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

Zenodo から SPIDER をダウンロードし、次のように置きます。

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

データセットのメモは [data/README.md](data/README.md) を見てください。

### Preprocess

```bash
python preprocess.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --sequences T2_SPACE
```

### Smoke Test

本実験の前に 1 epoch だけ動かします。

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

### Prediction Visualization

validation slice の入力画像・正解 mask・予測 mask・overlay を PNG で出力します。

```bash
python visualize_predictions.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_relaxed \
  --model_path outputs/t2_relaxed/checkpoints/best_model.keras \
  --sequences T2 \
  --min_classes 3 \
  --imbalance_threshold 0.90 \
  --max_slices_per_sequence 0 \
  --num_samples 12
```

出力先は `outputs/t2_relaxed/predictions/` です。各 PNG に対応する slice Dice は `prediction_summary.csv` に保存されます。

検証データ全体を採点し、全体・椎骨・脊柱管・椎間板ごとに低精度な画像を8枚ずつ確認する場合は、次を実行します。

中間画像を削除済みの場合は、保存した `filtered_files.txt` から検証用画像だけを別の場所へ再生成します。この処理は元のMHAと対象一覧を読み取るだけで、Google Driveの原本や保存済み一覧を変更しません。

```bash
python regenerate_selected_slices.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root /path/to/temporary_validation_slices \
  --file_list /path/to/experiment/filtered_files.txt \
  --split validation
```

```bash
python visualize_predictions.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root /path/to/temporary_validation_slices \
  --model_path outputs/all_sequences/checkpoints/best_model.keras \
  --filtered_file_list outputs/all_sequences/filtered_files.txt \
  --min_classes 4 \
  --imbalance_threshold 0.90 \
  --max_slices_per_sequence 1000 \
  --batch_size 2 \
  --worst_per_metric 8
```

`prediction_summary.csv` には検証データ全件のDiceと正解・予測領域の画素数が保存されます。低精度画像は `worst_dice_mean/`、`worst_dice_vertebrae/`、`worst_dice_spinal_canal/`、`worst_dice_ivds/` に分かれ、順位は `worst_case_summary.csv` で確認できます。

### Advanced: Key Command Line Arguments

#### `--sequences`

使う MRI シーケンスです。`T1`、`T2`、`T2_SPACE` に対応します。

#### `--target_height` / `--target_width`

2D スライスのサイズです。既定は `512 x 640` です。

#### `--imbalance_threshold`

スライス選別で許す、支配的な前景クラスの割合の上限です。既定は `0.55` です。

#### `--max_slices_per_sequence`

シーケンスごとに残すスライス数の上限です。既定は `1000` です。

#### `--focal_weight` / `--focal_gamma`

組み合わせ損失の設定です。既定は `0.6` と `4.0` です。

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

## Detailed Notes

以前の詳しい README は [docs/archive/readme-snapshots/README.ja.detailed.md](docs/archive/readme-snapshots/README.ja.detailed.md) に退避しています。

## Experiment Records

ローカル GPU / Colab での軽量な実験記録は [docs/experiments/README.md](docs/experiments/README.md) に保存します。モデル重みや前処理済み slice は Git 管理対象外です。

## Paper Reproduction

論文 reported Dice の完全再現に向けた監査・実行計画は [PAPER_REPRODUCTION_PLAN.md](PAPER_REPRODUCTION_PLAN.md) を参照してください。データ数・filtering・split の監査には `audit_reproduction.py` を使います。

## Pull Request Rule

PR description は必ず英語で `Why` / `What` / `How` を含めます。詳細は [docs/workflow/pr_description.md](docs/workflow/pr_description.md) を参照してください。

## Acknowledgements

このプロジェクトは SPIDER データセットと、Ahmed et al. (2025) が報告した腰椎 MRI セグメンテーション baseline に基づいています。リポジトリ構成は FastGS を参考にしており、特に研究コードと公開 Web サイトの分離を踏襲しています。

## Citation

このリポジトリが研究の助けになった場合は、元のデータセットと baseline の論文を引用してください。

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

**LumbarSeg が研究の助けになったら、このリポジトリに star を付けてもらえると嬉しいです。**

## Experiment harness

[HARNESS.md](HARNESS.md) を見てください。`python -m lumbarseg_cli smoke --download --json` は、データ取得から小さな学習、評価、metrics JSON までを 1 行で実行します。`doctor` は、この環境、Colab、研究室 PC（WSL2）の GPU・データ・依存関係を確認します。

## License

ライセンスファイルはまだないため、すべての権利は留保されています。SPIDER データセットと対象論文はそれぞれの条件に従います。先頭のリンクを見てください。
