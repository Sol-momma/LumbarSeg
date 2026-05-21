# LumbarSeg

腰椎MRIセグメンテーションの論文再現・改良を行う研究プロジェクト。

## 概要

**LumbarSeg** は、[Ahmed et al. (2025)](https://www.sciencedirect.com/science/article/pii/S2666827025000180) の腰椎MRIセグメンテーション手法を再現し、さらに精度を改良することを目指す卒業研究プロジェクト。

**タスク**: MRI画像から以下の4クラスをピクセル単位で自動セグメンテーション

| クラス | 構造                  | 臨床的意義           |
| ------ | --------------------- | -------------------- |
| 0      | 背景 (Background)     | —                    |
| 1      | 椎体 (Vertebrae)      | 骨折・変形の評価     |
| 2      | 脊柱管 (Spinal Canal) | 狭窄症の診断         |
| 3      | 椎間板 (IVDs)         | ヘルニア・変性の評価 |

## 手法

- **モデル**: Modified U-Net (Leaky ReLU, Glorot初期化, 512ch追加層)
- **損失関数**: Combined Loss = 0.6 x Focal Loss (gamma=4.0) + 0.4 x Dice Loss
- **データ前処理**: 18ラベル → 4クラス統合、クラス不均衡フィルタリング (閾値55%)

## データセット

[SPIDER Dataset](https://doi.org/10.5281/zenodo.10159290) (Nature Scientific Data 2024)

- 218患者 / 447 MRIシリーズ (T1, T2, T2 SPACE)
- オランダ4病院からのマルチセンター収集
- ライセンス: CC-BY 4.0

## プロジェクト目標

### Stage 1: 論文の再現

論文と同等の精度を達成する。

| 構造         | 論文 Dice | 比較 (nn-UNET) |
| ------------ | --------- | -------------- |
| IVDs         | 0.9688    | 0.86           |
| Vertebrae    | 0.9712    | 0.92           |
| Spinal Canal | 0.9671    | 0.92           |

### Stage 2: 精度の改良

アルゴリズムの改良により Dice係数をさらに向上させる。

## ドキュメント

| ドキュメント                                             | 内容                                                     |
| -------------------------------------------------------- | -------------------------------------------------------- |
| [論文概要・結果サマリ](docs/overview.md)                 | 論文の要点、データセット、結果の数値まとめ               |
| [プロジェクト目標・評価指標](docs/project_goals.md)      | Stage 1/2 の目標、T1/T2/T2 SPACE解説、Dice等の指標説明   |
| [教授向け発表アウトライン](docs/presentation_outline.md) | 15-20分の発表構成、各スライドのポイント                  |
| [データ前処理パイプライン](docs/preprocessing.md)        | 3D→2D変換、ラベルマッピング、フィルタリング手順          |
| [モデル構成・損失関数](docs/architecture.md)             | Modified U-Net詳細、Focal+Dice Loss数式、学習設定        |
| [SPIDERデータセット詳細](docs/dataset_spider.md)         | 病院別データ、分割、アノテーション方法、ベースライン結果 |
| [実装計画](docs/implementation_plan.md)                  | フェーズ分け、技術選定、未確定事項                       |
| [未解決事項](docs/open_questions.md)                     | 設計判断が必要なポイント、論文の矛盾点                   |
| [よく��る質問 (FAQ)](docs/faq.md)                        | 発表で想定されるQ&A (データ・モデル・結果・卒プロ)       |

## ディレクトリ構成

```
LumbarSeg/
├── README.md
├── CLAUDE.md
├── spine_segmentation.ipynb       # Colab向けの探索・実験ノートブック
├── preprocess.py                  # MHA→2D slice 前処理CLI
├── train.py                       # baseline学習CLI
├── evaluate.py                    # 学習済みモデル評価CLI
├── arguments/                     # FastGSを参考にしたCLI引数グループ
├── spine_baseline/                # ベースライン実装本体
│   ├── preprocessing.py           # ラベル変換、sagittal抽出、フィルタリング
│   ├── dataset.py                 # tf.data Dataset
│   ├── model.py                   # Modified U-Net
│   ├── losses.py                  # Focal + Dice loss
│   └── metrics.py                 # Dice / IoU / Precision / Recall / F1
├── docs/                          # ドキュメント
├── public/papers/                 # サイトで配信する論文PDF
├── data/                          # メタデータ
└── src/                           # Astroサイト
```

**注意**: MRI画像データ本体はGoogle Drive上 (`/content/drive/MyDrive/SPIDER/DataSet/`) に配置。

## 実行環境

- **コード実行**: Google Colab (GPU)
- **データ保存**: Google Drive
- **コード編集**: ローカル

## Baseline Quick Start

FastGS と同じく、実験の入口は root の CLI にまとめている。
Colab で `cd` してから以下を実行する。

### 1. 前処理のみ

依存関係:

```bash
pip install -r requirements-baseline.txt
```

```bash
python preprocess.py \
  --data_root /content/drive/MyDrive/SPIDER/DataSet \
  --output_root /content/drive/MyDrive/SPIDER/processed_baseline
```

特定シーケンスだけを処理する場合:

```bash
python preprocess.py --sequences T2_SPACE
```

### 2. 学習

```bash
python train.py \
  --data_root /content/drive/MyDrive/SPIDER/DataSet \
  --output_root /content/drive/MyDrive/SPIDER/processed_baseline \
  --batch_size 8 \
  --epochs 100
```

出力:

- `processed_baseline/images/`, `processed_baseline/masks/`
- `processed_baseline/filtered_files.txt`
- `processed_baseline/filtered_slice_stats.csv`
- `processed_baseline/checkpoints/best_model.keras`
- `processed_baseline/checkpoints/final_model.keras`
- `processed_baseline/checkpoints/training_log.csv`

### 3. 評価

```bash
python evaluate.py \
  --data_root /content/drive/MyDrive/SPIDER/DataSet \
  --output_root /content/drive/MyDrive/SPIDER/processed_baseline \
  --model_path /content/drive/MyDrive/SPIDER/processed_baseline/checkpoints/best_model.keras
```

短時間確認:

```bash
python evaluate.py --model_path /path/to/best_model.keras --limit 100
```

### 主な引数

| 引数 | 既定値 | 内容 |
| ---- | ------ | ---- |
| `--target_height` / `--target_width` | `512` / `640` | 論文の2D入力サイズ |
| `--num_classes` | `4` | 背景・椎体・脊柱管・椎間板 |
| `--imbalance_threshold` | `0.55` | foreground内の支配クラス割合の除外閾値 |
| `--focal_weight` | `0.6` | Combined Loss内のFocal重み |
| `--focal_gamma` | `4.0` | Focal Lossのgamma |
| `--dropout_rate` | `0.5` | 論文未記載のためbaseline既定値 |
| `--learning_rate` | `1e-4` | 論文未記載のためAdam baseline既定値 |

## ノートブックとWebサイト

このリポジトリには 2 つの実行対象があります。

- `spine_segmentation.ipynb`
  - 論文再現用のメインノートブック
  - **Google Colab で実行**
- Astro サイト
  - LumbarSeg の研究ページ、論文メモ、研究メモの表示用サイト
  - **ローカルで実行**

### Astro サイトをローカルで起動

前提:

- Node.js `>=22.12.0`
- `npm install` 済み

開発サーバー:

```bash
npm run dev
```

- 通常は `http://127.0.0.1:4321/LumbarSeg/`
- 研究メモページは `http://127.0.0.1:4321/LumbarSeg/notes/`
- 英語ページは `http://127.0.0.1:4321/LumbarSeg/en/`

本番ビルド:

```bash
npm run build
```

- 出力先は `dist/`

ビルド結果の確認:

```bash
npm run preview
```

コード整形:

```bash
npm run format
```

整形チェックのみ:

```bash
npm run format:check
```

- GitHub Actions でも `Prettier` workflow が `main` への push / pull request で走る

### Colab ノートブックの実行

`spine_segmentation.ipynb` は Astro ではなく、Google Colab で実行する。

### Colab での実行方法

1. `spine_segmentation.ipynb` を Google Colab で開く
2. ランタイム → ランタイムのタイプを変更 → GPU を選択
3. セルを上から順に実行

### 必要なライブラリ

- TensorFlow / Keras
- SimpleITK (MHAファイル読み込み)
- NumPy, Matplotlib
- scikit-learn (評価指標)

## 参考論文

1. **本論文**: Ahmed, I. et al. "Pioneering Precision in Lumbar Spine MRI Segmentation with Advanced Deep Learning and Data Enhancement." _Machine Learning with Applications_, Vol.20, 2025.

2. **データセット論文**: van der Graaf, J.W. et al. "Lumbar spine segmentation in MR images: a dataset and a public benchmark." _Nature Scientific Data_, 11:264, 2024.
