# LumbarSeg

**Languages / 语言 / 言語:** [English](README.md) · [简体](README.zh-CN.md) · [繁體](README.zh-TW.md) · **日本語**

腰椎 MRI の **4クラスセグメンテーション** を、Ahmed et al. (2025) のベースライン（Modified U-Net + Combined Loss）で再現する卒業研究用リポジトリです。

| リンク | |
| --- | --- |
| 論文（再現対象） | [Ahmed et al., 2025](https://www.sciencedirect.com/science/article/pii/S2666827025000180) |
| データセット | [SPIDER (Zenodo)](https://doi.org/10.5281/zenodo.10159290) |
| プロジェクトサイト | [lumbarseg.github.io](https://lumbarseg.github.io/) |

---

## リポジトリ分離

このリポジトリは、再現実験用のコードと実験ドキュメントに集中させます。Astro の Web サイトソースは、FastGS と同じようにコード本体とプロジェクトページを分けるため、別のローカルリポジトリ `../lumbarseg.github.io` に移動しました。

各リポジトリに置く内容は [REPOSITORY_SPLIT.md](REPOSITORY_SPLIT.md) を参照してください。

---

## 研究の目的

```mermaid
flowchart LR
  S1["Stage 1<br/>論文ベースライン再現"] --> S2["Stage 2<br/>論文を上回る改良"]
  S1 --> M1["Dice ≈ 0.97<br/>T2 SPACE"]
  S2 --> M2["アーキテクチャ・拡張・損失"]
```

### セグメンテーションクラス（4クラス）

```mermaid
flowchart LR
  subgraph spider["SPIDER 元ラベル"]
    R0["0"]
    R1["1–99"]
    R2["100"]
    R3["200+"]
  end
  subgraph four["4クラス"]
    C0["0 Background"]
    C1["1 椎体"]
    C2["2 脊柱管"]
    C3["3 椎間板"]
  end
  R0 --> C0
  R1 --> C1
  R2 --> C2
  R3 --> C3
```

| ID | 構造 | SPIDER 元ラベル |
| ---: | --- | --- |
| 0 | Background | `0` |
| 1 | Vertebrae（椎体） | `1–99` |
| 2 | Spinal Canal（脊柱管） | `100` |
| 3 | IVDs（椎間板） | `200+` |

### 論文の目標指標（T2 SPACE・参考）

| 構造 | Dice | IoU |
| --- | ---: | ---: |
| IVDs | 0.9688 | 0.9476 |
| Vertebrae | 0.9712 | 0.9461 |
| Spinal Canal | 0.9671 | 0.9501 |

---

## 全体の流れ（研究パイプライン）

```mermaid
flowchart LR
  subgraph input["入力"]
    A["SPIDER 3D MHA<br/>images + masks"]
  end
  subgraph prep["前処理 preprocess.py"]
    B["矢状 2D 抽出<br/>512×640"]
    C["4クラスへラベル変換"]
    D["スライスフィルタ<br/>4クラス未満・不均衡除外"]
  end
  subgraph train["学習 train.py"]
    E["Modified U-Net"]
    F["Combined Loss<br/>0.6×Focal + 0.4×Dice"]
    G["Early stopping<br/>val Mean IoU"]
  end
  subgraph eval["評価 evaluate.py"]
    H["Dice / IoU / F1 等"]
    I["validation_metrics.csv"]
  end
  A --> B --> C --> D --> E --> F --> G --> H --> I
```

---

## データの流れ（詳細）

```mermaid
flowchart TD
  Z["Zenodo: SPIDER .mha"] --> DR["--data_root / DataSet"]
  DR --> IMG["images/*.mha"]
  DR --> MSK["masks/*.mha"]

  IMG --> EX["extract_slices()"]
  MSK --> EX
  EX --> PNG["output_root/images/*.png<br/>output_root/masks/*.png"]

  PNG --> FL["filter_slices()"]
  FL --> FF["filtered_files.txt<br/>filtered_slice_stats.csv"]

  FF --> DS["TensorFlow Dataset<br/>train / val split"]
  DS --> UNET["build_modified_unet()"]
  UNET --> CKPT["checkpoints/best_model.keras"]
  CKPT --> EV["evaluate.py"]
  EV --> CSV["validation_metrics.csv"]
```

### フィルタ条件（論文準拠）

```mermaid
flowchart TD
  SL["候補スライス"] --> Q1{"マスクに<br/>4クラスある?"}
  Q1 -->|いいえ| X1["除外"]
  Q1 -->|はい| Q2{"前景最大占有率<br/>≤ 55%?"}
  Q2 -->|いいえ| X2["除外"]
  Q2 -->|はい| Q3{"シーケンス上限<br/>1000枚以内?"}
  Q3 -->|いいえ| X3["除外または間引き"]
  Q3 -->|はい| OK["学習に採用"]
```

---

## モデルと損失

```mermaid
flowchart TB
  IN["入力 512×640×1"] --> E1["Encoder L1–L2<br/>16–32 ch, DO 0.1"]
  E1 --> E2["Encoder L3–L4<br/>64–128 ch, DO 0.2"]
  E2 --> E3["Encoder L5<br/>256 ch, DO 0.3"]
  E3 --> BOT["Bottleneck<br/>512 ch"]
  BOT --> D1["Decoder + skip<br/>Custom Upsample Block"]
  D1 --> OUT["出力 512×640×4<br/>softmax"]

  OUT --> LOSS["Combined Loss"]
  LOSS --> F["0.6 × Focal γ=4"]
  LOSS --> D["0.4 × Dice"]

  subgraph train_cfg["学習設定"]
    A["Leaky ReLU α=0.1"]
    G["Glorot 初期化"]
    O["Adam lr=1e-4, batch=8, ≤100 ep"]
  end
```

---

## リポジトリ構成

```mermaid
flowchart TB
  ROOT["LumbarSeg/"]

  ROOT --> CLI["CLI エントリポイント"]
  CLI --> PRE["preprocess.py"]
  CLI --> TRN["train.py"]
  CLI --> EVA["evaluate.py"]

  ROOT --> ARG["arguments/"]
  ROOT --> PKG["spine_baseline/"]
  PKG --> PP["preprocessing.py"]
  PKG --> DS["dataset.py"]
  PKG --> MD["model.py"]
  PKG --> LS["losses.py"]
  PKG --> MT["metrics.py"]

  ROOT --> DATA["data/ メタデータ"]
  ROOT --> REQ["requirements-baseline.txt"]
  ROOT --> SPLIT["REPOSITORY_SPLIT.md"]

  PRE --> PP
  TRN --> PP
  TRN --> DS
  TRN --> MD
  TRN --> LS
  TRN --> MT
  EVA --> PP
  EVA --> DS
  EVA --> MT
```

---

## クイックスタート

```mermaid
flowchart TD
  A["クローン + pip install"] --> B["SPIDER を data_root に配置"]
  B --> C["preprocess.py"]
  C --> D["train.py 1 epoch スモーク"]
  D --> E["train.py 本学習"]
  E --> F["evaluate.py"]
  F --> G{"Dice ≈ 0.97?"}
  G -->|いいえ| H["ラベル・フィルタ・GPU を確認"]
  G -->|はい| I["Stage 2 改良"]
  H --> C
```

### 1. クローンと依存関係

```bash
git clone https://github.com/Sol-momma/LumbarSeg.git
cd LumbarSeg
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-baseline.txt
```

Nix を使う場合: `nix develop` のあと、上と同様に `.venv` を作成してください。

### 2. データ配置

[Zenodo](https://doi.org/10.5281/zenodo.10159290) から SPIDER を取得し、次の形にします（詳細は [data/README.md](data/README.md)）。

```mermaid
flowchart TB
  DR["--data_root<br/>SPIDER/DataSet/"]
  DR --> IMG["images/*.mha"]
  DR --> MSK["masks/*.mha"]
  DR --> CSV["SPIDER Lumbar Spine<br/>Segmentation Overview.csv"]
```

### 3. 前処理

```bash
python preprocess.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --sequences T2_SPACE
```

### 4. スモークテスト（必須）

本学習の前に **1 epoch** でパイプラインを確認します。

```bash
python train.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --sequences T2_SPACE \
  --epochs 1 \
  --batch_size 2
```

### 5. 本学習

```bash
python train.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --sequences T2_SPACE \
  --batch_size 8 \
  --epochs 100
```

出力例:

```mermaid
flowchart TB
  OUT["--output_root<br/>outputs/t2_space_baseline/"]
  OUT --> PNG["images/ + masks/"]
  OUT --> FLT["filtered_files.txt<br/>filtered_slice_stats.csv"]
  OUT --> CKPT["checkpoints/"]
  CKPT --> BEST["best_model.keras"]
  CKPT --> FINAL["final_model.keras"]
  CKPT --> LOG["training_log.csv"]
```

### 6. 評価

```bash
python evaluate.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --model_path outputs/t2_space_baseline/checkpoints/best_model.keras
```

---

## Google Colab（GPU 推奨）

```mermaid
flowchart TD
  C1["Google Drive マウント"] --> C2["LumbarSeg clone + pip"]
  C2 --> C3{"GPU 表示される?"}
  C3 -->|いいえ| C4["ランタイム → GPU"]
  C4 --> C3
  C3 -->|はい| C5["preprocess.py"]
  C5 --> C6["train.py 1 epoch"]
  C6 --> C7["evaluate.py"]
  C7 --> C8["train.py 100 epoch 本学習"]
```

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%cd /content
!test -d LumbarSeg && (cd LumbarSeg && git pull) || git clone https://github.com/Sol-momma/LumbarSeg.git
%cd /content/LumbarSeg
!pip install -q -r requirements-baseline.txt
```

```python
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))  # GPU が [] のときはランタイムを GPU に変更
```

```bash
!python preprocess.py \
  --data_root /content/drive/MyDrive/SPIDER/DataSet \
  --output_root /content/drive/MyDrive/SPIDER/outputs/t2_space_baseline \
  --sequences T2_SPACE

!python train.py \
  --data_root /content/drive/MyDrive/SPIDER/DataSet \
  --output_root /content/drive/MyDrive/SPIDER/outputs/t2_space_baseline \
  --sequences T2_SPACE \
  --epochs 1 --batch_size 2

!python evaluate.py \
  --data_root /content/drive/MyDrive/SPIDER/DataSet \
  --output_root /content/drive/MyDrive/SPIDER/outputs/t2_space_baseline \
  --model_path /content/drive/MyDrive/SPIDER/outputs/t2_space_baseline/checkpoints/best_model.keras
```

> **注意**: ランタイムが CPU のままだと学習が極端に遅くなります。Colab メニュー「ランタイムのタイプを変更」→ **GPU** を選んでから実行してください。

---

## 進捗（2026年6月時点）

```mermaid
flowchart LR
  P1["ベースライン CLI"]:::done
  P2["Colab スモーク"]:::done
  P3["GPU 本学習"]:::todo
  P4["論文 Dice 比較"]:::todo
  P5["マスク可視化"]:::todo
  P6["T1/T2 + Stage 2"]:::todo
  P1 --> P2 --> P3 --> P4 --> P5 --> P6
  classDef done fill:#d4edda,stroke:#28a745,color:#155724
  classDef todo fill:#fff3cd,stroke:#ffc107,color:#856404
```

---

## 主要 CLI 引数

| 引数 | 既定値 | 説明 |
| --- | ---: | --- |
| `--target_height` / `--target_width` | 512 / 640 | 入力スライスサイズ |
| `--sequences` | なし（全シーケンス） | `T1`, `T2`, `T2_SPACE`（カンマ区切り可） |
| `--imbalance_threshold` | 0.55 | 前景クラス最大比率の上限 |
| `--max_slices_per_sequence` | 1000 | シーケンスごとの保持上限（0 で無制限） |
| `--batch_size` | 8 | バッチサイズ |
| `--epochs` | 100 | 最大エポック数 |
| `--focal_weight` / `--focal_gamma` | 0.6 / 4.0 | Combined Loss |
| `--patience` | 15 | Early stopping |

---

## 参考文献

1. Ahmed, I. et al. *Pioneering Precision in Lumbar Spine MRI Segmentation with Advanced Deep Learning and Data Enhancement.* Machine Learning with Applications, Vol. 20, 2025.  
2. van der Graaf, J.W. et al. *Lumbar spine segmentation in MR images: a dataset and a public benchmark.* Scientific Data, 11:264, 2024.
