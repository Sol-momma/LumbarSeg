# LumbarSeg

**Languages / 语言 / 言語:** [English](README.md) · [简体](README.zh-CN.md) · **繁體** · [日本語](README.ja.md)

用於復現 Ahmed et al. (2025) 基線方法（Modified U-Net + Combined Loss）的**腰椎 MRI 四類分割**畢業研究倉庫。

| 連結 | |
| --- | --- |
| 目標論文 | [Ahmed et al., 2025](https://www.sciencedirect.com/science/article/pii/S2666827025000180) |
| 資料集 | [SPIDER (Zenodo)](https://doi.org/10.5281/zenodo.10159290) |
| 實驗頁面（可選） | [GitHub Pages](https://sol-momma.github.io/LumbarSeg/) |

---

## 研究目標

```mermaid
flowchart LR
  S1["階段 1<br/>復現論文基線"] --> S2["階段 2<br/>超越論文精度"]
  S1 --> M1["Dice ≈ 0.97<br/>T2 SPACE"]
  S2 --> M2["架構 / 增強 / 損失"]
```

### 分割類別（4 類）

```mermaid
flowchart LR
  subgraph spider["SPIDER 原始標籤"]
    R0["0"]
    R1["1–99"]
    R2["100"]
    R3["200+"]
  end
  subgraph four["4 類"]
    C0["0 背景"]
    C1["1 椎體"]
    C2["2 椎管"]
    C3["3 椎間盤"]
  end
  R0 --> C0
  R1 --> C1
  R2 --> C2
  R3 --> C3
```

| ID | 結構 | SPIDER 原始標籤 |
| ---: | --- | --- |
| 0 | 背景 (Background) | `0` |
| 1 | 椎體 (Vertebrae) | `1–99` |
| 2 | 椎管 (Spinal Canal) | `100` |
| 3 | 椎間盤 (IVDs) | `200+` |

### 論文目標指標（T2 SPACE，參考）

| 結構 | Dice | IoU |
| --- | ---: | ---: |
| IVDs | 0.9688 | 0.9476 |
| Vertebrae | 0.9712 | 0.9461 |
| Spinal Canal | 0.9671 | 0.9501 |

---

## 整體流程（研究流水線）

```mermaid
flowchart LR
  subgraph input["輸入"]
    A["SPIDER 3D MHA<br/>images + masks"]
  end
  subgraph prep["前處理 preprocess.py"]
    B["矢狀 2D 擷取<br/>512×640"]
    C["對應為 4 類標籤"]
    D["切片過濾<br/>剔除類別不足/不均衡"]
  end
  subgraph train["訓練 train.py"]
    E["Modified U-Net"]
    F["Combined Loss<br/>0.6×Focal + 0.4×Dice"]
    G["Early stopping<br/>val Mean IoU"]
  end
  subgraph eval["評估 evaluate.py"]
    H["Dice / IoU / F1 等"]
    I["validation_metrics.csv"]
  end
  A --> B --> C --> D --> E --> F --> G --> H --> I
```

---

## 資料流（詳細）

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

  FF --> DS["TensorFlow Dataset<br/>train / val 劃分"]
  DS --> UNET["build_modified_unet()"]
  UNET --> CKPT["checkpoints/best_model.keras"]
  CKPT --> EV["evaluate.py"]
  EV --> CSV["validation_metrics.csv"]
```

### 過濾條件（與論文一致）

```mermaid
flowchart TD
  SL["候選切片"] --> Q1{"遮罩含<br/>4 個類別?"}
  Q1 -->|否| X1["捨棄"]
  Q1 -->|是| Q2{"前景最大佔比<br/>≤ 55%?"}
  Q2 -->|否| X2["捨棄"]
  Q2 -->|是| Q3{"低於序列上限<br/>1000 張?"}
  Q3 -->|否| X3["捨棄或抽樣"]
  Q3 -->|是| OK["保留用於訓練"]
```

---

## 模型與損失

```mermaid
flowchart TB
  IN["輸入 512×640×1"] --> E1["Encoder L1–L2<br/>16–32 ch, DO 0.1"]
  E1 --> E2["Encoder L3–L4<br/>64–128 ch, DO 0.2"]
  E2 --> E3["Encoder L5<br/>256 ch, DO 0.3"]
  E3 --> BOT["Bottleneck<br/>512 ch"]
  BOT --> D1["Decoder + skip<br/>Custom Upsample Block"]
  D1 --> OUT["輸出 512×640×4<br/>softmax"]

  OUT --> LOSS["Combined Loss"]
  LOSS --> F["0.6 × Focal γ=4"]
  LOSS --> D["0.4 × Dice"]

  subgraph train_cfg["訓練設定"]
    A["Leaky ReLU α=0.1"]
    G["Glorot 初始化"]
    O["Adam lr=1e-4, batch=8, ≤100 ep"]
  end
```

---

## 倉庫結構

```mermaid
flowchart TB
  ROOT["LumbarSeg/"]

  ROOT --> CLI["CLI 入口"]
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

  ROOT --> DATA["data/ 中繼資料"]
  ROOT --> REQ["requirements-baseline.txt"]
  ROOT --> WEB["src/ Astro 可選"]

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

## 快速開始

```mermaid
flowchart TD
  A["克隆 + pip install"] --> B["將 SPIDER 放入 data_root"]
  B --> C["preprocess.py"]
  C --> D["train.py 1 epoch 冒煙測試"]
  D --> E["train.py 正式訓練"]
  E --> F["evaluate.py"]
  F --> G{"Dice ≈ 0.97?"}
  G -->|否| H["檢查標籤 / 過濾 / GPU"]
  G -->|是| I["階段 2 改進"]
  H --> C
```

### 1. 克隆與安裝

```bash
git clone https://github.com/Sol-momma/LumbarSeg.git
cd LumbarSeg
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-baseline.txt
```

使用 Nix：先執行 `nix develop`，再依上式建立 `.venv`。

### 2. 資料目錄

從 [Zenodo](https://doi.org/10.5281/zenodo.10159290) 取得 SPIDER，並依下列結構組織（詳見 [data/README.md](data/README.md)）：

```mermaid
flowchart TB
  DR["--data_root<br/>SPIDER/DataSet/"]
  DR --> IMG["images/*.mha"]
  DR --> MSK["masks/*.mha"]
  DR --> CSV["SPIDER Lumbar Spine<br/>Segmentation Overview.csv"]
```

### 3. 前處理

```bash
python preprocess.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --sequences T2_SPACE
```

### 4. 冒煙測試（必做）

正式訓練前請以 **1 個 epoch** 確認流水線正常：

```bash
python train.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --sequences T2_SPACE \
  --epochs 1 \
  --batch_size 2
```

### 5. 正式訓練

```bash
python train.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --sequences T2_SPACE \
  --batch_size 8 \
  --epochs 100
```

輸出範例：

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

### 6. 評估

```bash
python evaluate.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --model_path outputs/t2_space_baseline/checkpoints/best_model.keras
```

---

## Google Colab（建議使用 GPU）

```mermaid
flowchart TD
  C1["掛載 Google Drive"] --> C2["克隆 LumbarSeg + pip"]
  C2 --> C3{"可見 GPU?"}
  C3 -->|否| C4["執行階段 → GPU"]
  C4 --> C3
  C3 -->|是| C5["preprocess.py"]
  C5 --> C6["train.py 1 epoch"]
  C6 --> C7["evaluate.py"]
  C7 --> C8["train.py 100 epoch"]
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
print(tf.config.list_physical_devices("GPU"))  # 若為 []，請將執行階段切換為 GPU
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

> **注意：** 若執行階段仍為 CPU，訓練會非常慢。請在 Colab 中選擇 **執行階段 → 變更執行階段類型 → GPU** 後再執行。

---

## 進度（截至 2026 年 6 月）

```mermaid
flowchart LR
  P1["基線 CLI"]:::done
  P2["Colab 冒煙測試"]:::done
  P3["GPU 正式訓練"]:::todo
  P4["與論文 Dice 對比"]:::todo
  P5["遮罩疊加視覺化"]:::todo
  P6["T1/T2 + 階段 2"]:::todo
  P1 --> P2 --> P3 --> P4 --> P5 --> P6
  classDef done fill:#d4edda,stroke:#28a745,color:#155724
  classDef todo fill:#fff3cd,stroke:#ffc107,color:#856404
```

---

## 主要 CLI 參數

| 參數 | 預設值 | 說明 |
| --- | ---: | --- |
| `--target_height` / `--target_width` | 512 / 640 | 輸入切片尺寸 |
| `--sequences` | 全部序列 | `T1`、`T2`、`T2_SPACE`（逗號分隔） |
| `--imbalance_threshold` | 0.55 | 前景最大類佔比上限 |
| `--max_slices_per_sequence` | 1000 | 每序列保留上限（`0` 為不限制） |
| `--batch_size` | 8 | 批次大小 |
| `--epochs` | 100 | 最大 epoch 數 |
| `--focal_weight` / `--focal_gamma` | 0.6 / 4.0 | Combined Loss |
| `--patience` | 15 | Early stopping 耐心值 |

---

## 參考文獻

1. Ahmed, I. et al. *Pioneering Precision in Lumbar Spine MRI Segmentation with Advanced Deep Learning and Data Enhancement.* Machine Learning with Applications, Vol. 20, 2025.  
2. van der Graaf, J.W. et al. *Lumbar spine segmentation in MR images: a dataset and a public benchmark.* Scientific Data, 11:264, 2024.
