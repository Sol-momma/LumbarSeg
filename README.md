# LumbarSeg

**Languages / 语言 / 言語:** **English** · [简体](README.zh-CN.md) · [繁體](README.zh-TW.md) · [日本語](README.ja.md)

Graduation research repository for reproducing **four-class lumbar MRI segmentation** using the Ahmed et al. (2025) baseline (Modified U-Net + Combined Loss).

| Link | |
| --- | --- |
| Target paper | [Ahmed et al., 2025](https://www.sciencedirect.com/science/article/pii/S2666827025000180) |
| Dataset | [SPIDER (Zenodo)](https://doi.org/10.5281/zenodo.10159290) |
| Experiment page (optional) | [GitHub Pages](https://sol-momma.github.io/LumbarSeg/) |

---

## Research Goals

```mermaid
flowchart LR
  S1["Stage 1<br/>Reproduce paper baseline"] --> S2["Stage 2<br/>Improve beyond paper"]
  S1 --> M1["Dice ≈ 0.97<br/>T2 SPACE"]
  S2 --> M2["Architecture / augmentation / loss"]
```

### Segmentation Classes (4 classes)

```mermaid
flowchart LR
  subgraph spider["SPIDER raw labels"]
    R0["0"]
    R1["1–99"]
    R2["100"]
    R3["200+"]
  end
  subgraph four["4 classes"]
    C0["0 Background"]
    C1["1 Vertebrae"]
    C2["2 Spinal Canal"]
    C3["3 IVDs"]
  end
  R0 --> C0
  R1 --> C1
  R2 --> C2
  R3 --> C3
```

| ID | Structure | SPIDER source labels |
| ---: | --- | --- |
| 0 | Background | `0` |
| 1 | Vertebrae | `1–99` |
| 2 | Spinal Canal | `100` |
| 3 | IVDs (intervertebral discs) | `200+` |

### Target Metrics from the Paper (T2 SPACE, reference)

| Structure | Dice | IoU |
| --- | ---: | ---: |
| IVDs | 0.9688 | 0.9476 |
| Vertebrae | 0.9712 | 0.9461 |
| Spinal Canal | 0.9671 | 0.9501 |

---

## Overall Pipeline

```mermaid
flowchart LR
  subgraph input["Input"]
    A["SPIDER 3D MHA<br/>images + masks"]
  end
  subgraph prep["Preprocess preprocess.py"]
    B["Sagittal 2D extract<br/>512×640"]
    C["Map labels to 4 classes"]
    D["Slice filter<br/>drop incomplete / imbalanced"]
  end
  subgraph train["Train train.py"]
    E["Modified U-Net"]
    F["Combined Loss<br/>0.6×Focal + 0.4×Dice"]
    G["Early stopping<br/>val Mean IoU"]
  end
  subgraph eval["Evaluate evaluate.py"]
    H["Dice / IoU / F1, etc."]
    I["validation_metrics.csv"]
  end
  A --> B --> C --> D --> E --> F --> G --> H --> I
```

---

## Data Flow (Detail)

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

### Filtering Rules (paper-aligned)

```mermaid
flowchart TD
  SL["Candidate slice"] --> Q1{"Mask has<br/>4 classes?"}
  Q1 -->|No| X1["Discard"]
  Q1 -->|Yes| Q2{"Max foreground<br/>share ≤ 55%?"}
  Q2 -->|No| X2["Discard"]
  Q2 -->|Yes| Q3{"Under per-sequence<br/>cap 1000?"}
  Q3 -->|No| X3["Discard or subsample"]
  Q3 -->|Yes| OK["Keep for training"]
```

---

## Model and Loss

```mermaid
flowchart TB
  IN["Input 512×640×1"] --> E1["Encoder L1–L2<br/>16–32 ch, DO 0.1"]
  E1 --> E2["Encoder L3–L4<br/>64–128 ch, DO 0.2"]
  E2 --> E3["Encoder L5<br/>256 ch, DO 0.3"]
  E3 --> BOT["Bottleneck<br/>512 ch"]
  BOT --> D1["Decoder + skip<br/>Custom Upsample Block"]
  D1 --> OUT["Output 512×640×4<br/>softmax"]

  OUT --> LOSS["Combined Loss"]
  LOSS --> F["0.6 × Focal γ=4"]
  LOSS --> D["0.4 × Dice"]

  subgraph train_cfg["Training config"]
    A["Leaky ReLU α=0.1"]
    G["Glorot init"]
    O["Adam lr=1e-4, batch=8, ≤100 ep"]
  end
```

---

## Repository Layout

```mermaid
flowchart TB
  ROOT["LumbarSeg/"]

  ROOT --> CLI["CLI entrypoints"]
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

  ROOT --> DATA["data/ metadata"]
  ROOT --> REQ["requirements-baseline.txt"]
  ROOT --> WEB["src/ Astro site optional"]

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

## Quick Start

```mermaid
flowchart TD
  A["Clone + pip install"] --> B["Place SPIDER under data_root"]
  B --> C["preprocess.py"]
  C --> D["train.py 1 epoch smoke test"]
  D --> E["train.py full training"]
  E --> F["evaluate.py"]
  F --> G{"Dice ≈ 0.97?"}
  G -->|No| H["Debug labels / filter / GPU"]
  G -->|Yes| I["Stage 2 improvements"]
  H --> C
```

### 1. Clone and install

```bash
git clone https://github.com/Sol-momma/LumbarSeg.git
cd LumbarSeg
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-baseline.txt
```

With Nix: run `nix develop`, then create `.venv` as above.

### 2. Dataset layout

Download SPIDER from [Zenodo](https://doi.org/10.5281/zenodo.10159290) and arrange (see [data/README.md](data/README.md)):

```mermaid
flowchart TB
  DR["--data_root<br/>SPIDER/DataSet/"]
  DR --> IMG["images/*.mha"]
  DR --> MSK["masks/*.mha"]
  DR --> CSV["SPIDER Lumbar Spine<br/>Segmentation Overview.csv"]
```

### 3. Preprocess

```bash
python preprocess.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --sequences T2_SPACE
```

### 4. Smoke test (required)

Run **1 epoch** before full training to verify the pipeline:

```bash
python train.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --sequences T2_SPACE \
  --epochs 1 \
  --batch_size 2
```

### 5. Full training

```bash
python train.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --sequences T2_SPACE \
  --batch_size 8 \
  --epochs 100
```

Example outputs:

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

### 6. Evaluate

```bash
python evaluate.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --model_path outputs/t2_space_baseline/checkpoints/best_model.keras
```

---

## Google Colab (GPU recommended)

```mermaid
flowchart TD
  C1["Mount Google Drive"] --> C2["Clone LumbarSeg + pip install"]
  C2 --> C3{"GPU visible?"}
  C3 -->|No| C4["Runtime → GPU"]
  C4 --> C3
  C3 -->|Yes| C5["preprocess.py"]
  C5 --> C6["train.py smoke 1 ep"]
  C6 --> C7["evaluate.py"]
  C7 --> C8["train.py 100 ep full run"]
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
print(tf.config.list_physical_devices("GPU"))  # if [], switch runtime to GPU
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

> **Note:** Training on CPU is extremely slow. In Colab, use **Runtime → Change runtime type → GPU** before training.

---

## Progress (as of June 2026)

```mermaid
flowchart LR
  P1["Baseline CLI"]:::done
  P2["Colab smoke test"]:::done
  P3["GPU full training"]:::todo
  P4["Compare with paper Dice"]:::todo
  P5["Mask overlay viz"]:::todo
  P6["T1/T2 + Stage 2"]:::todo
  P1 --> P2 --> P3 --> P4 --> P5 --> P6
  classDef done fill:#d4edda,stroke:#28a745,color:#155724
  classDef todo fill:#fff3cd,stroke:#ffc107,color:#856404
```

---

## Key CLI Arguments

| Argument | Default | Description |
| --- | ---: | --- |
| `--target_height` / `--target_width` | 512 / 640 | Input slice size |
| `--sequences` | all | `T1`, `T2`, `T2_SPACE` (comma-separated) |
| `--imbalance_threshold` | 0.55 | Max dominant foreground fraction |
| `--max_slices_per_sequence` | 1000 | Cap per sequence (`0` = unlimited) |
| `--batch_size` | 8 | Batch size |
| `--epochs` | 100 | Max epochs |
| `--focal_weight` / `--focal_gamma` | 0.6 / 4.0 | Combined Loss |
| `--patience` | 15 | Early stopping patience |

---

## References

1. Ahmed, I. et al. *Pioneering Precision in Lumbar Spine MRI Segmentation with Advanced Deep Learning and Data Enhancement.* Machine Learning with Applications, Vol. 20, 2025.  
2. van der Graaf, J.W. et al. *Lumbar spine segmentation in MR images: a dataset and a public benchmark.* Scientific Data, 11:264, 2024.
