# LumbarSeg

**Languages / 语言 / 言語:** [English](README.md) · **中文** · [日本語](README.ja.md)

用于复现 Ahmed et al. (2025) 基线方法（Modified U-Net + Combined Loss）的**腰椎 MRI 四类分割**毕业研究仓库。

| 链接 | |
| --- | --- |
| 目标论文 | [Ahmed et al., 2025](https://www.sciencedirect.com/science/article/pii/S2666827025000180) |
| 数据集 | [SPIDER (Zenodo)](https://doi.org/10.5281/zenodo.10159290) |
| 实验页面（可选） | [GitHub Pages](https://sol-momma.github.io/LumbarSeg/) |

---

## 研究目标

1. **阶段 1**：用与论文相同的前处理、模型和损失，在 T2 SPACE 上复现 Dice ≈ 0.97  
2. **阶段 2**：复现成功后，通过架构、增强和损失改进超越论文精度

### 分割类别（4 类）

| ID | 结构 | SPIDER 原始标签 |
| ---: | --- | --- |
| 0 | 背景 (Background) | `0` |
| 1 | 椎体 (Vertebrae) | `1–99` |
| 2 | 椎管 (Spinal Canal) | `100` |
| 3 | 椎间盘 (IVDs) | `200+` |

### 论文目标指标（T2 SPACE，参考）

| 结构 | Dice | IoU |
| --- | ---: | ---: |
| IVDs | 0.9688 | 0.9476 |
| Vertebrae | 0.9712 | 0.9461 |
| Spinal Canal | 0.9671 | 0.9501 |

---

## 整体流程（研究流水线）

```mermaid
flowchart LR
  subgraph input["输入"]
    A["SPIDER 3D MHA<br/>images + masks"]
  end
  subgraph prep["前处理 preprocess.py"]
    B["矢状 2D 提取<br/>512×640"]
    C["映射为 4 类标签"]
    D["切片过滤<br/>剔除类别不足/不均衡"]
  end
  subgraph train["训练 train.py"]
    E["Modified U-Net"]
    F["Combined Loss<br/>0.6×Focal + 0.4×Dice"]
    G["Early stopping<br/>val Mean IoU"]
  end
  subgraph eval["评估 evaluate.py"]
    H["Dice / IoU / F1 等"]
    I["validation_metrics.csv"]
  end
  A --> B --> C --> D --> E --> F --> G --> H --> I
```

---

## 数据流（详细）

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

  FF --> DS["TensorFlow Dataset<br/>train / val 划分"]
  DS --> UNET["build_modified_unet()"]
  UNET --> CKPT["checkpoints/best_model.keras"]
  CKPT --> EV["evaluate.py"]
  EV --> CSV["validation_metrics.csv"]
```

### 过滤条件（与论文一致）

- 剔除掩膜中**少于 4 类**的切片  
- 剔除前景类最大占比 **超过 55%** 的切片（`--imbalance_threshold 0.55`）  
- 每个序列最多保留 **1000 张**（`--max_slices_per_sequence`，设为 `0` 表示不限制）

---

## 模型与损失

```mermaid
flowchart TB
  IN["输入 512×640×1"] --> ENC["Encoder<br/>16→32→64→128→256 ch"]
  ENC --> BOT["Bottleneck 512 ch"]
  BOT --> DEC["Decoder + Skip<br/>Custom Upsample Block"]
  DEC --> OUT["输出 512×640×4 softmax"]

  subgraph loss["Combined Loss"]
    L1["Focal Loss γ=4.0 × 0.6"]
    L2["Dice Loss × 0.4"]
  end
  OUT -.-> loss
```

- 激活函数：Leaky ReLU（α=0.1）  
- 初始化：Glorot uniform  
- Dropout：论文 schedule（0.1 / 0.2 / 0.3）  
- 优化：Adam，`lr=1e-4`，batch_size=8，最多 100 epoch

---

## 仓库结构

```text
LumbarSeg/
├── preprocess.py          # 仅前处理
├── train.py               # 前处理 + 训练
├── evaluate.py            # 验证集评估
├── arguments/             # CLI 参数（数据 / 模型 / 优化）
├── spine_baseline/        # 前处理、数据集、模型、损失、指标
├── data/                  # SPIDER 元数据（不含 MRI 体数据）
├── requirements-baseline.txt
├── flake.nix              # 可选：固定开发环境版本
└── src/                   # 可选：Astro 实验展示页
```

| 模块 | 作用 |
| --- | --- |
| `spine_baseline/preprocessing.py` | 读取 MHA、矢状提取、缩放、标签映射、过滤 |
| `spine_baseline/model.py` | Modified U-Net |
| `spine_baseline/losses.py` | Combined Loss |
| `spine_baseline/metrics.py` | Dice、Mean IoU 等 |
| `spine_baseline/dataset.py` | `tf.data` 流水线 |

---

## 快速开始

### 1. 克隆与安装

```bash
git clone https://github.com/Sol-momma/LumbarSeg.git
cd LumbarSeg
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-baseline.txt
```

使用 Nix：先执行 `nix develop`，再按上式创建 `.venv`。

### 2. 数据目录

从 [Zenodo](https://doi.org/10.5281/zenodo.10159290) 获取 SPIDER，并按如下组织（详见 [data/README.md](data/README.md)）：

```text
/path/to/SPIDER/DataSet/
├── images/
├── masks/
└── SPIDER Lumbar Spine Segmentation Overview.csv
```

### 3. 前处理

```bash
python preprocess.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --sequences T2_SPACE
```

### 4. 冒烟测试（必做）

正式训练前用 **1 个 epoch** 确认流水线正常：

```bash
python train.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --sequences T2_SPACE \
  --epochs 1 \
  --batch_size 2
```

### 5. 正式训练

```bash
python train.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --sequences T2_SPACE \
  --batch_size 8 \
  --epochs 100
```

输出示例：

```text
outputs/t2_space_baseline/
├── images/ masks/
├── filtered_files.txt
├── filtered_slice_stats.csv
└── checkpoints/
    ├── best_model.keras
    ├── final_model.keras
    └── training_log.csv
```

### 6. 评估

```bash
python evaluate.py \
  --data_root /path/to/SPIDER/DataSet \
  --output_root outputs/t2_space_baseline \
  --model_path outputs/t2_space_baseline/checkpoints/best_model.keras
```

---

## Google Colab（建议使用 GPU）

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
print(tf.config.list_physical_devices("GPU"))  # 若为 []，请将运行时切换为 GPU
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

> **注意：** 若运行时仍为 CPU，训练会非常慢。请在 Colab 中选择 **运行时 → 更改运行时类型 → GPU** 后再执行。

---

## 进度（截至 2026 年 6 月）

- [x] 基线 CLI（前处理、训练、评估）
- [x] Colab 冒烟测试：前处理 → 1 epoch 训练 → 评估
- [ ] 在 Colab **GPU** 上完成正式训练（100 epoch）
- [ ] 与论文 Dice 的定量对比
- [ ] 预测 mask 与 MRI 叠加可视化
- [ ] 扩展至 T1 / T2 及阶段 2 改进

---

## 主要 CLI 参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--target_height` / `--target_width` | 512 / 640 | 输入切片尺寸 |
| `--sequences` | 全部序列 | `T1`、`T2`、`T2_SPACE`（逗号分隔） |
| `--imbalance_threshold` | 0.55 | 前景最大类占比上限 |
| `--max_slices_per_sequence` | 1000 | 每序列保留上限（`0` 为不限制） |
| `--batch_size` | 8 | 批大小 |
| `--epochs` | 100 | 最大 epoch 数 |
| `--focal_weight` / `--focal_gamma` | 0.6 / 4.0 | Combined Loss |
| `--patience` | 15 | Early stopping 耐心值 |

---

## 参考文献

1. Ahmed, I. et al. *Pioneering Precision in Lumbar Spine MRI Segmentation with Advanced Deep Learning and Data Enhancement.* Machine Learning with Applications, Vol. 20, 2025.  
2. van der Graaf, J.W. et al. *Lumbar spine segmentation in MR images: a dataset and a public benchmark.* Scientific Data, 11:264, 2024.
