# モデルアーキテクチャ: Modified U-Net

## 全体構成

標準的なU-Net (Encoder-Decoder + Skip Connection) をベースに改良。

```
Input (512x640x1)
    │
    ▼
┌─────────────────────────┐
│   Contractive Path      │
│   (Conv + BN + Dropout  │
│    + MaxPool)           │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Bottleneck            │
│   (512ch 追加層)         │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Expansive Path        │
│   (Custom Upsample      │
│    Block + Skip Conn)   │
└─────────────────────────┘
    │
    ▼
Output (512x640x4)  ← 4クラスのsoftmax
```

## 主要な改良点

### 1. Leaky ReLU (alpha=0.1)

**問題**: 通常のReLUでは負の入力に対してゼロを返すため、一部のニューロンが永久に不活性化する ("dying ReLU" 問題)

**解決**: Leaky ReLU を使用し、負の領域でも小さな勾配 (0.1) を維持

```
LeakyReLU(x) = x       if x > 0
               0.1 * x  if x <= 0
```

### 2. Glorot Uniform 初期化 (Xavier)

**目的**: 学習の安定化、勾配消失/爆発の防止

```python
kernel_initializer='glorot_uniform'
```

重みを入力・出力ユニット数に基づいた一様分布で初期化。

### 3. Custom Upsample Block

Expansive Path で使用されるカスタムブロック:

```
Conv2DTranspose
    → Leaky ReLU (alpha=0.1)
    → Glorot Uniform Initializer
    → Concatenate (Skip Connection)
    → Conv + BN + Dropout
```

### 4. 512チャネル追加層

ボトルネック部に512チャネルの層を追加し、複雑な特徴の捕捉能力を強化。

### 5. Batch Normalization + Dropout

- Batch Normalization: 各層の出力を正規化し学習を安定化
- Dropout: 過学習防止 (具体的なrate値は論文に未記載)

## Encoder (Contractive Path) の推定構成

論文のFigure 7に基づく推定 (具体的なチャネル数は論文図から読み取り):

```
Level 1: Conv(64) → Conv(64) → MaxPool
Level 2: Conv(128) → Conv(128) → MaxPool
Level 3: Conv(256) → Conv(256) → MaxPool
Level 4: Conv(512) → Conv(512) → MaxPool
```

## Bottleneck

```
Conv(512) → Conv(512)   ← 追加の512ch層
```

## Decoder (Expansive Path) の推定構成

```
Level 4: Upsample(512) + Skip → Conv(256) → Conv(256)
Level 3: Upsample(256) + Skip → Conv(128) → Conv(128)
Level 2: Upsample(128) + Skip → Conv(64) → Conv(64)
Level 1: Upsample(64) + Skip → Conv(64) → Conv(64)
```

## 出力層

```
Conv2D(4, kernel_size=1, activation='softmax')
```

4クラスのピクセルごとの確率マップを出力。

---

## 損失関数: Combined Loss

### Focal Loss

```
L_focal = -Σ α_i * (1 - y_pred_i)^γ * y_true_i * log(y_pred_i)
```

- **γ (gamma) = 4.0** — 難しいサンプルへの集中度を制御
- クラス不均衡に対応し、分類困難なサンプルに重点を置く

### Dice Loss

```
L_dice = 1 - (2 * Σ y_true_i * y_pred_i + ε) / (Σ y_true_i + Σ y_pred_i + ε)
```

- 予測とGround Truthの重なり度合いを直接最適化

### Combined Loss

```
L_combined = α * L_focal + (1 - α) * L_dice
```

- **α = 0.6** — Focal Lossの重み
- **(1-α) = 0.4** — Dice Lossの重み

**注意**: 論文内にγの値について矛盾あり。
- Section 4 (Methodology): γ = 4.0 と記載
- その直後: α=0.6, γ=0.4 と記載 (γ=0.4はαとの混同の可能性)
- → **γ = 4.0 を採用するのが妥当**

---

## 学習設定

| パラメータ | 値 |
|---|---|
| エポック数 | 100 |
| バッチサイズ | 8 |
| Early Stopping | validation Mean IoU 基準 |
| Model Checkpointing | validation Mean IoU 基準 |
| Optimizer | 未記載 (Adam が一般的) |
| 学習率 | 未記載 |
| 入力サイズ | 512 x 640 (2D PNG) |
| 出力クラス数 | 4 |

## 評価指標

1. **Mean IoU** — 全クラス平均のIntersection over Union
2. **Dice Coefficient** — クラス別のSorensen-Dice指数
3. **ASD** — Average Surface Distance (mm)
4. **NSD** — Normalized Surface Distance (tolerance τ)
5. **Precision** — クラス別の適合率
6. **Recall** — クラス別の再現率
7. **F1 Score** — PrecisionとRecallの調和平均
