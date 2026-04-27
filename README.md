# Lumbar Spine MRI Segmentation

腰椎MRIセグメンテーションの深層学習モデル再現・改良プロジェクト。

## 概要

[Ahmed et al. (2025)](https://www.sciencedirect.com/science/article/pii/S2666827025000180) の論文を再現し、さらに精度を改良することを目指す卒業プロジェクト。

**タスク**: MRI画像から以下の4クラスをピクセル単位で自動セグメンテーション

| クラス | 構造 | 臨床的意義 |
|---|---|---|
| 0 | 背景 (Background) | — |
| 1 | 椎体 (Vertebrae) | 骨折・変形の評価 |
| 2 | 脊柱管 (Spinal Canal) | 狭窄症の診断 |
| 3 | 椎間板 (IVDs) | ヘルニア・変性の評価 |

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

| 構造 | 論文 Dice | 比較 (nn-UNET) |
|---|---|---|
| IVDs | 0.9688 | 0.86 |
| Vertebrae | 0.9712 | 0.92 |
| Spinal Canal | 0.9671 | 0.92 |

### Stage 2: 精度の改良
アルゴリズムの改良により Dice係数をさらに向上させる。

## ドキュメント

| ドキュメント | 内容 |
|---|---|
| [論文概要・結果サマリ](docs/overview.md) | 論文の要点、データセット、結果の数値まとめ |
| [プロジェクト目標・評価指標](docs/project_goals.md) | Stage 1/2 の目標、T1/T2/T2 SPACE解説、Dice等の指標説明 |
| [教授向け発表アウトライン](docs/presentation_outline.md) | 15-20分の発表構成、各スライドのポイント |
| [データ前処理パイプライン](docs/preprocessing.md) | 3D→2D変換、ラベルマッピング、フィルタリング手順 |
| [モデル構成・損失関数](docs/architecture.md) | Modified U-Net詳細、Focal+Dice Loss数式、学習設定 |
| [SPIDERデータセット詳細](docs/dataset_spider.md) | 病院別データ、分割、アノテーション方法、ベースライン結果 |
| [実装計画](docs/implementation_plan.md) | フェーズ分け、技術選定、未確定事項 |
| [未解決事項](docs/open_questions.md) | 設計判断が必要なポイント、論文の矛盾点 |
| [よく��る質問 (FAQ)](docs/faq.md) | 発表で想定されるQ&A (データ・モデル・結果・卒プロ) |

## ディレクトリ構成

```
SP/
├── README.md
├── CLAUDE.md
├── spine_segmentation.ipynb       # メインノートブック (Colab実行)
├── docs/                          # ドキュメント (上記リンク参照)
├── papers/                        # 論文PDF
│   ├── paper.pdf                  #   本論文 (Ahmed et al., 2025)
│   └── Dataset_Paper.pdf          #   データセット論文 (van der Graaf et al., 2024)
├── data/                          # メタデータ (CSVなど)
│   ├── *.csv                      #   SPIDER Overview (train/val分割情報)
│   └── *.json                     #   SPIDER Dataset メタデータ
└── outputs/                       # 学習結果・モデル保存 (今後作成)
```

**注意**: MRI画像データ本体はGoogle Drive上 (`/content/drive/MyDrive/SPIDER/DataSet/`) に配置。

## 実行環境

- **コード実行**: Google Colab (GPU)
- **データ保存**: Google Drive
- **コード編集**: ローカル

## 起動方法

このリポジトリには 2 つの実行対象があります。

- `spine_segmentation.ipynb`
  - 論文再現用のメインノートブック
  - **Google Colab で実行**
- Astro サイト
  - 論文メモと研究メモの表示用サイト
  - **ローカルで実行**

### Astro サイトをローカルで起動

前提:

- Node.js `>=22.12.0`
- `npm install` 済み

開発サーバー:

```bash
npm run dev
```

- 通常は `http://127.0.0.1:4321/`
- 研究メモページは `http://127.0.0.1:4321/notes/`
- 英語ページは `http://127.0.0.1:4321/en/`

本番ビルド:

```bash
npm run build
```

- 出力先は `dist/`

ビルド結果の確認:

```bash
npm run preview
```

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

1. **本論文**: Ahmed, I. et al. "Pioneering Precision in Lumbar Spine MRI Segmentation with Advanced Deep Learning and Data Enhancement." *Machine Learning with Applications*, Vol.20, 2025.

2. **データセット論文**: van der Graaf, J.W. et al. "Lumbar spine segmentation in MR images: a dataset and a public benchmark." *Nature Scientific Data*, 11:264, 2024.
