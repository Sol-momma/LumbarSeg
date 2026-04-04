# プロジェクト概要: 腰椎MRIセグメンテーション

## 論文情報

- **タイトル**: Pioneering Precision in Lumbar Spine MRI Segmentation with Advanced Deep Learning and Data Enhancement
- **著者**: Istiak Ahmed et al. (North South University / Friedrich-Alexander University)
- **掲載**: Machine Learning with Applications, Vol.20, 2025年6月
- **DOI**: 10.1016/j.mlwa.2025.100635

## 目的

腰椎MRI画像から以下の3構造 + 背景 = 4クラスを高精度に自動セグメンテーションする深層学習モデルを構築する。

| クラスID | 構造 | 英語名 |
|---|---|---|
| 0 | 背景 | Background |
| 1 | 椎体 | Vertebrae |
| 2 | 脊柱管 | Spinal Canal |
| 3 | 椎間板 | Intervertebral Discs (IVDs) |

## アプローチの3本柱

1. **データ前処理の徹底** — 16クラス→4クラス変換、ラベル修正、不均衡フィルタリング
2. **モデル改良** — U-Netベース + Leaky ReLU + Glorot初期化 + 512ch追加層
3. **損失関数の工夫** — Focal Loss + Dice Loss の組み合わせ

## データセット

SPIDERデータセット (Nature Scientific Data 2024) を使用。

| 項目 | 値 |
|---|---|
| 患者数 | 218名 |
| MRIシリーズ数 | 447 (T1: 196, T2: 210, T2 SPACE: 41) |
| 収集施設 | 4病院 (オランダ) |
| フォーマット | MHA (3D) |
| アノテーション | 椎体・椎間板・脊柱管の3D セグメンテーションマスク |
| ライセンス | CC-BY 4.0 |
| 入手先 | https://doi.org/10.5281/zenodo.10159290 |

### データ分割 (Dataset Paper)

| セット | Studies | Series |
|---|---|---|
| Training | 179 | 360 |
| Validation | 39 | 87 |
| Test (非公開) | 39 | 97 |

## 結果サマリ (T2 SPACE, クラス別)

| クラス | IoU | Dice | ASD | NSD | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Background | 0.9906 | 0.9950 | 0.0543 | 0.9951 | 0.9951 | 0.9955 | 0.9953 |
| IVDs | 0.9476 | 0.9688 | 0.0288 | 0.9994 | 0.9684 | 0.9778 | 0.9731 |
| Vertebrae | 0.9461 | 0.9712 | 0.0464 | 0.9944 | 0.9745 | 0.9701 | 0.9723 |
| Spinal Canal | 0.9501 | 0.9671 | 0.0361 | 0.9963 | 0.9761 | 0.9727 | 0.9744 |

## 比較 (nn-UNET vs Proposed)

| 構造 | Metric | Proposed | nn-UNET |
|---|---|---|---|
| IVDs | Mean Dice | 0.9688 | 0.86 |
| Vertebrae | Mean Dice | 0.9712 | 0.92 |
| Spinal Canal | Mean Dice | 0.9671 | 0.92 |
