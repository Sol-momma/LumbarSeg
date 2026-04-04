# 実装計画

## ディレクトリ構成 (案)

```
SP/
├── docs/                    # ドキュメント
├── data/                    # データ関連
│   ├── raw/                 # SPIDERデータセット (mha)
│   ├── processed/           # 前処理済み2D PNG
│   └── filtered/            # フィルタリング済みデータ
├── src/                     # ソースコード
│   ├── preprocessing/       # 前処理パイプライン
│   │   ├── extract_2d.py    # MHA→2D PNG抽出
│   │   ├── apta.py          # ラベル変換 (16→4クラス)
│   │   └── filter_data.py   # データフィルタリング
│   ├── model/               # モデル定義
│   │   ├── unet.py          # Modified U-Net
│   │   └── loss.py          # Combined Loss (Focal + Dice)
│   ├── training/            # 学習関連
│   │   ├── train.py         # 学習スクリプト
│   │   ├── dataset.py       # PyTorch/TF Dataset
│   │   └── augmentation.py  # データ拡張
│   └── evaluation/          # 評価関連
│       ├── metrics.py       # IoU, Dice, ASD, NSD, F1等
│       └── evaluate.py      # 評価スクリプト
├── notebooks/               # 実験用Jupyter
├── configs/                 # 設定ファイル
│   └── config.yaml          # ハイパーパラメータ
└── outputs/                 # 学習結果・モデル保存
```

## 実装フェーズ

### Phase 1: データ準備

1. **SPIDERデータセットのダウンロード**
   - Zenodo: https://doi.org/10.5281/zenodo.10159290
   - mha形式の3D MRI + マスク

2. **データ構造の確認**
   - mhaファイルのラベル値を調査
   - T1/T2/T2 SPACEの識別方法を確認
   - overview.csvの内容確認

3. **2D抽出 (`extract_2d.py`)**
   - SimpleITK or nibabel で mha読み込み
   - 矢状断スライスの抽出
   - 512x640にリサイズ
   - PNG保存

4. **ラベル変換 (`apta.py`)**
   - SPIDERのラベル値マッピングを調査
   - 16クラス→4クラスのマッピングテーブル作成
   - バッチ変換処理

5. **データフィルタリング (`filter_data.py`)**
   - 4クラス未満のスライス除外
   - クラス不均衡比率 > 55% の除外
   - T1/T2/T2 SPACEそれぞれで1000枚を目標

### Phase 2: モデル構築

6. **Dataset クラス (`dataset.py`)**
   - 画像とマスクのペア読み込み
   - 正規化
   - データ拡張 (optional)

7. **Modified U-Net (`unet.py`)**
   - Encoder: Conv + BN + LeakyReLU + MaxPool
   - Bottleneck: 512ch
   - Decoder: Conv2DTranspose + LeakyReLU + Glorot + Skip
   - Output: 4ch softmax

8. **Combined Loss (`loss.py`)**
   - Focal Loss (γ=4.0)
   - Dice Loss
   - Combined: α=0.6 * Focal + 0.4 * Dice

### Phase 3: 学習

9. **学習スクリプト (`train.py`)**
   - バッチサイズ 8
   - エポック数 100
   - Early Stopping (validation Mean IoU)
   - Model Checkpointing

### Phase 4: 評価

10. **評価指標 (`metrics.py`)**
    - Mean IoU, Dice, ASD, NSD, Precision, Recall, F1
    - クラス別 + 全体平均

11. **評価実行 (`evaluate.py`)**
    - テストデータでの評価
    - 結果の可視化 (予測マスク vs GT)
    - 学習曲線プロット

## 技術選定の選択肢

| 項目 | 選択肢A | 選択肢B | 備考 |
|---|---|---|---|
| フレームワーク | PyTorch | TensorFlow/Keras | 論文はKeras系の記述 (Conv2DTranspose等) |
| 3D画像読み込み | SimpleITK | nibabel | SimpleITKがmhaに標準対応 |
| 画像処理 | OpenCV | PIL/Pillow | OpenCVが高速 |
| 設定管理 | YAML | argparse | 再現性重視ならYAML |

**推奨**: 論文のコード記述がKeras寄りなので TensorFlow/Keras が最も忠実な再現になる。
ただしPyTorchでも問題なく実装可能。

## 未確定事項 (要確認)

- [ ] SPIDERデータのラベル値の実際のマッピング (ダウンロード後に確認)
- [ ] Optimizer の種類と学習率 (論文未記載 → Adam, lr=1e-3〜1e-4 で実験)
- [ ] Dropout rate (論文未記載 → 0.3〜0.5 で実験)
- [ ] データ拡張の有無と手法 (論文未記載)
- [ ] 学習/検証/テストの分割方法 (SPIDERのデフォルト分割を使用するか、独自に分割するか)
- [ ] NSDのtolerance τ の値

## 優先順位

1. まずデータをダウンロードしてラベル構造を確認
2. 前処理パイプラインを構築
3. シンプルなU-Netで学習を回して動作確認
4. 論文の改良点を段階的に追加
5. 評価指標を実装して結果を比較
