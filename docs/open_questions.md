# 未解決事項・判断が必要なポイント

## 要確認 (データダウンロード後)

### Q1: SPIDERマスクの実際のラベル値 [RESOLVED]

確認済みのラベル値:

- 0 = Background
- 1–8 = 個別椎体 (下から上に番号付け)
- 100 = Spinal Canal
- 201–208 = 個別IVD (下から上に番号付け)

4クラスマッピング: 0→0, 1–99→1 (Vertebrae), 100→2 (Spinal Canal), 200+→3 (IVDs)

### Q2: 画像の向き・軸の問題

- 論文Issue 3: 一部のスライスが矢状断でない → 手動で特定が必要
- 論文Issue 4: 一部が回転/反転 → どのファイルが該当するか
- **SimpleITKのメタデータ (Direction, Origin等) で判定可能か調査**

### Q3: Train/Val/Test の分割

- SPIDERのデフォルト分割 (Train:179, Val:39) をそのまま使うか？
- 論文では3000枚で学習しているが、Val/Testの分割比率は未記載
- **SPIDERのoverview.csvを確認する**

## 設計判断

### D1: フレームワーク選定

- 論文のコードはKeras風の記述 (Conv2DTranspose, LeakyReLU等)
- **TensorFlow/Keras で実装するのが論文に最も忠実**
- PyTorchでも同等に実装可能

### D2: 2D vs 3D 学習

- 論文は明確に **2D** アプローチ (512x640の2D PNGに変換して学習)
- Dataset Paperのベースラインは3Dアプローチ
- **論文に従い2Dで実装**

### D3: T1/T2/T2 SPACEの扱い

- 論文ではそれぞれ別にモデルを学習している？
- Table 2ではT2 SPACE, T1, T2ごとに別々の結果を報告
- **おそらく3つの別モデル、または全データ混合の1モデル → 要検討**

### D4: Optimizer と学習率

- 論文に記載なし
- 一般的なU-Net医用画像セグメンテーションでは:
  - Adam, lr=1e-4 が多い
  - SGD with momentum も使われる
- **Adam, lr=1e-4 をデフォルトとし、必要に応じて調整**

### D5: Dropout Rate [RESOLVED]

- 論文のFigure 7周辺の層説明に `0.1`, `0.2`, `0.3` のDropout rateが記載されている
- **実装ではこのスケジュールを既定値にし、`--dropout_rate` は比較実験用の上書きとして扱う**

### D6: データ拡張

- 本論文では明示的な記述なし
- Dataset Paperでは以下を使用:
  - Random elastic deformation
  - Random Gaussian noise
  - Random Gaussian smoothing
  - Random cropping (longitudinal axis)
- **基本的な拡張 (回転、反転、弾性変形) を実装し、ablation studyで効果を確認**

## 論文の矛盾・曖昧な点

### C1: γ の値

- Section 4: "gamma (γ = 4.0) value of 4.0"
- 直後: "Alpha (α) and Gamma (γ) Values of the combined loss function were set to 0.6 and 0.4"
- → 0.4 は (1-α) の値と混同している可能性が高い
- **γ = 4.0 を採用**

### C2: クラス不均衡比率の定義 [RESOLVED AS IMPLEMENTATION CHOICE]

- "Class Imbalance Ratio = Highest Class Weight / Lowest Class Weight"
- 55%の閾値: 比率が55%以上を除外とあるが、比率がパーセンテージで55%とはどういう意味か
- 最大/最小の比率なら通常1.0を超えるため、`0.55` との整合が取れない
- 論文本文では class weight を `class pixels / total pixels` と定義している
- **実装では、画像内の最大クラス重みが55%以上なら除外する、という運用解釈を採用**

### C3: 論文の再現性

- コードは公開されていない
- U-Netの正確なチャネル数・層数はFigure 7の図からの推定
- **段階的に実装し、結果を比較しながら調整する**
