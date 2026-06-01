# ベースラインからの改善提案

Ahmed et al. (2025) の再現ベースラインを固定したあと、卒研として差分を作りやすい改善案を優先度順に整理する。

## まず固定する比較条件

- 同じSPIDER splitを使う: training / validationを患者単位で混ぜない
- 同じ前処理を使う: 4クラス変換、4クラス未満除外、支配クラス割合55%超除外、各シーケンス最大1000枚
- 同じ入力を使う: 512 x 640 の2D sagittal slice
- 同じ評価を使う: class-wise Dice、IoU、ASD、NSD、Precision、Recall、F1
- T1、T2、T2 SPACEは混合結果だけでなく、シーケンス別にも報告する

## 優先度A: 再現性と失敗分析

1. 可視化評価を追加する
   - MRI画像、正解マスク、予測マスク、重ね合わせ画像を同じsliceで保存する
   - Diceが低い順に失敗例を出すと、改善理由を説明しやすい

2. シーケンス別モデルを比較する
   - 論文はT1 / T2 / T2 SPACE別に結果を示している
   - `--sequences T1`、`--sequences T2`、`--sequences T2_SPACE` で別学習し、混合学習と比較する

3. orientation補正を明示する
   - 論文は一部sliceが矢状断でない、回転/反転があると述べている
   - SimpleITKのDirection/Spacing/Originから自動検出し、補正ログをCSVに残す

## 優先度B: 精度改善

1. Attention U-Net
   - skip connectionにattention gateを入れ、椎間板や脊柱管の小さい領域へ集中させる
   - 既存U-Netからの差分が小さく、卒研で説明しやすい

2. U-Net++ または deep supervision
   - decoderの中間出力にも損失をかけ、境界と小構造の復元を安定させる
   - DiceだけでなくASD/NSD改善も期待できる

3. Boundary-aware loss
   - Combined LossにBoundary Loss、Surface Loss、またはHausdorff系lossを足す
   - 論文のDiceは高いので、次の改善点は境界距離指標に置くと差分が出やすい

4. Focal Tversky Loss
   - IVDsや脊柱管のような小クラスでfalse negativeを抑えたい場合に有効
   - `Focal + Dice` と `Focal Tversky + Dice` のablationにする

## 優先度C: データ改善

1. 医用画像向けaugmentation
   - elastic deformation、Gaussian noise、Gaussian smoothing、random cropを比較する
   - SPIDER dataset paperでも使われているため、根拠を説明しやすい

2. intensity normalizationの比較
   - 現在はsliceごとのmin-max正規化
   - volume単位z-score、percentile clipping、N4 bias correctionの有無を比較する

3. 2.5D入力
   - 中央sliceに前後sliceをチャネルとして加える
   - 2D U-Netの軽さを保ちつつ、3D文脈を少し使える

## 優先度D: 後処理

1. connected component filtering
   - 小さい孤立予測を削除し、ASD/NSDの悪化を抑える

2. anatomical order constraint
   - 椎体・椎間板が上下方向に交互に並ぶという制約を使う
   - 明らかに離れた偽陽性を除外できる

## 推奨する卒研の実験順

1. 論文準拠ベースラインをT2 SPACEで再現
2. 失敗sliceの可視化を作る
3. Attention U-Netを追加してDice/ASD/NSDを比較
4. Boundary-aware lossを追加して境界指標を比較
5. augmentationあり/なしのablationを行う
6. 最後にT1/T2/T2 SPACE別の得意・不得意を考察する
