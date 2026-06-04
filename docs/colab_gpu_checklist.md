# Google Colab GPU 実行ゴールとチェックリスト

## 目的

このチェックリストのゴールは、Ahmed et al. (2025) の腰椎MRIセグメンテーション手法を、SPIDERデータセット上で再現するために、Google Colab GPUで実行すべき作業を段階的に確認すること。

最初の到達目標は、**T2 SPACEのみでModified U-Netベースラインを学習し、validation metricsを出力すること**。その後、T1/T2/T2 SPACE全体、改善モデル、ablation studyへ進む。

## 現状

- [x] 論文の目的・対象構造を整理した
- [x] SPIDERデータセットの概要を整理した
- [x] 4クラスのラベルマッピングを確認した
- [x] 3D MHAから2Dスライスを作る前処理コードを用意した
- [x] クラス欠損・クラス不均衡スライスを除外するフィルタを用意した
- [x] 論文準拠のModified U-Netを実装した
- [x] Combined Lossを実装した
- [x] 学習スクリプトを用意した
- [x] 評価スクリプトを用意した
- [ ] Colab GPU上で1 epochのスモークテストを実行した
- [ ] T2 SPACEで本学習を実行した
- [ ] validation metricsを出力した
- [ ] 論文値と比較した
- [ ] 予測マスクを可視化した
- [ ] 改善手法を実装・比較した

## Colabで最初に実行すること

いきなり100 epochの本学習を実行しない。最初は以下の順番で、パイプライン全体が壊れていないかを確認する。

### 1. GPU確認

```python
import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
```

完了条件:

- [ ] GPUが1つ以上表示される

### 2. Google Driveをマウント

```python
from google.colab import drive

drive.mount("/content/drive")
```

完了条件:

- [ ] `/content/drive/MyDrive/SPIDER/DataSet/` に `images/`, `masks/`, `SPIDER Lumbar Spine Segmentation Overview.csv` がある

### 3. リポジトリ準備

```bash
!git clone https://github.com/Sol-momma/LumbarSeg.git
%cd LumbarSeg
!pip install -r requirements-baseline.txt
```

完了条件:

- [ ] `preprocess.py`, `train.py`, `evaluate.py` がColab上にある
- [ ] `SimpleITK`, `tensorflow`, `opencv-python`, `scipy`, `scikit-learn` がimportできる

### 4. T2 SPACEだけ前処理

```bash
!python preprocess.py \
  --data_root /content/drive/MyDrive/SPIDER/DataSet \
  --output_root /content/drive/MyDrive/SPIDER/outputs/t2_space_baseline \
  --sequences T2_SPACE
```

完了条件:

- [ ] `outputs/t2_space_baseline/images/` に `.npz` ファイルが作成される
- [ ] `outputs/t2_space_baseline/masks/` に `.npz` ファイルが作成される
- [ ] `filtered_files.txt` が作成される
- [ ] `filtered_slice_stats.csv` が作成される
- [ ] `Filtered files` が0ではない
- [ ] エラー件数が多すぎない

`Filtered files: 0` になった場合は、古いフィルタ実装を使っている可能性がある。`git pull origin main` で最新版に更新してから、同じ前処理コマンドを再実行する。抽出済みファイルがある場合、通常はMHA抽出をスキップしてフィルタだけ再実行される。

### 5. 1 epochのスモークテスト

```bash
!python train.py \
  --data_root /content/drive/MyDrive/SPIDER/DataSet \
  --output_root /content/drive/MyDrive/SPIDER/outputs/t2_space_baseline \
  --sequences T2_SPACE \
  --epochs 1 \
  --batch_size 2
```

完了条件:

- [ ] Train slicesが0ではない
- [ ] Validation slicesが0ではない
- [ ] `best_model.keras` が保存される
- [ ] `final_model.keras` が保存される
- [ ] `training_log.csv` が保存される
- [ ] loss, accuracy, mean_iou, dice_coefficientが表示される

### 6. スモークテストモデルを評価

```bash
!python evaluate.py \
  --data_root /content/drive/MyDrive/SPIDER/DataSet \
  --output_root /content/drive/MyDrive/SPIDER/outputs/t2_space_baseline \
  --model_path /content/drive/MyDrive/SPIDER/outputs/t2_space_baseline/checkpoints/best_model.keras \
  --limit 10
```

完了条件:

- [ ] `validation_metrics.csv` が作成される
- [ ] Background, Vertebrae, Spinal Canal, IVDs, Meanの行が出力される
- [ ] dice, iou, precision, recall, f1がNaNではない

## スモークテスト後に実行する本学習

スモークテストが成功してから、T2 SPACEで本学習を行う。

```bash
!python train.py \
  --data_root /content/drive/MyDrive/SPIDER/DataSet \
  --output_root /content/drive/MyDrive/SPIDER/outputs/t2_space_baseline \
  --sequences T2_SPACE \
  --epochs 100 \
  --batch_size 8
```

完了条件:

- [ ] Early Stoppingまたは100 epoch完了まで学習が進む
- [ ] `best_model.keras` が更新される
- [ ] `training_log.csv` に複数epochの記録が残る
- [ ] validation mean_iouが学習初期より改善している

## 本学習後の評価

```bash
!python evaluate.py \
  --data_root /content/drive/MyDrive/SPIDER/DataSet \
  --output_root /content/drive/MyDrive/SPIDER/outputs/t2_space_baseline \
  --model_path /content/drive/MyDrive/SPIDER/outputs/t2_space_baseline/checkpoints/best_model.keras
```

完了条件:

- [ ] `validation_metrics.csv` が保存される
- [ ] IVDs / Vertebrae / Spinal CanalのDiceを確認する
- [ ] 論文値と比較する

比較対象:

| 構造 | 論文Dice |
| --- | ---: |
| IVDs | 0.9688 |
| Vertebrae | 0.9712 |
| Spinal Canal | 0.9671 |

## 次の研究ステップ

T2 SPACEのbaseline結果が出たら、次の順番で進める。

- [ ] 失敗sliceを可視化する
- [ ] T1のみで同じ実験を行う
- [ ] T2のみで同じ実験を行う
- [ ] T1/T2/T2 SPACE混合学習を行う
- [ ] Attention U-Netを追加する
- [ ] Boundary LossまたはFocal Tversky Lossを追加する
- [ ] augmentationあり/なしを比較する
- [ ] ablation tableを作る

## 上司への現状報告文

現在は、論文再現に必要な前処理、モデル、損失関数、学習、評価のコード基盤は構築済みです。次の段階として、Google Colab GPU上でT2 SPACEデータを対象に1 epochのスモークテストを行い、前処理から評価までのパイプラインが正常に動作することを確認します。その後、100 epochの本学習を実行し、論文で報告されているDice約0.97との比較を行います。
