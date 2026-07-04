# 論文完全再現プラン

## 目的

Ahmed et al. (2025) の lumbar spine MRI segmentation baseline を、ローカル GPU / WSL2 環境で論文条件にできるだけ一致させて再現する。

最終目標は、T2 SPACE 条件で論文 reported Dice に近い class-wise Dice を確認し、差分が残る場合は「どの条件差分が原因か」を説明できる状態にすること。

参考:

- Paper: *Pioneering Precision in Lumbar Spine MRI Segmentation with Advanced Deep Learning and Data Enhancement*
- DOI: `10.1016/j.mlwa.2025.100635`
- arXiv: https://arxiv.org/abs/2409.06018
- Dataset: SPIDER, https://doi.org/10.5281/zenodo.10159290

## 現在の到達点

すでに論文 baseline 相当の実装はある。

- `preprocess.py`: SPIDER MHA から 2D sagittal slice を作成
- `train.py`: Modified U-Net baseline の学習
- `evaluate.py`: class-wise Dice / IoU / ASD / NSD / precision / recall / F1
- `visualize_predictions.py`: qualitative prediction panel の作成
- WSL2 + RTX 3060 Ti で TensorFlow GPU 実行確認済み

現在の最良実験:

| Item | Value |
| --- | --- |
| Sequence | T2 |
| Filtering | relaxed |
| Filtered slices | 1790 |
| Mean Dice | 0.889731 |
| Mean IoU | 0.829235 |

これは有効な baseline だが、論文の T2 SPACE reported Dice around 0.97 の完全再現ではない。

## 再現完了の定義

以下を満たしたら「論文再現完了」とする。

1. SPIDER の raw volume 数、sequence 数、train/validation split 数を記録する。
2. T1 / T2 / T2 SPACE それぞれで、抽出 slice 数と filtering 後 slice 数を記録する。
3. 論文条件に最も近い filtering 設定で T2 SPACE を学習する。
4. `best_model.keras` を validation set で評価し、class-wise metrics を保存する。
5. T2 SPACE の Dice が論文値に近いか比較する。
6. Dice が届かない場合、以下の差分を明記する。
   - retained slice 数
   - filtering 条件
   - sagittal axis / rotation / flip 補正
   - APTA / label correction の差分
   - train/validation split の差分
7. qualitative prediction examples を保存し、明らかな位置ずれやラベル崩れがないことを確認する。

## 実行フェーズ

## Phase 1: データ・filtering 監査

目的: T2 SPACE が 14 slice しか残らなかった原因を数で確認する。

実行コマンド:

```bash
python audit_reproduction.py \
  --data_root /mnt/c/Users/ctlab/somomma/DataSet \
  --output_root outputs/reproduction_audit_all \
  --extract \
  --force_reprocess \
  --write_slice_stats
```

出力:

```text
outputs/reproduction_audit_all/reproduction_raw_volume_audit.csv
outputs/reproduction_audit_all/reproduction_filtering_audit.csv
outputs/reproduction_audit_all/reproduction_slice_stats.csv
```

確認する列:

- `sequence`
- `extracted_slices`
- `removed_class_count`
- `removed_imbalance`
- `removed_sequence_cap`
- `kept`
- `train_slices`
- `validation_slices`

合格条件:

- T1/T2/T2_SPACE の raw volume 数が docs/reference/overview.md の `T1:196, T2:210, T2 SPACE:41` と整合する。
- T2 SPACE の extracted slice 数が十分あるか確認できる。
- strict 条件で落ちている理由が `removed_class_count` か `removed_imbalance` か特定できる。

## Phase 2: T2 SPACE reproduction run

Phase 1 の監査結果を見て、T2 SPACE で最も論文条件に近い設定を選ぶ。

最初に試す設定:

```bash
python train.py \
  --data_root /mnt/c/Users/ctlab/somomma/DataSet \
  --output_root outputs/t2_space_reproduction \
  --sequences T2_SPACE \
  --batch_size 2 \
  --epochs 100 \
  --min_classes 3 \
  --imbalance_threshold 0.90 \
  --max_slices_per_sequence 0 \
  --force_reprocess
```

評価:

```bash
python evaluate.py \
  --data_root /mnt/c/Users/ctlab/somomma/DataSet \
  --output_root outputs/t2_space_reproduction \
  --min_classes 3 \
  --imbalance_threshold 0.90 \
  --max_slices_per_sequence 0 \
  --model_path outputs/t2_space_reproduction/checkpoints/best_model.keras
```

出力:

```text
outputs/t2_space_reproduction/validation_metrics.csv
outputs/t2_space_reproduction/checkpoints/training_log.csv
```

合格条件:

- T2 SPACE の validation slice が少なすぎない。
- Vertebrae / Spinal Canal / IVDs がすべて学習されている。
- Mean Dice と class-wise Dice を論文値と比較できる。

## Phase 3: qualitative validation

T2 SPACE reproduction model で可視化する。

```bash
python visualize_predictions.py \
  --data_root /mnt/c/Users/ctlab/somomma/DataSet \
  --output_root outputs/t2_space_reproduction \
  --model_path outputs/t2_space_reproduction/checkpoints/best_model.keras \
  --sequences T2_SPACE \
  --min_classes 3 \
  --imbalance_threshold 0.90 \
  --max_slices_per_sequence 0 \
  --num_samples 12
```

確認:

```text
outputs/t2_space_reproduction/predictions/
outputs/t2_space_reproduction/predictions/prediction_summary.csv
```

合格条件:

- overlay が MRI 上の正しい構造に乗っている。
- Vertebrae / Spinal Canal / IVDs が視覚的に分離している。
- Dice が低い slice の失敗原因を説明できる。

## Phase 4: paper comparison report

結果を Markdown と CSV で保存する。

保存先:

```text
docs/experiments/t2_space_reproduction_YYYYMMDD.md
docs/experiments/t2_space_reproduction_metrics_YYYYMMDD.csv
docs/experiments/t2_space_reproduction_predictions_YYYYMMDD/
```

必ず書く内容:

- 実行環境
- raw volume count
- extracted slice count
- kept slice count
- train/validation slice count
- training command
- evaluation command
- class-wise metrics
- 論文値との差分表
- qualitative examples
- 未再現の場合の原因仮説

## Phase 5: 差分が残った場合の修正順

Dice が論文値に届かない場合は、以下の順で潰す。

1. **Data protocol**
   - T2 SPACE の raw volume が 41 series あるか
   - Overview CSV の split と一致しているか
   - validation slice が少なすぎないか

2. **Preprocessing**
   - sagittal axis inference が正しいか
   - rotation / flip の補正が必要な series がないか
   - mask label mapping が崩れていないか

3. **Filtering**
   - `min_classes=4` が厳しすぎないか
   - `imbalance_threshold=0.55` が T2 SPACE に厳しすぎないか
   - 論文の最終 1000 slices に近い selection になっているか

4. **Training**
   - batch size の違い
   - epoch 数
   - early stopping monitor
   - learning rate schedule

5. **Evaluation**
   - slice-wise average と pixel-wise aggregate の違い
   - background を mean に含めるか
   - class order の一致

## 直近の実行タスク

次に実行するコマンド:

```bash
python audit_reproduction.py \
  --data_root /mnt/c/Users/ctlab/somomma/DataSet \
  --output_root outputs/reproduction_audit_all \
  --extract \
  --force_reprocess \
  --write_slice_stats
```

その後に見るファイル:

```bash
cat outputs/reproduction_audit_all/reproduction_filtering_audit.csv
cat outputs/reproduction_audit_all/reproduction_raw_volume_audit.csv
```

この監査結果を見て、T2 SPACE の再学習条件を確定する。

## Phase 1 Audit Result

2026-06-28 の監査結果は [docs/experiments/reproduction_audit_20260628.md](docs/experiments/reproduction_audit_20260628.md) に保存した。

2026-06-29 に full public SPIDER dataset の監査も完了した:

```text
docs/experiments/reproduction_audit_full_20260629.md
```

次の T2 SPACE 再現条件:

```text
sequence=T2_SPACE
min_classes=4
imbalance_threshold=0.90
max_slices_per_sequence=1000
```

## Interrupted Preprocessing Repair

WSL や Windows storage の問題で前処理が中断された場合、image/mask の片方だけが残ったり、空の `.npz` が残ることがある。

その場合は、再実行前に以下で修復する。

```bash
python repair_processed_slices.py \
  --output_root ~/lumbarseg_runs/t2_space_reproduction_4cls090_cap1000 \
  --dry_run

python repair_processed_slices.py \
  --output_root ~/lumbarseg_runs/t2_space_reproduction_4cls090_cap1000
```

その後、`--force_reprocess` を付けずに `train.py` を再実行する。
