# Full Dataset Paper Reproduction Audit - 2026-06-29

## Summary

The full SPIDER public dataset was downloaded and audited with `audit_reproduction.py`.

This resolves the previous data availability issue. The local full dataset now matches the expected public SPIDER sequence counts:

| Sequence | Series |
| --- | ---: |
| T1 | 196 |
| T2 | 210 |
| T2_SPACE | 41 |
| Total | 447 |

The next reproduction run should use full T2_SPACE data.

## Raw Volume Audit

| Sequence | Subset | Series | Masks |
| --- | --- | ---: | ---: |
| T1 | training | 158 | 158 |
| T1 | validation | 38 | 38 |
| T2 | training | 172 | 172 |
| T2 | validation | 38 | 38 |
| T2_SPACE | training | 30 | 30 |
| T2_SPACE | validation | 11 | 11 |

## Extraction Summary

| Sequence | Extracted slices |
| --- | ---: |
| T1 | 4354 |
| T2 | 4597 |
| T2_SPACE | 5056 |

## Filtering Audit

| Config | Sequence | Kept | Train | Validation | Removed class count | Removed imbalance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| paper_default_4cls_055_cap1000 | T1 | 27 | 19 | 8 | 2728 | 1599 |
| paper_default_4cls_055_cap1000 | T2 | 40 | 33 | 7 | 2898 | 1659 |
| paper_default_4cls_055_cap1000 | T2_SPACE | 34 | 31 | 3 | 3607 | 1415 |
| relaxed_4cls_090_nocap | T1 | 1621 | 1308 | 313 | 2728 | 5 |
| relaxed_4cls_090_nocap | T2 | 1696 | 1384 | 312 | 2898 | 3 |
| relaxed_4cls_090_nocap | T2_SPACE | 1447 | 1056 | 391 | 3607 | 2 |
| relaxed_3cls_090_nocap | T1 | 2876 | 2299 | 577 | 1247 | 231 |
| relaxed_3cls_090_nocap | T2 | 3004 | 2431 | 573 | 1344 | 249 |
| relaxed_3cls_090_nocap | T2_SPACE | 2665 | 1906 | 759 | 2244 | 147 |
| no_imbalance_4cls_nocap | T1 | 1626 | 1312 | 314 | 2728 | 0 |
| no_imbalance_4cls_nocap | T2 | 1699 | 1386 | 313 | 2898 | 0 |
| no_imbalance_4cls_nocap | T2_SPACE | 1449 | 1057 | 392 | 3607 | 0 |
| no_imbalance_3cls_nocap | T1 | 3107 | 2492 | 615 | 1247 | 0 |
| no_imbalance_3cls_nocap | T2 | 3253 | 2641 | 612 | 1344 | 0 |
| no_imbalance_3cls_nocap | T2_SPACE | 2812 | 2022 | 790 | 2244 | 0 |

## Interpretation

The public full dataset is now available locally:

```text
images = 447
masks = 447
T1 = 196
T2 = 210
T2_SPACE = 41
```

The strict filtering implementation still retains too few slices:

```text
T2_SPACE paper_default_4cls_055_cap1000: 34 kept, 31 train, 3 validation
```

This confirms that the previous low T2_SPACE slice count was not only caused by missing data. The current strict filtering interpretation is still too aggressive.

For the next T2_SPACE reproduction run, use the closest practical paper-shaped setting:

```text
sequence=T2_SPACE
min_classes=4
imbalance_threshold=0.90
max_slices_per_sequence=1000
```

Rationale:

- Keeps all four classes per slice.
- Uses a relaxed imbalance threshold because `0.55` removes nearly all slices.
- Uses the paper-style 1000 slice cap.
- Uses full public T2_SPACE data.

If this underperforms, run the higher-coverage condition:

```text
sequence=T2_SPACE
min_classes=3
imbalance_threshold=0.90
max_slices_per_sequence=0
```

## Next Training Command

```bash
python train.py \
  --data_root /mnt/c/Users/ctlab/somomma/dataset \
  --output_root outputs/t2_space_reproduction_4cls090_cap1000 \
  --sequences T2_SPACE \
  --batch_size 2 \
  --epochs 100 \
  --min_classes 4 \
  --imbalance_threshold 0.90 \
  --max_slices_per_sequence 1000 \
  --force_reprocess
```

Evaluation:

```bash
python evaluate.py \
  --data_root /mnt/c/Users/ctlab/somomma/dataset \
  --output_root outputs/t2_space_reproduction_4cls090_cap1000 \
  --min_classes 4 \
  --imbalance_threshold 0.90 \
  --max_slices_per_sequence 1000 \
  --model_path outputs/t2_space_reproduction_4cls090_cap1000/checkpoints/best_model.keras
```

Visualization:

```bash
python visualize_predictions.py \
  --data_root /mnt/c/Users/ctlab/somomma/dataset \
  --output_root outputs/t2_space_reproduction_4cls090_cap1000 \
  --model_path outputs/t2_space_reproduction_4cls090_cap1000/checkpoints/best_model.keras \
  --sequences T2_SPACE \
  --min_classes 4 \
  --imbalance_threshold 0.90 \
  --max_slices_per_sequence 1000 \
  --num_samples 12
```
