# Paper Reproduction Audit - 2026-06-28

## Summary

`audit_reproduction.py` was run on the WSL2 lab environment to diagnose why strict T2_SPACE filtering retained only 14 slices.

The audit confirms that T2_SPACE has enough extracted slices for a reproduction attempt if the filtering conditions are relaxed. The next reproduction run should use:

```text
sequence=T2_SPACE
min_classes=3
imbalance_threshold=0.90
max_slices_per_sequence=0
```

## Command

```bash
python audit_reproduction.py \
  --data_root /mnt/c/Users/ctlab/somomma/DataSet \
  --output_root outputs/reproduction_audit_all \
  --extract \
  --force_reprocess \
  --write_slice_stats
```

## Raw Volume Audit

| Sequence | Subset | Series | Masks |
| --- | --- | ---: | ---: |
| T1 | training | 94 | 94 |
| T1 | validation | 22 | 22 |
| T2 | training | 100 | 100 |
| T2 | validation | 24 | 24 |
| T2_SPACE | training | 15 | 15 |
| T2_SPACE | validation | 4 | 4 |

Total local audited series:

| Sequence | Total series |
| --- | ---: |
| T1 | 116 |
| T2 | 124 |
| T2_SPACE | 19 |
| Total | 259 |

The local copy appears to contain 259 series, not the full 447 series described in the SPIDER dataset summary. This must be noted when comparing against the paper.

## Extraction Summary

| Item | Value |
| --- | ---: |
| Files processed | 259 |
| Total extracted slices | 7762 |
| Sagittal axis 2 | 226 series |
| Sagittal axis 0 | 33 series |
| Errors | 0 |

Extracted slices by sequence:

| Sequence | Extracted slices |
| --- | ---: |
| T1 | 2626 |
| T2 | 2770 |
| T2_SPACE | 2366 |

## Filtering Audit

| Config | Sequence | Kept | Train | Validation | Removed class count | Removed imbalance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| paper_default_4cls_055_cap1000 | T1 | 20 | 13 | 7 | 1668 | 938 |
| paper_default_4cls_055_cap1000 | T2 | 22 | 16 | 6 | 1765 | 983 |
| paper_default_4cls_055_cap1000 | T2_SPACE | 14 | 11 | 3 | 1697 | 655 |
| relaxed_4cls_090_nocap | T1 | 955 | 776 | 179 | 1668 | 3 |
| relaxed_4cls_090_nocap | T2 | 1003 | 811 | 192 | 1765 | 2 |
| relaxed_4cls_090_nocap | T2_SPACE | 668 | 544 | 124 | 1697 | 1 |
| relaxed_3cls_090_nocap | T1 | 1714 | 1379 | 335 | 769 | 143 |
| relaxed_3cls_090_nocap | T2 | 1790 | 1434 | 356 | 827 | 153 |
| relaxed_3cls_090_nocap | T2_SPACE | 1219 | 971 | 248 | 1067 | 80 |
| no_imbalance_4cls_nocap | T1 | 958 | 778 | 180 | 1668 | 0 |
| no_imbalance_4cls_nocap | T2 | 1005 | 812 | 193 | 1765 | 0 |
| no_imbalance_4cls_nocap | T2_SPACE | 669 | 545 | 124 | 1697 | 0 |
| no_imbalance_3cls_nocap | T1 | 1857 | 1498 | 359 | 769 | 0 |
| no_imbalance_3cls_nocap | T2 | 1943 | 1559 | 384 | 827 | 0 |
| no_imbalance_3cls_nocap | T2_SPACE | 1299 | 1040 | 259 | 1067 | 0 |

## Interpretation

Strict paper-like filtering is too aggressive in the current implementation:

```text
T2_SPACE strict: 2366 extracted -> 14 kept
```

The dominant issue is class-count filtering:

```text
removed_class_count = 1697
removed_imbalance = 655
```

Relaxing to `min_classes=3` and `imbalance_threshold=0.90` keeps enough T2_SPACE slices for a real reproduction attempt:

```text
T2_SPACE relaxed_3cls_090_nocap: 1219 kept, 971 train, 248 validation
```

This is the next condition to train.

## Next Command

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

Evaluation:

```bash
python evaluate.py \
  --data_root /mnt/c/Users/ctlab/somomma/DataSet \
  --output_root outputs/t2_space_reproduction \
  --min_classes 3 \
  --imbalance_threshold 0.90 \
  --max_slices_per_sequence 0 \
  --model_path outputs/t2_space_reproduction/checkpoints/best_model.keras
```

Visualization:

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
