# Paper vs. implementation gap (Ahmed et al., arXiv 2409.06018)

Compiled 2026-10-21. "論文に記載なし" = not stated in the paper. Each row names the config key that
controls it. `default` = implementation (reproduces the recorded 0.8497 run settings);
`paper` = `configs/experiments/paper.json`.

| Item | Paper | Implementation (`default`) | Config key | `paper` value |
| --- | --- | --- | --- | --- |
| Dice averaging unit | 論文に記載なし (reads as per-slice) | per-slice mean | `eval.dice_average_unit` = `slice` \| `case` \| `pixel` | `slice` (論文に記載なし) |
| Background in mean | Per-class Dice for all classes; the 0.97 target is foreground 3 classes (IVD 0.9688 / Vertebrae 0.9712 / Canal 0.9671). How the ≈0.969 mean is averaged: 論文に記載なし | headline = foreground macro (0.8497); 4-class mean (0.8836) reported for reference | `eval.include_background` (affects `mean_dice`/`mean_iou`; `foreground_macro_dice` is always foreground-only) | `false` |
| Train/val split | SPIDER training/validation subsets | Overview CSV `subset`, series-level; T2 SPACE 30/11 series | `data.split` | same |
| 1000-slice selection | 論文に記載なし | evenly spaced (`linspace`) over filename order, then split (1447 without cap) | `data.slice_cap_method` = `linspace_sorted` \| `random`, `data.slice_cap_seed` | `linspace_sorted`, seed 42 (論文に記載なし) |
| ≥4 classes per slice | yes | yes | `data.min_classes` | 4 |
| Imbalance filter | 55% (inconsistent with its own definition); T2 SPACE 37% | drop if largest foreground class share > 0.90 (0.55 keeps only 34 slices) | `data.imbalance_threshold` | 0.55 |
| Sequences for training | evaluated per sequence; training mix 論文に記載なし | T2 SPACE only | `data.sequences` | `T2_SPACE` (論文に記載なし) |
| Preprocessing | 512×640 PNG, manual axis correction, APTA | 512×640 npz, automatic axis inference, rule-based mapping, min-max | `data.target_height/width` | 512×640 |
| Epochs / batch | 100 epochs, batch 8, early stop on val mIoU | max 100, batch 2, patience 15 (best epoch 32) | `train.epochs`, `train.batch_size`, `train.patience` | 100, 8, patience 15 (論文に記載なし) |
| Loss | α=0.6, γ=0.4 | α=0.6, γ=4.0 (10× difference) | `train.focal_weight`, `train.focal_gamma` | 0.6, 0.4 |
| Optimizer / LR / augmentation | 論文に記載なし | Adam 1e-4, ReduceLROnPlateau, no augmentation | `train.learning_rate` | 1e-4 (論文に記載なし) |
| Empty-vs-empty Dice | 論文に記載なし | smoothing 1e-7 → 1.0 | `eval.smooth` | 1e-7 (論文に記載なし) |
| HD/ASD when one mask is empty (inf) | 論文に記載なし | inf propagates to the mean (Canal ASD = inf in the 0.8497 run) | `eval.surface_inf_policy` = `propagate` \| `exclude` \| `max_distance` | `propagate` (論文に記載なし) |
| NSD tolerance | 論文に記載なし | 1.0 mm | `eval.nsd_tolerance_mm` | 1.0 (論文に記載なし) |
