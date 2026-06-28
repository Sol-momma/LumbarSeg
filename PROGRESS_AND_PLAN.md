# LumbarSeg Progress and Next Plan

## Current Status

LumbarSeg now has a runnable TensorFlow/Keras baseline for SPIDER lumbar MRI segmentation, plus a working local GPU path through WSL2.

The most important confirmed result so far is the T2 relaxed baseline:

| Item | Value |
| --- | --- |
| Environment | WSL2 Ubuntu on Windows |
| GPU | NVIDIA GeForce RTX 3060 Ti |
| Sequence | T2 |
| Filtering | `min_classes=3`, `imbalance_threshold=0.90`, `max_slices_per_sequence=0` |
| Filtered slices | 1790 |
| Batch size | 2 |
| Epochs run | 32 |
| Mean Dice | 0.889731 |
| Mean IoU | 0.829235 |

Class-wise Dice:

| Class | Dice |
| --- | ---: |
| Background | 0.991183 |
| Vertebrae | 0.880898 |
| Spinal Canal | 0.859104 |
| IVDs | 0.827740 |
| Mean | 0.889731 |

This result is below the target paper-level Dice around 0.97, but it is the first usable local GPU baseline result.

## What We Have Done

### 1. Baseline CLI

Implemented the reproducible baseline pipeline:

- `preprocess.py`: converts SPIDER MHA volumes into 2D sagittal `.npz` slices.
- `train.py`: trains the Modified U-Net baseline.
- `evaluate.py`: computes class-wise validation metrics.
- `spine_baseline/`: contains preprocessing, dataset loading, model, losses, and metrics.
- `arguments/`: centralizes CLI arguments.

The baseline uses:

- Modified U-Net
- Leaky ReLU
- Glorot initialization
- Combined loss: `0.6 * Focal + 0.4 * Dice`
- Four classes: Background, Vertebrae, Spinal Canal, IVDs

### 2. Windows and WSL2 GPU Setup

We first tried native Windows TensorFlow GPU. That exposed a compatibility problem:

- Native Windows TensorFlow GPU support effectively stops at TensorFlow 2.10.
- TensorFlow 2.10 expects CUDA 11.x runtime such as `cudart64_110.dll`.
- The lab PC had CUDA 12.1/12.6 driver support, so native Windows TensorFlow was not the cleanest route.

We then switched to WSL2, where TensorFlow could see the GPU successfully:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

This returned:

```text
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 3. Initial Strict Experiments

Strict T2_SPACE filtering kept only 14 slices:

| Experiment | Filtered slices | Mean Dice |
| --- | ---: | ---: |
| T2_SPACE strict | 14 | 0.335 |
| T2 strict | small validation set | 0.282 |

These results were not useful for paper-level reproduction because the usable slice count was too small and validation was unstable.

### 4. T2 Relaxed Baseline

We relaxed filtering to increase the number of usable slices:

```bash
python train.py \
  --data_root /mnt/c/Users/ctlab/somomma/DataSet \
  --output_root outputs/t2_relaxed \
  --sequences T2 \
  --batch_size 2 \
  --epochs 100 \
  --min_classes 3 \
  --imbalance_threshold 0.90 \
  --max_slices_per_sequence 0 \
  --force_reprocess
```

This produced 1790 filtered T2 slices and a usable baseline:

```text
Mean Dice: 0.889731
Mean IoU: 0.829235
```

The corresponding lightweight experiment record was added under:

```text
docs/experiments/t2_relaxed_20260625.md
docs/experiments/t2_relaxed_metrics_20260625.csv
docs/experiments/t2_relaxed_training_log_tail_20260625.csv
```

### 5. Qualitative Prediction Visualization

We added a prediction visualization CLI:

```text
visualize_predictions.py
```

It renders:

- Input MRI slice
- Ground truth mask
- Predicted mask
- Prediction overlay
- Per-slice Dice summary CSV

Example command:

```bash
python visualize_predictions.py \
  --data_root /mnt/c/Users/ctlab/somomma/DataSet \
  --output_root outputs/t2_relaxed \
  --model_path outputs/t2_relaxed/checkpoints/best_model.keras \
  --sequences T2 \
  --min_classes 3 \
  --imbalance_threshold 0.90 \
  --max_slices_per_sequence 0 \
  --num_samples 12
```

Qualitative examples were generated and prepared under:

```text
docs/experiments/t2_relaxed_predictions_20260628/
```

### 6. PR Documentation Rule

We established a project rule that all PR descriptions must include:

```markdown
## Why

## What

## How
```

Optional sections:

```markdown
## Verification

## Notes
```

This rule is documented in:

```text
docs/workflow/pr_description.md
```

## Current Open Issues

### 1. T2_SPACE Reproduction Is Not Solved Yet

The target paper result is around Dice 0.97, but our successful run is T2 relaxed, not T2_SPACE paper-level reproduction.

Important open questions:

- Does the available local SPIDER copy contain enough T2_SPACE data?
- Is strict filtering too aggressive for T2_SPACE?
- Is the paper's filtering/slice selection different from our current implementation?

### 2. Qualitative Validation Is Not Complete

The visualization pipeline exists, but we still need to inspect the images carefully.

Important checks:

- Are predictions anatomically aligned with the MRI?
- Are vertebrae, spinal canal, and IVDs separated correctly?
- Are low-Dice slices caused by model failure, preprocessing, or label imbalance?

### 3. Failure Case Analysis Is Missing

The current visualization samples slices evenly. It does not yet automatically find the worst slices.

We need to identify:

- worst mean Dice slices
- worst Vertebrae Dice slices
- worst Spinal Canal Dice slices
- worst IVD Dice slices

### 4. Main Branch State Should Be Confirmed

After each PR merge, local `main` should be updated and checked:

```bash
git switch main
git pull
git log --oneline --decorate -5
```

This matters because the work has been split across multiple PRs.

## Next Plan

## Phase 1: Confirm Qualitative Baseline

Goal: confirm that Dice 0.889731 corresponds to anatomically plausible segmentation.

Tasks:

- Open all PNGs in `docs/experiments/t2_relaxed_predictions_20260628/`.
- Compare input, ground truth, prediction, and overlay.
- Record obvious failure patterns.
- Pay special attention to slices with low per-slice Dice in `prediction_summary.csv`.

Deliverable:

```text
docs/experiments/t2_relaxed_failure_analysis_20260628.md
```

Acceptance criteria:

- At least 5 representative success/failure cases are described.
- Failure causes are categorized as model, preprocessing, label imbalance, or uncertain.

## Phase 2: Add Worst-Case Visualization

Goal: make failure analysis repeatable.

Implementation:

Extend `visualize_predictions.py` with sorting options:

```bash
--sort_by dice_mean
--sort_by dice_vertebrae
--sort_by dice_spinal_canal
--sort_by dice_ivds
--sort_order ascending
```

Expected behavior:

- Run inference on the selected split.
- Compute per-slice Dice.
- Sort slices by the selected score.
- Render the worst `--num_samples` slices.

Deliverables:

```text
visualize_predictions.py
README.ja.md
docs/experiments/t2_relaxed_worst_predictions_YYYYMMDD/
```

Acceptance criteria:

- Can generate worst-case prediction panels.
- Produces `prediction_summary.csv`.
- Works with the existing T2 relaxed baseline command.

## Phase 3: Re-check Data Protocol

Goal: understand why T2_SPACE had only 14 usable slices and whether that blocks paper reproduction.

Tasks:

- Count raw files by sequence.
- Count extracted slices by sequence before filtering.
- Count kept slices under multiple filtering settings.
- Compare T2, T2_SPACE, and all-sequence settings.

Suggested command matrix:

```text
T2_SPACE strict
T2_SPACE relaxed
T2 strict
T2 relaxed
all sequences relaxed
```

Deliverable:

```text
docs/experiments/sequence_filtering_audit_YYYYMMDD.md
```

Acceptance criteria:

- Table showing raw volumes, extracted slices, kept slices, train slices, validation slices.
- Clear decision on whether T2_SPACE reproduction is feasible with the current local data.

## Phase 4: Improve Dice Beyond 0.89

Goal: move from usable baseline toward publishable improvement.

Candidate experiments:

1. Filtering and sampling
   - Tune `min_classes`
   - Tune `imbalance_threshold`
   - Balance samples by class presence

2. Training settings
   - Learning rate sweep
   - Patience adjustment
   - More stable validation monitoring

3. Augmentation
   - brightness/contrast
   - small affine transforms
   - elastic deformation if medically reasonable

4. Loss improvements
   - class-weighted Dice
   - Tversky / Focal Tversky
   - boundary-aware loss

5. Architecture improvements
   - Attention U-Net
   - residual blocks
   - deep supervision

Deliverable:

```text
docs/experiments/improvement_ablation_table_YYYYMMDD.md
```

Acceptance criteria:

- At least 3 controlled experiments.
- Same train/validation protocol.
- Class-wise Dice table.
- Qualitative examples for each major result.

## Phase 5: Prepare Research Narrative

Goal: make the work presentable to the professor and usable for thesis/paper writing.

Narrative:

1. We reproduced a TensorFlow/Keras baseline pipeline for SPIDER lumbar MRI segmentation.
2. We found that strict T2_SPACE reproduction was blocked by too few retained slices.
3. We established a usable T2 relaxed baseline with Mean Dice 0.889731.
4. We added qualitative validation to check anatomical plausibility.
5. We will analyze failures and use them to guide targeted improvements.

Deliverables:

```text
docs/presentation/
docs/research/
docs/experiments/
```

Acceptance criteria:

- One-page progress summary.
- Updated experiment table.
- Representative prediction images.
- Next experiment proposal with expected impact.

## Immediate Next Action

The next concrete task should be:

```text
Run the paper reproduction audit with audit_reproduction.py.
```

Why:

- The baseline code is runnable, but paper-level reproduction is not complete.
- T2_SPACE strict filtering retained only 14 slices, so the first blocker is data protocol mismatch.
- The next experiment must quantify raw volumes, extracted slices, kept slices, and train/validation split counts before another long training run.

Proposed branch:

```bash
git switch -c codex/paper-reproduction-audit
```

Run:

```bash
python audit_reproduction.py \
  --data_root /mnt/c/Users/ctlab/somomma/DataSet \
  --output_root outputs/reproduction_audit_all \
  --extract \
  --force_reprocess \
  --write_slice_stats
```

Then inspect:

```bash
cat outputs/reproduction_audit_all/reproduction_filtering_audit.csv
cat outputs/reproduction_audit_all/reproduction_raw_volume_audit.csv
```

Detailed plan:

```text
PAPER_REPRODUCTION_PLAN.md
```

Proposed PR description structure:

```markdown
## Why

The baseline code is runnable, but the paper-level T2 SPACE result has not been reproduced yet. The current T2_SPACE strict setting retains too few slices, so we need a reproducible audit of data counts, filtering effects, and split counts before running more long GPU jobs.

## What

- Add `audit_reproduction.py`.
- Add `PAPER_REPRODUCTION_PLAN.md`.
- Document the audit workflow from `README.ja.md`.

## How

The audit CLI can optionally extract all SPIDER slices, compute raw volume counts, evaluate multiple non-destructive filtering configurations, and write CSV summaries for raw volumes, filtering, and split counts.
```
