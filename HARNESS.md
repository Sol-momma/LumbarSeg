# Experiment harness (`python -m lumbarseg_cli`)

One CLI for SPIDER download, preprocessing, training, evaluation, environment checks and cleanup.
Run `python -m lumbarseg_cli --help` and `python -m lumbarseg_cli <command> --help` for every flag.

## One line: data → tiny training → evaluation → metrics JSON

```bash
pip install -r requirements-baseline.txt      # CPU box: tensorflow-cpu is enough
python -m lumbarseg_cli smoke --download --json
```

`--download` fetches only the T2 SPACE series (41 images + 41 masks, 1.62 GB) from
[Zenodo 10.5281/zenodo.10159290](https://doi.org/10.5281/zenodo.10159290) using HTTP range reads of
`images.zip`, so the full 3.7 GB archive is never downloaded. Check first with
`python -m lumbarseg_cli prepare-data --download --dry-run`. Data is written to the profile's
`data_root`, never into this repository (SPIDER is CC-BY-4.0; we do not redistribute it).

The smoke metrics prove the pipeline runs. They are not accuracy numbers.

## Commands

| Command | Purpose | Destructive? |
| --- | --- | --- |
| `doctor` | deps, `nvidia-smi`, TensorFlow GPU, `torch.cuda` (if installed), WSL detection, data_root, processed slices, disk | read-only, exit 3 if not ready |
| `prepare-data [--download] [--force]` | fetch from Zenodo, extract + filter slices into `<output_root>/<config>/processed` | `--dry-run` |
| `train [--smoke]` | train Modified U-Net; writes `run/` (file lists, config, checkpoints) | `--dry-run` |
| `evaluate` | evaluate a model with the fixed counting rules, append to `experiments/results.jsonl` | `--no-record` |
| `score-csv CSV` | recompute headline Dice from a legacy `validation_metrics.csv` | read-only |
| `cleanup run\|processed\|all` | delete outputs under `output_root` only | `--dry-run` |
| `smoke [--download]` | prepare-data → train --smoke → evaluate | — |

Every command takes `--json`, `--profile`, `--config`, `--set KEY=VALUE`. Errors print `error:` and `next:`
(what to do), and with `--json` a `{"ok": false, "error", "next"}` object. Exit codes: 0 ok, 1 failure,
2 usage/config error, 3 environment not ready.

## Where it runs (profiles)

| Profile | data_root | output_root | device |
| --- | --- | --- | --- |
| `box` | `/workspace/lumbarseg-data/DataSet` | `/workspace/lumbarseg-data/runs` | CPU |
| `colab` | `/content/drive/MyDrive/SPIDER/DataSet` | `/content/drive/MyDrive/SPIDER/runs` | GPU |
| `lab` | `/mnt/c/Users/ctlab/somomma/DataSet` | `~/lumbarseg_runs` (Linux side; `/mnt/c` is slow) | WSL2, RTX 3060 Ti 8 GB |

Auto-detected (Colab → `colab`, WSL → `lab`, else `box`); override with `--profile` or `LUMBARSEG_PROFILE`.
Paths: `LUMBARSEG_DATA_ROOT`, `LUMBARSEG_OUTPUT_ROOT`, `LUMBARSEG_RESULTS`.
Lab GPU: TensorFlow needs `pip install 'tensorflow[and-cuda]'` inside WSL2 and the NVIDIA *Windows* driver.

## Experiment configs

- `default` — implementation values; the settings of the recorded 0.8497 run.
- `paper` — values the paper states (γ=0.4, batch 8, imbalance 0.55). Unstated items keep implementation values and are listed in `_not_in_paper`.
- `smoke` — 128×160, 8 train / 4 val slices, 1 epoch.

Full comparison: [docs/experiments/paper_gap.md](docs/experiments/paper_gap.md).

## Evaluation protocol (fixed in `config.eval`)

| Rule | Default | Options | Paper |
| --- | --- | --- | --- |
| Dice averaging unit | `slice`: per-slice Dice, mean over slices | `case` (pool pixels per series), `pixel` (pool all) | 論文に記載なし |
| Background | headline `foreground_macro_dice` = mean of Vertebrae, Spinal Canal, IVDs; `include_background` controls `mean_dice` / `mean_iou` | `true` / `false` | per-class targets are foreground; averaging 論文に記載なし |
| Empty prediction and empty truth | Dice = 1 (smoothing `1e-7`) | `eval.smooth` | 論文に記載なし |
| HD / HD95 / ASD with inf (one mask empty) | `propagate`: the mean becomes inf; `*_n_inf` counts the inf slices | `exclude`, `max_distance` (image diagonal in mm) | 論文に記載なし |
| Surface metrics unit | per slice, in mm from the stored spacing | — | 論文に記載なし |
| NSD tolerance | 1.0 mm | `eval.nsd_tolerance_mm` | 論文に記載なし |

The slice/default rules match `spine_baseline/metrics.py` (`evaluate.py`), which produced 0.8497.
`tests/test_harness_metrics.py` checks that they agree.

## Results log

`evaluate` appends one JSON line to `experiments/results.jsonl`:
`timestamp` (UTC), `experiment`, `commit`, `dirty`, `profile`, `model`, the full `config` (data, model, train, eval),
`split` (validation list path, slice and case counts, SHA-256 of the list, `limit`) and `metrics` (per-class dice/iou/asd/hd/hd95/nsd with inf counts,
`foreground_macro_dice`, `mean_dice`, `all_class_mean_dice`). inf is written as `"inf"`. `smoke` writes to `experiments/smoke.local.jsonl` (gitignored) unless `LUMBARSEG_RESULTS` is set, so only real experiments land in `results.jsonl`.
