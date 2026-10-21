# train
## Sub-features
`--smoke` (128×160, 8/4 slices, 1 epoch), full training with `--config default|paper`, `--set` overrides, `--dry-run`.
## How to get to it (user POV)
`python -m lumbarseg_cli train --smoke`
## Driving it with the CLI
JSON: `.epochs_run`, `.best_model`. Check that `run/train_files.txt`, `run/validation_files.txt`, `run/config.json` and `run/checkpoints/best_model.keras` exist.
## Gotchas
Requires prepare-data for the same `--config` (processed dirs are per config). On CPU, a full 512×640 epoch is slow. Use `--smoke` for checks.
