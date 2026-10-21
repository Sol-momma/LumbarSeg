# evaluate / score-csv
## Sub-features
Fixed counting rules from `config.eval`, per-class dice/iou/asd/hd/hd95/nsd with inf counts, record appended to results.jsonl (commit, dirty, config, split sha256, metrics).
## How to get to it (user POV)
`python -m lumbarseg_cli evaluate --model M.keras --file_list frozen.txt --processed_root DIR`
## Driving it with the CLI
`--json` → `.metrics.foreground_macro_dice`, `.split.sha256`, `.recorded`. Compare `wc -l` of the results file before and after (+1). `--no-record` must leave it unchanged.
`score-csv docs/experiments/t2_space_reproduction_4cls090_cap1000_20260711/validation_metrics.csv` must give 0.8497.
## Gotchas
The 0.8497 model weights are not in the repo (`*.keras` is gitignored), so a model-level reproduction needs the lab PC checkpoint or a retrain.
