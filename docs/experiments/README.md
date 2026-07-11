# Experiment Records

このディレクトリには、ローカル GPU や Colab で実行した軽量な実験記録を保存する。

大きな成果物は Git に含めない。

- Include: Markdown summaries, metrics CSV, small training log excerpts.
- Exclude: `.keras` models, preprocessed `.npz` slices, full `outputs/` directories.

## Records

- [T2 relaxed baseline - 2026-06-25](t2_relaxed_20260625.md)
- [T2 relaxed prediction examples - 2026-06-28](t2_relaxed_predictions_20260628/README.md)
- [Paper reproduction audit - 2026-06-28](reproduction_audit_20260628.md)
- [Full dataset paper reproduction audit - 2026-06-29](reproduction_audit_full_20260629.md)
- [T2 SPACE reproduction metrics - 2026-07-11](t2_space_reproduction_4cls090_cap1000_20260711/validation_metrics.csv)

## Automation

WSL GPU 側で次の再現実験を実行する。

```bash
bash scripts/run_reproduction_experiment.sh all_4cls090_cap1000
```

実行済み実験の判断表を更新する。

```bash
python scripts/summarize_reproduction_results.py
```
