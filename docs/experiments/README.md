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

精度0.97キャンペーンの合格条件と実験順は
[T2 SPACE 精度0.97キャンペーン計画](t2_space_accuracy_097_plan.md)を参照する。

WSL GPU 側で次の再現実験を実行する。

```bash
bash scripts/run_reproduction_experiment.sh all_4cls090_cap1000
```

実行済み実験の判断表を更新する。

```bash
python scripts/summarize_reproduction_results.py
```

検証CSVが新しい合格条件を満たすか確認する。

```bash
python scripts/check_reproduction_target.py /path/to/validation_metrics.csv
```

単一GPUで複数のT2 SPACEプリセットを順番に実行する。キューは通常シェルからの起動を拒否するため、tmux内で実行する。

```bash
tmux new-session -s lumbarseg-goal
bash scripts/run_reproduction_goal_queue.sh t2_space_4cls090_cap1000
```

開始後は`Ctrl-b`、続けて`d`で画面から離れる。再接続は`tmux attach -t lumbarseg-goal`を使う。
