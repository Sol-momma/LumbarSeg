---
name: verify-lumbarseg
description: Drive the LumbarSeg experiment CLI (python -m lumbarseg_cli) end to end — doctor, Zenodo download, preprocessing, smoke training, evaluation, results.jsonl — and judge the output yourself. Use before/after changing training, data or evaluation code, or when asked to prove the harness works on the box, Colab or the lab PC (WSL2).
---

# Verify LumbarSeg via its CLI

The surface is a short-lived Python CLI. No server. Every command supports `--json`; parse it instead of reading logs.
Logs go to stderr, the result object goes to stdout.

## Launch (install once)

```bash
cd <repo>
python3 -m venv .venv && .venv/bin/pip install -r requirements-baseline.txt   # CPU box: tensorflow-cpu also works
export PY=.venv/bin/python
```
Ready when `$PY -m lumbarseg_cli --help` prints the command list.
Profiles: `box` (CPU, `/workspace/lumbarseg-data`), `colab`, `lab` (WSL2, `/mnt/c/Users/ctlab/somomma/DataSet`). Auto-detected; force with `--profile` or `LUMBARSEG_PROFILE`.
Isolation: point `LUMBARSEG_OUTPUT_ROOT` and `LUMBARSEG_RESULTS` at a scratch dir so you never overwrite someone else's run in the shared output root.

## Doctor (run first, read-only)

```bash
$PY -m lumbarseg_cli doctor --json
```
Pass = `.ok == true` (exit 0). Exit 3 = a required check failed; each failed check has a `next` field. GPU checks (`nvidia-smi`, `tensorflow GPU`) are required only for GPU profiles; `torch.cuda` is never required.

## Drive

Use `scripts/verify.sh`, which runs the whole journey into a scratch dir and checks it:
```bash
.cursor/skills/verify-lumbarseg/scripts/verify.sh            # needs data under data_root (or add --download)
.cursor/skills/verify-lumbarseg/scripts/verify.sh --download  # fetch T2 SPACE (1.62 GB) first if missing
```
It prints `PASS`/`FAIL` lines and exits non-zero on the first failure. Manual steps: see `features/`.

## Judging results (what counts as correct)

- `smoke` JSON: `.ok == true`, `.metrics.foreground_macro_dice` in [0, 1], `.recorded` is a path, and that file gained exactly one line.
- Smoke Dice values mean nothing for accuracy (1 epoch, 8 slices, 128×160). Never report them as accuracy.
- Accuracy claims need `evaluate` on the full frozen validation list (`split.limit == null`) with the `default` (or named) config. Compare `metrics.foreground_macro_dice` to the recorded 0.8497 only if `config.eval` matches (`slice`, background excluded, `propagate`).
- `score-csv` on `docs/experiments/t2_space_reproduction_4cls090_cap1000_20260711/validation_metrics.csv` must print 0.8497. This checks the aggregation only, not a model.
- An inf ASD/HD with `*_n_inf > 0` is expected under `propagate` (canal missed on some slice). It is not a crash.

## Evidence

Save the stdout JSON of each step and `git rev-parse HEAD` under `$EVIDENCE` (verify.sh: `/tmp/lumbarseg-verify-<timestamp>/evidence`). Proof = the action's JSON plus the side effect (new results line, `run/checkpoints/best_model.keras` exists, files in data_root). For `--dry-run`, check that nothing was written: verify.sh compares file counts before and after.

## Cleanup

```bash
LUMBARSEG_OUTPUT_ROOT=<scratch> $PY -m lumbarseg_cli cleanup all --config smoke --dry-run --json   # check the targets first
LUMBARSEG_OUTPUT_ROOT=<scratch> $PY -m lumbarseg_cli cleanup all --config smoke --json
```
`cleanup` deletes only `run/` and `processed/` inside `output_root`. It never touches data_root or results logs. Evidence lives outside output_root and is kept. Do not kill processes by name. Stop only PIDs you started.

## Gotchas

- `prepare-data` with no `.mha` files exits 3 with `next: ... --download --dry-run`.
- On WSL, `/mnt/c` I/O is slow. Keep `output_root` on the Linux filesystem (lab profile does).
- `tests/test_reproduction_goal_queue.py::test_queue_runs_sequentially_and_stops_after_target` already fails on main (not harness-related).
