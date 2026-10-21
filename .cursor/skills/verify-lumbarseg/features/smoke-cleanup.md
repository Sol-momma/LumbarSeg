# smoke / cleanup
## Sub-features
One-line prepare → train --smoke → evaluate; cleanup of run/processed with dry-run.
## How to get to it (user POV)
`python -m lumbarseg_cli smoke --download --json`; `python -m lumbarseg_cli cleanup all --dry-run`
## Driving it with the CLI
`scripts/verify.sh` covers both in a scratch output root.
## Gotchas
smoke writes to `experiments/smoke.local.jsonl` unless `LUMBARSEG_RESULTS` is set. cleanup refuses paths outside output_root.
