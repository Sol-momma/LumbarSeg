# doctor
## Sub-features
deps, nvidia-smi, TensorFlow GPU count, torch.cuda (optional), WSL detection, data_root counts, processed slices, disk, results log.
## How to get to it (user POV)
`python -m lumbarseg_cli doctor [--profile lab]`
## Driving it with the CLI
`$PY -m lumbarseg_cli doctor --json | jq '.ok, (.checks[] | select(.ok|not))'`. Exit 0 = ready, 3 = not ready.
## Gotchas
On the box there is no GPU, so the GPU checks show `warn` (not required for `box`). With `--profile lab` on the box they are required, so doctor exits 3. That is expected.
