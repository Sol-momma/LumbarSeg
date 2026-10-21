# prepare-data
## Sub-features
Zenodo range download (configured sequences only), size dry-run, slice extraction, filtering (imbalance threshold, 1000 cap, cap method/seed).
## How to get to it (user POV)
`python -m lumbarseg_cli prepare-data --download --dry-run`, then without `--dry-run`.
## Driving it with the CLI
Dry-run JSON: `.download_plan.files` (82 for T2 SPACE), `.download_plan.bytes_to_download`. Real run JSON: `.filtering.kept`, `.filtering.kept_by_sequence`. Confirm side effects: `ls $data_root/images/*.mha | wc -l`, and `<output_root>/<config>/processed/filtered_files.txt` exists.
## Gotchas
Existing files with the right size are skipped. `--all-sequences` downloads about 3.8 GB.
