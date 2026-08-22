from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys

from sklearn.model_selection import train_test_split

# Running ``python scripts/...`` places scripts/ rather than the repository root
# on sys.path. Keep the documented CLI independent of an editable installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spine_baseline.file_lists import (
    AUTHOR_DIAGNOSTIC_WARNING,
    COHORT_MODE_AUTHOR_DIAGNOSTIC_SLICE,
    read_file_list,
    validate_disjoint_cohorts,
)


SPLIT_MODE = "author_diagnostic_random_slice_90_10"
VALIDATION_FRACTION = 0.10


@dataclass(frozen=True)
class AuthorDiagnosticSplit:
    train_files: tuple[str, ...]
    validation_files: tuple[str, ...]
    shared_series: tuple[str, ...]
    seed: int


def _validate_source_files(files: list[str]) -> list[str]:
    for filename in files:
        entry = Path(filename)
        if entry.name != filename or entry.suffix.casefold() != ".npz":
            raise ValueError(f"Source list must contain .npz basenames only: {filename!r}")
    if len(files) < 2:
        raise ValueError("Author diagnostic split requires at least two slices")
    # Sorting makes the seed operate on content rather than on incidental file
    # discovery order. This is deterministic, but it remains only a diagnostic
    # reconstruction because the author's exact source order was not published.
    return sorted(files, key=lambda item: (item.casefold(), item))


def create_author_diagnostic_split(files: list[str], seed: int) -> AuthorDiagnosticSplit:
    ordered = _validate_source_files(files)
    train_files, validation_files = train_test_split(
        ordered,
        test_size=VALIDATION_FRACTION,
        random_state=seed,
        shuffle=True,
    )
    train_files = sorted(train_files, key=lambda item: (item.casefold(), item))
    validation_files = sorted(validation_files, key=lambda item: (item.casefold(), item))
    report = validate_disjoint_cohorts(
        train_files,
        validation_files,
        mode=COHORT_MODE_AUTHOR_DIAGNOSTIC_SLICE,
    )
    return AuthorDiagnosticSplit(
        train_files=tuple(train_files),
        validation_files=tuple(validation_files),
        shared_series=report.shared_series,
        seed=seed,
    )


def _list_bytes(files: tuple[str, ...]) -> bytes:
    return ("\n".join(files) + "\n").encode("utf-8")


def write_split(output_dir: Path, source_path: Path, split: AuthorDiagnosticSplit) -> None:
    if output_dir.exists():
        raise ValueError(f"Output directory already exists; refusing to mix split evidence: {output_dir}")
    output_dir.mkdir(parents=True)

    train_bytes = _list_bytes(split.train_files)
    validation_bytes = _list_bytes(split.validation_files)
    (output_dir / "train_files.txt").write_bytes(train_bytes)
    (output_dir / "validation_files.txt").write_bytes(validation_bytes)

    source_bytes = source_path.read_bytes()
    rows = [
        ("split_mode", SPLIT_MODE),
        ("seed", str(split.seed)),
        ("validation_fraction", f"{VALIDATION_FRACTION:.2f}"),
        ("source_file_list", str(source_path.resolve())),
        ("source_file_list_sha256", hashlib.sha256(source_bytes).hexdigest()),
        ("source_slices", str(len(split.train_files) + len(split.validation_files))),
        ("train_slices", str(len(split.train_files))),
        ("validation_slices", str(len(split.validation_files))),
        ("train_file_list_sha256", hashlib.sha256(train_bytes).hexdigest()),
        ("validation_file_list_sha256", hashlib.sha256(validation_bytes).hexdigest()),
        ("shared_series_count", str(len(split.shared_series))),
        ("shared_series", ",".join(split.shared_series)),
        ("final_generalization_evidence", "false"),
        ("warning", AUTHOR_DIAGNOSTIC_WARNING),
    ]
    (output_dir / "split_config.tsv").write_text(
        "key\tvalue\n" + "".join(f"{key}\t{value}\n" for key, value in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = ArgumentParser(
        description=(
            "Create a deterministic author-style random 90/10 slice split. "
            "This is diagnostic paper-alignment evidence, not final generalization evidence."
        )
    )
    parser.add_argument("--source-file-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        split = create_author_diagnostic_split(read_file_list(args.source_file_list), args.seed)
        write_split(args.output_dir, args.source_file_list, split)
    except (OSError, ValueError) as exc:
        print(f"Invalid author diagnostic split: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote deterministic 90/10 diagnostic split to: {args.output_dir}")
    print(f"WARNING: {AUTHOR_DIAGNOSTIC_WARNING}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
