from __future__ import annotations

from argparse import ArgumentParser
from csv import DictReader, DictWriter
import hashlib
from pathlib import Path
import sys

import numpy as np


FIELDS = ["file", "image_sha256", "mask_sha256"]


def read_file_list(path: Path) -> list[str]:
    files = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not files:
        raise ValueError(f"File list is empty: {path}")
    if len(files) != len(set(files)):
        raise ValueError(f"File list contains duplicates: {path}")
    return files


def hash_npz(path: Path) -> str:
    """Hash logical NPZ contents, excluding ZIP timestamps and compression details."""
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        for key in sorted(archive.files):
            array = np.ascontiguousarray(archive[key])
            for value in (key, array.dtype.str, repr(array.shape)):
                encoded = value.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_rows(output_root: Path, files: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for filename in files:
        if Path(filename).name != filename:
            raise ValueError(f"Slice list must contain plain filenames: {filename}")
        image_path = output_root / "images" / filename
        mask_path = output_root / "masks" / filename
        if not image_path.is_file() or not mask_path.is_file():
            raise ValueError(f"Missing validation image or mask: {filename}")
        rows.append(
            {
                "file": filename,
                "image_sha256": hash_npz(image_path),
                "mask_sha256": hash_npz(mask_path),
            }
        )
    return rows


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = DictReader(handle, delimiter="\t")
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Cohort manifest columns must be: {', '.join(FIELDS)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Cohort manifest is empty: {path}")
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = ArgumentParser(description="Generate or verify logical hashes for a validation cohort.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--file-list", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", type=Path, help="Write a new cohort manifest.")
    mode.add_argument("--verify", type=Path, help="Verify against an existing cohort manifest.")
    args = parser.parse_args()

    try:
        rows = build_rows(args.output_root, read_file_list(args.file_list))
        if args.write is not None:
            write_manifest(args.write, rows)
            print(f"Wrote validation cohort manifest: {args.write}")
            return 0
        expected = read_manifest(args.verify)
        if rows != expected:
            raise ValueError("Validation image or mask contents differ from the frozen cohort")
    except (OSError, KeyError, ValueError) as exc:
        print(f"Invalid validation cohort: {exc}", file=sys.stderr)
        return 2

    print(f"Verified validation cohort: {args.verify}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
