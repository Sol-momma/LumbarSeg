"""Regenerate only an experiment's selected train or validation slices."""

from argparse import ArgumentParser
from pathlib import Path

from spine_baseline.preprocessing import extract_slices, split_train_val


def read_file_list(path: Path) -> list[str]:
    files = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not files:
        raise ValueError(f"No slice names found in {path}")
    if len(files) != len(set(files)):
        raise ValueError(f"Duplicate slice names found in {path}")
    return files


def select_split(data_root: Path, files: list[str], split: str) -> list[str]:
    if split == "all":
        return files
    train_files, val_files, unmatched = split_train_val(data_root, files)
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} slice names do not match the overview CSV; first={unmatched[:5]}"
        )
    return train_files if split == "train" else val_files


def main() -> None:
    parser = ArgumentParser(
        description="Regenerate only slices named by a saved filtered_files.txt without refiltering it."
    )
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--file_list", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "all"], default="validation")
    parser.add_argument("--target_height", type=int, default=512)
    parser.add_argument("--target_width", type=int, default=640)
    parser.add_argument("--force_reprocess", action="store_true")
    parser.add_argument(
        "--orientation_mode",
        choices=("legacy", "metadata", "manifest"),
        default="legacy",
    )
    parser.add_argument("--orientation_manifest", type=Path, default=None)
    args = parser.parse_args()

    source_files = read_file_list(args.file_list)
    selected_files = select_split(args.data_root, source_files, args.split)
    if not selected_files:
        raise ValueError(f"No {args.split} slices were selected")

    stats = extract_slices(
        data_root=args.data_root,
        output_root=args.output_root,
        target_height=args.target_height,
        target_width=args.target_width,
        force=args.force_reprocess,
        selected_files=set(selected_files),
        orientation_mode=args.orientation_mode,
        orientation_manifest=args.orientation_manifest,
    )

    image_names = {path.name for path in (args.output_root / "images").glob("*.npz")}
    mask_names = {path.name for path in (args.output_root / "masks").glob("*.npz")}
    expected = set(selected_files)
    missing = expected - (image_names & mask_names)
    unexpected = (image_names | mask_names) - expected
    if stats["errors"] or missing or unexpected:
        raise RuntimeError(
            "Selected-slice regeneration did not match the saved experiment list: "
            f"errors={stats['errors']}, missing={len(missing)}, unexpected={len(unexpected)}"
        )

    # This exact count check is the safety boundary: downstream analysis must
    # never silently use a different validation population than the trained run.
    print(f"Regenerated and verified {len(expected)} {args.split} image/mask pairs")
    print(f"Source volumes read: {stats['files_processed']}")
    print(f"Output root: {args.output_root}")


if __name__ == "__main__":
    main()
