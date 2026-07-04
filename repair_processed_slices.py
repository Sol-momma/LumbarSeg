from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def validate_npz(path: Path, expected_key: str) -> str | None:
    try:
        with np.load(path) as sample:
            # Accessing the array forces NumPy to read the archive member.
            # A file can exist with a valid-looking name after an interrupted
            # preprocessing run, but still fail here with EOFError or a zip
            # error because WSL was stopped mid-write.
            if expected_key not in sample:
                return f"missing key: {expected_key}"
            _ = sample[expected_key].shape
    except Exception as exc:
        return str(exc)
    return None


def remove_file(path: Path, dry_run: bool) -> None:
    if not dry_run:
        path.unlink(missing_ok=True)


def main() -> None:
    parser = ArgumentParser(description="Repair interrupted preprocessing outputs by removing corrupt slice pairs.")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    image_dir = args.output_root / "images"
    mask_dir = args.output_root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise SystemExit(f"Expected images/ and masks/ under {args.output_root}")

    image_files = {path.name: path for path in image_dir.glob("*.npz")}
    mask_files = {path.name: path for path in mask_dir.glob("*.npz")}

    image_only = set(image_files) - set(mask_files)
    mask_only = set(mask_files) - set(image_files)
    corrupt_pairs = []

    for filename in sorted(set(image_files) & set(mask_files)):
        image_error = validate_npz(image_files[filename], "image")
        mask_error = validate_npz(mask_files[filename], "mask")
        if image_error or mask_error:
            corrupt_pairs.append((filename, image_error, mask_error))

    for filename in sorted(image_only):
        remove_file(image_files[filename], args.dry_run)
    for filename in sorted(mask_only):
        remove_file(mask_files[filename], args.dry_run)
    for filename, _, _ in corrupt_pairs:
        # Training assumes image/mask filenames are paired. If either side is
        # corrupt, deleting both sides lets the next preprocessing run recreate
        # a consistent pair without mixing old and newly generated slices.
        remove_file(image_files[filename], args.dry_run)
        remove_file(mask_files[filename], args.dry_run)

    print(f"Output root: {args.output_root}")
    print(f"Images without masks: {len(image_only)}")
    print(f"Masks without images: {len(mask_only)}")
    print(f"Corrupt pairs: {len(corrupt_pairs)}")
    print("Dry run only. No files were removed." if args.dry_run else "Repair complete.")

    for filename in sorted(image_only):
        print(f"image_only,{filename}")
    for filename in sorted(mask_only):
        print(f"mask_only,{filename}")
    for filename, image_error, mask_error in corrupt_pairs:
        print(f"corrupt,{filename},image={image_error},mask={mask_error}")


if __name__ == "__main__":
    main()
