# SPIDERデータセットの3D MRIデータを2D画像に変換する

from argparse import ArgumentParser

from arguments import add_data_args, get_data_params
from spine_baseline.preprocessing import extract_slices, filter_slices


def main() -> None:
    parser = ArgumentParser(description="Preprocess SPIDER MHA volumes for the baseline model.")
    add_data_args(parser)
    data = get_data_params(parser.parse_args())

    stats = extract_slices(
        data_root=data.data_root,
        output_root=data.output_root,
        target_height=data.target_height,
        target_width=data.target_width,
        sequences=data.sequences,
        force=data.force_reprocess,
        orientation_mode=data.orientation_mode,
        orientation_manifest=data.orientation_manifest,
    )
    kept_files, filter_stats = filter_slices(
        data.output_root,
        data.min_classes,
        data.imbalance_threshold,
        data.max_slices_per_sequence,
    )

    print("Extraction stats:", stats)
    print("Filtering stats:", filter_stats)
    print(f"Filtered files: {len(kept_files)}")


if __name__ == "__main__":
    main()
