from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from arguments import add_data_args, get_data_params
from spine_baseline.filtering import dominant_foreground_fraction


@dataclass(frozen=True)
class FilterConfig:
    name: str
    min_classes: int
    imbalance_threshold: float
    max_slices_per_sequence: int | None


DEFAULT_CONFIGS = [
    FilterConfig("paper_default_4cls_055_cap1000", 4, 0.55, 1000),
    FilterConfig("relaxed_4cls_090_nocap", 4, 0.90, None),
    FilterConfig("relaxed_3cls_090_nocap", 3, 0.90, None),
    FilterConfig("no_imbalance_4cls_nocap", 4, 1.00, None),
    FilterConfig("no_imbalance_3cls_nocap", 3, 1.00, None),
]


def classify_sequence(filename: str) -> str:
    name = filename.lower()
    if "t2_space" in name or "t2_sag_space" in name or "space" in name:
        return "T2_SPACE"
    if "t2" in name:
        return "T2"
    if "t1" in name:
        return "T1"
    return "Unknown"


def parse_sequences(sequences: str | None) -> set[str] | None:
    if not sequences:
        return None
    return {item.strip().upper() for item in sequences.split(",") if item.strip()}


def class_fractions(mask: np.ndarray, num_classes: int = 4) -> dict[int, float]:
    counts = np.bincount(mask.astype(np.int64).ravel(), minlength=num_classes)
    total = counts.sum()
    if total == 0:
        return {class_id: 0.0 for class_id in range(num_classes)}
    return {class_id: float(counts[class_id] / total) for class_id in range(num_classes)}


def get_series_id(slice_filename: str) -> str:
    return slice_filename.removesuffix(".npz").rsplit("_s", 1)[0]


def split_train_val(data_root: Path, kept_files: list[str]) -> tuple[list[str], list[str], list[str]]:
    overview = load_overview(data_root)
    train_ids = set(overview.loc[overview["subset"] == "training", "new_file_name"])
    val_ids = set(overview.loc[overview["subset"] == "validation", "new_file_name"])
    train_files = [name for name in kept_files if get_series_id(name) in train_ids]
    val_files = [name for name in kept_files if get_series_id(name) in val_ids]
    unmatched = [name for name in kept_files if get_series_id(name) not in train_ids and get_series_id(name) not in val_ids]
    return train_files, val_files, unmatched


def metadata_value(sample, key: str, fallback: str) -> str:
    if key not in sample:
        return fallback
    value = sample[key]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def parse_filter_configs(configs: list[str] | None) -> list[FilterConfig]:
    if not configs:
        return DEFAULT_CONFIGS

    parsed = []
    for raw_config in configs:
        parts = raw_config.split(":")
        if len(parts) != 4:
            raise ValueError(
                "Filter configs must use name:min_classes:imbalance_threshold:max_slices_per_sequence. "
                "Use 0 for no cap."
            )
        name, min_classes, threshold, cap = parts
        cap_value = int(cap)
        parsed.append(
            FilterConfig(
                name=name,
                min_classes=int(min_classes),
                imbalance_threshold=float(threshold),
                max_slices_per_sequence=cap_value if cap_value > 0 else None,
            )
        )
    return parsed


def load_overview(data_root: Path) -> pd.DataFrame:
    csv_path = data_root / "SPIDER Lumbar Spine Segmentation Overview.csv"
    overview = pd.read_csv(csv_path)
    overview["new_file_name"] = overview["new_file_name"].astype(str)
    overview["sequence"] = overview["new_file_name"].map(classify_sequence)
    return overview


def audit_raw_volumes(data_root: Path, sequences: str | None) -> pd.DataFrame:
    image_dir = data_root / "images"
    mask_dir = data_root / "masks"
    overview = load_overview(data_root)
    allowed_sequences = parse_sequences(sequences)

    rows = []
    for image_path in sorted(image_dir.glob("*.mha")):
        sequence = classify_sequence(image_path.name)
        if allowed_sequences is not None and sequence not in allowed_sequences:
            continue
        series_id = image_path.stem
        subset_values = overview.loc[overview["new_file_name"] == series_id, "subset"].tolist()
        rows.append(
            {
                "series_id": series_id,
                "sequence": sequence,
                "has_mask": (mask_dir / image_path.name).exists(),
                "subset": subset_values[0] if subset_values else "unmatched",
            }
        )

    if not rows:
        return pd.DataFrame(columns=["sequence", "subset", "series_count", "mask_count"])

    raw = pd.DataFrame(rows)
    return (
        raw.groupby(["sequence", "subset"], dropna=False)
        .agg(series_count=("series_id", "count"), mask_count=("has_mask", "sum"))
        .reset_index()
        .sort_values(["sequence", "subset"])
    )


def load_slice_stats(output_root: Path, sequences: str | None) -> pd.DataFrame:
    mask_dir = output_root / "masks"
    allowed_sequences = parse_sequences(sequences)
    rows = []

    for mask_file in tqdm(sorted(mask_dir.glob("*.npz")), desc="Auditing preprocessed slices"):
        with np.load(mask_file) as sample:
            mask = sample["mask"]
            sequence = metadata_value(sample, "sequence", classify_sequence(get_series_id(mask_file.name)))
            series_id = metadata_value(sample, "series_id", get_series_id(mask_file.name))

        if allowed_sequences is not None and sequence not in allowed_sequences:
            continue

        fractions = class_fractions(mask)
        rows.append(
            {
                "file": mask_file.name,
                "series_id": series_id,
                "sequence": sequence,
                "num_classes": int(len(np.unique(mask))),
                "max_foreground_class_fraction": dominant_foreground_fraction(mask),
                "max_class_fraction": max(fractions.values()),
                "background_fraction": fractions.get(0, 0.0),
                "vertebrae_fraction": fractions.get(1, 0.0),
                "canal_fraction": fractions.get(2, 0.0),
                "ivd_fraction": fractions.get(3, 0.0),
            }
        )

    if not rows:
        raise ValueError(f"No preprocessed mask slices found under {mask_dir}")

    return pd.DataFrame(rows)


def apply_filter_config(slice_stats: pd.DataFrame, config: FilterConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    kept_frames = []

    for sequence, sequence_rows in slice_stats.groupby("sequence", sort=True):
        sequence_rows = sequence_rows.sort_values("file").reset_index(drop=True)
        class_ok = sequence_rows["num_classes"] >= config.min_classes
        imbalance_ok = sequence_rows["max_foreground_class_fraction"] <= config.imbalance_threshold
        eligible = sequence_rows.loc[class_ok & imbalance_ok].copy()

        if config.max_slices_per_sequence is None or len(eligible) <= config.max_slices_per_sequence:
            capped = eligible
            removed_cap = 0
        else:
            selected_indices = np.linspace(0, len(eligible) - 1, config.max_slices_per_sequence, dtype=int)
            capped = eligible.iloc[selected_indices].copy()
            removed_cap = len(eligible) - len(capped)

        rows.append(
            {
                "config": config.name,
                "sequence": sequence,
                "min_classes": config.min_classes,
                "imbalance_threshold": config.imbalance_threshold,
                "max_slices_per_sequence": config.max_slices_per_sequence or 0,
                "extracted_slices": len(sequence_rows),
                "removed_class_count": int((~class_ok).sum()),
                "removed_imbalance": int((class_ok & ~imbalance_ok).sum()),
                "removed_sequence_cap": int(removed_cap),
                "kept": len(capped),
            }
        )
        kept_frames.append(capped.assign(config=config.name))

    summary = pd.DataFrame(rows)
    kept = pd.concat(kept_frames, ignore_index=True) if kept_frames else pd.DataFrame()
    return summary, kept


def add_split_counts(summary: pd.DataFrame, kept: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    rows = []
    for (config, sequence), group in kept.groupby(["config", "sequence"], sort=True):
        train_files, val_files, unmatched = split_train_val(data_root, group["file"].tolist())
        rows.append(
            {
                "config": config,
                "sequence": sequence,
                "train_slices": len(train_files),
                "validation_slices": len(val_files),
                "unmatched_slices": len(unmatched),
            }
        )
    split_counts = pd.DataFrame(rows)
    if split_counts.empty:
        summary[["train_slices", "validation_slices", "unmatched_slices"]] = 0
        return summary
    merged = summary.merge(split_counts, on=["config", "sequence"], how="left").fillna(
        {"train_slices": 0, "validation_slices": 0, "unmatched_slices": 0}
    )
    for column in ["train_slices", "validation_slices", "unmatched_slices"]:
        merged[column] = merged[column].astype(int)
    return merged


def main() -> None:
    parser = ArgumentParser(description="Audit SPIDER preprocessing and filtering against reproduction targets.")
    add_data_args(parser)
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Run extraction before auditing. Use this for a fresh all-sequence audit output directory.",
    )
    parser.add_argument(
        "--audit_output_path",
        type=Path,
        default=None,
        help="CSV path for filtering summary. Defaults to output_root/reproduction_filtering_audit.csv.",
    )
    parser.add_argument(
        "--write_slice_stats",
        action="store_true",
        help="Also write per-slice class fraction stats to output_root/reproduction_slice_stats.csv.",
    )
    parser.add_argument(
        "--filter_config",
        action="append",
        help=(
            "Custom filter config as name:min_classes:imbalance_threshold:max_slices_per_sequence. "
            "Can be repeated. Use cap 0 for no cap."
        ),
    )
    args = parser.parse_args()
    data = get_data_params(args)

    if args.extract:
        from spine_baseline.preprocessing import extract_slices

        extract_stats = extract_slices(
            data_root=data.data_root,
            output_root=data.output_root,
            target_height=data.target_height,
            target_width=data.target_width,
            sequences=data.sequences,
            force=data.force_reprocess,
            orientation_mode=data.orientation_mode,
            orientation_manifest=data.orientation_manifest,
        )
        print("Extraction stats:", extract_stats)

    raw_summary = audit_raw_volumes(data.data_root, data.sequences)
    raw_path = data.output_root / "reproduction_raw_volume_audit.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_summary.to_csv(raw_path, index=False)

    slice_stats = load_slice_stats(data.output_root, data.sequences)
    if args.write_slice_stats:
        slice_stats.to_csv(data.output_root / "reproduction_slice_stats.csv", index=False)

    configs = parse_filter_configs(args.filter_config)
    summaries = []
    kept_frames = []
    for config in configs:
        summary, kept = apply_filter_config(slice_stats, config)
        summaries.append(summary)
        kept_frames.append(kept)

    filtering_summary = pd.concat(summaries, ignore_index=True)
    kept_all = pd.concat(kept_frames, ignore_index=True) if kept_frames else pd.DataFrame()
    filtering_summary = add_split_counts(filtering_summary, kept_all, data.data_root)
    audit_path = args.audit_output_path or (data.output_root / "reproduction_filtering_audit.csv")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    filtering_summary.to_csv(audit_path, index=False)

    print("Raw volume audit:")
    print(raw_summary.to_string(index=False))
    print()
    print("Filtering audit:")
    print(filtering_summary.to_string(index=False))
    print()
    print(f"Saved raw volume audit to: {raw_path}")
    print(f"Saved filtering audit to: {audit_path}")


if __name__ == "__main__":
    main()
