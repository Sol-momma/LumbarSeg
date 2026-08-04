from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import tensorflow as tf

from arguments import add_data_args, add_model_args, add_optimization_args, get_param_groups
from spine_baseline.constants import CLASS_COLORS, CLASS_NAMES
from spine_baseline.dataset import load_sample
from spine_baseline.losses import combined_loss
from spine_baseline.metrics import dice_coefficient, mean_iou
from spine_baseline.preprocessing import classify_sequence, filter_slices, get_series_id, split_train_val


WORST_CASE_METRICS = (
    "dice_mean",
    "dice_vertebrae",
    "dice_spinal_canal",
    "dice_ivds",
)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    colors = np.asarray(CLASS_COLORS, dtype=np.uint8)
    return colors[mask]


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image_rgb = np.repeat((image[..., np.newaxis] * 255.0).astype(np.uint8), 3, axis=-1)
    mask_rgb = colorize_mask(mask)
    foreground = mask > 0
    overlay = image_rgb.copy()
    overlay[foreground] = (
        (1.0 - alpha) * image_rgb[foreground].astype(np.float32)
        + alpha * mask_rgb[foreground].astype(np.float32)
    ).astype(np.uint8)
    return overlay


def dice_per_class(pred: np.ndarray, true: np.ndarray, num_classes: int) -> dict[str, float]:
    scores = {}
    for class_id in range(num_classes):
        pred_mask = pred == class_id
        true_mask = true == class_id
        denom = pred_mask.sum() + true_mask.sum()
        if denom == 0:
            score = 1.0
        else:
            score = 2.0 * np.logical_and(pred_mask, true_mask).sum() / denom
        scores[f"dice_{CLASS_NAMES[class_id].lower().replace(' ', '_')}"] = float(score)
    scores["dice_mean"] = float(np.mean(list(scores.values())))
    return scores


def choose_files(files: list[str], num_samples: int, start_index: int, strategy: str, seed: int) -> list[str]:
    if not files:
        raise ValueError("No files are available for visualization.")
    if num_samples <= 0 or num_samples >= len(files):
        return files[start_index:]

    available = files[start_index:]
    if not available:
        raise ValueError(f"start_index {start_index} is outside the file list of length {len(files)}.")

    if strategy == "first":
        return available[:num_samples]
    if strategy == "random":
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(available), size=min(num_samples, len(available)), replace=False))
        return [available[index] for index in indices]

    indices = np.linspace(0, len(available) - 1, min(num_samples, len(available)), dtype=int)
    return [available[index] for index in indices]


def rank_rows(
    rows: list[dict],
    metric: str,
    num_samples: int,
    sort_order: str = "ascending",
) -> list[dict]:
    """Return a deterministic best/worst slice ranking.

    Filename is the tie breaker so repeated runs produce the same qualitative
    sample even when several slices have identical rounded Dice values.
    """
    reverse = sort_order == "descending"
    ranked = sorted(rows, key=lambda row: (row[metric], row["file"]), reverse=reverse)
    return ranked if num_samples <= 0 else ranked[:num_samples]


def score_prediction(
    filename: str,
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    num_classes: int,
) -> dict[str, float | int | str]:
    """Build a compact diagnostic row without retaining the prediction mask."""
    scores: dict[str, float | int | str] = {
        "file": filename,
        "series_id": get_series_id(filename),
        "sequence": classify_sequence(get_series_id(filename)),
    }
    try:
        scores["slice_index"] = int(Path(filename).stem.rsplit("_s", 1)[1])
    except (IndexError, ValueError):
        scores["slice_index"] = -1
    scores.update(dice_per_class(pred_mask, true_mask, num_classes))

    # Area ratios help distinguish a missed small structure from a spatially
    # misplaced prediction during failure analysis. Pixels are kept as exact
    # counts because every processed slice has the same 512x640 canvas.
    for class_id, class_name in enumerate(CLASS_NAMES):
        key = class_name.lower().replace(" ", "_")
        true_pixels = int(np.count_nonzero(true_mask == class_id))
        pred_pixels = int(np.count_nonzero(pred_mask == class_id))
        scores[f"true_pixels_{key}"] = true_pixels
        scores[f"pred_pixels_{key}"] = pred_pixels
        scores[f"pred_to_true_area_{key}"] = (
            float(pred_pixels / true_pixels) if true_pixels else float("nan")
        )
    return scores


def score_files(
    model,
    files: list[str],
    output_root: Path,
    num_classes: int,
    batch_size: int,
) -> list[dict]:
    """Score every requested slice in batches while keeping memory bounded."""
    rows = []
    safe_batch_size = max(1, batch_size)
    for start in range(0, len(files), safe_batch_size):
        batch_files = files[start : start + safe_batch_size]
        loaded = [load_sample(filename, output_root, num_classes) for filename in batch_files]
        images = np.stack([sample[0] for sample in loaded])
        true_masks = np.argmax(np.stack([sample[1] for sample in loaded]), axis=-1).astype(np.uint8)
        predictions = model.predict(images, verbose=0)
        pred_masks = np.argmax(predictions, axis=-1).astype(np.uint8)
        rows.extend(
            score_prediction(filename, pred_mask, true_mask, num_classes)
            for filename, pred_mask, true_mask in zip(batch_files, pred_masks, true_masks)
        )
        print(f"Scored {min(start + safe_batch_size, len(files))}/{len(files)} slices", flush=True)
    return rows


def plot_prediction(
    image: np.ndarray,
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
    filename: str,
    scores: dict[str, float],
    output_path: Path,
) -> None:
    panels = [
        ("Input", np.repeat((image[..., np.newaxis] * 255.0).astype(np.uint8), 3, axis=-1)),
        ("Ground truth", colorize_mask(true_mask)),
        ("Prediction", colorize_mask(pred_mask)),
        ("Overlay", overlay_mask(image, pred_mask)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), constrained_layout=True)
    title = f"{filename} | mean Dice {scores['dice_mean']:.3f}"
    fig.suptitle(title, fontsize=12)
    for axis, (panel_title, panel_image) in zip(axes, panels):
        axis.imshow(panel_image)
        axis.set_title(panel_title)
        axis.axis("off")
    legend_handles = [
        Patch(
            facecolor=np.asarray(CLASS_COLORS[class_id], dtype=np.float32) / 255.0,
            edgecolor="white",
            label=CLASS_NAMES[class_id],
        )
        for class_id in range(1, len(CLASS_NAMES))
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles), frameon=False)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def render_rows(
    model,
    rows: list[dict],
    output_root: Path,
    output_dir: Path,
    num_classes: int,
) -> list[dict]:
    """Render only selected slices, avoiding hundreds of unnecessary PNGs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for index, row in enumerate(rows):
        filename = str(row["file"])
        image, mask_onehot = load_sample(filename, output_root, num_classes)
        pred = model.predict(image[np.newaxis, ...], verbose=0)[0]
        pred_mask = np.argmax(pred, axis=-1).astype(np.uint8)
        true_mask = np.argmax(mask_onehot, axis=-1).astype(np.uint8)
        scores = dice_per_class(pred_mask, true_mask, num_classes)
        output_path = output_dir / f"{index:03d}_{Path(filename).stem}.png"
        plot_prediction(image[..., 0], true_mask, pred_mask, filename, scores, output_path)
        rendered.append({**row, "png": str(output_path.relative_to(output_dir.parent))})
    return rendered


def resolve_split(split: str, data_root: Path, kept_files: list[str]) -> list[str]:
    train_files, val_files, _ = split_train_val(data_root, kept_files)
    if split == "train":
        return train_files
    if split == "validation":
        return val_files
    return kept_files


def main() -> None:
    parser = ArgumentParser(description="Render qualitative prediction panels for a trained baseline model.")
    add_data_args(parser)
    add_model_args(parser)
    add_optimization_args(parser)
    parser.add_argument("--model_path", required=True, help="Path to a trained .keras model.")
    parser.add_argument(
        "--prediction_output_dir",
        type=Path,
        default=None,
        help="Directory for rendered PNGs. Defaults to output_root/predictions.",
    )
    parser.add_argument(
        "--filtered_file_list",
        type=Path,
        default=None,
        help="Optional saved filtered_files.txt. Using it avoids recalculating and overwriting the experiment selection.",
    )
    parser.add_argument("--split", choices=["train", "validation", "all"], default="validation")
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--selection", choices=["even", "first", "random", "worst"], default="even")
    parser.add_argument(
        "--sort_by",
        choices=WORST_CASE_METRICS,
        default="dice_mean",
        help="Metric used when --selection worst is selected.",
    )
    parser.add_argument("--sort_order", choices=["ascending", "descending"], default="ascending")
    parser.add_argument(
        "--worst_per_metric",
        type=int,
        default=0,
        help="Render this many worst slices for mean Dice and each foreground class in one run.",
    )
    args = parser.parse_args()
    data, model_params, opt = get_param_groups(args)

    output_dir = args.prediction_output_dir or (data.output_root / "predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.filtered_file_list is not None:
        kept_files = [
            line.strip()
            for line in args.filtered_file_list.read_text().splitlines()
            if line.strip()
        ]
    else:
        kept_files, _ = filter_slices(
            data.output_root,
            data.min_classes,
            data.imbalance_threshold,
            data.max_slices_per_sequence,
        )
    split_files = resolve_split(args.split, data.data_root, kept_files)
    missing_processed = [
        name for name in split_files
        if not (data.output_root / "images" / name).exists()
        or not (data.output_root / "masks" / name).exists()
    ]
    if missing_processed:
        raise FileNotFoundError(
            f"Processed image/mask pairs are missing for {len(missing_processed)} selected slices; "
            f"first={missing_processed[:5]}"
        )

    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={
            "loss_fn": combined_loss(alpha=opt.focal_weight, gamma=opt.focal_gamma),
            "mean_iou": mean_iou(model_params.num_classes),
            "dice_coefficient": dice_coefficient(model_params.num_classes),
        },
        compile=False,
    )

    score_all = args.selection == "worst" or args.worst_per_metric > 0
    files_to_score = split_files if score_all else choose_files(
        split_files,
        args.num_samples,
        args.start_index,
        args.selection,
        opt.seed,
    )
    rows = score_files(
        model,
        files_to_score,
        data.output_root,
        model_params.num_classes,
        opt.batch_size,
    )
    summary_path = output_dir / "prediction_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    rendered_rows = []
    if args.worst_per_metric > 0:
        for metric in WORST_CASE_METRICS:
            ranked = rank_rows(rows, metric, args.worst_per_metric, "ascending")
            category_dir = output_dir / f"worst_{metric}"
            category_rows = render_rows(
                model,
                ranked,
                data.output_root,
                category_dir,
                model_params.num_classes,
            )
            rendered_rows.extend(
                {"rank_metric": metric, "rank": rank, **row}
                for rank, row in enumerate(category_rows, start=1)
            )
    else:
        selected_rows = (
            rank_rows(rows, args.sort_by, args.num_samples, args.sort_order)
            if args.selection == "worst"
            else rows
        )
        rendered_rows = render_rows(
            model,
            selected_rows,
            data.output_root,
            output_dir,
            model_params.num_classes,
        )

    worst_summary_path = output_dir / "worst_case_summary.csv"
    pd.DataFrame(rendered_rows).to_csv(worst_summary_path, index=False)
    print(f"Rendered {len(rendered_rows)} prediction panels to: {output_dir}")
    print(f"Saved per-slice Dice summary to: {summary_path}")
    print(f"Saved rendered-case ranking to: {worst_summary_path}")


if __name__ == "__main__":
    main()
