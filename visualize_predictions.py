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
from spine_baseline.preprocessing import filter_slices, split_train_val


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
    parser.add_argument("--split", choices=["train", "validation", "all"], default="validation")
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--selection", choices=["even", "first", "random"], default="even")
    args = parser.parse_args()
    data, model_params, opt = get_param_groups(args)

    output_dir = args.prediction_output_dir or (data.output_root / "predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    kept_files, _ = filter_slices(
        data.output_root,
        data.min_classes,
        data.imbalance_threshold,
        data.max_slices_per_sequence,
    )
    split_files = resolve_split(args.split, data.data_root, kept_files)
    selected_files = choose_files(split_files, args.num_samples, args.start_index, args.selection, opt.seed)

    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={
            "loss_fn": combined_loss(alpha=opt.focal_weight, gamma=opt.focal_gamma),
            "mean_iou": mean_iou(model_params.num_classes),
            "dice_coefficient": dice_coefficient(model_params.num_classes),
        },
        compile=False,
    )

    rows = []
    for index, filename in enumerate(selected_files):
        image, mask_onehot = load_sample(filename, data.output_root, model_params.num_classes)
        pred = model.predict(image[np.newaxis, ...], verbose=0)[0]
        pred_mask = np.argmax(pred, axis=-1).astype(np.uint8)
        true_mask = np.argmax(mask_onehot, axis=-1).astype(np.uint8)
        image_2d = image[..., 0]
        scores = dice_per_class(pred_mask, true_mask, model_params.num_classes)
        output_path = output_dir / f"{index:03d}_{Path(filename).stem}.png"
        plot_prediction(image_2d, true_mask, pred_mask, filename, scores, output_path)
        rows.append({"file": filename, "png": output_path.name, **scores})

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "prediction_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Rendered {len(selected_files)} prediction panels to: {output_dir}")
    print(f"Saved per-slice Dice summary to: {summary_path}")


if __name__ == "__main__":
    main()
