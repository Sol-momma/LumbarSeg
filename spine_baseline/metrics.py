import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import ndimage
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from spine_baseline.constants import CLASS_NAMES
from spine_baseline.dataset import load_sample, load_slice_spacing


OVERLAP_METRIC_NAMES = ("dice", "iou", "precision", "recall", "f1")
_SMOOTH = 1e-7


def _get_series_id(slice_filename: str) -> str:
    """Parse the preprocessing filename without importing its heavy I/O stack.

    ``preprocessing.py`` imports OpenCV and SimpleITK. Evaluation itself should
    remain importable in lightweight analysis environments, so this stable file
    naming rule is repeated locally instead of adding those runtime dependencies.
    """
    return slice_filename.removesuffix(".npz").rsplit("_s", 1)[0]


def mean_iou(num_classes: int):
    def metric(y_true, y_pred):
        y_pred_argmax = tf.argmax(y_pred, axis=-1)
        y_true_argmax = tf.argmax(y_true, axis=-1)

        iou_sum = 0.0
        for class_id in range(num_classes):
            pred_class = tf.cast(tf.equal(y_pred_argmax, class_id), tf.float32)
            true_class = tf.cast(tf.equal(y_true_argmax, class_id), tf.float32)
            intersection = tf.reduce_sum(pred_class * true_class)
            union = tf.reduce_sum(pred_class) + tf.reduce_sum(true_class) - intersection
            iou_sum += (intersection + 1e-7) / (union + 1e-7)
        return iou_sum / num_classes

    metric.__name__ = "mean_iou"
    return metric


def dice_coefficient(num_classes: int):
    def metric(y_true, y_pred):
        y_pred_argmax = tf.argmax(y_pred, axis=-1)
        y_true_argmax = tf.argmax(y_true, axis=-1)

        dice_sum = 0.0
        for class_id in range(num_classes):
            pred_class = tf.cast(tf.equal(y_pred_argmax, class_id), tf.float32)
            true_class = tf.cast(tf.equal(y_true_argmax, class_id), tf.float32)
            intersection = tf.reduce_sum(pred_class * true_class)
            dice_sum += (2.0 * intersection + 1e-7) / (
                tf.reduce_sum(pred_class) + tf.reduce_sum(true_class) + 1e-7
            )
        return dice_sum / num_classes

    metric.__name__ = "dice_coefficient"
    return metric


def surface_mask(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return np.logical_xor(mask, eroded)


def directed_surface_distances(source: np.ndarray, target: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    source_surface = surface_mask(source)
    target_surface = surface_mask(target)
    if not np.any(source_surface) and not np.any(target_surface):
        return np.array([0.0], dtype=np.float32)
    if not np.any(source_surface) or not np.any(target_surface):
        return np.array([np.inf], dtype=np.float32)
    distance_map = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    return distance_map[source_surface]


def average_surface_distance(pred_mask: np.ndarray, true_mask: np.ndarray, spacing: np.ndarray) -> float:
    pred_to_true = directed_surface_distances(pred_mask, true_mask, spacing)
    true_to_pred = directed_surface_distances(true_mask, pred_mask, spacing)
    distances = np.concatenate([pred_to_true, true_to_pred])
    if np.isinf(distances).any():
        return float("inf")
    return float(np.mean(distances))


def normalized_surface_dice(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    spacing: np.ndarray,
    tolerance: float,
) -> float:
    pred_to_true = directed_surface_distances(pred_mask, true_mask, spacing)
    true_to_pred = directed_surface_distances(true_mask, pred_mask, spacing)
    distances = np.concatenate([pred_to_true, true_to_pred])
    if np.isinf(distances).any():
        return 0.0
    return float(np.mean(distances <= tolerance))


def _confusion_matrix(true_class: np.ndarray, pred_class: np.ndarray, num_classes: int) -> np.ndarray:
    """Build a compact count matrix without retaining another full-size mask copy."""
    encoded = true_class.astype(np.int64) * num_classes + pred_class.astype(np.int64)
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def _overlap_scores(confusion_matrices: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate per-class overlap scores for one or more confusion matrices.

    Dice and IoU intentionally use the same smoothing as the historical
    ``evaluate_classwise`` implementation. In particular, a class absent from
    both prediction and truth scores 1.0 for Dice/IoU, while precision, recall,
    and F1 keep sklearn's previous ``zero_division=0`` behavior.
    """
    matrices = np.asarray(confusion_matrices, dtype=np.float64)
    if matrices.ndim == 2:
        matrices = matrices[np.newaxis, ...]

    true_positive = np.diagonal(matrices, axis1=1, axis2=2)
    true_count = matrices.sum(axis=2)
    pred_count = matrices.sum(axis=1)
    false_positive = pred_count - true_positive
    false_negative = true_count - true_positive

    dice_denominator = 2.0 * true_positive + false_positive + false_negative
    iou_denominator = true_positive + false_positive + false_negative
    dice = (2.0 * true_positive + _SMOOTH) / (dice_denominator + _SMOOTH)
    iou = (true_positive + _SMOOTH) / (iou_denominator + _SMOOTH)
    precision = np.divide(
        true_positive,
        pred_count,
        out=np.zeros_like(true_positive),
        where=pred_count != 0,
    )
    recall = np.divide(
        true_positive,
        true_count,
        out=np.zeros_like(true_positive),
        where=true_count != 0,
    )
    f1_denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        f1_denominator,
        out=np.zeros_like(true_positive),
        where=f1_denominator != 0,
    )
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate_overlap_metrics(
    slice_confusions: list[np.ndarray],
    series_ids: list[str],
    num_classes: int,
) -> pd.DataFrame:
    """Return slice-, pixel-, and series-level macro overlap summaries.

    The output is deliberately separate from ``validation_metrics.csv``. That
    file is consumed by the reproduction target checker, so adding rows or
    changing its schema would silently alter an established experiment
    contract. ``scope`` contains per-class rows plus explicit all-class and
    foreground-only means, making the background choice visible in every
    downstream comparison.
    """
    if not slice_confusions:
        raise ValueError("No slice confusion matrices provided")
    if len(slice_confusions) != len(series_ids):
        raise ValueError("slice_confusions and series_ids must have the same length")

    confusion_stack = np.stack(slice_confusions).astype(np.int64, copy=False)
    expected_shape = (num_classes, num_classes)
    if confusion_stack.shape[1:] != expected_shape:
        raise ValueError(
            f"Expected confusion matrices with shape {expected_shape}, got {confusion_stack.shape[1:]}"
        )

    series_confusions: dict[str, np.ndarray] = {}
    for series_id, confusion in zip(series_ids, confusion_stack):
        if series_id not in series_confusions:
            series_confusions[series_id] = np.zeros(expected_shape, dtype=np.int64)
        series_confusions[series_id] += confusion

    aggregation_inputs = {
        # Every slice contributes equal weight regardless of its foreground
        # area. This matches the historical Dice/IoU calculation.
        "slice_macro": confusion_stack,
        # Pool before calculating scores so every pixel contributes equally.
        "pixel_pooled": confusion_stack.sum(axis=0, keepdims=True),
        # Pool slices within a series first, then give each series equal weight.
        "series_macro": np.stack(list(series_confusions.values())),
    }

    class_names = [CLASS_NAMES[class_id] for class_id in range(num_classes)]
    rows = []
    for aggregation, matrices in aggregation_inputs.items():
        sample_scores = _overlap_scores(matrices)
        class_scores = {name: values.mean(axis=0) for name, values in sample_scores.items()}

        for class_id, class_name in enumerate(class_names):
            rows.append({
                "aggregation": aggregation,
                "scope": class_name,
                **{name: values[class_id] for name, values in class_scores.items()},
                "slice_count": len(slice_confusions),
                "series_count": len(series_confusions),
            })

        scope_indices = {
            "all_classes": np.arange(num_classes),
            # Class 0 is the confirmed background label in this project. Keep
            # this choice explicit instead of relying on a trailing slice.
            "foreground_classes": np.arange(1, num_classes),
        }
        for scope, indices in scope_indices.items():
            rows.append({
                "aggregation": aggregation,
                "scope": scope,
                **{name: values[indices].mean() for name, values in class_scores.items()},
                "slice_count": len(slice_confusions),
                "series_count": len(series_confusions),
            })

    return pd.DataFrame(
        rows,
        columns=[
            "aggregation",
            "scope",
            *OVERLAP_METRIC_NAMES,
            "slice_count",
            "series_count",
        ],
    )


def evaluate_classwise_with_aggregations(
    model,
    file_list: list[str],
    output_root,
    num_classes: int,
    limit: int | None = None,
    nsd_tolerance: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_files = file_list if limit is None else file_list[:limit]
    if not eval_files:
        raise ValueError("No files provided for evaluation")

    all_dice = {class_id: [] for class_id in range(num_classes)}
    all_iou = {class_id: [] for class_id in range(num_classes)}
    all_asd = {class_id: [] for class_id in range(num_classes)}
    all_nsd = {class_id: [] for class_id in range(num_classes)}
    all_preds = []
    all_trues = []
    slice_confusions = []
    series_ids = []

    for filename in tqdm(eval_files, desc="Evaluating"):
        image, mask_onehot = load_sample(filename, output_root, num_classes)
        spacing = load_slice_spacing(filename, output_root)
        pred = model.predict(image[np.newaxis, ...], verbose=0)[0]
        pred_class_2d = np.argmax(pred, axis=-1)
        true_class_2d = np.argmax(mask_onehot, axis=-1)
        pred_class = pred_class_2d.flatten()
        true_class = true_class_2d.flatten()

        all_preds.append(pred_class)
        all_trues.append(true_class)
        slice_confusions.append(_confusion_matrix(true_class, pred_class, num_classes))
        series_ids.append(_get_series_id(filename))

        for class_id in range(num_classes):
            pred_mask = pred_class == class_id
            true_mask = true_class == class_id
            intersection = np.logical_and(pred_mask, true_mask).sum()
            union = np.logical_or(pred_mask, true_mask).sum()
            all_dice[class_id].append((2.0 * intersection + 1e-7) / (pred_mask.sum() + true_mask.sum() + 1e-7))
            all_iou[class_id].append((intersection + 1e-7) / (union + 1e-7))
            pred_mask_2d = pred_class_2d == class_id
            true_mask_2d = true_class_2d == class_id
            all_asd[class_id].append(average_surface_distance(pred_mask_2d, true_mask_2d, spacing))
            all_nsd[class_id].append(
                normalized_surface_dice(pred_mask_2d, true_mask_2d, spacing, tolerance=nsd_tolerance)
            )

    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    rows = []
    for class_id in range(num_classes):
        rows.append({
            "class": CLASS_NAMES[class_id],
            "dice": np.mean(all_dice[class_id]),
            "iou": np.mean(all_iou[class_id]),
            "asd": np.mean(all_asd[class_id]),
            "nsd": np.mean(all_nsd[class_id]),
            "precision": precision_score(all_trues == class_id, all_preds == class_id, zero_division=0),
            "recall": recall_score(all_trues == class_id, all_preds == class_id, zero_division=0),
            "f1": f1_score(all_trues == class_id, all_preds == class_id, zero_division=0),
        })

    results = pd.DataFrame(rows)
    mean_row = {"class": "Mean"}
    for column in ["dice", "iou", "asd", "nsd", "precision", "recall", "f1"]:
        mean_row[column] = results[column].mean()
    classwise_results = pd.concat([results, pd.DataFrame([mean_row])], ignore_index=True)
    aggregation_results = aggregate_overlap_metrics(slice_confusions, series_ids, num_classes)
    return classwise_results, aggregation_results


def evaluate_classwise(
    model,
    file_list: list[str],
    output_root,
    num_classes: int,
    limit: int | None = None,
    nsd_tolerance: float = 1.0,
) -> pd.DataFrame:
    """Preserve the historical class-wise evaluation API and result schema."""
    classwise_results, _ = evaluate_classwise_with_aggregations(
        model,
        file_list,
        output_root,
        num_classes,
        limit=limit,
        nsd_tolerance=nsd_tolerance,
    )
    return classwise_results
