import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from spine_baseline.constants import CLASS_NAMES
from spine_baseline.dataset import load_sample


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


def evaluate_classwise(model, file_list: list[str], output_root, num_classes: int, limit: int | None = None) -> pd.DataFrame:
    eval_files = file_list if limit is None else file_list[:limit]
    if not eval_files:
        raise ValueError("No files provided for evaluation")

    all_dice = {class_id: [] for class_id in range(num_classes)}
    all_iou = {class_id: [] for class_id in range(num_classes)}
    all_preds = []
    all_trues = []

    for filename in tqdm(eval_files, desc="Evaluating"):
        image, mask_onehot = load_sample(filename, output_root, num_classes)
        pred = model.predict(image[np.newaxis, ...], verbose=0)[0]
        pred_class = np.argmax(pred, axis=-1).flatten()
        true_class = np.argmax(mask_onehot, axis=-1).flatten()

        all_preds.append(pred_class)
        all_trues.append(true_class)

        for class_id in range(num_classes):
            pred_mask = pred_class == class_id
            true_mask = true_class == class_id
            intersection = np.logical_and(pred_mask, true_mask).sum()
            union = np.logical_or(pred_mask, true_mask).sum()
            all_dice[class_id].append((2.0 * intersection + 1e-7) / (pred_mask.sum() + true_mask.sum() + 1e-7))
            all_iou[class_id].append((intersection + 1e-7) / (union + 1e-7))

    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    rows = []
    for class_id in range(num_classes):
        rows.append({
            "class": CLASS_NAMES[class_id],
            "dice": np.mean(all_dice[class_id]),
            "iou": np.mean(all_iou[class_id]),
            "precision": precision_score(all_trues == class_id, all_preds == class_id, zero_division=0),
            "recall": recall_score(all_trues == class_id, all_preds == class_id, zero_division=0),
            "f1": f1_score(all_trues == class_id, all_preds == class_id, zero_division=0),
        })

    results = pd.DataFrame(rows)
    mean_row = {"class": "Mean"}
    for column in ["dice", "iou", "precision", "recall", "f1"]:
        mean_row[column] = results[column].mean()
    return pd.concat([results, pd.DataFrame([mean_row])], ignore_index=True)
