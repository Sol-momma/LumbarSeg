from pathlib import Path

import tensorflow as tf


def spinal_canal_boundary_mask(y_true, class_id: int = 2):
    """Return a one-pixel inner/outer boundary band around the true canal.

    The inverse-square-root experiment improved recall but produced even more
    false-positive canal pixels.  A 3x3 morphological gradient lets the next
    ablation focus on where the outline is wrong without increasing the weight
    of the canal centre or distant background.
    """
    canal = tf.cast(y_true[..., class_id : class_id + 1], tf.float32)
    dilated = tf.nn.max_pool2d(canal, ksize=3, strides=1, padding="SAME")
    eroded = 1.0 - tf.nn.max_pool2d(1.0 - canal, ksize=3, strides=1, padding="SAME")
    return tf.squeeze(tf.clip_by_value(dilated - eroded, 0.0, 1.0), axis=-1)


def validate_loss_configuration(
    focal_class_weight_mode: str,
    focal_canal_boundary_boost: float,
) -> None:
    """Reject configurations that would mix two experimental changes."""
    if focal_canal_boundary_boost < 0.0:
        raise ValueError("--focal_canal_boundary_boost must be zero or greater")
    if focal_canal_boundary_boost > 0.0 and focal_class_weight_mode != "none":
        raise ValueError(
            "Boundary focal boost cannot be combined with focal class weights; "
            "compare one loss change at a time"
        )


def write_loss_config(
    path: Path,
    *,
    focal_weight: float,
    focal_gamma: float,
    focal_class_weight_mode: str,
    focal_canal_boundary_boost: float,
) -> None:
    """Persist the exact loss definition beside the resulting checkpoint."""
    values = {
        "loss_name": "combined_loss",
        "combined_focal_weight": focal_weight,
        "combined_dice_weight": 1.0 - focal_weight,
        "focal_gamma": focal_gamma,
        "focal_class_weight_mode": focal_class_weight_mode,
        "focal_canal_boundary_boost": focal_canal_boundary_boost,
        "boundary_class_id": 2,
        "boundary_definition": "ground_truth_morphological_gradient_3x3",
        "boundary_normalization": "weighted_mean",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "key\tvalue\n" + "".join(f"{key}\t{value}\n" for key, value in values.items()),
        encoding="utf-8",
    )


def focal_loss(
    y_true,
    y_pred,
    gamma: float = 4.0,
    class_weights=None,
    canal_boundary_boost: float = 0.0,
):
    """Multi-class focal loss from the paper.

    The paper includes a per-class alpha_i term but does not publish class
    weights. By default this uses equal weights; pass class_weights for
    controlled ablations.
    """
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    focal = -tf.pow(1.0 - y_pred, gamma) * y_true * tf.math.log(y_pred)
    if class_weights is not None:
        focal *= tf.reshape(tf.cast(class_weights, y_pred.dtype), (1, 1, 1, -1))
    per_pixel_focal = tf.reduce_sum(focal, axis=-1)
    if canal_boundary_boost == 0.0:
        # Keep the default path byte-for-byte equivalent to the established
        # baseline formula.  This protects all previous experiments and makes
        # the new CLI option a true one-factor ablation.
        return tf.reduce_mean(per_pixel_focal)

    boundary = tf.cast(spinal_canal_boundary_mask(y_true), per_pixel_focal.dtype)
    spatial_weights = 1.0 + tf.cast(canal_boundary_boost, per_pixel_focal.dtype) * boundary
    # Normalising by the applied weights avoids changing the global loss scale
    # (and therefore the effective learning rate) merely because a slice has a
    # longer canal outline.
    return tf.reduce_sum(per_pixel_focal * spatial_weights) / tf.reduce_sum(spatial_weights)


def dice_loss(y_true, y_pred, epsilon: float = 1e-6):
    numerator = 2.0 * tf.reduce_sum(y_true * y_pred, axis=(1, 2)) + epsilon
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2)) + epsilon
    dice = numerator / denominator
    return 1.0 - tf.reduce_mean(dice)


def combined_loss(
    alpha: float = 0.6,
    gamma: float = 4.0,
    class_weights=None,
    canal_boundary_boost: float = 0.0,
):
    def loss_fn(y_true, y_pred):
        return alpha * focal_loss(
            y_true,
            y_pred,
            gamma=gamma,
            class_weights=class_weights,
            canal_boundary_boost=canal_boundary_boost,
        ) + (1.0 - alpha) * dice_loss(y_true, y_pred)

    loss_fn.__name__ = "combined_loss"
    return loss_fn
