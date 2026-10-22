import tensorflow as tf


def boundary_map(y_true, cls: int = 3, kernel: int = 5):
    """1 on pixels within kernel//2 of the class `cls` border (dilation - erosion of its mask), else 0."""
    m = y_true[..., cls:cls + 1]
    dil = tf.nn.max_pool2d(m, kernel, 1, "SAME")
    ero = -tf.nn.max_pool2d(-m, kernel, 1, "SAME")
    return dil - ero  # (B,H,W,1)


def focal_loss(y_true, y_pred, gamma: float = 4.0, class_weights=None, pixel_weights=None):
    """Multi-class focal loss from the paper.

    The paper includes a per-class alpha_i term but does not publish class
    weights. By default this uses equal weights; pass class_weights for
    controlled ablations.
    """
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    focal = -tf.pow(1.0 - y_pred, gamma) * y_true * tf.math.log(y_pred)
    if class_weights is not None:
        focal *= tf.reshape(tf.cast(class_weights, y_pred.dtype), (1, 1, 1, -1))
    focal = tf.reduce_sum(focal, axis=-1)
    if pixel_weights is not None:
        focal *= pixel_weights[..., 0]
    return tf.reduce_mean(focal)


def dice_loss(y_true, y_pred, epsilon: float = 1e-6, class_weights=None):
    numerator = 2.0 * tf.reduce_sum(y_true * y_pred, axis=(1, 2)) + epsilon
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2)) + epsilon
    dice = numerator / denominator
    if class_weights is None:
        return 1.0 - tf.reduce_mean(dice)
    w = tf.reshape(tf.cast(class_weights, dice.dtype), (1, -1))
    return 1.0 - tf.reduce_mean(tf.reduce_sum(dice * w, axis=-1) / tf.reduce_sum(w))


def combined_loss(alpha: float = 0.6, gamma: float = 4.0, class_weights=None, dice_class_weights=None,
                  boundary_weight: float = 0.0, boundary_class: int = 3, boundary_kernel: int = 5):
    """alpha*focal + (1-alpha)*Dice. class_weights scale focal per class; dice_class_weights weight the
    per-class Dice average. boundary_weight>0 multiplies the focal term by (1 + boundary_weight) on the
    border band of class `boundary_class` (IVD = 3). All defaults = original behaviour."""
    def loss_fn(y_true, y_pred):
        pw = None
        if boundary_weight:
            pw = 1.0 + boundary_weight * boundary_map(tf.cast(y_true, y_pred.dtype), boundary_class, boundary_kernel)
        return alpha * focal_loss(y_true, y_pred, gamma=gamma, class_weights=class_weights, pixel_weights=pw) + (
            1.0 - alpha
        ) * dice_loss(y_true, y_pred, class_weights=dice_class_weights)

    loss_fn.__name__ = "combined_loss"
    return loss_fn
