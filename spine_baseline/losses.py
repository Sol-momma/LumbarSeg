import tensorflow as tf


def focal_loss(y_true, y_pred, gamma: float = 4.0, class_weights=None):
    """Multi-class focal loss from the paper.

    The paper includes a per-class alpha_i term but does not publish class
    weights. By default this uses equal weights; pass class_weights for
    controlled ablations.
    """
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    focal = -tf.pow(1.0 - y_pred, gamma) * y_true * tf.math.log(y_pred)
    if class_weights is not None:
        focal *= tf.reshape(tf.cast(class_weights, y_pred.dtype), (1, 1, 1, -1))
    return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))


def dice_loss(y_true, y_pred, epsilon: float = 1e-6):
    numerator = 2.0 * tf.reduce_sum(y_true * y_pred, axis=(1, 2)) + epsilon
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2)) + epsilon
    dice = numerator / denominator
    return 1.0 - tf.reduce_mean(dice)


def combined_loss(alpha: float = 0.6, gamma: float = 4.0, class_weights=None):
    def loss_fn(y_true, y_pred):
        return alpha * focal_loss(y_true, y_pred, gamma=gamma, class_weights=class_weights) + (
            1.0 - alpha
        ) * dice_loss(y_true, y_pred)

    loss_fn.__name__ = "combined_loss"
    return loss_fn
