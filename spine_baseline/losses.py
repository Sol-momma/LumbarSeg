import tensorflow as tf


def focal_loss(y_true, y_pred, gamma: float = 4.0):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    focal = -y_true * tf.pow(1.0 - y_pred, gamma) * tf.math.log(y_pred)
    return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))


def dice_loss(y_true, y_pred, epsilon: float = 1e-6):
    numerator = 2.0 * tf.reduce_sum(y_true * y_pred, axis=(1, 2)) + epsilon
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2)) + epsilon
    dice = numerator / denominator
    return 1.0 - tf.reduce_mean(dice)


def combined_loss(alpha: float = 0.6, gamma: float = 4.0):
    def loss_fn(y_true, y_pred):
        return alpha * focal_loss(y_true, y_pred, gamma=gamma) + (1.0 - alpha) * dice_loss(y_true, y_pred)

    return loss_fn
