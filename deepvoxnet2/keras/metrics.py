import tensorflow as tf
from pymirc.metrics.tf_metrics import generalized_dice_coeff


def binary_dice_score(y_true, y_pred, sample_weight=None, **kwargs):
    if "threshold" not in kwargs:
        kwargs["threshold"] = 0.5

    return generalized_dice_coeff(y_true, y_pred, **kwargs)


def binary_volume_difference(y_true, y_pred, sample_weight=None, threshold=None):
    if threshold is not None:
        y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)

    return tf.math.reduce_sum(y_pred, axis=(1, 2, 3, 4)) - tf.math.reduce_sum(y_true, axis=(1, 2, 3, 4))
