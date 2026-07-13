"""Dice score and IoU score; registered for .keras model loading."""
import tensorflow as tf
from tensorflow.keras import metrics as keras_metrics


@tf.keras.utils.register_keras_serializable(package="swin_dl")
class DiceScore(keras_metrics.Metric):
    """Dice coefficient (F1 for segmentation): 2*|X∩Y| / (|X|+|Y|).

    Accumulates global intersection/union across all batches (streaming),
    instead of averaging per-batch ratios, so empty masks no longer score 1.0.
    """

    def __init__(self, smooth=1e-6, threshold=0.5, name="dice_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = float(smooth)
        self.threshold = float(threshold)
        self.intersection = self.add_weight(name="intersection", initializer="zeros", dtype=tf.float32)
        self.total = self.add_weight(name="total", initializer="zeros", dtype=tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({"smooth": self.smooth, "threshold": self.threshold})
        return config

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]) > self.threshold, tf.float32)
        self.intersection.assign_add(tf.reduce_sum(y_true * y_pred))
        self.total.assign_add(tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))

    def result(self):
        return (2.0 * self.intersection + self.smooth) / (self.total + self.smooth)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.total.assign(0.0)


@tf.keras.utils.register_keras_serializable(package="swin_dl")
class IoUScore(keras_metrics.Metric):
    """IoU (Jaccard): |X∩Y| / |X∪Y|.

    Accumulates global intersection/union across all batches (streaming),
    instead of averaging per-batch ratios, so empty masks no longer score 1.0.
    """

    def __init__(self, smooth=1e-6, threshold=0.5, name="iou_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = float(smooth)
        self.threshold = float(threshold)
        self.intersection = self.add_weight(name="intersection", initializer="zeros", dtype=tf.float32)
        self.union = self.add_weight(name="union", initializer="zeros", dtype=tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({"smooth": self.smooth, "threshold": self.threshold})
        return config

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]) > self.threshold, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        self.intersection.assign_add(intersection)
        self.union.assign_add(tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection)

    def result(self):
        return (self.intersection + self.smooth) / (self.union + self.smooth)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)
