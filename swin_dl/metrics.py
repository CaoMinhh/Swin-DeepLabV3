"""Dice score and IoU score; registered for .keras model loading."""
import tensorflow as tf
from tensorflow.keras import metrics as keras_metrics


@tf.keras.utils.register_keras_serializable(package="swin_dl")
class DiceScore(keras_metrics.Metric):
    """Dice coefficient (F1 for segmentation): 2*|X∩Y| / (|X|+|Y|)."""

    def __init__(self, smooth=1e-6, name="dice_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = float(smooth)
        self.dice_sum = self.add_weight(name="dice_sum", initializer="zeros", dtype=tf.float32)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({"smooth": self.smooth})
        return config

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        self.dice_sum.assign_add(dice)
        self.count.assign_add(1.0)

    def result(self):
        return self.dice_sum / tf.maximum(self.count, 1.0)

    def reset_state(self):
        self.dice_sum.assign(0.0)
        self.count.assign(0.0)


@tf.keras.utils.register_keras_serializable(package="swin_dl")
class IoUScore(keras_metrics.Metric):
    """IoU (Jaccard): |X∩Y| / |X∪Y|."""

    def __init__(self, smooth=1e-6, name="iou_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = float(smooth)
        self.iou_sum = self.add_weight(name="iou_sum", initializer="zeros", dtype=tf.float32)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({"smooth": self.smooth})
        return config

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        self.iou_sum.assign_add(iou)
        self.count.assign_add(1.0)

    def result(self):
        return self.iou_sum / tf.maximum(self.count, 1.0)

    def reset_state(self):
        self.iou_sum.assign(0.0)
        self.count.assign(0.0)
