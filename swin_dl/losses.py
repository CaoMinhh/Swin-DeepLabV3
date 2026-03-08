"""Dice loss for segmentation; registered for .keras model loading."""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="swin_dl")
class DiceLoss(tf.keras.losses.Loss):
    """Dice loss = 1 - Dice coefficient. For binary segmentation (sigmoid output)."""

    def __init__(self, smooth=1e-6, name="dice_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = float(smooth)

    def get_config(self):
        config = super().get_config()
        config.update({"smooth": self.smooth})
        return config

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice
