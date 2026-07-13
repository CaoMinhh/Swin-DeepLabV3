"""Loss functions for segmentation; registered for .keras model loading."""
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


@tf.keras.utils.register_keras_serializable(package="swin_dl")
class FocalLoss(tf.keras.losses.Loss):
    """Binary focal loss = -alpha * (1 - p_t)^gamma * log(p_t).

    Args:
        alpha: Weight for positive class. Default 0.25.
        gamma: Focusing parameter. Default 2.0.
    """

    def __init__(self, alpha=0.25, gamma=2.0, name="focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha, "gamma": self.gamma})
        return config

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_factor = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        return alpha_factor * modulating_factor * bce


@tf.keras.utils.register_keras_serializable(package="swin_dl")
class BCEDiceLoss(tf.keras.losses.Loss):
    """Hybrid BCE + Dice loss: lambda * BCE + (1 - lambda) * Dice.

    Args:
        lambda_bce: Weight for BCE component in [0, 1]. Default 0.5.
        smooth: Smoothing factor for Dice. Default 1e-6.
    """

    def __init__(self, lambda_bce=0.5, smooth=1e-6, name="bce_dice_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.lambda_bce = float(lambda_bce)
        self.smooth = float(smooth)

    def get_config(self):
        config = super().get_config()
        config.update({"lambda_bce": self.lambda_bce, "smooth": self.smooth})
        return config

    def call(self, y_true, y_pred):
        y_true_flat = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_flat = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        bce_loss = tf.keras.losses.binary_crossentropy(y_true_flat, y_pred_flat)
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        return self.lambda_bce * bce_loss + (1.0 - self.lambda_bce) * dice_loss
