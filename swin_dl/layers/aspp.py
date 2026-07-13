"""ASPP (Atrous Spatial Pyramid Pooling) for DeepLab."""
import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="swin_dl")
class ASPP(layers.Layer):
    def __init__(self, filters=256, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.conv1 = layers.Conv2D(filters, 1, padding='same', activation='relu')
        self.conv6 = layers.Conv2D(filters, 3, padding='same', dilation_rate=6, activation='relu')
        self.conv12 = layers.Conv2D(filters, 3, padding='same', dilation_rate=12, activation='relu')
        self.conv18 = layers.Conv2D(filters, 3, padding='same', dilation_rate=18, activation='relu')
        self.pool_conv = layers.Conv2D(filters, 1, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.project = layers.Conv2D(filters, 1, padding='same', activation='relu')

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "dropout_rate": self.dropout_rate,
        })
        return config

    def call(self, x):
        size = tf.shape(x)[1:3]
        pooled = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        # Cast resize output back to x.dtype: tf.image.resize always returns float32,
        # which breaks mixed-precision (BF16) when concatenated with Conv2D outputs.
        f5 = tf.cast(tf.image.resize(self.pool_conv(pooled), size), x.dtype)
        out = tf.concat([self.conv1(x), self.conv6(x), self.conv12(x), self.conv18(x), f5], axis=-1)
        return self.project(self.dropout(out))
