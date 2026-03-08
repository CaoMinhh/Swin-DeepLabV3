"""Swin Transformer: Window Attention and SwinBlock."""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="swin_dl")
class WindowAttention(layers.Layer):
    def __init__(self, dim, window_size, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3)
        self.proj = layers.Dense(dim)
        # Pre-compute relative position index
        coords = np.arange(window_size)
        coords_h, coords_w = np.meshgrid(coords, coords, indexing='ij')
        coords_flat = np.stack([coords_h.flatten(), coords_w.flatten()])
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.transpose(1, 2, 0)
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.rel_idx = rel.sum(-1).flatten().astype(np.int32)

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "window_size": self.window_size,
            "num_heads": self.num_heads,
        })
        return config

    def build(self, input_shape):
        n = (2 * self.window_size - 1) ** 2
        self.bias_table = self.add_weight(shape=(n, self.num_heads), initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, x):
        B = tf.shape(x)[0]
        N = self.window_size ** 2
        C = self.dim
        qkv = tf.reshape(self.qkv(x), [B, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = tf.matmul(q * self.scale, k, transpose_b=True)
        bias = tf.gather(self.bias_table, self.rel_idx)
        bias = tf.reshape(bias, [N, N, -1])
        attn = attn + tf.transpose(bias, [2, 0, 1])[tf.newaxis, :, :, :]
        attn = tf.nn.softmax(attn, axis=-1)
        x = tf.transpose(tf.matmul(attn, v), [0, 2, 1, 3])
        return self.proj(tf.reshape(x, [B, N, C]))


@tf.keras.utils.register_keras_serializable(package="swin_dl")
class SwinBlock(layers.Layer):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = tf.keras.Sequential([
            layers.Dense(dim * 4, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(dim)
        ])

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "shift_size": self.shift_size,
            "dropout_rate": self.dropout_rate,
        })
        return config

    def call(self, x):
        B = tf.shape(x)[0]
        H, W, C = x.shape[1], x.shape[2], self.dim
        shortcut = x
        x = self.norm1(x)
        if self.shift_size > 0:
            x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        # Window partition
        x = tf.reshape(x, [B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [-1, self.window_size ** 2, C])
        # Attention
        x = self.attn(x)
        # Window reverse
        x = tf.reshape(x, [B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, C])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [B, H, W, C])
        if self.shift_size > 0:
            x = tf.roll(x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        x = shortcut + x
        return x + self.mlp(self.norm2(x))
