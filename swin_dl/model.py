"""Swin-DeepLabV3: backbone Swin + decoder DeepLabV3."""
import tensorflow as tf
from tensorflow.keras import layers, Model

from .layers import SwinBlock, PatchMerging, ASPP


@tf.keras.utils.register_keras_serializable(package="swin_dl")
class SwinDeepLabV3(Model):
    def __init__(self, input_shape=(256, 256, 3), num_classes=1, embed_dim=96,
                 depths=None, num_heads=None, window_size=8,
                 dropout_rate=0.1, **kwargs):
        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        super().__init__(**kwargs)
        self._input_shape = tuple(input_shape) if isinstance(input_shape, (list, tuple)) else input_shape
        self._num_classes = num_classes
        self._embed_dim = embed_dim
        self._depths = list(depths)
        self._num_heads = list(num_heads)
        self._window_size = window_size
        self._dropout_rate = dropout_rate
        self.input_size = self._input_shape[:2]

        # Patch embedding
        self.patch_embed = tf.keras.Sequential([
            layers.Conv2D(embed_dim, 4, strides=4, padding='valid'),
            layers.LayerNormalization(epsilon=1e-5)
        ])

        # Swin stages
        self.all_blocks = []
        self.merges = []
        self.stage_config = []
        block_idx = 0
        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            dim = embed_dim * (2 ** i)
            self.stage_config.append((block_idx, depth))
            for j in range(depth):
                shift = 0 if j % 2 == 0 else window_size // 2
                self.all_blocks.append(SwinBlock(dim, heads, window_size, shift, dropout_rate=dropout_rate))
                block_idx += 1
            if i < len(depths) - 1:
                self.merges.append(PatchMerging(dim))

        # Decoder
        self.aspp = ASPP(256, dropout_rate=dropout_rate)
        self.low_proj = layers.Conv2D(48, 1, padding='same', activation='relu')
        self.decoder = tf.keras.Sequential([
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Conv2D(256, 3, padding='same', activation='relu')
        ])
        self.classifier = layers.Conv2D(num_classes, 1, activation='sigmoid')

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self._input_shape,
            "num_classes": self._num_classes,
            "embed_dim": self._embed_dim,
            "depths": self._depths,
            "num_heads": self._num_heads,
            "window_size": self._window_size,
            "dropout_rate": self._dropout_rate,
        })
        return config

    def call(self, inputs):
        x = self.patch_embed(inputs)
        features = []

        merge_idx = 0
        for stage_idx, (start, num_blocks) in enumerate(self.stage_config):
            for i in range(num_blocks):
                x = self.all_blocks[start + i](x)
            features.append(x)
            if merge_idx < len(self.merges):
                x = self.merges[merge_idx](x)
                merge_idx += 1

        # Decoder
        high_feat = self.aspp(features[-1])
        low_feat = self.low_proj(features[0])

        x = tf.image.resize(high_feat, tf.shape(low_feat)[1:3])
        x = tf.concat([x, low_feat], axis=-1)
        x = self.decoder(x)
        x = tf.image.resize(x, self.input_size)
        return self.classifier(x)
