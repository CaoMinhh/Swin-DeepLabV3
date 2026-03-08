"""
Default configuration for Swin-DeepLabV3.
Import and modify before creating the model if needed.
"""
# Default model config
DEFAULT_INPUT_SHAPE = (256, 256, 1)
DEFAULT_NUM_CLASSES = 1
DEFAULT_EMBED_DIM = 64
DEFAULT_DEPTHS = [2, 2, 3, 2]
DEFAULT_NUM_HEADS = [2, 4, 8, 16]
DEFAULT_WINDOW_SIZE = 8
DEFAULT_DROPOUT_RATE = 0.2

def get_default_model_kwargs():
    """Return default keyword arguments for SwinDeepLabV3 constructor."""
    return dict(
        input_shape=DEFAULT_INPUT_SHAPE,
        num_classes=DEFAULT_NUM_CLASSES,
        embed_dim=DEFAULT_EMBED_DIM,
        depths=DEFAULT_DEPTHS,
        num_heads=DEFAULT_NUM_HEADS,
        window_size=DEFAULT_WINDOW_SIZE,
        dropout_rate=DEFAULT_DROPOUT_RATE,
    )
