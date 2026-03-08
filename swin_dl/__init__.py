"""
Swin-DeepLabV3: segmentation with Swin Transformer backbone and DeepLabV3 decoder.
Import this package before loading a .keras model so all custom layers/losses/metrics are registered.
"""
from .model import SwinDeepLabV3
from .layers import WindowAttention, SwinBlock, PatchMerging, ASPP
from .losses import DiceLoss
from .metrics import DiceScore, IoUScore


def get_custom_objects():
    """Return dict of custom_objects for loading model: load_model(..., custom_objects=swin_dl.get_custom_objects())."""
    return {
        "SwinDeepLabV3": SwinDeepLabV3,
        "WindowAttention": WindowAttention,
        "SwinBlock": SwinBlock,
        "PatchMerging": PatchMerging,
        "ASPP": ASPP,
        "DiceLoss": DiceLoss,
        "DiceScore": DiceScore,
        "IoUScore": IoUScore,
    }


__all__ = [
    "SwinDeepLabV3",
    "WindowAttention",
    "SwinBlock",
    "PatchMerging",
    "ASPP",
    "DiceLoss",
    "DiceScore",
    "IoUScore",
    "get_custom_objects",
]
