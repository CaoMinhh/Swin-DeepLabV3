"""
Swin-DeepLabV3: breast ultrasound segmentation with Swin Transformer encoder
and DeepLabV3 decoder (ASPP + skip connection).

Import this package before loading a .keras checkpoint so custom layers,
losses, and metrics are registered.
"""
from .model import SwinDeepLabV3
from .layers import WindowAttention, SwinBlock, PatchMerging, ASPP
from .losses import DiceLoss, FocalLoss, BCEDiceLoss
from .metrics import DiceScore, IoUScore
from .config import get_default_model_kwargs


def get_custom_objects():
    """Return custom_objects dict for tf.keras.models.load_model()."""
    return {
        "SwinDeepLabV3": SwinDeepLabV3,
        "WindowAttention": WindowAttention,
        "SwinBlock": SwinBlock,
        "PatchMerging": PatchMerging,
        "ASPP": ASPP,
        "DiceLoss": DiceLoss,
        "FocalLoss": FocalLoss,
        "BCEDiceLoss": BCEDiceLoss,
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
    "FocalLoss",
    "BCEDiceLoss",
    "DiceScore",
    "IoUScore",
    "get_default_model_kwargs",
    "get_custom_objects",
]
