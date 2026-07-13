"""
Run inference with a trained SwinDeepLabV3 model.

Supports a single image, a directory of images, or a .npy array.

Usage:
  python inference.py --model outputs/models/dataset_dice.keras --input image.png --output pred.png
  python inference.py --model outputs/models/dataset_dice.keras --input path/to/images/ --output outputs/predictions/
  python inference.py --model outputs/models/dataset_dice.keras --input dataset/X_test.npy --output outputs/predictions/pred.npy
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import training_utils  # noqa: E402

import tensorflow as tf  # noqa: E402

import swin_dl  # noqa: F401, E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
INPUT_SIZE = (256, 256)


def load_image(path: str) -> np.ndarray:
    """Load and preprocess a single image to (1, H, W, 3) float32 in [0, 1]."""
    img = Image.open(path).convert("RGB").resize(INPUT_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]


def load_npy(path: str) -> np.ndarray:
    """Load .npy array and normalize to float32 [0, 1] with shape (N, H, W, 3)."""
    arr = np.load(path).astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    if arr.ndim == 3:
        if arr.shape[-1] in (1, 3):
            arr = arr[np.newaxis, ...]
        else:
            arr = arr[..., np.newaxis]
    if arr.shape[-1] == 1:
        arr = np.concatenate([arr, arr, arr], axis=-1)
    return arr


def collect_image_paths(path: str) -> list[str]:
    """Return sorted image file paths from a file or directory."""
    if os.path.isfile(path):
        return [path]
    paths = []
    for root, _, files in os.walk(path):
        for name in sorted(files):
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                paths.append(os.path.join(root, name))
    return paths


def save_mask(pred: np.ndarray, path: str, threshold: float = 0.5):
    """Save a single binary mask prediction as PNG."""
    mask = (pred.squeeze() >= threshold).astype(np.uint8) * 255
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(mask).save(path)


def main():
    parser = argparse.ArgumentParser(description="Inference with SwinDeepLabV3")
    parser.add_argument("--model", type=str, required=True, help="Path to .keras checkpoint")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image, directory of images, or .npy file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .png (single image), directory (batch), or .npy (array input)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model, custom_objects=swin_dl.get_custom_objects())

    if args.input.lower().endswith(".npy"):
        x = load_npy(args.input)
        print(f"Loaded array: {x.shape}")
        y_pred = model.predict(x, batch_size=args.batch_size, verbose=1)
        if args.output.lower().endswith(".npy"):
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            np.save(args.output, y_pred)
            print(f"Saved predictions → {args.output}")
        else:
            os.makedirs(args.output, exist_ok=True)
            for i, pred in enumerate(y_pred):
                out_path = os.path.join(args.output, f"pred_{i:04d}.png")
                save_mask(pred, out_path, threshold=args.threshold)
            print(f"Saved {len(y_pred)} masks → {args.output}")
        return

    image_paths = collect_image_paths(args.input)
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {args.input}")

    if len(image_paths) == 1 and args.output.lower().endswith(".png"):
        x = load_image(image_paths[0])
        y_pred = model.predict(x, batch_size=1, verbose=0)
        save_mask(y_pred[0], args.output, threshold=args.threshold)
        print(f"Saved mask → {args.output}")
        return

    os.makedirs(args.output, exist_ok=True)
    for path in image_paths:
        x = load_image(path)
        y_pred = model.predict(x, batch_size=1, verbose=0)
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.output, f"{stem}_mask.png")
        save_mask(y_pred[0], out_path, threshold=args.threshold)
    print(f"Saved {len(image_paths)} masks → {args.output}")


if __name__ == "__main__":
    main()
