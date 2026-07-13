"""
Evaluate a trained SwinDeepLabV3 checkpoint on the test set.

Expected data layout (default directory: ./dataset):
  dataset/X_test.npy, dataset/Y_test.npy

Usage:
  python evaluation.py --model outputs/models/dataset_dice.keras
  python evaluation.py --model outputs/models/dataset_dice.keras --dataset dataset --split test
  python evaluation.py --model outputs/models/dataset_dice.keras --save-pred outputs/predictions/test_pred.npy
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import training_utils  # noqa: E402

import tensorflow as tf  # noqa: E402

import swin_dl  # noqa: F401, E402
from swin_dl import DiceScore, IoUScore
from training_utils import load_split


def compute_metrics(y_true, y_pred):
    """Compute global Dice and IoU scores."""
    dice_metric = DiceScore(smooth=1e-6)
    iou_metric = IoUScore(smooth=1e-6)
    dice_metric.update_state(y_true, y_pred)
    iou_metric.update_state(y_true, y_pred)
    return float(dice_metric.result()), float(iou_metric.result())


def per_sample_metrics(y_true, y_pred):
    """Compute per-sample Dice and IoU lists."""
    dice_metric = DiceScore(smooth=1e-6)
    iou_metric = IoUScore(smooth=1e-6)
    dices, ious = [], []
    for i in range(len(y_true)):
        dice_metric.reset_state()
        iou_metric.reset_state()
        dice_metric.update_state(y_true[i : i + 1], y_pred[i : i + 1])
        iou_metric.update_state(y_true[i : i + 1], y_pred[i : i + 1])
        dices.append(float(dice_metric.result()))
        ious.append(float(iou_metric.result()))
    return dices, ious


def main():
    parser = argparse.ArgumentParser(description="Evaluate SwinDeepLabV3")
    parser.add_argument("--model", type=str, required=True, help="Path to .keras checkpoint")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="Directory with X_train/Y_train/X_test/Y_test.npy (default: dataset)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train"],
        help="Which split to evaluate (default: test)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--save-pred",
        type=str,
        default="",
        help="If set, save raw predictions (.npy) to this path",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model, custom_objects=swin_dl.get_custom_objects())

    print(f"Loading data: {args.dataset}")
    x_train, y_train, x_test, y_test = load_split(args.dataset)
    if args.split == "train":
        x_eval, y_eval = x_train, y_train
    else:
        x_eval, y_eval = x_test, y_test
    print(f"Evaluating split='{args.split}' ({len(x_eval)} samples)")

    print("Predicting...")
    y_pred = model.predict(x_eval, batch_size=args.batch_size, verbose=1)

    dice, iou = compute_metrics(y_eval, y_pred)
    dices, ious = per_sample_metrics(y_eval, y_pred)

    print("\n--- Evaluation metrics ---")
    print(f"  Dice (global): {dice:.6f}  ({dice * 100:.2f}%)")
    print(f"  IoU  (global): {iou:.6f}  ({iou * 100:.2f}%)")
    print(f"  Dice per sample: mean={np.mean(dices):.6f}, std={np.std(dices):.6f}")
    print(f"  IoU  per sample: mean={np.mean(ious):.6f}, std={np.std(ious):.6f}")

    if args.save_pred:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_pred)), exist_ok=True)
        np.save(args.save_pred, y_pred)
        print(f"\nPredictions saved → {args.save_pred}")


if __name__ == "__main__":
    main()
