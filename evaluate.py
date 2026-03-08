"""
Evaluation script: load trained model, predict on test set, compute Dice and IoU.
Reuses load_dataset from train and DiceScore/IoUScore from swin_dl.
Usage: python evaluate.py [--model PATH] [--data-dir DIR] [--binary-y] [--save-pred PATH]
"""
import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

import swin_dl  # noqa: F401 - register custom layers/loss/metrics for load_model
from swin_dl import DiceScore, IoUScore
from train import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate Swin-DeepLabV3 on test set")
    parser.add_argument("--model", type=str, default="weight/swin_deeplab_v3_trained.keras", help="Path to .keras model")
    parser.add_argument("--data-dir", type=str, default="dataset_chuan", help="Directory with X_test.npy, Y_test.npy")
    parser.add_argument("--binary-y", action="store_true", help="Use if training used --binary-y")
    parser.add_argument("--save-pred", type=str, default="", help="If set, save predictions to this .npy path")
    parser.add_argument("--batch-size", type=int, default=8, help="Prediction batch size")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"Error: model file not found: {args.model}")
        return

    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)

    print(f"Loading test data from: {args.data_dir}")
    _, _, x_test, y_test = load_dataset(args.data_dir, binary_y=args.binary_y)
    print(f"Test: X {x_test.shape}, Y {y_test.shape}")

    print("Predicting...")
    y_pred = model.predict(x_test, batch_size=args.batch_size, verbose=1)

    # Global Dice and IoU using swin_dl metrics
    dice_metric = DiceScore(smooth=1e-6)
    iou_metric = IoUScore(smooth=1e-6)
    dice_metric.update_state(y_test, y_pred)
    iou_metric.update_state(y_test, y_pred)
    dice = float(dice_metric.result())
    iou = float(iou_metric.result())
    print(f"\n--- Test metrics ---")
    print(f"  Dice score: {dice:.6f}")
    print(f"  IoU score:  {iou:.6f}")

    # Per-sample Dice / IoU (reuse same metric class)
    dices, ious = [], []
    for i in range(len(y_test)):
        dice_metric.reset_state()
        iou_metric.reset_state()
        dice_metric.update_state(y_test[i : i + 1], y_pred[i : i + 1])
        iou_metric.update_state(y_test[i : i + 1], y_pred[i : i + 1])
        dices.append(float(dice_metric.result()))
        ious.append(float(iou_metric.result()))
    print(f"  Dice per sample: mean={np.mean(dices):.6f}, std={np.std(dices):.6f}")
    print(f"  IoU  per sample: mean={np.mean(ious):.6f}, std={np.std(ious):.6f}")

    if args.save_pred:
        np.save(args.save_pred, y_pred)
        print(f"\nPredictions saved to: {args.save_pred}")


if __name__ == "__main__":
    main()
