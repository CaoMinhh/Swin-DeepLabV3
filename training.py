"""
Train SwinDeepLabV3 on a train/test split.

Expected data layout (default directory: ./dataset):
  dataset/X_train.npy  (N, H, W, 3)  uint8 or float32
  dataset/Y_train.npy  (N, H, W, 1) or (N, H, W)  uint8 (0-1 / 0-255) or float32
  dataset/X_test.npy   (M, H, W, 3)
  dataset/Y_test.npy   (M, H, W, 1) or (M, H, W)

Usage:
  python training.py
  python training.py --dataset dataset --loss dice
  python training.py --val-split 0.1        # hold out 10% of train for validation

TensorBoard:
  tensorboard --logdir outputs/tensorboard
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import training_utils  # noqa: E402 — GPU env before TF import

import tensorflow as tf  # noqa: E402

from swin_dl import SwinDeepLabV3, DiceLoss, DiceScore, IoUScore, get_default_model_kwargs
from swin_dl.losses import BCEDiceLoss, FocalLoss
from training_utils import (
    append_csv_row,
    evaluate_model,
    load_split,
    make_callbacks,
    make_logger,
    make_tf_dataset,
    save_history,
    setup_gpu,
)

LOSS_BUILDERS = {
    "bce": lambda: tf.keras.losses.BinaryCrossentropy(name="bce_loss"),
    "focal": lambda: FocalLoss(alpha=0.25, gamma=2.0, name="focal_loss"),
    "dice": lambda: DiceLoss(smooth=1e-6, name="dice_loss"),
    "bce_dice": lambda: BCEDiceLoss(lambda_bce=0.5, smooth=1e-6, name="bce_dice_loss"),
}


def build_model(loss_name: str, model_kwargs: dict, strategy, use_jit: bool):
    """Build and compile SwinDeepLabV3 inside distribute strategy scope."""
    with strategy.scope():
        model = SwinDeepLabV3(**model_kwargs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=LOSS_BUILDERS[loss_name](),
            metrics=[
                DiceScore(smooth=1e-6, name="dice_score"),
                IoUScore(smooth=1e-6, name="iou_score"),
            ],
            jit_compile=use_jit,
        )
        dummy = np.zeros((1,) + tuple(model_kwargs["input_shape"]), dtype=np.float32)
        _ = model(dummy)
    return model


def split_validation(x_train, y_train, val_split: float, seed: int = 42):
    """Hold out a fraction of the training set for validation."""
    n = len(x_train)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_split))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return (
        x_train[train_idx],
        y_train[train_idx],
        x_train[val_idx],
        y_train[val_idx],
    )


def main():
    parser = argparse.ArgumentParser(description="Train SwinDeepLabV3 (train/test split)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="Directory with X_train/Y_train/X_test/Y_test.npy (default: dataset)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="dice",
        choices=list(LOSS_BUILDERS.keys()),
        help="Loss function (default: dice)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20, help="EarlyStopping patience")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help="Fraction of train held out for validation. If 0, the test set is "
        "used for validation monitoring (default: 0.0)",
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mp", action="store_true", help="Disable BF16 mixed precision")
    parser.add_argument("--no-xla", action="store_true", help="Disable XLA JIT")
    args = parser.parse_args()
    args.xla = not args.no_xla

    strategy = setup_gpu(mixed_precision=not args.no_mp, xla=args.xla)
    model_kwargs = get_default_model_kwargs()

    dataset_name = os.path.basename(os.path.normpath(args.dataset))
    model_path = os.path.join(args.output_dir, "models", f"{dataset_name}_{args.loss}.keras")
    log_path = os.path.join(args.output_dir, "logs", f"train_{dataset_name}_{args.loss}.log")
    csv_path = os.path.join(args.output_dir, "results", f"train_{dataset_name}_{args.loss}.csv")
    tb_log_dir = os.path.join(args.output_dir, "tensorboard", dataset_name, args.loss)
    log = make_logger(log_path)

    log.info("=" * 60)
    log.info(
        f"Training SwinDeepLabV3 — dataset={dataset_name} loss={args.loss} "
        f"epochs={args.epochs} batch={args.batch_size} "
        f"BF16={not args.no_mp} XLA={args.xla}"
    )
    log.info("=" * 60)

    x_train, y_train, x_test, y_test = load_split(args.dataset)
    log.info(f"  Train: X {x_train.shape}  Y {y_train.shape}")
    log.info(f"  Test:  X {x_test.shape}  Y {y_test.shape}")

    if args.val_split > 0.0:
        x_tr, y_tr, x_val, y_val = split_validation(
            x_train, y_train, args.val_split, seed=args.seed
        )
        log.info(
            f"  Val split={args.val_split}: train={len(x_tr)} val={len(x_val)} "
            f"(test held out for final evaluation)"
        )
    else:
        x_tr, y_tr = x_train, y_train
        x_val, y_val = x_test, y_test
        log.info("  No val split: monitoring on the test set")

    use_jit = args.xla and not isinstance(strategy, tf.distribute.MirroredStrategy)
    model = build_model(args.loss, model_kwargs, strategy, use_jit)

    train_ds = make_tf_dataset(x_tr, y_tr, args.batch_size, shuffle=True)
    val_ds = make_tf_dataset(x_val, y_val, args.batch_size, shuffle=False)

    callbacks = make_callbacks(model_path, patience=args.patience, tb_log_dir=tb_log_dir)

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.time() - t0

    stopped_epoch = len(history.history["loss"])
    history_path = os.path.join(
        args.output_dir, "models", f"{dataset_name}_{args.loss}_history.json"
    )
    save_history(history, history_path)
    log.info(f"  Saved history → {history_path}")
    log.info(f"  Best model   → {model_path}  (epoch {stopped_epoch}/{args.epochs})")
    log.info(f"  TensorBoard  → {tb_log_dir}")

    import swin_dl  # noqa: F401

    best = tf.keras.models.load_model(model_path, custom_objects=swin_dl.get_custom_objects())
    res = evaluate_model(best, x_test, y_test, batch_size=args.batch_size)
    row = {
        "dataset": dataset_name,
        "loss": args.loss,
        "dice_pct": round(res["dice"] * 100, 4),
        "iou_pct": round(res["iou"] * 100, 4),
        "stopped_epoch": stopped_epoch,
        "time_s": round(elapsed, 1),
    }
    append_csv_row(csv_path, row)

    log.info("")
    log.info("=" * 60)
    log.info("Test set evaluation (best checkpoint)")
    log.info("=" * 60)
    log.info(f"  Dice: {row['dice_pct']:.2f}%   IoU: {row['iou_pct']:.2f}%")
    log.info(f"  Time: {elapsed:.0f}s")
    log.info(f"  Results → {csv_path}")

    summary_path = os.path.join(
        args.output_dir, "results", f"train_{dataset_name}_{args.loss}_summary.json"
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    log.info(f"  Summary → {summary_path}")


if __name__ == "__main__":
    main()
