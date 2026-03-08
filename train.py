"""
Main script: build, compile, fit, save and test load model.
- Train: python train.py
- Dry-run (build + summary + save/load only, no training): python train.py --dry-run
Dataset: dataset/BRISC_X_train.npy, BRISC_Y_train.npy, BRISC_X_test.npy, BRISC_Y_test.npy
"""
import argparse
import os
import numpy as np
import tensorflow as tf
from swin_dl import SwinDeepLabV3, DiceLoss, DiceScore, IoUScore


def load_brisc_dataset(data_dir="dataset"):
    """Load BRISC dataset from directory (expects .npy files)."""
    x_train = np.load(os.path.join(data_dir, "BRISC_X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "BRISC_Y_train.npy"))
    x_test = np.load(os.path.join(data_dir, "BRISC_X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "BRISC_Y_test.npy"))
    # Ensure float32 and mask has trailing channel (H, W, 1)
    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    if y_train.ndim == 3:
        y_train = y_train[..., np.newaxis]
    if y_test.ndim == 3:
        y_test = y_test[..., np.newaxis]
    y_train = np.asarray(y_train, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)
    print(f"Train: X {x_train.shape}, Y {y_train.shape} | Test: X {x_test.shape}, Y {y_test.shape}")
    return x_train, y_train, x_test, y_test


def get_model():
    """Build and compile the model."""
    model = SwinDeepLabV3(
        input_shape=(256, 256, 3),
        num_classes=1,
        embed_dim=64,
        depths=[2, 2, 3, 2],
        num_heads=[2, 4, 8, 16],
        window_size=8,
        dropout_rate=0.2,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=DiceLoss(smooth=1e-6),
        metrics=[DiceScore(smooth=1e-6), IoUScore(smooth=1e-6)],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train or dry-run Swin-DeepLabV3")
    parser.add_argument("--dry-run", action="store_true", help="Build, summary, save/load only; no fit")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Directory containing BRISC_*.npy")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--save", type=str, default="swin_deeplab_v3_trained.keras", help="Path to save model")
    args = parser.parse_args()

    print("TensorFlow:", tf.__version__)
    model = get_model()

    # Build and print summary
    dummy = np.zeros((1, 256, 256, 3), dtype=np.float32)
    _ = model(dummy)
    model.summary()
    n_params = sum(tf.reduce_prod(w.shape).numpy() for w in model.trainable_weights)
    print(f"\nTotal trainable parameters: {n_params:,}")

    if args.dry_run:
        # Dry-run: save -> load -> predict
        path = "/weight/swin_deeplab_v3.keras"
        model.save(path)
        print(f"\nSaved: {path}")
        import swin_dl  # noqa: F401
        loaded = tf.keras.models.load_model(path)
        out = loaded.predict(dummy, verbose=0)
        print(f"Load & predict OK, output shape: {out.shape}")
        return

    # Load BRISC dataset
    x_train, y_train, x_val, y_val = load_brisc_dataset(args.data_dir)

    # Fit
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print("\nHistory:", history.history)

    model.save(args.save)
    print(f"Saved: {args.save}")


if __name__ == "__main__":
    main()
