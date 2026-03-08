"""
Main script: build, compile, fit, save and test load model.
- Train: python train.py
- Dry-run (build + summary + save/load only, no training): python train.py --dry-run
Dataset: dataset/BRISC_X_train.npy, BRISC_Y_train.npy, BRISC_X_test.npy, BRISC_Y_test.npy
"""
import argparse
import json
import os

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
# Reduce TensorFlow / GPU log verbosity (set before importing tf)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=ERROR only


import numpy as np
import tensorflow as tf

# Suppress TensorFlow Python logger (info/warnings)
tf.get_logger().setLevel("ERROR")

from swin_dl import SwinDeepLabV3, DiceLoss, DiceScore, IoUScore


def load_dataset(data_dir="dataset", normalize_x=True, normalize_y=True, binary_y=False):
    """Load dataset from directory (expects .npy files). Normalize X,Y from [0,255] to [0,1]."""
    x_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "Y_train.npy"))
    x_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "Y_test.npy"))
    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    if normalize_x:
        x_train, x_test = x_train / 255.0, x_test / 255.0
    if y_train.ndim == 3:
        y_train = y_train[..., np.newaxis]
    if y_test.ndim == 3:
        y_test = y_test[..., np.newaxis]
    y_train = np.asarray(y_train, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)
    if binary_y:
        y_train = (y_train >= 127).astype(np.float32)
        y_test = (y_test >= 127).astype(np.float32)
    elif normalize_y:
        y_train, y_test = y_train / 255.0, y_test / 255.0
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
    parser.add_argument("--data-dir", type=str, default="dataset_chuan", help="Directory containing BRISC_*.npy")
    parser.add_argument("--epochs", type=int, default=250, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--save", type=str, default="/weight/swin_deeplab_v3_trained.keras", help="Path to save model")
    parser.add_argument("--binary-y", action="store_true", help="Binarize Y mask (>=127 -> 1, else 0); default: normalize to [0,1]")
    parser.add_argument("--history", type=str, default="", help="Path to save training history JSON (default: <save_base>_history.json)")
    args = parser.parse_args()

    print("TensorFlow:", tf.__version__)
    model = get_model()

    # Build and print summary (input: 3 channels RGB)
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

    # Load dataset (X,Y normalized to [0,1]; use --binary-y for 0/1 mask)
    x_train, y_train, x_val, y_val = load_dataset(
        args.data_dir, binary_y=args.binary_y
    )

    # Fit
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        # validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print("\nHistory:", history.history)

    model.save(args.save)
    print(f"Saved: {args.save}")

    # Save history for later visualization (loss, metrics per epoch)
    history_path = args.history or args.save.replace(".keras", "_history.json")
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)
    print(f"History saved: {history_path}")


if __name__ == "__main__":
    main()
