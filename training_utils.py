"""
Shared training/evaluation utilities.

GPU acceleration stack (applied in setup_gpu()):
  1. Mixed Precision BF16  — Tensor Cores, ~2x throughput vs FP32, numerically
                             safer than FP16 (no inf/nan from narrow dynamic range)
  2. TF32                  — already ON by default on Ampere, leave as-is
  3. XLA JIT compilation   — fuses ops across the graph, reduces kernel launches;
                             enabled via jit_compile=True on model.compile()
  4. tf.data prefetch       — overlaps CPU preprocessing with GPU compute
  5. ModelCheckpoint        — saves best val_dice weights, not last epoch
  6. EarlyStopping          — stops when no improvement for `patience` epochs
  7. Large batch size       — 32 is the recommended default for A100 80GB;
                             adjust per GPU memory headroom

Note: Flash Attention does NOT exist in TensorFlow core. The Window Attention in
Swin operates on small 8x8 windows (64 tokens), so memory is not the bottleneck;
XLA fused kernels + BF16 Tensor Cores give equivalent benefit at this scale.
"""
import csv
import json
import logging
import os
import sys

import numpy as np
import tensorflow as tf

# ──────────────────────────────────────────────────────────────
# GPU setup
# ──────────────────────────────────────────────────────────────

def setup_gpu(mixed_precision: bool = True, xla: bool = True) -> "tf.distribute.Strategy":
    """Configure A100(s) for maximum training throughput.

    Returns a tf.distribute.Strategy:
      - MirroredStrategy  if 2+ GPUs detected (data-parallel across all GPUs)
      - OneDeviceStrategy if 1 GPU
      - OneDeviceStrategy("/cpu:0") if no GPU

    Build your model inside `with strategy.scope():` to enable multi-GPU.

    Args:
        mixed_precision: Enable BF16 mixed precision (default True).
        xla: Enable XLA JIT compilation (default True).
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] {len(gpus)} device(s): {[g.name for g in gpus]}")
    else:
        print("[GPU] No GPU found, running on CPU")

    if mixed_precision:
        # BF16 preferred over FP16 on A100:
        #   - Same 10-bit mantissa → same Tensor Core throughput (~2x FP32)
        #   - 8-bit exponent (same as FP32) → no loss scaling needed, no NaN risk
        #   NOTE: tf.image.resize always outputs float32 — cast back to x.dtype
        #         in any layer that uses resize (ASPP, decoder). Handled in
        #         aspp.py and model.py.
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        print("[GPU] Mixed precision: mixed_bfloat16")
    else:
        print("[GPU] Mixed precision: OFF (float32)")

    if xla:
        tf.config.optimizer.set_jit(True)
        print("[GPU] XLA JIT: ON (global)")

    if len(gpus) >= 2:
        strategy = tf.distribute.MirroredStrategy()
        print(f"[GPU] MirroredStrategy: {strategy.num_replicas_in_sync} replicas")
    elif len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
        print("[GPU] OneDeviceStrategy: GPU:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
        print("[GPU] OneDeviceStrategy: CPU")

    return strategy


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

def _prepare_x(x: np.ndarray) -> np.ndarray:
    """Cast to float32, scale to [0, 1] if needed, ensure 3 channels."""
    x = x.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    if x.ndim == 3:
        x = x[..., np.newaxis]
    if x.shape[-1] == 1:
        x = np.concatenate([x, x, x], axis=-1)
    return x


def _prepare_y(y: np.ndarray) -> np.ndarray:
    """Cast to float32 in [0, 1], ensure a trailing channel dim."""
    y = y.astype(np.float32)
    if y.max() > 1.0:
        y = y / 255.0
    if y.ndim == 3:
        y = y[..., np.newaxis]
    return y


def load_split(data_dir: str):
    """Load X_train/Y_train/X_test/Y_test.npy from data_dir.

    Accepts X as (N, H, W, 3) or (N, H, W) and Y as (N, H, W, 1) or (N, H, W).
    Pixel values may be uint8 [0, 255] or float32 [0, 1]; both are normalized
    to float32 [0, 1].

    Returns (x_train, y_train, x_test, y_test).
    """
    paths = {
        "X_train": os.path.join(data_dir, "X_train.npy"),
        "Y_train": os.path.join(data_dir, "Y_train.npy"),
        "X_test": os.path.join(data_dir, "X_test.npy"),
        "Y_test": os.path.join(data_dir, "Y_test.npy"),
    }
    missing = [name for name, p in paths.items() if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(f'{m}.npy' for m in missing)} in {data_dir}"
        )
    x_train = _prepare_x(np.load(paths["X_train"]))
    y_train = _prepare_y(np.load(paths["Y_train"]))
    x_test = _prepare_x(np.load(paths["X_test"]))
    y_test = _prepare_y(np.load(paths["Y_test"]))
    return x_train, y_train, x_test, y_test


def make_tf_dataset(x, y, batch_size: int, shuffle: bool = False) -> tf.data.Dataset:
    """Build a tf.data.Dataset with prefetch for GPU pipeline overlap."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x), seed=42)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ──────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────

def make_callbacks(model_path: str, patience: int = 20, tb_log_dir: str = None):
    """Return [ModelCheckpoint(best only), EarlyStopping, TensorBoard].

    ModelCheckpoint saves the .keras file ONLY when val_dice_score improves,
    so the saved file always contains the best weights — ready for inference.

    Args:
        model_path:  Path for the .keras checkpoint.
        patience:    EarlyStopping patience (epochs without improvement).
        tb_log_dir:  TensorBoard log directory. If None, TensorBoard is skipped.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="val_dice_score",
        mode="max",
        save_best_only=True,
        save_weights_only=False,   # full .keras for inference
        verbose=0,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_dice_score",
        mode="max",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
    callbacks = [checkpoint, early_stop]

    if tb_log_dir is not None:
        os.makedirs(tb_log_dir, exist_ok=True)
        tb = tf.keras.callbacks.TensorBoard(
            log_dir=tb_log_dir,
            histogram_freq=0,        # weight histograms are expensive — keep off
            write_graph=False,       # graph is large; skip for speed
            update_freq="epoch",     # log scalars every epoch
            profile_batch=0,         # disable profiling
        )
        callbacks.append(tb)

    return callbacks


# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────

def make_logger(log_path: str) -> logging.Logger:
    """Return a logger that writes to both stdout and a .log file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger


# ──────────────────────────────────────────────────────────────
# Results / CSV
# ──────────────────────────────────────────────────────────────

def append_csv_row(csv_path: str, row: dict) -> None:
    """Append one row to CSV immediately (safe against crashes mid-run)."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_history(history, path: str) -> None:
    """Serialize Keras History to JSON."""
    serializable = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate_model(model, x_val, y_val, batch_size: int = 32) -> dict:
    """Evaluate model on a set, return dice and iou as floats."""
    val_ds = make_tf_dataset(x_val, y_val, batch_size=batch_size, shuffle=False)
    results = model.evaluate(val_ds, verbose=0, return_dict=True)
    return {
        "dice": float(results.get("dice_score", 0.0)),
        "iou": float(results.get("iou_score", 0.0)),
    }
