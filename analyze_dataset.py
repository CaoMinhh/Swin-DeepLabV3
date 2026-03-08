"""
Analyze dataset: check X (RGB vs 1ch replicated) and Y (binary vs [0,1]).
Usage: python analyze_dataset.py [data_dir]
Default: dataset_chuan
"""
import os
import sys
import numpy as np


def find_npy_pairs(data_dir):
    """Find (X, Y) .npy pairs in data_dir."""
    files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
    pairs = []
    for stem in ["X_train", "X_test"]:
        fn = stem + ".npy"
        if fn in files:
            yname = fn.replace("X_train", "Y_train").replace("X_test", "Y_test")
            if yname in files:
                pairs.append((os.path.join(data_dir, fn), os.path.join(data_dir, yname)))
    # Fallback: any X* and Y* with same suffix (e.g. BRISC_X_train / BRISC_Y_train)
    if not pairs:
        xs = sorted([f for f in files if "X" in f and "train" in f])
        ys = sorted([f for f in files if "Y" in f and "train" in f])
        for xf, yf in zip(xs, ys):
            pairs.append((os.path.join(data_dir, xf), os.path.join(data_dir, yf)))
        xs = sorted([f for f in files if "X" in f and "test" in f])
        ys = sorted([f for f in files if "Y" in f and "test" in f])
        for xf, yf in zip(xs, ys):
            pairs.append((os.path.join(data_dir, xf), os.path.join(data_dir, yf)))
    if not pairs and files:
        pairs = [(os.path.join(data_dir, files[0]), os.path.join(data_dir, files[1]) if len(files) > 1 else None)]
    return pairs


def analyze_x(x, name="X"):
    """Check if X is real RGB or 1 channel replicated to 3."""
    print(f"\n--- {name} (inputs) ---")
    print(f"  shape: {x.shape}, dtype: {x.dtype}")
    if x.ndim != 4:
        print(f"  Expected (N,H,W,C), got ndim={x.ndim}. Skip channel check.")
        return
    n, h, w, c = x.shape
    print(f"  N={n}, H={h}, W={w}, C={c}")
    if c != 3:
        print(f"  Channels = {c} (not 3). No RGB replication check.")
        return
    # Per-channel stats
    for i in range(3):
        ch = x[..., i]
        print(f"  channel[{i}]: min={ch.min():.4f}, max={ch.max():.4f}, mean={ch.mean():.4f}, std={ch.std():.4f}")
    # Check if R==G==B (1ch replicated)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    eq_rg = np.allclose(r, g)
    eq_rb = np.allclose(r, b)
    eq_gb = np.allclose(g, b)
    if eq_rg and eq_rb and eq_gb:
        print("  => Conclusion: channels are IDENTICAL (likely 1 channel replicated to 3).")
    else:
        diff_rg = np.abs(r - g).max()
        diff_rb = np.abs(r - b).max()
        print(f"  => Conclusion: channels DIFFER (real RGB). max |R-G|={diff_rg:.4f}, |R-B|={diff_rb:.4f}")


def analyze_y(y, name="Y"):
    """Check if Y is binary (0/1) or continuous in [0,1]."""
    print(f"\n--- {name} (labels/masks) ---")
    print(f"  shape: {y.shape}, dtype: {y.dtype}")
    y_flat = y.flatten()
    uniq = np.unique(y_flat)
    print(f"  min={y_flat.min():.6f}, max={y_flat.max():.6f}, mean={y_flat.mean():.6f}")
    print(f"  number of unique values: {len(uniq)}")
    if len(uniq) <= 20:
        print(f"  unique values: {uniq}")
    else:
        print(f"  unique (first 20): {uniq[:20]}")
        print(f"  unique (last 5):   {uniq[-5:]}")
    # Binary check
    only_01 = np.all((y_flat == 0) | (y_flat == 1))
    in_01 = np.all((y_flat >= 0) & (y_flat <= 1))
    if only_01:
        print("  => Conclusion: BINARY (only 0 and 1).")
    elif in_01:
        print("  => Conclusion: continuous in [0, 1] (not strictly 0/1).")
    else:
        print("  => Conclusion: values outside [0, 1] (check normalization).")


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset_chuan"
    if not os.path.isdir(data_dir):
        print(f"Error: not a directory: {data_dir}")
        sys.exit(1)
    print(f"Data dir: {os.path.abspath(data_dir)}")
    files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
    print(f".npy files: {files}")
    for f in files:
        arr = np.load(os.path.join(data_dir, f))
        print(f"  {f}: shape={arr.shape}, dtype={arr.dtype}")
    pairs = find_npy_pairs(data_dir)
    if not pairs:
        npy_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        print(f"No X/Y pairs found. .npy files: {npy_files}")
        # Try loading first two as X, Y by convention
        if len(npy_files) >= 2:
            x_path = os.path.join(data_dir, npy_files[0])
            y_path = os.path.join(data_dir, npy_files[1])
            pairs = [(x_path, y_path)]
    for i, (x_path, y_path) in enumerate(pairs):
        label = f"pair_{i+1}" if len(pairs) > 1 else ""
        print(f"\n{'='*60}")
        print(f"Files: X={os.path.basename(x_path)}, Y={os.path.basename(y_path) if y_path else 'N/A'}")
        x = np.load(x_path)
        analyze_x(x, name=f"X {label}")
        if y_path and os.path.isfile(y_path):
            y = np.load(y_path)
            analyze_y(y, name=f"Y {label}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
