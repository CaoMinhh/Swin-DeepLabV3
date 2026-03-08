# Swin-DeepLabV3

Semantic segmentation with **Swin Transformer** backbone and **DeepLabV3-style** decoder (ASPP). TensorFlow/Keras implementation with Dice loss and Dice/IoU metrics.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy

```bash
pip install tensorflow numpy
```

## Dataset

Place your data as NumPy arrays in a directory (e.g. `dataset_chuan/`):

- `X_train.npy` — shape `(N, H, W, 3)` or `(N, H, W, 1)`, dtype uint8 or float32
- `Y_train.npy` — shape `(N, H, W, 1)` or `(N, H, W)`, masks (uint8 0–255 or float32)
- `X_test.npy`, `Y_test.npy` — same format for test set

The script normalizes X and Y to [0, 1] by default (divide by 255). Use `--binary-y` if masks should be binarized (≥127 → 1).

## Project structure

```
.
├── swin_dl/                 # Model package
│   ├── __init__.py
│   ├── config.py            # Default model config
│   ├── model.py             # SwinDeepLabV3
│   ├── losses.py            # DiceLoss
│   ├── metrics.py           # DiceScore, IoUScore
│   └── layers/
│       ├── attention.py     # WindowAttention, SwinBlock
│       ├── merging.py       # PatchMerging
│       └── aspp.py          # ASPP
├── train.py                 # Training script
├── evaluate.py              # Evaluation (load model, predict, Dice/IoU)
├── analyze_dataset.py       # Dataset analysis (RGB vs 1ch, binary vs [0,1])
└── README.md
```

## Training

```bash
# Train (default: dataset_chuan, 50 epochs, save model + history JSON)
python train.py

# Options
python train.py --data-dir dataset_chuan --epochs 100 --batch-size 8
python train.py --binary-y                          # Binarize masks
python train.py --save my_model.keras --history my_history.json
python train.py --dry-run                           # Build + save/load test only, no training
```

Outputs:

- Trained model: `swin_deeplab_v3_trained.keras` (or `--save`)
- Training history: `<save_base>_history.json` (for plotting loss/metrics)

## Evaluation

```bash
# Evaluate on test set (Dice, IoU)
python evaluate.py --model swin_deeplab_v3_trained.keras --data-dir dataset_chuan

# Save predictions
python evaluate.py --save-pred predictions.npy
python evaluate.py --binary-y   # If training used --binary-y
```

## Dataset analysis

Check whether X is real RGB or 1-channel replicated, and whether Y is binary or continuous:

```bash
python analyze_dataset.py dataset_chuan
```

## Loading a saved model

Custom layers/losses/metrics are registered under the `swin_dl` package. Import it before loading:

```python
import swin_dl  # Registers custom objects
model = tf.keras.models.load_model("swin_deeplab_v3_trained.keras")
pred = model.predict(x)
```

Or pass custom objects explicitly:

```python
from swin_dl import get_custom_objects
model = tf.keras.models.load_model("model.keras", custom_objects=get_custom_objects())
```

## Model config

Default input shape is `(256, 256, 3)`. For 1-channel input use `(256, 256, 1)`. Config and defaults are in `swin_dl/config.py`; override when building the model in code.

## Citation

If you use this code in your research, please cite:

```bib
@article{swin_deeplabv3,
  title   = {Swin-DeepLabV3: Enhanced Semantic Segmentation Through Global-Local Feature Fusion Using Swin Transformer and Atrous Spatial Pyramid Pooling},
  author  = {Tran Cao Minh and Ha Minh Tan and Nguyen Huynh Thong and Kien Cao-Van and Si Duy Truong and Thi Ngoc My Truong and Dinh Thang Nguyen and Tuan Anh Huynh},
  journal = {},
  year    = {},
  volume  = {},
  pages   = {}
}
```
