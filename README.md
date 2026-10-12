# Wafer Map Defect Classification on WM-811K

A small CNN, trained from scratch in PyTorch, that reads a semiconductor wafer map and names its spatial defect pattern. The whole pipeline is one script (`wafer_map_pipeline.py`) that runs top to bottom.

## Summary

Wafer maps are grids of passed and failed dies. The spatial pattern of the failures (a ring at the edge, a cluster in the center, a scratch) points to the process step that caused them. This project classifies each map into one of nine classes, even though 85% of the labeled wafers have no defect pattern at all.

On the held-out test set (17,295 wafers) the model reaches 95.5% accuracy, 0.81 macro-F1 and 0.89 mean average precision. The "always predict none" baseline has 85.2% accuracy but only 0.10 macro-F1, so macro-F1 is the metric to read.

## Data

[WM-811K wafer map dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) (`LSWMD.pkl`).

- 811,457 wafer maps, of which 172,950 (21.3%) carry an expert label.
- Each map is a grid: `0` = no die, `1` = die passed, `2` = die failed.
- 346 distinct raw map shapes.
- Nine classes: Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none.

| Class | Count | Share |
|---|---:|---:|
| none | 147,431 | 85.24% |
| Edge-Ring | 9,680 | 5.60% |
| Edge-Loc | 5,189 | 3.00% |
| Center | 4,294 | 2.48% |
| Loc | 3,593 | 2.08% |
| Scratch | 1,193 | 0.69% |
| Random | 866 | 0.50% |
| Donut | 555 | 0.32% |
| Near-full | 149 | 0.09% |

## Method

1. **Load and clean labels.** `failureType` is a nested array left over from a MATLAB export, so it is unwrapped to a plain string. Unlabeled wafers are dropped.
2. **Resize.** Every map is resized to 48x48 with nearest-neighbor sampling, so pixel values stay exactly `{0, 1, 2}` and no fractional die states are invented. Values are scaled to `{0, 0.5, 1}`.
3. **Split.** Stratified 80/10/10 into 138,360 train, 17,295 validation and 17,295 test wafers.
4. **Leakage check.** Wafers come in lots of up to 25. The script reports how many lots appear in both train and test (64.5% here), as a diagnostic.
5. **Class weights.** Inverse-frequency weights in the loss, from 0.13 for none up to 129 for Near-full.
6. **Model.** Three conv blocks (Conv, BatchNorm, ReLU, MaxPool) with 16, 32 and 64 channels, adaptive average pooling to 3x3, then a 128-unit dense layer with dropout 0.3. 98,537 trainable parameters.
7. **Training.** Adam, learning rate 1e-3, batch size 32, 25 epochs. The checkpoint with the best validation macro-F1 is kept, because accuracy would reward always predicting none.
8. **Evaluation.** The test set is used once, after the checkpoint is chosen.

## Results

Best validation macro-F1 was 0.828 (epoch 19). Test results:

| Metric | Value |
|---|---:|
| Accuracy | 0.955 |
| Macro-F1 | 0.809 |
| Macro recall | 0.861 |
| Weighted F1 | 0.957 |
| Mean average precision | 0.892 |
| Baseline "always none": accuracy / macro-F1 | 0.852 / 0.102 |

Per-class test results:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Edge-Ring | 0.937 | 0.978 | 0.957 |
| Near-full | 0.875 | 0.933 | 0.903 |
| Center | 0.824 | 0.935 | 0.876 |
| Donut | 0.780 | 0.821 | 0.800 |
| Random | 0.778 | 0.814 | 0.795 |
| Edge-Loc | 0.676 | 0.863 | 0.758 |
| Loc | 0.594 | 0.769 | 0.670 |
| Scratch | 0.460 | 0.672 | 0.546 |
| none | 0.993 | 0.965 | 0.979 |

What the outputs show:

- The largest confusions come from none being predicted as a defect: Edge-Loc (167 wafers), Loc (138), Scratch (78) and Center (73). Class weighting trades precision on rare classes for recall.
- Scratch and Loc are the weakest classes. Both are small, localized patterns that are easy to confuse with each other and with none.
- Mean confidence is 0.970 on correct predictions and 0.728 on wrong ones, so low confidence is a usable signal for flagging wafers for review.
- Validation macro-F1 swings widely between epochs (min 0.449, max 0.828), which is why the best checkpoint is kept.

## Limitations

- **Lot leakage.** The split is by wafer, not by lot. 64.5% of lots appear in both train and test, so test scores are optimistic for lots the model has never seen. A lot-grouped split would be the next step.
- **Run-to-run variance.** GPU non-determinism and noisy validation F1 move results by a few points. Quote a range, not one number.
- **Resolution.** Resizing to 48x48 loses detail on large maps, which likely hurts thin patterns such as Scratch.
- **Labels.** Only 21% of wafers are labeled, and label noise is possible. The most confident mistakes are saved as a figure for inspection.

## Usage

```bash
pip install numpy pandas matplotlib scikit-learn torch
export WM811K_DIR=/path/to/wm811k-wafer-map   # folder containing LSWMD.pkl
export OUT_DIR=outputs                        # optional, default: outputs
export EPOCHS=25                              # optional, default: 25
python wafer_map_pipeline.py | tee run_log.txt
```

The default `WM811K_DIR` points to the Kaggle input path. The script runs on CPU or GPU and uses seed 0.

## Outputs

Everything is written to `OUT_DIR`.

Tables (CSV): class counts, split counts, training history, test classification report, confusion matrix, top confusions, average precision.

Figures (PNG): class distribution, one example per class, training curves, confusion matrix (raw and row-normalized), per-class F1 vs. training size, precision-recall curves, confidence histogram, per-class metrics, random test predictions, most confident mistakes.

Other: `wafer_cnn.pt` (best checkpoint) and `summary.json` (headline numbers).

## Stack

Python, PyTorch, NumPy, pandas, scikit-learn, Matplotlib.
