# =============================================================================
# Wafer map defect classification on WM-811K
# One file, run top to bottom:  python wafer_map_pipeline.py | tee run_log.txt
#
# Question this script answers:
#   Can a small CNN, trained from scratch, read a wafer map (a grid of passed
#   and failed dies) and name the spatial defect pattern (Center, Donut,
#   Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, or none), even though
#   "none" makes up about 85% of the labeled data?
#
# Data: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map  (LSWMD.pkl)
# Each wafer map is a grid: 0 = no die here (outside the wafer),
#                           1 = die passed, 2 = die failed.
# Only about 21% of the 811,457 wafers carry an expert pattern label.
#
# Every figure is saved as a PNG and shown on screen. Every table is saved as
# a CSV. Everything lands in OUT_DIR.
# =============================================================================

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ---- config ----------------------------------------------------------------
DATA_DIR = os.environ.get("WM811K_DIR", "/kaggle/input/datasets/qingyi/wm811k-wafer-map")
OUT_DIR = os.environ.get("OUT_DIR", "outputs")
IMG_SIZE = 48          # every wafer map is resized to IMG_SIZE x IMG_SIZE
BATCH_SIZE = 32
EPOCHS = int(os.environ.get("EPOCHS", 25))
LEARNING_RATE = 1e-3
SEED = 0

CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
           "Near-full", "Random", "Scratch", "none"]   # "none" = no defect pattern
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 78)
print("CONFIG")
print("=" * 78)
print("device        :", device)
print("data dir      :", DATA_DIR)
print("output dir    :", OUT_DIR)
print("image size    :", IMG_SIZE, "x", IMG_SIZE)
print("batch size    :", BATCH_SIZE, "  epochs:", EPOCHS, "  lr:", LEARNING_RATE, "  seed:", SEED)
print("classes       :", CLASSES)

# =============================================================================
# 1. Load the pickle
# =============================================================================
print("\n" + "=" * 78)
print("1. LOAD DATA")
print("=" * 78)

pkl_candidates = glob.glob(os.path.join(DATA_DIR, "**", "*.pkl"), recursive=True)
print(f"Found {len(pkl_candidates)} .pkl file(s) under {DATA_DIR}:")
for p in pkl_candidates:
    print("  ", p)
assert len(pkl_candidates) > 0, f"No .pkl file found under {DATA_DIR}"

raw = pd.read_pickle(pkl_candidates[0])
print("\nShape  :", raw.shape)
print("Columns:", list(raw.columns))   # the dataset itself spells it 'trianTestLabel'
print("\nFirst rows (waferMap column shortened for display):")
print(raw.drop(columns=["waferMap"]).head())

# =============================================================================
# 2. Clean the labels and count what we have
# =============================================================================
# failureType is stored as a 1-element nested array (a leftover from the MATLAB
# export). unwrap_label pulls out the plain string, or None if it is empty.
print("\n" + "=" * 78)
print("2. LABELS AND CLASS BALANCE")
print("=" * 78)


def unwrap_label(value):
    while isinstance(value, (list, np.ndarray)):
        if len(value) == 0:
            return None
        value = value[0]
    return value


raw["failureType"] = raw["failureType"].apply(unwrap_label)

print("Label counts (None = unlabeled):")
print(raw["failureType"].value_counts(dropna=False))

labeled = raw[raw["failureType"].isin(CLASSES)].reset_index(drop=True)
print(f"\n{len(labeled)} of {len(raw)} wafers carry a usable label "
      f"({len(labeled) / len(raw):.1%}).")

class_counts = labeled["failureType"].value_counts().reindex(CLASSES)
class_table = pd.DataFrame({"count": class_counts,
                            "share_%": (100 * class_counts / class_counts.sum()).round(2)})
print("\nLabeled wafers per class:")
print(class_table)
class_table.to_csv(f"{OUT_DIR}/01_class_counts.csv")

fig, ax = plt.subplots(figsize=(8, 4))
class_counts.plot(kind="bar", ax=ax)
ax.set_title("Labeled wafer maps per class")
ax.set_ylabel("count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_class_distribution.png", dpi=150)
plt.show()

# =============================================================================
# 3. One real example per class
# =============================================================================
print("\n" + "=" * 78)
print("3. ONE EXAMPLE PER CLASS (raw size, before resizing)")
print("=" * 78)

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for ax, cls in zip(axes.flat, CLASSES):
    example = labeled.loc[labeled["failureType"] == cls, "waferMap"].iloc[0]
    ax.imshow(example, cmap="gray")
    ax.set_title(f"{cls}  {np.asarray(example).shape}")
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_example_per_class.png", dpi=150)
plt.show()

shapes = labeled["waferMap"].apply(lambda w: np.asarray(w).shape)
print("Number of distinct raw wafer map shapes:", shapes.nunique())
print("Most common shapes:")
print(shapes.value_counts().head(5))

# =============================================================================
# 4. Resize every wafer map to a fixed size
# =============================================================================
# A CNN needs a fixed input shape. Nearest-neighbour sampling keeps the values
# exactly {0, 1, 2}. A normal interpolating resize would invent fractional
# values that do not correspond to any real die state.
print("\n" + "=" * 78)
print("4. RESIZE TO FIXED SIZE")
print("=" * 78)


def resize_wafer(wafer, size=IMG_SIZE):
    wafer = np.asarray(wafer)
    h, w = wafer.shape
    row_idx = np.clip((np.arange(size) * h / size).astype(int), 0, h - 1)
    col_idx = np.clip((np.arange(size) * w / size).astype(int), 0, w - 1)
    return wafer[np.ix_(row_idx, col_idx)]


X = np.stack([resize_wafer(w) for w in labeled["waferMap"]]).astype(np.float32)
X = X / 2.0                       # {0, 1, 2} -> {0, 0.5, 1}
y = labeled["failureType"].map(CLASS_TO_IDX).to_numpy()

print("X shape:", X.shape, " y shape:", y.shape)
print("Unique pixel values after resize:", np.unique(X))

# =============================================================================
# 5. Train / validation / test split
# =============================================================================
# 80 / 10 / 10, stratified so every split keeps the class proportions. Without
# that, a rare class like Near-full could land with no test examples by chance.
print("\n" + "=" * 78)
print("5. SPLIT, LEAKAGE CHECK, CLASS WEIGHTS")
print("=" * 78)

idx = np.arange(len(y))
idx_train, idx_temp = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=SEED,
                                     stratify=y[idx_temp])

X_train, y_train = X[idx_train], y[idx_train]
X_val, y_val = X[idx_val], y[idx_val]
X_test, y_test = X[idx_test], y[idx_test]
print(f"train: {len(y_train)}   val: {len(y_val)}   test: {len(y_test)}")

split_table = pd.DataFrame({
    "train": np.bincount(y_train, minlength=len(CLASSES)),
    "val": np.bincount(y_val, minlength=len(CLASSES)),
    "test": np.bincount(y_test, minlength=len(CLASSES)),
}, index=CLASSES)
print("\nWafers per class in each split:")
print(split_table)
split_table.to_csv(f"{OUT_DIR}/02_split_counts.csv")

# Baseline: a model that always says "none". Shows why accuracy alone misleads.
none_idx = CLASS_TO_IDX["none"]
majority_acc = (y_test == none_idx).mean()
majority_f1 = f1_score(y_test, np.full_like(y_test, none_idx), average="macro", zero_division=0)
print(f"\nBaseline 'always predict none': accuracy={majority_acc:.3f}  macro-F1={majority_f1:.3f}")

# Leakage check (a diagnostic, not a fix). Wafers are split one by one, but they
# come in lots of up to 25. If a lot appears in both train and test, test
# wafers may share fab conditions with training wafers, which can make the test
# score look better than it would on a lot the model has never seen.
lots = labeled["lotName"].to_numpy()
train_lots, test_lots = set(lots[idx_train]), set(lots[idx_test])
overlap = train_lots & test_lots
print(f"\nLots shared between train and test: {len(overlap)} of "
      f"{len(train_lots | test_lots)} total lots "
      f"({len(overlap) / len(train_lots | test_lots):.1%})")
print("-> The test score below is therefore optimistic for completely unseen lots.")

# Inverse-frequency class weights, so rare patterns count more in the loss.
counts = np.bincount(y_train, minlength=len(CLASSES)).astype(np.float64)
class_weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32)
print("\nclass weights:", {c: round(w, 2) for c, w in zip(CLASSES, class_weights.tolist())})


def to_loader(X, y, shuffle):
    ds = TensorDataset(torch.from_numpy(X).unsqueeze(1), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


train_loader = to_loader(X_train, y_train, shuffle=True)
val_loader = to_loader(X_val, y_val, shuffle=False)
test_loader = to_loader(X_test, y_test, shuffle=False)

# =============================================================================
# 6. The model: a small CNN trained from scratch
# =============================================================================
print("\n" + "=" * 78)
print("6. MODEL")
print("=" * 78)


class WaferCNN(nn.Module):
    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = WaferCNN().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(model)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nTrainable parameters: {n_params:,}")

# =============================================================================
# 7. Train
# =============================================================================
# Macro-F1 (the average F1 over all 9 classes, each counted equally) is the
# selection metric, because accuracy would reward always predicting "none".
# The checkpoint with the best validation macro-F1 is kept.
# Note: train metrics are computed while training (dropout on, weights moving),
# so they run lower than validation metrics early on.
print("\n" + "=" * 78)
print("7. TRAINING")
print("=" * 78)


def run_epoch(loader, train):
    model.train(mode=train)
    total_loss = 0.0
    all_preds, all_true = [], []
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_true.append(yb.cpu().numpy())
    preds, true = np.concatenate(all_preds), np.concatenate(all_true)
    return total_loss / len(loader.dataset), f1_score(true, preds, average="macro", zero_division=0)


history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
best_val_f1, best_epoch, best_state = -1.0, 0, None

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_f1 = run_epoch(train_loader, train=True)
    va_loss, va_f1 = run_epoch(val_loader, train=False)
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(va_loss)
    history["train_f1"].append(tr_f1)
    history["val_f1"].append(va_f1)
    if va_f1 > best_val_f1:
        best_val_f1, best_epoch = va_f1, epoch
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    print(f"epoch {epoch:3d}/{EPOCHS}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
          f"train_macroF1={tr_f1:.3f}  val_macroF1={va_f1:.3f}")

model.load_state_dict(best_state)
print(f"\nBest val macro-F1: {best_val_f1:.3f} (epoch {best_epoch})")
val_f1_arr = np.array(history["val_f1"])
print(f"Val macro-F1 across epochs: min={val_f1_arr.min():.3f}  max={val_f1_arr.max():.3f}  "
      f"std={val_f1_arr.std():.3f}  (large swings are why the best checkpoint is kept)")
pd.DataFrame(history).rename_axis("epoch").to_csv(f"{OUT_DIR}/03_training_history.csv")

# =============================================================================
# 8. Training curves
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="val")
axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
axes[1].plot(history["train_f1"], label="train")
axes[1].plot(history["val_f1"], label="val")
axes[1].axvline(best_epoch - 1, color="gray", linestyle=":", label="best val epoch")
axes[1].set_title("Macro F1"); axes[1].set_xlabel("epoch"); axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_training_curves.png", dpi=150)
plt.show()

# =============================================================================
# 9. Test set: classification report
# =============================================================================
print("\n" + "=" * 78)
print("9. TEST SET RESULTS (held out, used once, after picking the checkpoint)")
print("=" * 78)

model.eval()
all_preds, all_true, all_probs = [], [], []
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb.to(device))
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_true.append(yb.numpy())
        all_probs.append(logits.softmax(dim=1).cpu().numpy())
y_pred, y_true = np.concatenate(all_preds), np.concatenate(all_true)
y_prob = np.concatenate(all_probs)

report = classification_report(y_true, y_pred, target_names=CLASSES,
                               output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).T
print(report_df.round(3))
report_df.to_csv(f"{OUT_DIR}/04_test_classification_report.csv")

cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
print("\nConfusion matrix (rows = true, columns = predicted):")
print(cm_df)
cm_df.to_csv(f"{OUT_DIR}/05_confusion_matrix.csv")

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Test set confusion matrix")
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
fig.colorbar(im)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_confusion_matrix.png", dpi=150)
plt.show()

# =============================================================================
# 10. Row-normalized confusion matrix
# =============================================================================
# Raw counts are dominated by "none". Normalizing each row shows, for the
# wafers that truly have a pattern, where they end up.
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion matrix, normalized by true class")
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        ax.text(j, i, f"{cm_norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                color="white" if cm_norm[i, j] > 0.5 else "black")
fig.colorbar(im)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/08_confusion_matrix_normalized.png", dpi=150)
plt.show()

# Largest off-diagonal cells: the confusions that matter most.
off = [(CLASSES[i], CLASSES[j], cm[i, j], cm_norm[i, j])
       for i in range(len(CLASSES)) for j in range(len(CLASSES)) if i != j]
off_df = pd.DataFrame(off, columns=["true", "predicted", "count", "share_of_true_class"])
off_df = off_df.sort_values("count", ascending=False).head(8).reset_index(drop=True)
print("\nTop 8 confusions by count:")
print(off_df.round(3))
off_df.to_csv(f"{OUT_DIR}/06_top_confusions.csv", index=False)

# =============================================================================
# 11. Per-class F1 vs training-set size
# =============================================================================
# If F1 tracks training size, weak classes are weak for lack of examples.
train_counts = np.bincount(y_train, minlength=len(CLASSES))
f1_per_class = report_df.loc[CLASSES, "f1-score"].to_numpy()

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(train_counts, f1_per_class)
for i, c in enumerate(CLASSES):
    ax.annotate(c, (train_counts[i], f1_per_class[i]), fontsize=8,
                xytext=(5, 5), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel("training examples for this class (log scale)")
ax.set_ylabel("test F1")
ax.set_title("Per-class F1 vs. training-set size")
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/09_f1_vs_train_size.png", dpi=150)
plt.show()

# =============================================================================
# 12. Precision-recall curves per class
# =============================================================================
# PR curves instead of ROC, because ROC looks too good under heavy imbalance.
ap_scores = {}
fig, ax = plt.subplots(figsize=(8, 7))
for i, c in enumerate(CLASSES):
    y_true_bin = (y_true == i).astype(int)
    prec, rec, _ = precision_recall_curve(y_true_bin, y_prob[:, i])
    ap_scores[c] = average_precision_score(y_true_bin, y_prob[:, i])
    ax.plot(rec, prec, label=f"{c} (AP={ap_scores[c]:.2f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-recall curve per class (one-vs-rest)")
ax.legend(fontsize=8, loc="lower left")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/10_precision_recall_curves.png", dpi=150)
plt.show()

ap_df = pd.Series(ap_scores, name="average_precision").to_frame()
print("\nAverage precision per class:")
print(ap_df.round(3))
print(f"Mean average precision: {ap_df['average_precision'].mean():.3f}")
ap_df.to_csv(f"{OUT_DIR}/07_average_precision.csv")

# =============================================================================
# 13. Prediction confidence: right vs wrong
# =============================================================================
confidence = y_prob.max(axis=1)
correct_mask = y_pred == y_true
print(f"\nMean confidence when correct: {confidence[correct_mask].mean():.3f}")
print(f"Mean confidence when wrong  : {confidence[~correct_mask].mean():.3f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(confidence[correct_mask], bins=30, alpha=0.6, label=f"correct (n={correct_mask.sum()})")
ax.hist(confidence[~correct_mask], bins=30, alpha=0.6, label=f"incorrect (n={(~correct_mask).sum()})")
ax.set_xlabel("model confidence (max softmax probability)")
ax.set_ylabel("count")
ax.set_title("Confidence when right vs. when wrong")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/11_confidence_histogram.png", dpi=150)
plt.show()

# =============================================================================
# 14. Per-class precision / recall / F1 bar chart
# =============================================================================
per_class = report_df.loc[CLASSES, ["precision", "recall", "f1-score"]]
per_class.plot(kind="bar", figsize=(10, 4))
plt.title("Per-class test metrics")
plt.ylim(0, 1)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_per_class_metrics.png", dpi=150)
plt.show()

# =============================================================================
# 15. Random test predictions (green = correct, red = wrong)
# =============================================================================
rng = np.random.default_rng(SEED)
sample_idx = rng.choice(len(y_test), size=min(12, len(y_test)), replace=False)

fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for ax, i in zip(axes.flat, sample_idx):
    ax.imshow(X_test[i], cmap="gray")
    true_c, pred_c = CLASSES[y_true[i]], CLASSES[y_pred[i]]
    ax.set_title(f"true: {true_c}\npred: {pred_c}",
                 color="green" if true_c == pred_c else "red", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_sample_predictions.png", dpi=150)
plt.show()

# =============================================================================
# 16. The most confident mistakes
# =============================================================================
# Confident errors point at systematic confusions, label noise, or resize
# artifacts, which random errors do not.
wrong_idx = np.where(~correct_mask)[0]
top_wrong = wrong_idx[np.argsort(-confidence[wrong_idx])][: min(12, len(wrong_idx))]

if len(top_wrong) == 0:
    print("No misclassified test examples, nothing to plot here.")
else:
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, top_wrong):
        ax.imshow(X_test[i], cmap="gray")
        ax.set_title(f"true: {CLASSES[y_true[i]]}\npred: {CLASSES[y_pred[i]]} "
                     f"({confidence[i]:.0%} conf.)", color="red", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/12_most_confident_mistakes.png", dpi=150)
    plt.show()

# =============================================================================
# 17. Save the model and print the final summary
# =============================================================================
torch.save(model.state_dict(), f"{OUT_DIR}/wafer_cnn.pt")

summary = {
    "labeled_wafers": int(len(y)),
    "total_wafers": int(len(raw)),
    "n_train": int(len(y_train)), "n_val": int(len(y_val)), "n_test": int(len(y_test)),
    "parameters": int(n_params),
    "lots_shared_train_test_%": round(100 * len(overlap) / len(train_lots | test_lots), 1),
    "baseline_always_none_accuracy": round(float(majority_acc), 3),
    "baseline_always_none_macro_f1": round(float(majority_f1), 3),
    "best_val_macro_f1": round(float(best_val_f1), 3),
    "best_epoch": int(best_epoch),
    "test_accuracy": round(report["accuracy"], 3),
    "test_macro_f1": round(report["macro avg"]["f1-score"], 3),
    "test_macro_recall": round(report["macro avg"]["recall"], 3),
    "test_weighted_f1": round(report["weighted avg"]["f1-score"], 3),
    "test_mean_average_precision": round(float(ap_df["average_precision"].mean()), 3),
}
with open(f"{OUT_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 78)
print("FINAL SUMMARY")
print("=" * 78)
for k, v in summary.items():
    print(f"{k:34s}: {v}")
print(f"\nSaved model, figures and tables to: {OUT_DIR}/")
print("Results shift by a few points between runs (GPU non-determinism and noisy "
      "validation F1), so quote a range, not one number.")







