import glob
import os
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
 
# ---- config ----
DATA_DIR = "/kaggle/input/datasets/qingyi/wm811k-wafer-map"
IMG_SIZE = 48          # every wafer map gets resized to IMG_SIZE x IMG_SIZE
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3
SEED = 0
 
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
           "Near-full", "Random", "Scratch", "none"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
 
torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


###### Since data is avilable at Kaggle, it was fetched and ran directly at kaggle using pickl

#### This part checks for the existence of the data and then print the found path on consol
pkl_candidates = glob.glob(os.path.join(DATA_DIR, "**", "*.pkl"), recursive=True)
print(f"Found {len(pkl_candidates)} .pkl file(s) under {DATA_DIR}:")
for p in pkl_candidates:
    print(" ", p)
 
assert len(pkl_candidates) > 0, (
    f"No .pkl file found under {DATA_DIR}. "
    f"Contents of that folder: {os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else 'PATH DOES NOT EXIST'}"
)
 
raw = pd.read_pickle(pkl_candidates[0])
print("\nShape:", raw.shape)
print("Columns:", list(raw.columns))
raw.head()


##### Inspecting the raw data for labels

def unwrap_label(value):
    while isinstance(value, (list, np.ndarray)):
        if len(value) == 0:
            return None
        value = value[0]
    return value
 
raw["failureType"] = raw["failureType"].apply(unwrap_label)
 
print("Label counts (including unlabeled = NaN or not one of the 9 classes):")
print(raw["failureType"].value_counts(dropna=False))
 
# %%
labeled = raw[raw["failureType"].isin(CLASSES)].reset_index(drop=True)
print(f"\n{len(labeled)} of {len(raw)} wafers carry a usable label "
      f"({len(labeled) / len(raw):.1%}).")
 
fig, ax = plt.subplots(figsize=(8, 4))
labeled["failureType"].value_counts().reindex(CLASSES).plot(kind="bar", ax=ax)
ax.set_title("Labeled wafer maps per class")
ax.set_ylabel("count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("01_class_distribution.png", dpi=150)
plt.show()


# %% Visualizing each class which is shown in label 

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for ax, cls in zip(axes.flat, CLASSES):
    example = labeled.loc[labeled["failureType"] == cls, "waferMap"].iloc[0]
    ax.imshow(example, cmap="gray")
    ax.set_title(cls)
    ax.axis("off")
plt.tight_layout()
plt.savefig("02_example_per_class.png", dpi=150)
plt.show()




# ## 4. Resize every wafer map to a fixed size
#
# Wafer maps come in different sizes, but a CNN needs a fixed input shape.
# Nearest-neighbor resizing keeps the pixel values exactly {0, 1, 2} —
# a normal (interpolated) resize would invent fractional values in
# between, which don't correspond to a real die state.
 
# %%
def resize_wafer(wafer, size=IMG_SIZE):
    wafer = np.asarray(wafer)
    h, w = wafer.shape
    row_idx = np.clip((np.arange(size) * h / size).astype(int), 0, h - 1)
    col_idx = np.clip((np.arange(size) * w / size).astype(int), 0, w - 1)
    return wafer[np.ix_(row_idx, col_idx)]
 
X = np.stack([resize_wafer(w) for w in labeled["waferMap"]]).astype(np.float32)
X = X / 2.0  # {0, 1, 2} -> {0, 0.5, 1}
y = labeled["failureType"].map(CLASS_TO_IDX).to_numpy()
 
print("X shape:", X.shape, " y shape:", y.shape)
 
