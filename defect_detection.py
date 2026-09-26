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
           "Near-full", "Random", "Scratch", "none"]  ### here none means that they are not faulty. So no fault as given in the list is found. 
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
# Wafer maps come in different sizes, but a CNN needs a fixed input shape. using Nearest-neighbor resizing keeps the pixel values exactly {0, 1, 2}. 
# Becausee a normal (interpolated) resize would invent fractional values in between, which don't correspond to a real die state.
 
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

##### Data stratification 
#### since y is the labeled data so the y is used to make splits

idx = np.arrange(len(y))
idx_train, idx_temp = train_test_split(idx, test_size=0.2, random_state = SEED, stratify = y)
idx_val, idx_test = train_test_split(idx_temp, test_size= 0.2, random_state = SEED, startify= y[idx_temp])


X_train, y_train = X[idx_train], y[idx_train]
X_val, y_val = X[idx_val], y[idx_val]
X_test, y_test = X[idx_test], y[idx_test]

print(f"train: {len(y_train)}   val: {len(y_val)}   test: {len(y_test)}")



##### Class imbalance handling with weights penalization




# Diagnostic, not a fix: wafers are split individually here, but they  actually come in lots of up to 25. If the same lot shows up in both
# train and test, those test wafers may share lot-level fab conditions with training examples the model has already seen, which can make test
# performance look slightly better than it would on a genuinely unseen
# lot. This just reports how much of that is happening; a stricter alternative would group-split by lotName instead of by wafer
# (sklearn's GroupShuffleSplit), at the cost of losing exact stratification by class.



#### Since wafers are made in lots, so this step needs contextual knowledge. This inspects how many lots are common in training and testing lots
### output wwas Lots shared between train and test: 6789 of 10522 total lots (64.5%)

lots = labeled["lotName"].to_numpy()
train_lots, test_lots = set(lots[idx_train]), set(lots[idx_test])
overlap = train_lots & test_lots
print(f"Lots shared between train and test: {len(overlap)} of "
      f"{len(train_lots | test_lots)} total lots "
      f"({len(overlap) / len(train_lots | test_lots):.1%})")


#### Since in this pipeline Pytorch is being used therefore, now the images in y, X are being transformed to pytorch using TensorDataset within a function to_loader
### atfer this conversion, the shufllin is only maintained for test

def to_loader(X, y, shuffle):
    ds = TensorDataset(torch.from_numpy(X).unsqueeze(1), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)
 
train_loader = to_loader(X_train, y_train, shuffle=True)
val_loader = to_loader(X_val, y_val, shuffle=False)
test_loader = to_loader(X_test, y_test, shuffle=False)


###  This is the place where class imbalance is being handled using an approach so called "Inverse Frequency" weights 
# Inverse-frequency class weights: "none" alone is usually ~85% of the labeled data, so an unweighted loss would mostly just learn to predict
# "none" every time and still look accurate.

counts = np.bicount(y_train, minlenght = len(CLASSES).astype(np.float64))
class_weights =torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype = torch.float32 )
print("\nclass weights:", {c: round(w, 2) for c, w in zip(CLASSES, class_weights.tolist())})























