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
