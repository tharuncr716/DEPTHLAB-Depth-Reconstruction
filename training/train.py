
import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import pandas as pd

from pathlib import Path
from torch.utils.data import DataLoader

PROJECT_DIR = Path(
    "/content/drive/MyDrive/P4_Depth_Reconstruction"
)

DATA_PATH = "/content/nyu_small.h5"

SPLIT_PATH = os.path.join(
    PROJECT_DIR,
    "dataset/processed/small_splits.json"
)

MODEL_DIR = os.path.join(PROJECT_DIR, "models")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, MODEL_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "training"))

from depth_cnn import DepthRefinementCNN
from dataset_loader import NYUDepthDataset


# ==========================================
# CONFIGURATION
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.0005

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# LOSS FUNCTION
# ==========================================

def depth_loss(prediction, target):

    l1 = F.l1_loss(prediction, target)

    pred_dx = (
        prediction[:, :, :, 1:]
        - prediction[:, :, :, :-1]
    )

    target_dx = (
        target[:, :, :, 1:]
        - target[:, :, :, :-1]
    )

    pred_dy = (
        prediction[:, :, 1:, :]
        - prediction[:, :, :-1, :]
    )

    target_dy = (
        target[:, :, 1:, :]
        - target[:, :, :-1, :]
    )

    gradient_loss = (
        F.l1_loss(pred_dx, target_dx)
        + F.l1_loss(pred_dy, target_dy)
    )

    return l1 + 0.1 * gradient_loss


# ==========================================
# DATASETS
# ==========================================

train_dataset = NYUDepthDataset(
    DATA_PATH,
    SPLIT_PATH,
    split="train"
)

val_dataset = NYUDepthDataset(
    DATA_PATH,
    SPLIT_PATH,
    split="validation",
    factor=4
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)


# ==========================================
# MODEL
# ==========================================

model = DepthRefinementCNN(
    channels=32,
    blocks=5
).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_val_loss = float("inf")
history = []


# ==========================================
# TRAINING
# ==========================================

for epoch in range(EPOCHS):

    start = time.time()

    model.train()
    train_total = 0.0

    for batch_index, (rgb, low_depth, target) in enumerate(
        train_loader
    ):

        rgb = rgb.to(DEVICE)
        low_depth = low_depth.to(DEVICE)
        target = target.to(DEVICE)

        optimizer.zero_grad()

        prediction = model(rgb, low_depth)

        loss = depth_loss(prediction, target)

        loss.backward()
        optimizer.step()

        train_total += loss.item()

    avg_train = train_total / len(train_loader)

    # ======================================
    # VALIDATION
    # ======================================

    model.eval()
    val_total = 0.0

    with torch.no_grad():

        for rgb, low_depth, target in val_loader:

            rgb = rgb.to(DEVICE)
            low_depth = low_depth.to(DEVICE)
            target = target.to(DEVICE)

            prediction = model(rgb, low_depth)

            loss = depth_loss(prediction, target)

            val_total += loss.item()

    avg_val = val_total / len(val_loader)

    duration = time.time() - start

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train: {avg_train:.6f} | "
        f"Validation: {avg_val:.6f} | "
        f"Time: {duration:.1f}s"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": avg_train,
        "validation_loss": avg_val
    })

    if avg_val < best_val_loss:

        best_val_loss = avg_val

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss
            },
            os.path.join(MODEL_DIR, "best_depth_model.pth")
        )

        print("Best checkpoint saved!")

    pd.DataFrame(history).to_csv(
        os.path.join(OUTPUT_DIR, "training_history.csv"),
        index=False
    )

print("\nTraining completed!")
print("Best validation loss:", best_val_loss)
