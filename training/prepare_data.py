
import h5py
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Original dataset
MAT_PATH = r"C:\Users\tharu\Downloads\nyu_depth_v2_labeled (1).mat"

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = PROJECT_DIR / "dataset" / "splits"
PREVIEW_DIR = PROJECT_DIR / "outputs" / "dataset_preview"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

# Reproducible random split
SEED = 42
rng = np.random.default_rng(SEED)

with h5py.File(MAT_PATH, "r") as f:

    total = f["images"].shape[0]

    print("Total dataset images:", total)

    # Shuffle sample indices
    indices = rng.permutation(total)

    # 70% training, 15% validation, 15% testing
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    splits = {
        "train": indices[:train_end].tolist(),
        "validation": indices[train_end:val_end].tolist(),
        "test": indices[val_end:].tolist()
    }

    # Save split information
    split_file = SPLIT_DIR / "dataset_splits.json"

    with open(split_file, "w") as file:
        json.dump(splits, file, indent=4)

    print("\nDataset split completed:")
    for name, values in splits.items():
        print(f"{name}: {len(values)} images")

    # Select one test image for comparison
    sample_index = splits["test"][0]

    rgb = np.transpose(
        f["images"][sample_index], (2, 1, 0)
    )

    depth = np.transpose(
        f["depths"][sample_index], (1, 0)
    )

    # Check depth values
    if not np.isfinite(depth).all():
        raise ValueError("Invalid depth values found!")

    print("\nSample index:", sample_index)
    print("RGB shape:", rgb.shape)
    print("Depth shape:", depth.shape)
    print("Depth range:", depth.min(), "to", depth.max())

    # Generate 2x, 4x and 8x low-resolution depth maps
    factors = [2, 4, 8]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(depth, cmap="viridis")
    axes[0, 1].set_title("Ground Truth Depth")
    axes[0, 1].axis("off")

    axes[0, 2].axis("off")

    for i, factor in enumerate(factors):

        height, width = depth.shape

        # Downsample depth
        low_res = cv2.resize(
            depth,
            (width // factor, height // factor),
            interpolation=cv2.INTER_AREA
        )

        # Upsample using bicubic interpolation
        upsampled = cv2.resize(
            low_res,
            (width, height),
            interpolation=cv2.INTER_CUBIC
        )

        axes[1, i].imshow(upsampled, cmap="viridis")
        axes[1, i].set_title(f"{factor}x Bicubic Upsampling")
        axes[1, i].axis("off")

        print(
            f"{factor}x: Low-resolution shape = {low_res.shape}"
        )

    plt.tight_layout()

    output_file = PREVIEW_DIR / "upsampling_comparison.png"
    plt.savefig(output_file, dpi=150)
    plt.show()

print("\nDataset preparation completed!")
print("Split file:", split_file)
print("Preview:", output_file)
