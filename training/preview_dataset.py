
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

file_path = r"C:\Users\tharu\Downloads\nyu_depth_v2_labeled (1).mat"

output_dir = Path("outputs/dataset_preview")
output_dir.mkdir(parents=True, exist_ok=True)

with h5py.File(file_path, "r") as f:

    # Read the first RGB image
    rgb = f["images"][0]

    # Convert channel-first (3, 640, 480)
    # to channel-last (640, 480, 3)
    rgb = np.transpose(rgb, (2, 1, 0))

    # Read corresponding ground-truth depth
    depth = np.transpose(f["depths"][0], (1, 0))

    print("RGB image shape:", rgb.shape)
    print("Depth map shape:", depth.shape)

    print("Depth minimum:", np.nanmin(depth))
    print("Depth maximum:", np.nanmax(depth))
    print("Depth mean:", np.nanmean(depth))
    print("Invalid depth pixels:", np.count_nonzero(~np.isfinite(depth)))

    # Display and save RGB and depth
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("Original RGB Image")
    axes[0].axis("off")

    im = axes[1].imshow(depth, cmap="viridis")
    axes[1].set_title("Ground Truth Depth Map")
    axes[1].axis("off")

    plt.colorbar(im, ax=axes[1], label="Depth value")
    plt.tight_layout()

    plt.savefig(output_dir / "sample_000.png", dpi=150)
    plt.show()

print("Preview saved successfully!")
