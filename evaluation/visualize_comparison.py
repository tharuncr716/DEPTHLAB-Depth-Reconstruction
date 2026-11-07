
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = ROOT_DIR / "dataset" / "processed" / "nyu_small.h5"
SPLIT_PATH = ROOT_DIR / "dataset" / "processed" / "small_splits.json"
MODEL_PATH = ROOT_DIR / "models" / "best_depth_model.pth"

OUTPUT_DIR = ROOT_DIR / "outputs" / "visual_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_DIR))

from models.depth_cnn import DepthRefinementCNN
from training.dataset_loader import NYUDepthDataset
from classical.classical_methods import apply_methods


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FACTORS = [2, 4, 8]
SAMPLE_INDICES = [0, 23, 52]

HYBRID_ALPHA = {
    2: 0.75,
    4: 0.50,
    8: 0.00
}


def load_model():

    model = DepthRefinementCNN(channels=32, blocks=5)

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


def create_low_depth(gt, factor):

    height, width = gt.shape

    low = cv2.resize(
        gt,
        (width // factor, height // factor),
        interpolation=cv2.INTER_AREA
    )

    return low


def prepare_rgb(rgb):

    rgb = np.clip(rgb, 0, 1)

    return (rgb * 255).astype(np.uint8)


def predict_cnn(model, rgb, bicubic):

    rgb_tensor = torch.from_numpy(
        np.transpose(rgb, (2, 0, 1)).copy()
    ).float().unsqueeze(0).to(DEVICE)

    depth_tensor = torch.from_numpy(
        bicubic.copy()
    ).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(rgb_tensor, depth_tensor)

    return (
        prediction.squeeze()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def save_comparison(rgb, gt, bicubic, jbf, cnn, hybrid,
                    factor, sample_index):

    # Use the same depth display range for all depth maps.
    vmin = float(np.min(gt))
    vmax = float(np.max(gt))

    cnn_error = np.abs(gt - cnn)
    hybrid_error = np.abs(gt - hybrid)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    fig.suptitle(
        f"Depth Reconstruction Comparison | "
        f"Test Sample {sample_index} | {factor}x",
        fontsize=16
    )

    # Original RGB image.
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original RGB")

    # Ground truth.
    axes[0, 1].imshow(
        gt, cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[0, 1].set_title("Ground Truth")

    # Bicubic.
    axes[0, 2].imshow(
        bicubic, cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[0, 2].set_title("Bicubic")

    # Joint Bilateral.
    axes[0, 3].imshow(
        jbf, cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[0, 3].set_title("Joint Bilateral")

    # CNN.
    axes[1, 0].imshow(
        cnn, cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[1, 0].set_title("CNN Prediction")

    # Hybrid.
    axes[1, 1].imshow(
        hybrid, cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[1, 1].set_title("Hybrid Prediction")

    # CNN absolute error.
    axes[1, 2].imshow(cnn_error, cmap="magma")
    axes[1, 2].set_title("CNN Absolute Error")

    # Hybrid absolute error.
    axes[1, 3].imshow(hybrid_error, cmap="magma")
    axes[1, 3].set_title("Hybrid Absolute Error")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = (
        OUTPUT_DIR /
        f"sample_{sample_index}_{factor}x_comparison.png"
    )

    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():

    print("DEPTHLAB - VISUAL COMPARISON")
    print("Device:", DEVICE)

    model = load_model()

    dataset = NYUDepthDataset(
        h5_path=DATASET_PATH,
        split_path=SPLIT_PATH,
        split="test",
        factor=4
    )

    try:

        for sample_index in SAMPLE_INDICES:

            if sample_index >= len(dataset):
                print(f"Skipping invalid sample: {sample_index}")
                continue

            rgb_tensor, _, gt_tensor = dataset[sample_index]

            rgb = (
                rgb_tensor.numpy()
                .transpose(1, 2, 0)
                .astype(np.float32)
            )

            gt = (
                gt_tensor.squeeze()
                .numpy()
                .astype(np.float32)
            )

            rgb_uint8 = prepare_rgb(rgb)

            for factor in FACTORS:

                print(
                    f"\nProcessing sample {sample_index}, "
                    f"factor {factor}x"
                )

                low_depth = create_low_depth(gt, factor)

                bicubic = cv2.resize(
                    low_depth,
                    (gt.shape[1], gt.shape[0]),
                    interpolation=cv2.INTER_CUBIC
                )

                # Classical methods.
                classical = apply_methods(
                    rgb_uint8,
                    low_depth
                )

                jbf = np.asarray(
                    classical["Joint Bilateral"],
                    dtype=np.float32
                )

                # CNN.
                cnn = predict_cnn(
                    model,
                    rgb,
                    bicubic
                )

                # Hybrid.
                alpha = HYBRID_ALPHA[factor]

                hybrid = (
                    alpha * cnn
                    + (1 - alpha) * jbf
                )

                hybrid = np.clip(hybrid, 0.1, None)

                save_comparison(
                    rgb,
                    gt,
                    bicubic,
                    jbf,
                    cnn,
                    hybrid,
                    factor,
                    sample_index
                )

    finally:

        if getattr(dataset, "file", None) is not None:
            dataset.file.close()

    print("\nVisual comparison completed.")
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
