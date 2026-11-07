
import h5py
import cv2
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from skimage.metrics import structural_similarity


# =====================================================
# 1. PROJECT PATHS
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MAT_PATH = r"C:\Users\tharu\Downloads\nyu_depth_v2_labeled (1).mat"

SPLIT_PATH = (
    PROJECT_DIR / "dataset" / "splits" / "dataset_splits.json"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "classical"


# =====================================================
# 2. EVALUATION METRICS
# =====================================================

def calculate_metrics(gt, prediction):

    gt = np.asarray(gt, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)

    if gt.shape != prediction.shape:
        raise ValueError("Ground truth and prediction shapes differ.")

    valid = (
        np.isfinite(gt)
        & np.isfinite(prediction)
        & (gt > 0)
    )

    if not np.any(valid):
        raise ValueError("No valid pixels available for evaluation.")

    gt_valid = gt[valid]
    pred_valid = prediction[valid]

    error = gt_valid - pred_valid

    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))

    data_range = float(gt_valid.max() - gt_valid.min())

    if data_range <= 0:
        psnr = float("nan")
        ssim = float("nan")
    else:
        if mse == 0:
            psnr = float("inf")
        else:
            psnr = float(
                10 * np.log10((data_range ** 2) / mse)
            )

        # SSIM is calculated on the full image.
        # NYU labeled depth maps are normally dense.
        # Invalid pixels are replaced with GT values
        # to avoid penalizing missing predictions twice.
        gt_clean = np.where(valid, gt, 0).astype(np.float32)
        pred_clean = np.where(
            valid, prediction, gt_clean
        ).astype(np.float32)

        ssim = float(
            structural_similarity(
                gt_clean,
                pred_clean,
                data_range=data_range
            )
        )

    return rmse, mae, psnr, ssim


# =====================================================
# 3. GUIDED FILTER
# =====================================================

def guided_filter(rgb, depth, radius=8, eps=0.0001):

    guide = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    ).astype(np.float32) / 255.0

    source = depth.astype(np.float32)

    kernel_size = 2 * radius + 1

    def box_filter(image):
        return cv2.boxFilter(
            image,
            ddepth=-1,
            ksize=(kernel_size, kernel_size),
            normalize=True,
            borderType=cv2.BORDER_REFLECT
        )

    mean_guide = box_filter(guide)
    mean_source = box_filter(source)

    mean_guide_source = box_filter(guide * source)

    covariance = (
        mean_guide_source
        - mean_guide * mean_source
    )

    mean_guide_squared = box_filter(guide * guide)

    variance = (
        mean_guide_squared
        - mean_guide * mean_guide
    )

    a = covariance / (variance + eps)

    b = mean_source - a * mean_guide

    mean_a = box_filter(a)
    mean_b = box_filter(b)

    output = mean_a * guide + mean_b

    return output.astype(np.float32)


# =====================================================
# 4. JOINT BILATERAL FILTER
# =====================================================

def joint_bilateral_filter(
    rgb,
    depth,
    radius=3,
    sigma_color=25.0,
    sigma_space=3.0
):

    guide = rgb.astype(np.float32)
    source = depth.astype(np.float32)

    height, width = source.shape

    padded_guide = cv2.copyMakeBorder(
        guide,
        radius, radius, radius, radius,
        cv2.BORDER_REFLECT
    )

    padded_depth = cv2.copyMakeBorder(
        source,
        radius, radius, radius, radius,
        cv2.BORDER_REFLECT
    )

    weighted_sum = np.zeros_like(
        source, dtype=np.float32
    )

    weight_sum = np.zeros_like(
        source, dtype=np.float32
    )

    color_denominator = 2.0 * sigma_color ** 2
    space_denominator = 2.0 * sigma_space ** 2

    for dy in range(-radius, radius + 1):

        for dx in range(-radius, radius + 1):

            y_start = radius + dy
            x_start = radius + dx

            neighbour_guide = padded_guide[
                y_start:y_start + height,
                x_start:x_start + width
            ]

            neighbour_depth = padded_depth[
                y_start:y_start + height,
                x_start:x_start + width
            ]

            color_difference = np.sum(
                (guide - neighbour_guide) ** 2,
                axis=2
            )

            color_weight = np.exp(
                -color_difference / color_denominator
            )

            spatial_distance = dx ** 2 + dy ** 2

            spatial_weight = np.exp(
                -spatial_distance / space_denominator
            )

            weight = color_weight * spatial_weight

            weighted_sum += weight * neighbour_depth
            weight_sum += weight

    output = weighted_sum / np.maximum(
        weight_sum, 1e-8
    )

    return output.astype(np.float32)


# =====================================================
# 5. CLASSICAL UPSAMPLING METHODS
# =====================================================

def apply_methods(rgb, low_depth):

    height, width = rgb.shape[:2]

    low_depth = np.asarray(
        low_depth, dtype=np.float32
    )

    # Method 1: Nearest neighbour
    nearest = cv2.resize(
        low_depth,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    # Method 2: Bilinear
    bilinear = cv2.resize(
        low_depth,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    # Method 3: Bicubic
    bicubic = cv2.resize(
        low_depth,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

    # Method 4: Bilateral filter
    bilateral = cv2.bilateralFilter(
        bicubic.astype(np.float32),
        d=9,
        sigmaColor=0.1,
        sigmaSpace=7
    )

    # Method 5: Joint bilateral filter
    joint_bilateral = joint_bilateral_filter(
        rgb,
        bicubic,
        radius=3,
        sigma_color=25.0,
        sigma_space=3.0
    )

    # Method 6: Guided filter
    guided = guided_filter(
        rgb,
        bicubic,
        radius=8,
        eps=0.0001
    )

    return {
        "Nearest": nearest,
        "Bilinear": bilinear,
        "Bicubic": bicubic,
        "Bilateral": bilateral,
        "Joint Bilateral": joint_bilateral,
        "Guided Filter": guided
    }


# =====================================================
# 6. SINGLE-IMAGE EXPERIMENT
# =====================================================

def run_single_image_experiment():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("OpenCV version:", cv2.__version__)
    print("\nLoading dataset split...")

    with open(SPLIT_PATH, "r") as file:
        splits = json.load(file)

    sample_index = splits["test"][0]

    print("Testing sample:", sample_index)

    with h5py.File(MAT_PATH, "r") as f:

        rgb = np.transpose(
            f["images"][sample_index],
            (2, 1, 0)
        )

        gt = np.transpose(
            f["depths"][sample_index],
            (1, 0)
        )

    rgb = np.ascontiguousarray(rgb)
    gt = np.ascontiguousarray(
        gt.astype(np.float32)
    )

    print("RGB shape:", rgb.shape)
    print("Depth shape:", gt.shape)

    all_metrics = []

    for factor in [2, 4, 8]:

        print("\n================================")
        print(f"Processing {factor}x upsampling")
        print("================================")

        height, width = gt.shape

        low_depth = cv2.resize(
            gt,
            (width // factor, height // factor),
            interpolation=cv2.INTER_AREA
        )

        print("Low resolution:", low_depth.shape)

        start_time = time.perf_counter()

        results = apply_methods(
            rgb,
            low_depth
        )

        total_time = (
            time.perf_counter() - start_time
        )

        print(
            "All methods completed in:",
            round(total_time, 3),
            "seconds"
        )

        fig, axes = plt.subplots(
            2, 4,
            figsize=(18, 9)
        )

        axes = axes.flatten()

        axes[0].imshow(
            gt,
            cmap="viridis"
        )
        axes[0].set_title("Ground Truth")

        axes[1].imshow(rgb)
        axes[1].set_title("RGB Guidance")

        for i, (method, prediction) in enumerate(
            results.items(),
            start=2
        ):

            rmse, mae, psnr, ssim = calculate_metrics(
                gt,
                prediction
            )

            all_metrics.append({
                "Image_ID": sample_index,
                "Scale": f"{factor}x",
                "Method": method,
                "RMSE": rmse,
                "MAE": mae,
                "PSNR": psnr,
                "SSIM": ssim
            })

            axes[i].imshow(
                prediction,
                cmap="viridis",
                vmin=float(gt.min()),
                vmax=float(gt.max())
            )

            axes[i].set_title(
                f"{method}\nRMSE: {rmse:.4f}"
            )

            print(
                f"{method:18s} "
                f"RMSE={rmse:.5f} "
                f"MAE={mae:.5f} "
                f"PSNR={psnr:.3f} "
                f"SSIM={ssim:.4f}"
            )

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        image_path = (
            OUTPUT_DIR / f"comparison_{factor}x.png"
        )

        plt.savefig(
            image_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print("Saved:", image_path)

    # Save metrics
    df = pd.DataFrame(all_metrics)

    csv_path = (
        OUTPUT_DIR / "classical_metrics_sample.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print("\n================================")
    print("FINAL CLASSICAL METHOD RESULTS")
    print("================================")

    print(df.to_string(index=False))

    print("\nMetrics saved:", csv_path)

    print("\nAll classical experiments completed!")


# =====================================================
# 7. MAIN ENTRY POINT
# =====================================================

# This block runs ONLY when this file is executed
# directly, not when another script imports it.

if __name__ == "__main__":
    run_single_image_experiment()
