
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = ROOT_DIR / "dataset" / "processed" / "nyu_small.h5"
SPLIT_PATH = ROOT_DIR / "dataset" / "processed" / "small_splits.json"
MODEL_PATH = ROOT_DIR / "models" / "best_depth_model.pth"

OUTPUT_DIR = ROOT_DIR / "outputs" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# IMPORT PROJECT MODULES
# ---------------------------------------------------------

sys.path.insert(0, str(ROOT_DIR))

from models.depth_cnn import DepthRefinementCNN
from training.dataset_loader import NYUDepthDataset
from classical.classical_methods import (
    apply_methods,
    calculate_metrics
)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

FACTORS = [2, 4, 8]

# Hybrid weights selected using the validation set.
HYBRID_ALPHA = {
    2: 0.75,
    4: 0.50,
    8: 0.00
}

# Set to None to evaluate all test samples.
# Set to 10 for a quick initial test.
MAX_IMAGES = None

METHOD_NAMES = [
    "Nearest",
    "Bilinear",
    "Bicubic",
    "Bilateral",
    "Joint Bilateral",
    "Guided Filter",
    "CNN",
    "Hybrid"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------
# METRIC HANDLING
# ---------------------------------------------------------

def get_metrics(gt, prediction):
    """
    Calculates metrics and supports either dictionary
    or tuple output from calculate_metrics().
    """

    result = calculate_metrics(gt, prediction)

    # Case 1: calculate_metrics returns a dictionary.
    if isinstance(result, dict):

        lower_keys = {
            str(key).lower(): key
            for key in result.keys()
        }

        def find_value(possible_names):
            for name in possible_names:
                if name.lower() in lower_keys:
                    return result[lower_keys[name.lower()]]

            raise KeyError(
                f"Metric not found. Available keys: {list(result.keys())}"
            )

        rmse = find_value(["rmse", "root_mean_squared_error"])
        mae = find_value(["mae", "mean_absolute_error"])
        psnr = find_value(["psnr"])
        ssim = find_value(["ssim"])

    # Case 2: calculate_metrics returns a tuple/list.
    elif isinstance(result, (tuple, list, np.ndarray)):

        if len(result) != 4:
            raise ValueError(
                "calculate_metrics() must return four values: "
                "RMSE, MAE, PSNR, SSIM. "
                f"Received {len(result)} values: {result}"
            )

        # Expected order:
        # RMSE, MAE, PSNR, SSIM
        rmse, mae, psnr, ssim = result

    else:
        raise TypeError(
            "Unsupported calculate_metrics() return type: "
            f"{type(result)}"
        )

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "PSNR": float(psnr),
        "SSIM": float(ssim)
    }


# ---------------------------------------------------------
# IMAGE PREPARATION
# ---------------------------------------------------------

def create_low_resolution_depth(gt_depth, factor):
    """
    Creates a synthetic low-resolution depth map
    from the ground-truth depth map.
    """

    height, width = gt_depth.shape

    low_width = max(1, width // factor)
    low_height = max(1, height // factor)

    low_depth = cv2.resize(
        gt_depth,
        (low_width, low_height),
        interpolation=cv2.INTER_AREA
    )

    return low_depth


def resize_depth_to_original(low_depth, original_shape):
    """
    Upsamples low-resolution depth using bicubic
    interpolation to the original RGB dimensions.
    """

    height, width = original_shape

    return cv2.resize(
        low_depth,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )


def prepare_rgb_for_classical(rgb):
    """
    Converts RGB from float [0,1] to uint8 [0,255].
    """

    rgb = np.clip(rgb, 0.0, 1.0)

    return (rgb * 255.0).astype(np.uint8)


# ---------------------------------------------------------
# LOAD CNN MODEL
# ---------------------------------------------------------

def load_cnn_model():

    print("\nLoading CNN model...")

    model = DepthRefinementCNN(
        channels=32,
        blocks=5
    )

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    print("CNN checkpoint loaded successfully.")

    return model


# ---------------------------------------------------------
# CNN PREDICTION
# ---------------------------------------------------------

def run_cnn(model, rgb, bicubic_depth):
    """
    CNN forward pass.

    The model accepts two separate inputs:
        model(rgb_tensor, depth_tensor)
    """

    rgb_tensor = torch.from_numpy(
        np.transpose(rgb, (2, 0, 1)).copy()
    ).float().unsqueeze(0).to(DEVICE)

    depth_tensor = torch.from_numpy(
        bicubic_depth.copy()
    ).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    start_time = time.perf_counter()

    with torch.no_grad():
        prediction = model(rgb_tensor, depth_tensor)

    elapsed_time = time.perf_counter() - start_time

    prediction = (
        prediction.squeeze()
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return prediction, elapsed_time


# ---------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------

def main():

    print("\n========================================")
    print(" DEPTHLAB - MODEL EVALUATION")
    print("========================================")

    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Checkpoint: {MODEL_PATH}")

    # Check required files.
    for file_path in [DATASET_PATH, SPLIT_PATH, MODEL_PATH]:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    # Load CNN.
    model = load_cnn_model()

    # Load test dataset.
    dataset = NYUDepthDataset(
        h5_path=DATASET_PATH,
        split_path=SPLIT_PATH,
        split="test",
        factor=4
    )

    total_images = len(dataset)

    if MAX_IMAGES is not None:
        total_images = min(total_images, MAX_IMAGES)

    print(f"\nTest samples: {total_images}")

    all_rows = []
    timing_rows = []

    # -----------------------------------------------------
    # EVALUATE EACH UPSAMPLING FACTOR
    # -----------------------------------------------------

    for factor in FACTORS:

        print("\n========================================")
        print(f" EVALUATING UPSAMPLING FACTOR: {factor}x")
        print("========================================")

        alpha = HYBRID_ALPHA[factor]

        print(f"Hybrid alpha: {alpha}")

        for image_index in range(total_images):

            # Dataset returns:
            # RGB tensor, bicubic depth tensor, GT depth tensor.
            rgb_tensor, _, gt_tensor = dataset[image_index]

            rgb = (
                rgb_tensor
                .numpy()
                .transpose(1, 2, 0)
                .astype(np.float32)
            )

            gt_depth = (
                gt_tensor
                .squeeze()
                .numpy()
                .astype(np.float32)
            )

            # Create low-resolution depth for this factor.
            low_depth = create_low_resolution_depth(
                gt_depth,
                factor
            )

            # Bicubic upsampling.
            bicubic_depth = resize_depth_to_original(
                low_depth,
                gt_depth.shape
            )

            rgb_uint8 = prepare_rgb_for_classical(rgb)

            # -------------------------------------------------
            # CLASSICAL METHODS
            # -------------------------------------------------

            classical_start = time.perf_counter()

            classical_results = apply_methods(
                rgb_uint8,
                low_depth
            )

            classical_elapsed = (
                time.perf_counter() - classical_start
            )

            # Check expected classical method names.
            for method_name in [
                "Nearest",
                "Bilinear",
                "Bicubic",
                "Bilateral",
                "Joint Bilateral",
                "Guided Filter"
            ]:

                if method_name not in classical_results:
                    raise KeyError(
                        f"'{method_name}' missing from apply_methods(). "
                        f"Available methods: {list(classical_results.keys())}"
                    )

                prediction = np.asarray(
                    classical_results[method_name],
                    dtype=np.float32
                )

                if prediction.shape != gt_depth.shape:
                    raise ValueError(
                        f"{method_name} output shape {prediction.shape} "
                        f"does not match GT shape {gt_depth.shape}"
                    )

                metrics = get_metrics(
                    gt_depth,
                    prediction
                )

                all_rows.append({
                    "Image_Index": image_index,
                    "Factor": factor,
                    "Method": method_name,
                    **metrics
                })

            # Classical methods are calculated together,
            # so this is the combined runtime per image.
            timing_rows.append({
                "Image_Index": image_index,
                "Factor": factor,
                "Method": "Classical_Batch",
                "Time_Seconds": classical_elapsed
            })

            # -------------------------------------------------
            # CNN
            # -------------------------------------------------

            cnn_prediction, cnn_elapsed = run_cnn(
                model,
                rgb,
                bicubic_depth
            )

            cnn_metrics = get_metrics(
                gt_depth,
                cnn_prediction
            )

            all_rows.append({
                "Image_Index": image_index,
                "Factor": factor,
                "Method": "CNN",
                **cnn_metrics
            })

            timing_rows.append({
                "Image_Index": image_index,
                "Factor": factor,
                "Method": "CNN",
                "Time_Seconds": cnn_elapsed
            })

            # -------------------------------------------------
            # HYBRID MODEL
            # -------------------------------------------------

            jbf_prediction = np.asarray(
                classical_results["Joint Bilateral"],
                dtype=np.float32
            )

            hybrid_prediction = (
                alpha * cnn_prediction
                + (1.0 - alpha) * jbf_prediction
            )

            hybrid_prediction = np.clip(
                hybrid_prediction,
                0.1,
                None
            )

            hybrid_metrics = get_metrics(
                gt_depth,
                hybrid_prediction
            )

            all_rows.append({
                "Image_Index": image_index,
                "Factor": factor,
                "Method": "Hybrid",
                **hybrid_metrics
            })

            # -------------------------------------------------
            # PROGRESS
            # -------------------------------------------------

            if (
                (image_index + 1) % 5 == 0
                or image_index + 1 == total_images
            ):

                print(
                    f"Factor {factor}x: "
                    f"Processed {image_index + 1}/{total_images} images"
                )

    # Close HDF5 file if it was opened.
    if getattr(dataset, "file", None) is not None:
        dataset.file.close()

    # -----------------------------------------------------
    # SAVE PER-IMAGE RESULTS
    # -----------------------------------------------------

    results_df = pd.DataFrame(all_rows)

    per_image_path = (
        OUTPUT_DIR / "comparison_per_image.csv"
    )

    results_df.to_csv(
        per_image_path,
        index=False
    )

    # -----------------------------------------------------
    # GENERATE SUMMARY
    # -----------------------------------------------------

    summary_df = (
        results_df
        .groupby(["Factor", "Method"])[
            ["RMSE", "MAE", "PSNR", "SSIM"]
        ]
        .mean()
        .reset_index()
    )

    summary_df = summary_df.sort_values(
        ["Factor", "RMSE"]
    )

    summary_path = (
        OUTPUT_DIR / "comparison_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # -----------------------------------------------------
    # SAVE TIMING RESULTS
    # -----------------------------------------------------

    timing_df = pd.DataFrame(timing_rows)

    timing_path = (
        OUTPUT_DIR / "classical_batch_timing.csv"
    )

    timing_df.to_csv(
        timing_path,
        index=False
    )

    # -----------------------------------------------------
    # DISPLAY FINAL RESULTS
    # -----------------------------------------------------

    print("\n========================================")
    print(" EVALUATION COMPLETED")
    print("========================================")

    for factor in FACTORS:

        print(f"\n========== {factor}x RESULTS ==========")

        factor_results = summary_df[
            summary_df["Factor"] == factor
        ]

        print(
            factor_results.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

    print("\n========================================")
    print(" OUTPUT FILES")
    print("========================================")

    print(f"\nPer-image results:\n{per_image_path}")
    print(f"\nSummary results:\n{summary_path}")
    print(f"\nTiming results:\n{timing_path}")

    print("\nEvaluation finished successfully.")


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
