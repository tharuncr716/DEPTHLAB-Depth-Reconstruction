
import sys
import json
import time
import h5py
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from skimage.metrics import structural_similarity

# Import methods from your classical implementation
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "classical"))

from classical_methods import (
    apply_methods,
    calculate_metrics
)

MAT_PATH = r"C:\Users\tharu\Downloads\nyu_depth_v2_labeled (1).mat"

SPLIT_PATH = PROJECT_DIR / "dataset" / "splits" / "dataset_splits.json"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "classical"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(SPLIT_PATH, "r") as f:
    splits = json.load(f)

test_indices = splits["test"]

factors = [2, 4, 8]

records = []

print("Total test images:", len(test_indices))

start_all = time.perf_counter()

with h5py.File(MAT_PATH, "r") as f:

    for position, sample_index in enumerate(test_indices):

        rgb = np.transpose(
            f["images"][sample_index], (2, 1, 0)
        )

        gt = np.transpose(
            f["depths"][sample_index], (1, 0)
        ).astype(np.float32)

        rgb = np.ascontiguousarray(rgb)
        gt = np.ascontiguousarray(gt)

        for factor in factors:

            height, width = gt.shape

            low_depth = cv2.resize(
                gt,
                (width // factor, height // factor),
                interpolation=cv2.INTER_AREA
            )

            method_start = time.perf_counter()

            results = apply_methods(rgb, low_depth)

            elapsed = time.perf_counter() - method_start

            for method, prediction in results.items():

                prediction = np.nan_to_num(
                    prediction,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )

                rmse, mae, psnr, ssim = calculate_metrics(
                    gt, prediction
                )

                records.append({
                    "Image_ID": sample_index,
                    "Scale": f"{factor}x",
                    "Method": method,
                    "RMSE": rmse,
                    "MAE": mae,
                    "PSNR": psnr,
                    "SSIM": ssim,
                    "Method_Runtime_Seconds": elapsed
                })

        if (position + 1) % 10 == 0:
            print(
                f"Processed {position + 1}/{len(test_indices)} images"
            )

# Save per-image results
results_df = pd.DataFrame(records)

results_df.to_csv(
    OUTPUT_DIR / "classical_test_per_image.csv",
    index=False
)

# Average metrics across all test images
summary = (
    results_df
    .groupby(["Scale", "Method"])
    [["RMSE", "MAE", "PSNR", "SSIM",
      "Method_Runtime_Seconds"]]
    .mean()
    .reset_index()
)

summary.to_csv(
    OUTPUT_DIR / "classical_test_summary.csv",
    index=False
)

print("\nFULL TEST SET RESULTS")
print(summary.to_string(index=False))

print("\nTotal execution time:",
      round(time.perf_counter() - start_all, 2),
      "seconds")

print("\nSaved:")
print(OUTPUT_DIR / "classical_test_per_image.csv")
print(OUTPUT_DIR / "classical_test_summary.csv")
