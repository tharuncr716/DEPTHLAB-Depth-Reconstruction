
import sys
import time
import threading
from pathlib import Path

import cv2
import h5py
import numpy as np
import psutil
import torch
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = ROOT_DIR / "dataset" / "processed" / "nyu_small.h5"
SPLIT_PATH = ROOT_DIR / "dataset" / "processed" / "small_splits.json"
MODEL_PATH = ROOT_DIR / "models" / "best_depth_model.pth"

OUTPUT_DIR = ROOT_DIR / "outputs" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_DIR))

from models.depth_cnn import DepthRefinementCNN
from classical.classical_methods import apply_methods


DEVICE = torch.device("cpu")
FACTORS = [2, 4, 8]


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

    model.eval()

    return model


def get_sample():

    with h5py.File(DATASET_PATH, "r") as f:

        import json

        with open(SPLIT_PATH, "r") as file:
            splits = json.load(file)

        sample_index = splits["test"][0]

        rgb = np.transpose(
            f["images"][sample_index],
            (1, 2, 0)
        ).astype(np.float32) / 255.0

        depth = f["depths"][sample_index][:].astype(np.float32)

    return rgb, depth


class MemoryMonitor:

    def __init__(self, interval=0.01):

        self.process = psutil.Process()
        self.interval = interval
        self.peak_mb = 0
        self.running = False
        self.thread = None

    def monitor(self):

        while self.running:

            try:
                memory_mb = (
                    self.process.memory_info().rss / (1024 ** 2)
                )

                self.peak_mb = max(self.peak_mb, memory_mb)

            except psutil.Error:
                break

            time.sleep(self.interval)

    def start(self):

        self.peak_mb = (
            self.process.memory_info().rss / (1024 ** 2)
        )

        self.running = True

        self.thread = threading.Thread(
            target=self.monitor,
            daemon=True
        )

        self.thread.start()

    def stop(self):

        self.running = False

        if self.thread is not None:
            self.thread.join()

        current_mb = (
            self.process.memory_info().rss / (1024 ** 2)
        )

        self.peak_mb = max(self.peak_mb, current_mb)

        return self.peak_mb


def resize_depth(depth, factor):

    height, width = depth.shape

    low = cv2.resize(
        depth,
        (width // factor, height // factor),
        interpolation=cv2.INTER_AREA
    )

    bicubic = cv2.resize(
        low,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

    return low, bicubic


def run_cnn(model, rgb, bicubic):

    rgb_tensor = torch.from_numpy(
        rgb.transpose(2, 0, 1).copy()
    ).float().unsqueeze(0)

    depth_tensor = torch.from_numpy(
        bicubic.copy()
    ).float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        return model(rgb_tensor, depth_tensor)


def measure_memory(method_name, function):

    monitor = MemoryMonitor()

    monitor.start()

    start = time.perf_counter()

    function()

    elapsed = time.perf_counter() - start

    peak_memory = monitor.stop()

    return {
        "Method": method_name,
        "Peak_Process_RAM_MB": peak_memory,
        "Execution_Time_Seconds": elapsed
    }


def main():

    print("\n====================================")
    print(" DEPTHLAB - MEMORY ANALYSIS")
    print("====================================")

    print("Device:", DEVICE)

    print("\nLoading model...")

    model = load_model()

    model_size_mb = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    ) / (1024 ** 2)

    print(f"Model parameter size: {model_size_mb:.3f} MB")

    rgb, gt = get_sample()

    rgb_uint8 = (
        np.clip(rgb, 0, 1) * 255
    ).astype(np.uint8)

    results = []

    for factor in FACTORS:

        print(f"\nEvaluating {factor}x...")

        low_depth, bicubic = resize_depth(gt, factor)

        # Warm-up CNN before measurement.
        run_cnn(model, rgb, bicubic)

        cnn_result = measure_memory(
            "CNN",
            lambda: run_cnn(model, rgb, bicubic)
        )

        cnn_result["Factor"] = factor
        results.append(cnn_result)

        # Measure all classical methods together.
        classical_result = measure_memory(
            "Classical_Batch",
            lambda: apply_methods(rgb_uint8, low_depth)
        )

        classical_result["Factor"] = factor
        results.append(classical_result)

    df = pd.DataFrame(results)

    output_path = OUTPUT_DIR / "memory_summary.csv"

    df.to_csv(output_path, index=False)

    print("\n====================================")
    print(" MEMORY RESULTS")
    print("====================================")

    print(df.to_string(index=False))

    print("\nModel parameter size:", f"{model_size_mb:.3f} MB")

    print("\nSaved:", output_path)

    print(
        "\nNote: Peak RAM is the total process RSS, "
        "not the isolated memory consumption of each method."
    )


if __name__ == "__main__":
    main()
