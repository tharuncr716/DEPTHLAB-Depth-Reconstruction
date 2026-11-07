
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT_DIR / "outputs" / "evaluation" / "comparison_summary.csv"

GRAPH_DIR = ROOT_DIR / "outputs" / "evaluation" / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

metrics = ["RMSE", "MAE", "PSNR", "SSIM"]

# Keep methods in a consistent order.
method_order = [
    "Nearest",
    "Bilinear",
    "Bicubic",
    "Bilateral",
    "Joint Bilateral",
    "Guided Filter",
    "CNN",
    "Hybrid"
]

for metric in metrics:

    plt.figure(figsize=(12, 6))

    for method in method_order:

        method_data = df[df["Method"] == method].sort_values("Factor")

        if not method_data.empty:
            plt.plot(
                method_data["Factor"],
                method_data[metric],
                marker="o",
                linewidth=2,
                label=method
            )

    plt.title(
        f"{metric} Comparison Across Upsampling Factors",
        fontsize=14
    )

    plt.xlabel("Upsampling Factor")
    plt.ylabel(metric)

    plt.xticks([2, 4, 8], ["2x", "4x", "8x"])

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = GRAPH_DIR / f"{metric.lower()}_comparison.png"

    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved: {output_path}")

print("\nAll evaluation graphs generated successfully.")
