
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]

CSV_PATH = (
    ROOT_DIR / "outputs" / "evaluation"
    / "classical_batch_timing.csv"
)

df = pd.read_csv(CSV_PATH)

print("\n========== RUNTIME ANALYSIS ==========\n")

summary = (
    df.groupby(["Factor", "Method"])["Time_Seconds"]
    .agg(["mean", "median", "min", "max"])
    .reset_index()
)

summary.columns = [
    "Factor",
    "Method",
    "Mean_Seconds",
    "Median_Seconds",
    "Min_Seconds",
    "Max_Seconds"
]

print(summary.to_string(index=False))

output_path = (
    ROOT_DIR / "outputs" / "evaluation"
    / "runtime_summary.csv"
)

summary.to_csv(output_path, index=False)

print("\nSaved runtime summary:", output_path)

print("\n========== AVERAGE RUNTIME PER IMAGE ==========\n")

for factor in [2, 4, 8]:

    factor_data = summary[
        summary["Factor"] == factor
    ]

    print(f"\n{factor}x:")

    for _, row in factor_data.iterrows():

        print(
            f"{row['Method']}: "
            f"{row['Mean_Seconds']:.6f} seconds/image"
        )
