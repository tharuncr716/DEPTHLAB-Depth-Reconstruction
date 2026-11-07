
import h5py
import json
import cv2
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

MAT_PATH = r"C:\Users\tharu\Downloads\nyu_depth_v2_labeled (1).mat"

SPLIT_PATH = PROJECT / "dataset" / "splits" / "dataset_splits.json"

OUTPUT = PROJECT / "dataset" / "processed" / "nyu_small.h5"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

TARGET_WIDTH = 320
TARGET_HEIGHT = 240

# Smaller experimental subset
LIMITS = {
    "train": 300,
    "validation": 60,
    "test": 60
}

with open(SPLIT_PATH, "r") as f:
    splits = json.load(f)

selected = {}

for name, count in LIMITS.items():
    selected[name] = splits[name][:count]

    print(name, "samples:", len(selected[name]))

all_indices = (
    selected["train"]
    + selected["validation"]
    + selected["test"]
)

# Store original sample indices and their new positions
positions = {
    int(original): position
    for position, original in enumerate(all_indices)
}

new_splits = {
    name: [
        positions[int(index)]
        for index in indices
    ]
    for name, indices in selected.items()
}

with h5py.File(MAT_PATH, "r") as source:

    with h5py.File(OUTPUT, "w") as destination:

        count = len(all_indices)

        images = destination.create_dataset(
            "images",
            shape=(count, 3, TARGET_HEIGHT, TARGET_WIDTH),
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4
        )

        depths = destination.create_dataset(
            "depths",
            shape=(count, TARGET_HEIGHT, TARGET_WIDTH),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4
        )

        for position, original_index in enumerate(all_indices):

            rgb = np.transpose(
                source["images"][original_index],
                (2, 1, 0)
            )

            depth = np.transpose(
                source["depths"][original_index],
                (1, 0)
            )

            rgb_small = cv2.resize(
                rgb,
                (TARGET_WIDTH, TARGET_HEIGHT),
                interpolation=cv2.INTER_AREA
            )

            depth_small = cv2.resize(
                depth,
                (TARGET_WIDTH, TARGET_HEIGHT),
                interpolation=cv2.INTER_AREA
            )

            images[position] = np.transpose(
                rgb_small, (2, 0, 1)
            )

            depths[position] = depth_small

            if (position + 1) % 50 == 0:
                print(
                    f"Processed {position + 1}/{count}"
                )

        destination.attrs["original_indices"] = np.array(
            all_indices
        )

        print("\nSmall dataset created successfully!")

with open(
    OUTPUT.parent / "small_splits.json", "w"
) as f:
    json.dump(new_splits, f, indent=4)

print("Dataset:", OUTPUT)
print("Splits:", OUTPUT.parent / "small_splits.json")
print("Original samples:", len(all_indices))
