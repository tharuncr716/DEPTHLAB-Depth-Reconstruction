
import h5py
import cv2
import json
import numpy as np
import torch

from torch.utils.data import Dataset


class NYUDepthDataset(Dataset):

    def __init__(
        self,
        h5_path,
        split_path,
        split="train",
        factor=None
    ):
        self.h5_path = str(h5_path)
        self.split = split
        self.factor = factor
        self.file = None

        with open(split_path, "r") as f:
            splits = json.load(f)

        self.indices = splits[split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")

        sample_index = self.indices[index]

        rgb = np.transpose(
            self.file["images"][sample_index],
            (1, 2, 0)
        )

        depth = self.file["depths"][sample_index][:]

        rgb = rgb.astype(np.float32) / 255.0
        depth = depth.astype(np.float32)

        if not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError(f"Invalid depth in sample {sample_index}")

        height, width = depth.shape

        if self.split == "train" and self.factor is None:
            factor = int(np.random.choice([2, 4, 8]))
        else:
            factor = self.factor if self.factor else 4

        low_depth = cv2.resize(
            depth,
            (width // factor, height // factor),
            interpolation=cv2.INTER_AREA
        )

        upsampled = cv2.resize(
            low_depth,
            (width, height),
            interpolation=cv2.INTER_CUBIC
        )

        if self.split == "train" and np.random.rand() < 0.5:
            rgb = np.flip(rgb, axis=1).copy()
            depth = np.flip(depth, axis=1).copy()
            upsampled = np.flip(upsampled, axis=1).copy()

        rgb_tensor = torch.from_numpy(
            np.transpose(rgb, (2, 0, 1)).copy()
        ).float()

        low_tensor = torch.from_numpy(
            upsampled[None].copy()
        ).float()

        target_tensor = torch.from_numpy(
            depth[None].copy()
        ).float()

        return rgb_tensor, low_tensor, target_tensor
