
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class DepthRefinementCNN(nn.Module):

    def __init__(self, channels=32, blocks=5):

        super().__init__()

        # Input: RGB (3 channels) + interpolated depth (1)
        self.input_layer = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(blocks)]
        )

        self.output_layer = nn.Conv2d(
            channels, 1, 3, padding=1
        )

        # Start with zero depth correction
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, rgb, depth):

        # RGB: B x 3 x H x W
        # Depth: B x 1 x H x W

        x = torch.cat([rgb, depth], dim=1)

        features = self.input_layer(x)

        features = self.residual_blocks(features)

        correction = self.output_layer(features)

        # Residual learning
        refined_depth = depth + correction

        return torch.clamp(refined_depth, min=0.1)
