import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class Encoder(nn.Module):
    """
    Input:  (B, 3, 256, 256)
    Output: (B, embedding_dim, 32, 32)

    3 capas Conv2d con stride=2 reducen 256->128->64->32.
    2 ResBlocks refinan las features sin cambiar resolución.
    """
    def __init__(self, in_channels=3, hidden_channels=[64, 128, 256], embedding_dim=256):
        super().__init__()

        layers = []
        prev = in_channels
        for ch in hidden_channels:
            layers += [
                nn.Conv2d(prev, ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ]
            prev = ch

        # Conv 1x1 para ajustar a embedding_dim si es necesario
        layers += [nn.Conv2d(prev, embedding_dim, kernel_size=3, stride=1, padding=1)]

        self.conv = nn.Sequential(*layers)
        self.res = nn.Sequential(
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
        )

    def forward(self, x):
        return self.res(self.conv(x))
