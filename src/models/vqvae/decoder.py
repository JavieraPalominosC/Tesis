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


class Decoder(nn.Module):
    """
    Input:  (B, embedding_dim, 32, 32)
    Output: (B, 3, 256, 256)

    Espejo exacto del encoder:
    3 ConvTranspose2d con stride=2: 32->64->128->256
    Tanh al final para output en [-1, 1].
    """
    def __init__(self, out_channels=3, hidden_channels=[256, 128, 64], embedding_dim=256):
        super().__init__()

        self.res = nn.Sequential(
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
        )

        # hidden_channels = [256, 128, 64] → 3 upsamples → 32*8 = 256
        layers = []
        prev = embedding_dim
        for ch in hidden_channels:
            layers += [
                nn.ConvTranspose2d(prev, ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ]
            prev = ch

        # Conv final 1x1 sin upsample para llegar a out_channels
        layers += [
            nn.Conv2d(prev, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        ]

        self.deconv = nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv(self.res(x))
