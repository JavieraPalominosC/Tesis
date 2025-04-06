import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        # Encoder: Conv1D → AvgPool → Linear → z
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),  # ← ahora 8 en vez de 4
            nn.Flatten(),             # (batch, 32*8)
            nn.Linear(32 * 8, latent_dim)
        )

        # Decoder: Linear → Unflatten → Upsample → Conv1D → salida
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 8),
            nn.Unflatten(1, (32, 8)),
            nn.Upsample(size=input_dim),  # upsample a 50
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x: (batch, timesteps, 1) → (batch, 1, timesteps)
        x = x.transpose(1, 2)
        z = self.encoder(x)                # (batch, latent_dim)
        x_rec = self.decoder(z)            # (batch, 1, timesteps)
        x_rec = x_rec.transpose(1, 2)      # (batch, timesteps, 1)
        return x_rec,