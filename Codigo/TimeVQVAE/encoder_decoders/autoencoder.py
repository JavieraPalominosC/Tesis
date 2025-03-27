import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        """
        :param input_dim: Longitud de la serie de tiempo después de la interpolación
        :param latent_dim: Dimensión del espacio latente
        """
        super().__init__()

        # 🔹 Encoder: Comprime la serie de tiempo a una representación latente `z`
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)  # `z` (representación comprimida)
        )

        # 🔹 Decoder: Reconstruye la serie a partir de `z`
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)  # Reconstrucción de la serie
        )

    def forward(self, x):
        z = self.encoder(x)  # Extraer representación latente
        x_reconstructed = self.decoder(z)  # Reconstrucción
        return x_reconstructed, z
