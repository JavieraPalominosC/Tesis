"""
VQ-VAE para detección de outliers en curvas de luz de supernovas.

Basado en:
- VQ-VAE original: Van Den Oord et al. (2017) - arXiv:1711.00937
- TimeVQVAE-AD: Lee et al. (2023) - arXiv:2311.12550

Arquitectura:
    Encoder CNN → Vector Quantizer (Codebook) → Decoder CNN

Las imágenes de entrada son 2grid: 6 bandas (u, g, r, i, z, Y) apiladas verticalmente.
Cada imagen es RGB de tamaño configurable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# ENCODER
# ─────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Encoder CNN que comprime la imagen 2grid a un espacio latente.
    
    Entrada:  (B, 3, H, W)  - imagen RGB
    Salida:   (B, latent_dim, H', W')  - representación latente continua z_e
    """
    def __init__(self, in_channels=3, latent_dim=256, hidden_dims=[64, 128, 256]):
        super().__init__()
        
        layers = []
        prev_ch = in_channels
        for h_dim in hidden_dims:
            layers += [
                nn.Conv2d(prev_ch, h_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU(inplace=True),
            ]
            prev_ch = h_dim
        
        # Capa final sin activación (salida al codebook)
        layers.append(nn.Conv2d(prev_ch, latent_dim, kernel_size=3, stride=1, padding=1))
        
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


# ─────────────────────────────────────────────
# VECTOR QUANTIZER
# ─────────────────────────────────────────────

class VectorQuantizer(nn.Module):
    """
    Cuantizador vectorial con straight-through estimator.
    
    Mapea z_e (continuo) → z_q (discreto) usando el codebook.
    El codebook aprende K vectores de embeddings de dimensión D.
    
    Referencia: Eq. (1), (2), (3) en VQ-VAE paper (1711.00937)
    
    Args:
        num_embeddings (K): Número de vectores en el codebook
        embedding_dim (D):  Dimensión de cada vector
        commitment_cost:    Peso del término de commitment loss (β en el paper)
    """
    def __init__(self, num_embeddings=512, embedding_dim=256, commitment_cost=0.25):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook: K vectores de dimensión D
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(
            -1 / num_embeddings, 1 / num_embeddings
        )

    def forward(self, z_e):
        """
        Args:
            z_e: (B, D, H, W) - salida continua del encoder
        Returns:
            z_q:        (B, D, H, W) - representación cuantizada
            loss:       scalar       - VQ loss + commitment loss
            indices:    (B, H, W)    - índices del codebook (tokens s)
        """
        # (B, D, H, W) → (B, H, W, D)
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        flat_z = z_e.view(-1, self.embedding_dim)  # (B*H*W, D)
        
        # Distancias al codebook: ||z_e - e_k||^2
        distances = (
            flat_z.pow(2).sum(1, keepdim=True)
            - 2 * flat_z @ self.embeddings.weight.t()
            + self.embeddings.weight.pow(2).sum(1)
        )  # (B*H*W, K)
        
        # Índice más cercano en el codebook
        indices = distances.argmin(1)  # (B*H*W,)
        
        # Lookup en el codebook
        z_q = self.embeddings(indices).view(z_e.shape)  # (B, H, W, D)
        
        # VQ Loss: ||sg[z_e] - e||^2 + β * ||z_e - sg[e]||^2
        vq_loss = F.mse_loss(z_q.detach(), z_e)
        commitment_loss = F.mse_loss(z_q, z_e.detach())
        loss = vq_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator: gradientes pasan como si z_q = z_e
        z_q = z_e + (z_q - z_e).detach()
        
        # (B, H, W, D) → (B, D, H, W)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        indices = indices.view(z_e.shape[:-1])  # (B, H, W)
        
        return z_q, loss, indices


# ─────────────────────────────────────────────
# DECODER
# ─────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Decoder CNN que reconstruye la imagen desde z_q.
    
    Entrada:  (B, latent_dim, H', W')  - representación cuantizada z_q
    Salida:   (B, 3, H, W)             - imagen reconstruida
    """
    def __init__(self, latent_dim=256, out_channels=3, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_ch = latent_dim
        for h_dim in hidden_dims:
            layers += [
                nn.ConvTranspose2d(prev_ch, h_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU(inplace=True),
            ]
            prev_ch = h_dim
        
        # Capa final con Sigmoid (imágenes normalizadas entre 0 y 1)
        layers += [
            nn.ConvTranspose2d(prev_ch, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        ]
        
        self.decoder = nn.Sequential(*layers)

    def forward(self, z_q):
        return self.decoder(z_q)


# ─────────────────────────────────────────────
# VQ-VAE COMPLETO
# ─────────────────────────────────────────────

class VQVAE(nn.Module):
    """
    VQ-VAE completo para imágenes 2grid de curvas de luz de supernovas.
    
    Pipeline:
        imagen → Encoder → z_e → VectorQuantizer → z_q → Decoder → reconstrucción
    
    La pérdida total es:
        L = ||x - x_hat||^2 + ||sg[z_e] - e||^2 + β * ||z_e - sg[e]||^2
    
    Args:
        in_channels:      Canales de entrada (3 para RGB)
        latent_dim:       Dimensión del espacio latente
        num_embeddings:   Tamaño del codebook (K)
        commitment_cost:  Peso del commitment loss (β)
        hidden_dims:      Canales ocultos del encoder/decoder
    """
    def __init__(
        self,
        in_channels=3,
        latent_dim=256,
        num_embeddings=512,
        commitment_cost=0.25,
        hidden_dims=[64, 128, 256],
    ):
        super().__init__()
        
        self.encoder = Encoder(in_channels, latent_dim, hidden_dims)
        self.vq = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(latent_dim, in_channels, hidden_dims[::-1])

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) - imagen 2grid normalizada entre 0 y 1
        Returns:
            x_hat:   (B, 3, H, W) - imagen reconstruida
            vq_loss: scalar       - pérdida del codebook
            indices: (B, H', W')  - tokens del codebook (para anomaly detection)
        """
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, vq_loss, indices

    def encode(self, x):
        """Devuelve los tokens del codebook para una imagen."""
        z_e = self.encoder(x)
        _, _, indices = self.vq(z_e)
        return indices

    def decode_from_indices(self, indices):
        """Reconstruye imagen desde índices del codebook."""
        B, H, W = indices.shape
        z_q = self.vq.embeddings(indices.view(-1))
        z_q = z_q.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)

    def reconstruction_error(self, x):
        """
        Calcula el error de reconstrucción por imagen.
        Usado como anomaly score: mayor error → más anómalo.
        
        Returns:
            error: (B,) - MSE por imagen
        """
        x_hat, _, _ = self.forward(x)
        error = F.mse_loss(x_hat, x, reduction='none')
        return error.mean(dim=[1, 2, 3])  # promedio por imagen
