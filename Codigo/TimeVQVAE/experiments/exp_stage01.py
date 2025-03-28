import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
import pytorch_lightning as pl

from encoder_decoders.autoencoder import Autoencoder  # Nuevo Autoencoder
from vector_quantization import VectorQuantize
from utils import quantize, linear_warmup_cosine_annealingLR


class ExpStage1(pl.LightningModule):
    def __init__(self, in_channels: int, input_length: int, config: dict):
        """
        :param input_length: length of input time series
        :param config: configs/config.yaml
        """
        super().__init__()
        self.input_length = input_length
        self.config = config

        # 🔹 Definir Autoencoder
        self.autoencoder = Autoencoder(input_length, config["VQ-VAE"]["latent_dim"])  # ✅ Ahora sí existe

        self.vq_model = VectorQuantize(
          dim=config["VQ-VAE"]["latent_dim"], 
          codebook_size=config["VQ-VAE"]["codebook_size"], 
          kmeans_init=config["VQ-VAE"]["kmeans_init"], 
          codebook_dim=config["VQ-VAE"]["codebook_dim"])
            # ✅ Pasamos solo los valores necesarios



        # 🔹 Decoder
        self.decoder = Autoencoder(input_length, config["VQ-VAE"]["latent_dim"]).decoder  # ✅ Ahora sí existe

    def forward(self, batch, batch_idx, return_x_rec: bool = False):
        """
        :param batch: input batch of time series (batch, channels, length)
        """
        x = batch  # Solo usamos la serie, ignoramos etiquetas si las hay
        print(x.shape)

        # 🔹 Paso 1: Autoencoder → Extraer Representación Latente
        z = self.autoencoder.encoder(x.view(x.shape[0], -1))

        # 🔹 Paso 2: Cuantización Vectorial
        z = (z - z.mean()) / (z.std() + 1e-5)  # 🔹 Normalización
        z = z.unsqueeze(1)  # Añade una dimensión: (batch_size, 1, latent_dim)

        # 🔹 Debugging
        print(f"Shape de z después de reshape: {z.shape}")  # Debería ser (20, 1, 128)

        z_q, s, vq_loss, perplexity = quantize(z, self.vq_model)  # ✅ Ahora con la forma correcta


        # 🔹 Paso 3: Decodificación
        x_rec = self.decoder(z_q)  # (batch, channels, length)
        x_rec = x_rec.transpose(1, 2)
        print("Forma de x_rec después de decodificación:", x_rec.shape)

        if return_x_rec:
            return x_rec

        # 🔹 Pérdidas
        recons_loss = F.mse_loss(x, x_rec.transpose(1, 2))
        total_loss = recons_loss + vq_loss["loss"]

        return {"recons_loss": recons_loss, "vq_loss": vq_loss, "total_loss": total_loss, "perplexity": perplexity}

    def training_step(self, batch, batch_idx):
        losses = self.forward(batch, batch_idx)

        # 🔹 Visualización con wandb en validación
        if batch_idx == 0 and not self.training:
            self.log_wandb(batch, losses)

        self.log('train/loss', losses["total_loss"])
        self.log('train/recons_loss', losses["recons_loss"])
        self.log('train/vq_loss', losses["vq_loss"]["loss"])
        self.log('train/perplexity', losses["perplexity"])
        return losses["total_loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        losses = self.forward(batch, batch_idx)

        # 🔹 Visualización con wandb
        if batch_idx == 0:
            self.log_wandb(batch, losses)

        self.log('val/loss', losses["total_loss"])
        self.log('val/recons_loss', losses["recons_loss"])
        self.log('val/vq_loss', losses["vq_loss"]["loss"])
        self.log('val/perplexity', losses["perplexity"])
        return losses

    def log_wandb(self, batch, losses):
        """🔹 Función para graficar series originales vs reconstruidas en wandb"""
        x = batch  # Asumiendo que x tiene la forma (batch_size, length, channels)
        x_rec = self.forward(batch, batch_idx=0, return_x_rec=True)

        # Seleccionar un índice de ejemplo aleatorio del batch
        b = np.random.randint(0, x.shape[0])

        fig, ax = plt.subplots(figsize=(6, 3))
        plt.title(f'step-{self.global_step} (blue:GT, orange:reconstructed)')

        # Asegurarse de eliminar la dimensión del canal para la graficación
        x_plotted = x[b, :, 0].cpu().numpy()  # Removemos la dimensión de canal para x
        x_rec_plotted = x_rec[b, :, 0].detach().cpu().numpy()  # Removemos la dimensión de canal para x_rec

        ax.plot(x_plotted, label="Original", color='blue')
        ax.plot(x_rec_plotted, label="Reconstrucción", color='orange', alpha=0.7)
        ax.legend()

        plt.tight_layout()
        wandb.log({"x vs x_rec (val)": wandb.Image(plt)})
        plt.close()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.config['exp_params']['lr'])
        total_steps = self.config['trainer_params']['max_steps']['stage1']
        warmup_steps = int(self.config['exp_params']['linear_warmup_rate'] * total_steps)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.config['exp_params']['lr'], total_steps=total_steps, pct_start=self.config['exp_params']['linear_warmup_rate'], anneal_strategy='cos', final_div_factor=(self.config['exp_params']['lr'] / self.config['exp_params']['min_lr']))

        return {'optimizer': opt, 'lr_scheduler': scheduler}
