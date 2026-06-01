import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import wandb
import numpy as np
from .encoder import Encoder
from .decoder import Decoder
from .codebook import VectorQuantizerEMA as VectorQuantizer


class CodebookClassifier(nn.Module):
    """
    Clasificador MLP sobre la representación cuantizada z_q.

    Por qué sobre z_q y no z_e:
        - z_q es el espacio discreto que usa el decoder y el prior MaskGIT
        - El gradiente fluye hacia el encoder vía straight-through estimator,
          presionando al codebook a organizarse semánticamente
        - Resultado esperado: UMAP del codebook con clustering por tipo de SN

    Pipeline:
        z_q (B, D, 32, 32)
          → GlobalAvgPool → (B, D)
          → Linear(D → D//2) → ReLU → Dropout
          → Linear(D//2 → num_classes)
    """
    def __init__(self, embedding_dim=256, num_classes=17, dropout=0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes),
        )

    def forward(self, z_q):
        return self.mlp(self.pool(z_q))   # (B, num_classes)


class VQVAE(L.LightningModule):
    def __init__(self, in_channels=3, hidden_channels=[64, 128, 256],
                 embedding_dim=256, num_embeddings=512,
                 commitment_cost=0.25, learning_rate=2e-4,
                 num_classes=0, cls_weight=1.0, cls_dropout=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes   = num_classes
        self.cls_weight    = cls_weight

        self.encoder    = Encoder(in_channels, hidden_channels, embedding_dim)
        self.codebook   = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder    = Decoder(in_channels, list(reversed(hidden_channels)), embedding_dim)

        self.classifier = (
            CodebookClassifier(embedding_dim, num_classes, cls_dropout)
            if num_classes > 0 else None
        )
        self._fixed_examples = None

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices, perplexity, cb_loss, cm_loss = self.codebook(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, vq_loss, indices, perplexity, cb_loss, cm_loss, z_q

    def on_fit_start(self):
        dm = self.trainer.datamodule
        dm.setup()
        val_loader = dm.val_dataloader()
        batch = next(iter(val_loader))
        if isinstance(batch, (list, tuple)) and not isinstance(batch, torch.Tensor):
            images = batch[0]
        else:
            images = batch
        self._fixed_examples = images[:8].detach().cpu()
        print(f"Ejemplos fijos guardados: {self._fixed_examples.shape}", flush=True)

    def _step(self, batch, stage):
        # Desempacar batch — soporta solo imágenes o (imágenes, labels)
        if isinstance(batch, (list, tuple)) and not isinstance(batch, torch.Tensor):
            x      = batch[0]
            labels = batch[1] if (len(batch) > 1 and self.classifier is not None) else None
        else:
            x      = batch
            labels = None

        x_hat, vq_loss, _, perplexity, cb_loss, cm_loss, z_q = self(x)
        recon_loss = F.mse_loss(x_hat, x)
        loss       = recon_loss + vq_loss

        cls_loss = torch.tensor(0.0, device=self.device)
        acc      = torch.tensor(0.0, device=self.device)
        if self.classifier is not None and labels is not None:
            valid = labels >= 0
            if valid.any():
                logits   = self.classifier(z_q[valid])
                cls_loss = F.cross_entropy(logits, labels[valid])
                loss     = loss + self.cls_weight * cls_loss
                preds    = logits.argmax(dim=1)
                acc      = (preds == labels[valid]).float().mean()

        self.log(f"{stage}/loss",            loss,       prog_bar=True,  on_step=True, on_epoch=False, sync_dist=False)
        self.log(f"{stage}/recon_loss",      recon_loss, prog_bar=True,  on_step=True, on_epoch=False, sync_dist=False)
        self.log(f"{stage}/vq_loss",         vq_loss,    prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)
        self.log(f"{stage}/codebook_loss",   cb_loss,    prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)
        self.log(f"{stage}/commitment_loss", cm_loss,    prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)
        self.log(f"{stage}/perplexity",      perplexity, prog_bar=True,  on_step=True, on_epoch=False, sync_dist=False)
        if self.classifier is not None:
            self.log(f"{stage}/cls_loss", cls_loss, prog_bar=True,  on_step=True, on_epoch=False, sync_dist=False)
            self.log(f"{stage}/acc",      acc,      prog_bar=True,  on_step=True, on_epoch=False, sync_dist=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
        if batch_idx == 0 and self._fixed_examples is not None:
            self._log_reconstructions()
        return loss

    def _log_reconstructions(self):
        x = self._fixed_examples.to(self.device)
        with torch.no_grad():
            x_hat, _, _, _, _, _, _ = self(x)
        x_show     = (x.cpu().clamp(-1, 1) + 1) / 2
        x_hat_show = (x_hat.cpu().clamp(-1, 1) + 1) / 2
        images = []
        for i in range(min(4, len(x))):
            orig  = x_show[i].permute(1, 2, 0).numpy()
            recon = x_hat_show[i].permute(1, 2, 0).numpy()
            pair  = np.concatenate([orig, recon], axis=1)
            images.append(wandb.Image(pair, caption=f"Orig | Recon (epoch {self.current_epoch})"))
        self.logger.experiment.log({"reconstructions": images, "epoch": self.current_epoch})
        del x, x_hat, x_show, x_hat_show
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def encode_to_indices(self, x):
        z_e = self.encoder(x)
        _, _, indices, _, _, _ = self.codebook(z_e)
        return indices

    def decode_from_indices(self, indices):
        B, H, W = indices.shape
        z_q = self.codebook.embedding[indices.view(-1)]
        z_q = z_q.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)

    def reconstruction_error(self, x):
        x_hat, _, _, _, _, _, _ = self(x)
        return F.mse_loss(x_hat, x, reduction='none').mean(dim=[1, 2, 3])
