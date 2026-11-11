"""VQ-VAE para superficies GP (1–2 canales). No toca el VQVAE del 2grid."""
import numpy as np
import torch
import wandb

from .model import VQVAE


class VQVAEGP(VQVAE):
    """Igual que VQVAE; solo adapta el log de reconstrucciones a 1–2 canales."""

    def _log_reconstructions(self):
        x = self._fixed_examples.to(self.device)
        with torch.no_grad():
            x_hat, _, _, _, _, _, _ = self(x)
        x_show = (x.cpu().clamp(-1, 1) + 1) / 2
        x_hat_show = (x_hat.cpu().clamp(-1, 1) + 1) / 2
        images = []
        for i in range(min(4, len(x))):
            orig = x_show[i]
            recon = x_hat_show[i]
            if orig.shape[0] == 1:
                orig = orig.repeat(3, 1, 1)
                recon = recon.repeat(3, 1, 1)
            elif orig.shape[0] == 2:
                z = torch.zeros(1, *orig.shape[1:])
                orig = torch.cat([orig, z], dim=0)
                recon = torch.cat([recon[:2], z], dim=0)
            orig = orig.permute(1, 2, 0).numpy()
            recon = recon.permute(1, 2, 0).numpy()
            pair = np.concatenate([orig, recon], axis=1)
            images.append(
                wandb.Image(pair, caption=f"Orig | Recon (epoch {self.current_epoch})")
            )
        self.logger.experiment.log(
            {"reconstructions": images, "epoch": self.current_epoch}
        )
        del x, x_hat, x_show, x_hat_show
        torch.cuda.empty_cache()
