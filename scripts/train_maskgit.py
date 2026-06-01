import sys
sys.path.insert(0, '.')
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
import wandb

from src.models.maskgit import MaskGIT


class TokenDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = torch.from_numpy(tokens).long()

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]


class MaskGITWithViz(MaskGIT):
    """MaskGIT con visualizaciones extra en wandb."""

    def __init__(self, image_dir, vqvae_ckpt=None, n_viz=4, **kwargs):
        super().__init__(**kwargs)
        self.image_dir  = Path(image_dir)
        self.n_viz      = n_viz
        self.vqvae      = None

        # Cargar VQ-VAE para decodificar tokens si se pasa checkpoint
        if vqvae_ckpt:
            self._load_vqvae(vqvae_ckpt)

        # Guardar algunos SNIDs fijos para visualizar siempre los mismos
        pngs = sorted(self.image_dir.glob("*.png"))[:n_viz]
        self.viz_snids = [p.stem for p in pngs]

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def _load_vqvae(self, ckpt_path):
        from src.models.vqvae.model import VQVAE
        self.vqvae = VQVAE.load_from_checkpoint(ckpt_path)
        self.vqvae.eval()
        for p in self.vqvae.parameters():
            p.requires_grad = False

    def _step(self, batch, stage):
        tokens = batch
        masked, mask = self._mask_tokens(tokens)
        logits = self(masked)
        loss = F.cross_entropy(logits[mask], tokens[mask])

        # Log con on_epoch=True para que EarlyStopping lo vea
        self.log(f"{stage}/loss", loss, prog_bar=True,
                 on_step=(stage == "train"), on_epoch=True, sync_dist=False)
        return loss

    def on_validation_epoch_end(self):
        """Al final de cada época de validación, loguear visualizaciones."""
        if not isinstance(self.logger, WandbLogger):
            return

        self._log_token_histogram()
        self._log_visual_examples()

    def _log_token_histogram(self):
        """Histograma de tokens predichos sobre un batch de val."""
        # Usar los tokens de viz_snids como proxy
        all_predicted = []
        for snid in self.viz_snids[:self.n_viz]:
            img_path = self.image_dir / f"{snid}.png"
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            x   = self.transform(img).unsqueeze(0).to(self.device)
            if self.vqvae is not None:
                with torch.no_grad():
                    _, _, tokens, _, _, _ = self.vqvae(x)  # tokens: (1, H, W)
                    all_predicted.append(tokens.flatten().cpu().numpy())

        if all_predicted:
            import numpy as np
            all_tok = np.concatenate(all_predicted)
            self.logger.experiment.log({
                "token_histogram": wandb.Histogram(all_tok, num_bins=64),
                "epoch": self.current_epoch,
            })

    def _log_visual_examples(self):
        """Loguear imágenes originales + mapa de tokens."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        images_to_log = []
        for snid in self.viz_snids[:self.n_viz]:
            img_path = self.image_dir / f"{snid}.png"
            if not img_path.exists():
                continue

            img = Image.open(img_path).convert("RGB")
            x   = self.transform(img).unsqueeze(0).to(self.device)

            if self.vqvae is None:
                continue

            with torch.no_grad():
                _, _, tokens, _, _, _ = self.vqvae(x)  # tokens: (1, 32, 32)
                token_map = tokens[0].cpu().numpy()

                # Anomaly score con MaskGIT
                flat = tokens.view(1, -1)               # (1, 1024)
                score = self.anomaly_score(flat).item()

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(img)
            axes[0].set_title(f"Original\n{snid}")
            axes[0].axis("off")

            im = axes[1].imshow(token_map, cmap="tab20", aspect="auto")
            axes[1].set_title(f"Token map\nscore={score:.2f}")
            axes[1].axis("off")
            plt.colorbar(im, ax=axes[1], fraction=0.046)
            plt.tight_layout()

            # Convertir figura a imagen wandb
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(h, w, 3)
            images_to_log.append(wandb.Image(buf, caption=f"{snid} | score={score:.2f}"))
            plt.close(fig)

        if images_to_log:
            self.logger.experiment.log({
                "ejemplos_visuales": images_to_log,
                "epoch": self.current_epoch,
            })


def main():
    torch.cuda.empty_cache()

    tokens = np.load('data/tokens/tokens.npy')
    print(f"Tokens cargados: {tokens.shape}", flush=True)

    n       = len(tokens)
    n_train = int(n * 0.8)
    n_val   = n - n_train

    dataset  = TokenDataset(tokens)
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                              num_workers=4, pin_memory=False)

    print(f"Train: {n_train:,} | Val: {n_val:,}", flush=True)

    VQVAE_CKPT = "results/vqvae/checkpoints/vqvae-epoch=06-train/loss=0.0003.ckpt"

    model = MaskGITWithViz(
        image_dir   = "data/images/elasticc_1/2grid",
        vqvae_ckpt  = VQVAE_CKPT if os.path.exists(VQVAE_CKPT) else None,
        n_viz       = 4,
        # hiperparámetros MaskGIT
        num_tokens    = 512,
        seq_len       = 1024,
        hidden_dim    = 256,
        n_layers      = 4,
        n_heads       = 4,
        ff_mult       = 4,
        dropout       = 0.1,
        learning_rate = 1e-4,
    )

    os.makedirs('results/maskgit/checkpoints', exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath    = 'results/maskgit/checkpoints',
        filename   = "maskgit-{epoch:02d}-{val/loss:.4f}",
        monitor    = "val/loss",
        mode       = "min",
        save_top_k = 3,
    )
    early_stop = EarlyStopping(
        monitor  = "val/loss",
        patience = 5,
        mode     = "min",
    )

    logger = WandbLogger(
        project  = "vqvae-supernovas",
        name     = "maskgit-elasticc1-v2",
        id       = "d1c0pnkj",
        resume   = "must",
        save_dir = "results/maskgit/logs",
    )

    trainer = L.Trainer(
        max_epochs           = 100,
        accelerator          = "gpu",
        devices              = 1,
        precision            = "16-mixed",
        gradient_clip_val    = 1.0,
        callbacks            = [checkpoint, early_stop],
        logger               = logger,
        log_every_n_steps    = 100,
        val_check_interval   = 1.0,
        num_sanity_val_steps = 2,
    )

    trainer.fit(model, train_loader, val_loader,
                ckpt_path="results/maskgit/checkpoints/maskgit-epoch=35-val/loss=2.5925.ckpt")
    print(f"Mejor checkpoint: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()
