import sys
sys.path.insert(0, '.')
import os
import math
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
import json

from src.models.maskgit import MaskGIT

# Grupos para anomaly score por tipo
GROUP_MAP = {
    0:'SNIa', 1:'SNIa', 2:'SNIa',
    3:'SNII', 4:'SNII', 5:'SNII', 11:'SNII',
    6:'SNIbc', 7:'SNIbc', 8:'SNIbc', 9:'SNIbc', 10:'SNIbc',
    12:'SNIIn', 13:'SNIIn',
    14:'SLSN', 15:'SLSN',
    16:'PISN',
}
GROUP_NAMES = ['SNIa', 'SNIbc', 'SNII', 'SNIIn', 'SLSN', 'PISN']


class TokenDataset(Dataset):
    def __init__(self, tokens, labels=None):
        self.tokens = torch.from_numpy(tokens).long()
        self.labels = labels  # array de ints o None

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.tokens[idx], int(self.labels[idx])
        return self.tokens[idx], -1


class MaskGITWithViz(MaskGIT):

    def __init__(self, image_dir, vqvae_ckpt=None, n_viz=4, **kwargs):
        super().__init__(**kwargs)
        self.image_dir = Path(image_dir)
        self.n_viz     = n_viz
        self.vqvae     = None

        if vqvae_ckpt and os.path.exists(vqvae_ckpt):
            self._load_vqvae(vqvae_ckpt)

        pngs = sorted(self.image_dir.glob("*.png"))[:n_viz]
        self.viz_snids = [p.stem for p in pngs]

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # Acumuladores para anomaly score por grupo
        self._val_scores  = []
        self._val_labels  = []

    def _load_vqvae(self, ckpt_path):
        from src.models.vqvae.model import VQVAE
        self.vqvae = VQVAE.load_from_checkpoint(ckpt_path)
        self.vqvae.eval()
        for p in self.vqvae.parameters():
            p.requires_grad = False

    def _approx_anomaly_score(self, tokens):
        """
        Aproximación rápida del anomaly score: un forward pass con 50% de tokens enmascarados.
        tokens: (B, T)  →  score: (B,)
        """
        B, T = tokens.shape
        masked = tokens.clone()
        mask   = torch.zeros(B, T, dtype=torch.bool, device=tokens.device)
        perm   = torch.stack([torch.randperm(T, device=tokens.device) for _ in range(B)])
        n_mask = T // 2
        for i in range(B):
            mask[i, perm[i, :n_mask]] = True
            masked[i, mask[i]] = self.mask_token

        with torch.no_grad():
            logits    = self(masked)                                      # (B, T, V)
            log_probs = F.log_softmax(logits, dim=-1)                     # (B, T, V)
            token_lp  = log_probs.gather(2, tokens.unsqueeze(2)).squeeze(2)  # (B, T)
            score     = -(token_lp * mask.float()).sum(1) / mask.float().sum(1)
        return score

    def _step(self, batch, stage):
        tokens, labels = batch
        masked, mask = self._mask_tokens(tokens)
        logits = self(masked)
        loss   = F.cross_entropy(logits[mask], tokens[mask])
        if not torch.isfinite(loss):
            return None

        # Perplexity del prior
        perplexity = torch.exp(loss.detach())

        self.log(f"{stage}/loss",       loss,       prog_bar=True,
                 on_step=(stage=="train"), on_epoch=True, sync_dist=False)
        self.log(f"{stage}/perplexity", perplexity, prog_bar=False,
                 on_step=False, on_epoch=True, sync_dist=False)

        # Acumular scores para val
        if stage == "val":
            scores = self._approx_anomaly_score(tokens).cpu().numpy()
            self._val_scores.append(scores)
            self._val_labels.append(labels.cpu().numpy())

        return loss

    def on_validation_epoch_start(self):
        self._val_scores = []
        self._val_labels = []

    def on_validation_epoch_end(self):
        if not isinstance(self.logger, WandbLogger):
            return

        # ── Anomaly score por grupo ────────────────────────────────
        if self._val_scores:
            all_scores = np.concatenate(self._val_scores)
            all_labels = np.concatenate(self._val_labels)

            # Histograma global
            self.logger.experiment.log({
                "anomaly_score/histogram": wandb.Histogram(all_scores[np.isfinite(all_scores)], num_bins=64),
                "epoch": self.current_epoch,
            })

            # Score medio por grupo
            group_scores = {}
            for label_idx, group_name in GROUP_MAP.items():
                mask = all_labels == label_idx
                if mask.sum() > 0:
                    group_scores[group_name] = float(all_scores[mask].mean())

            # Agregar por nombre de grupo (promedio de subtipos)
            group_mean = {}
            for gname in GROUP_NAMES:
                vals = [v for k, v in group_scores.items() if GROUP_MAP.get(
                    next((i for i, g in GROUP_MAP.items() if g == gname and
                          all_labels[all_labels == i].shape[0] > 0), -1), '') == gname]
                # Forma más directa:
                idxs = [i for i, g in GROUP_MAP.items() if g == gname]
                mask = np.isin(all_labels, idxs)
                if mask.sum() > 0:
                    group_mean[f"anomaly_score/group/{gname}"] = float(all_scores[mask].mean())

            self.logger.experiment.log({**group_mean, "epoch": self.current_epoch})

            # Tabla resumen
            table = wandb.Table(columns=["grupo", "score_medio", "n_muestras"])
            for gname in GROUP_NAMES:
                idxs = [i for i, g in GROUP_MAP.items() if g == gname]
                mask = np.isin(all_labels, idxs)
                if mask.sum() > 0:
                    table.add_data(gname, float(all_scores[mask].mean()), int(mask.sum()))
            self.logger.experiment.log({
                "anomaly_score/tabla_grupos": table,
                "epoch": self.current_epoch,
            })

        # ── Visualizaciones ───────────────────────────────────────
        self._log_visual_examples()

    def _log_visual_examples(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if self.vqvae is None:
            return

        images_to_log = []
        for snid in self.viz_snids[:self.n_viz]:
            img_path = self.image_dir / f"{snid}.png"
            if not img_path.exists():
                continue

            img = Image.open(img_path).convert("RGB")
            x   = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                _, _, tokens, _, _, _, _ = self.vqvae(x)
                token_map = tokens[0].cpu().numpy()
                flat  = tokens.view(1, -1)
                score = self._approx_anomaly_score(flat).item()

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(img)
            axes[0].set_title(f"Original\n{snid}")
            axes[0].axis("off")
            im = axes[1].imshow(token_map, cmap="tab20", aspect="auto")
            axes[1].set_title(f"Token map\nscore={score:.3f}")
            axes[1].axis("off")
            plt.colorbar(im, ax=axes[1], fraction=0.046)
            plt.tight_layout()

            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf  = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
            images_to_log.append(wandb.Image(buf, caption=f"{snid} | score={score:.3f}"))
            plt.close(fig)

        if images_to_log:
            self.logger.experiment.log({
                "ejemplos_visuales": images_to_log,
                "epoch": self.current_epoch,
            })


def main():
    torch.cuda.empty_cache()

    # ── Tokens y labels ───────────────────────────────────────────
    tokens = np.load('data/tokens/tokens_cls.npy')
    print(f"Tokens cargados: {tokens.shape}", flush=True)

    # Cargar paths para mapear a labels
    with open('data/tokens/paths_cls.txt') as f:
        paths = [l.strip() for l in f.readlines()]
    with open('data/folds/labels.json') as f:
        labels_map = json.load(f)

    # Extraer SNID del path y mapear a label
    labels = np.array([
        labels_map.get(Path(p).stem, -1) for p in paths
    ], dtype=np.int64)
    print(f"Labels cargados: {labels.shape}, únicos: {np.unique(labels[labels>=0]).shape[0]} clases", flush=True)

    n       = len(tokens)
    n_train = int(n * 0.8)
    n_val   = n - n_train

    dataset  = TokenDataset(tokens, labels)
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                              num_workers=4, pin_memory=False)

    print(f"Train: {n_train:,} | Val: {n_val:,}", flush=True)

    VQVAE_CKPT = "results/vqvae/checkpoints_cls01/vqvae-epoch=07-train/loss=0.1188.ckpt"

    model = MaskGITWithViz(
        image_dir     = "data/images/elasticc_1/2grid",
        vqvae_ckpt    = VQVAE_CKPT,
        n_viz         = 4,
        num_tokens    = 512,
        seq_len       = 1024,
        hidden_dim    = 256,
        n_layers      = 4,
        n_heads       = 4,
        ff_mult       = 4,
        dropout       = 0.1,
        learning_rate = 1e-4,
    )

    os.makedirs('results/maskgit_cls/checkpoints', exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath    = 'results/maskgit_cls/checkpoints',
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
        name     = "maskgit-cls-v1",
        save_dir = "results/maskgit_cls/logs",
    )

    trainer = L.Trainer(
        max_epochs           = 100,
        accelerator          = "gpu",
        devices              = 1,
        precision            = "16-mixed",
        gradient_clip_val       = 0.5,
        gradient_clip_algorithm = "norm",
        callbacks            = [checkpoint, early_stop],
        logger               = logger,
        log_every_n_steps    = 100,
        val_check_interval   = 1.0,
        num_sanity_val_steps = 2,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path="results/maskgit_cls/checkpoints/maskgit-epoch=41-val/loss=3.0498.ckpt")
    print(f"Mejor checkpoint: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()
