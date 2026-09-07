"""
MaskGIT en waterfall_global, pero la mascara de entrenamiento es una
ventana rectangular contigua en el mapa 32x32 (no posiciones al azar).

La fraccion enmascarada sigue el schedule coseno del MaskGIT original.
El score de validacion (_approx_anomaly_score) no cambia: sigue siendo
50% random, igual que en eval.

Uso: python scripts/train_maskgit_window.py waterfall_global
"""
import sys
sys.path.insert(0, '.')
import os
import math
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from scripts.train_maskgit_cls import TokenDataset, MaskGITWithViz

REP = sys.argv[1] if len(sys.argv) > 1 else 'waterfall_global'
assert REP in ('overlay_lines', 'points', 'waterfall', 'waterfall_global')

LABELS_PATHS = {
    'overlay_lines':    'data/labels_overlay_full.json',
    'points':           'data/labels_points_full.json',
    'waterfall':        'data/labels_waterfall_full.json',
    'waterfall_global': 'data/labels_waterfall_global_full.json',
}


class MaskGITWindow(MaskGITWithViz):
    """Enmascara un rectangulo 2D contiguo en vez de token indices sueltos."""

    def __init__(self, image_dir='.', vqvae_ckpt=None, n_viz=0, **kwargs):
        # Se salta MaskGITWithViz.__init__ a proposito: ese init hace glob sobre
        # el directorio de imagenes, que aca tiene 1.39M PNGs. Solo se hereda su
        # _step, que desempaqueta (tokens, label) del TokenDataset.
        from src.models.maskgit import MaskGIT
        MaskGIT.__init__(self, **kwargs)
        self.image_dir = Path(image_dir)
        self.n_viz = 0
        self.vqvae = None
        self.viz_snids = []
        self._val_scores = []
        self._val_labels = []

    def _mask_tokens(self, tokens):
        B, T = tokens.shape
        H = W = int(round(math.sqrt(T)))
        if H * W != T:
            return super()._mask_tokens(tokens)

        r = torch.rand(B, device=tokens.device)
        r = (1 - torch.cos(r * math.pi)) / 2
        n_mask = (r * T).long().clamp(min=1)

        masked = tokens.clone()
        mask = torch.zeros(B, T, dtype=torch.bool, device=tokens.device)
        for i in range(B):
            n = int(n_mask[i].item())
            w = int(torch.randint(1, W + 1, (1,), device=tokens.device).item())
            h = min(H, max(1, (n + w - 1) // w))
            w = min(W, max(1, (n + h - 1) // h))
            r0 = 0 if h == H else int(torch.randint(0, H - h + 1, (1,), device=tokens.device).item())
            c0 = 0 if w == W else int(torch.randint(0, W - w + 1, (1,), device=tokens.device).item())
            grid = torch.zeros(H, W, dtype=torch.bool, device=tokens.device)
            grid[r0:r0 + h, c0:c0 + w] = True
            flat = grid.reshape(-1)
            idx = flat.nonzero(as_tuple=False).view(-1)
            if idx.numel() > n:
                keep = idx[torch.randperm(idx.numel(), device=tokens.device)[:n]]
                flat[:] = False
                flat[keep] = True
            elif idx.numel() < n:
                extra = n - int(idx.numel())
                hole = (~flat).nonzero(as_tuple=False).view(-1)
                add = hole[torch.randperm(hole.numel(), device=tokens.device)[:extra]]
                flat[add] = True
            mask[i] = flat
            masked[i, mask[i]] = self.mask_token
        return masked, mask


def main():
    tokens = np.load(f'data/tokens/tokens_{REP}_full.npy')

    with open(f'data/tokens/paths_{REP}_full.txt') as f:
        paths = [l.strip() for l in f.readlines()]
    with open(LABELS_PATHS[REP]) as f:
        labels_map = json.load(f)

    labels = np.array([
        labels_map.get(Path(p).stem, -1) for p in paths
    ], dtype=np.int64)
    print(f"[{REP}] ventana 2D | tokens {tokens.shape} | labels {np.unique(labels[labels >= 0]).shape[0]}",
          flush=True)

    n = len(tokens)
    n_train = int(n * 0.8)
    n_val = n - n_train
    dataset = TokenDataset(tokens, labels)
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                            num_workers=4, pin_memory=False)
    print(f"[{REP}] Train: {n_train:,} | Val: {n_val:,}", flush=True)

    model = MaskGITWindow(
        num_tokens=512,
        seq_len=1024,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        ff_mult=4,
        dropout=0.1,
        learning_rate=1e-4,
    )

    ckpt_dir = f'results/maskgit_window/{REP}/checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='maskgit-window-{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=1,
        save_last=True,
    )
    early_stop = EarlyStopping(monitor='val/loss', patience=5, mode='min')

    # Permite retomar si SLURM corta el job por tiempo.
    resume = Path(ckpt_dir) / 'last.ckpt'
    if not resume.exists():
        # El nombre lleva "val/loss", y esa barra hace que Lightning escriba un
        # subdirectorio, asi que hay que buscar recursivamente.
        prev = sorted(Path(ckpt_dir).rglob('*.ckpt'), key=lambda p: p.stat().st_mtime)
        resume = prev[-1] if prev else None
    if resume is not None:
        print(f"[{REP}] retomando desde {resume}", flush=True)

    trainer = L.Trainer(
        max_epochs=50,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        gradient_clip_val=0.5,
        gradient_clip_algorithm='norm',
        callbacks=[checkpoint, early_stop],
        logger=False,
        enable_model_summary=False,
        log_every_n_steps=100,
        val_check_interval=1.0,
        num_sanity_val_steps=2,
        default_root_dir=f'results/maskgit_window/{REP}',
    )
    trainer.fit(model, train_loader, val_loader,
                ckpt_path=str(resume) if resume is not None else None)
    print(f"[{REP}] Mejor checkpoint: {checkpoint.best_model_path}", flush=True)


if __name__ == '__main__':
    main()
