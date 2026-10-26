"""MaskGIT para una rama de la comparacion de codificacion de flujo.

Mismos hiperparametros que train_maskgit_full.py; lo unico que cambia entre
ramas es el conjunto de tokens, que viene de imagenes renderizadas con una
codificacion distinta.

Uso: python scripts/train_maskgit_enc.py <rama> [corrida] [mascara]
     rama:    actual | mad100 | mad300
     corrida: vacio para el subconjunto, "full" para el dataset completo
     mascara: aleatoria (por defecto) | ventana
"""
import sys
sys.path.insert(0, '.')
import os
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from scripts.mascara_ventana import enmascarar_ventana
from scripts.train_maskgit_cls import TokenDataset, MaskGITWithViz
from src.models.maskgit import MaskGIT

RAMA = sys.argv[1]
assert RAMA in ('actual', 'mad100', 'mad300', 'mad300t')
CORRIDA = sys.argv[2] if len(sys.argv) > 2 else ''
MASCARA = sys.argv[3] if len(sys.argv) > 3 else 'aleatoria'
assert MASCARA in ('aleatoria', 'ventana')
SUF = f'_{CORRIDA}' if CORRIDA else ''
VEN = '_ventana' if MASCARA == 'ventana' else ''
ROOT = f'data/enc_cmp/{RAMA}'


class MaskGITPlain(MaskGITWithViz):
    """Solo hereda _step de MaskGITWithViz; se salta el glob del init."""

    def __init__(self, **kwargs):
        MaskGIT.__init__(self, **kwargs)
        self.image_dir = Path(ROOT) / 'images'
        self.n_viz = 0
        self.vqvae = None
        self.viz_snids = []
        self._val_scores = []
        self._val_labels = []

    def _mask_tokens(self, tokens):
        if MASCARA == 'ventana':
            out = enmascarar_ventana(tokens, self.mask_token)
            if out is not None:
                return out
        return super()._mask_tokens(tokens)


def main():
    tokens = np.load(f'{ROOT}/tokens{SUF}.npy')
    with open(f'{ROOT}/paths{SUF}.txt') as f:
        paths = [l.strip() for l in f]

    # FOLDS_SPLIT fija el split desde un archivo de folds en vez de sortearlo.
    # Lo usa el experimento de clases excluidas por dos razones: los tokens se
    # generan sobre las 1.39M imagenes (hacen falta las excluidas para evaluar)
    # pero el prior solo puede ver las clases vistas, y ademas el prior tiene que
    # compartir la particion del VQ-VAE para que "validacion" signifique lo mismo
    # en toda la cadena.
    folds_split = os.environ.get('FOLDS_SPLIT', '')
    labels_file = os.environ.get('LABELS_FILE', f'{ROOT}/labels.json')

    with open(labels_file) as f:
        labels_map = json.load(f)

    if folds_split:
        with open(folds_split) as f:
            folds = json.load(f)['0']
        idx_de = {p: i for i, p in enumerate(paths)}
        i_train = [idx_de[p] for p in folds['train'] if p in idx_de]
        i_val = [idx_de[p] for p in folds['val'] if p in idx_de]
        usados = i_train + i_val
        labels = np.array([labels_map.get(Path(paths[i]).stem, -1) for i in usados],
                          dtype=np.int64)
        dataset = TokenDataset(tokens[usados], labels)
        train_ds = torch.utils.data.Subset(dataset, list(range(len(i_train))))
        val_ds = torch.utils.data.Subset(dataset, list(range(len(i_train), len(usados))))
        n_train = len(i_train)
        print(f'[{RAMA}{SUF}] split desde {folds_split}: '
              f'{len(i_train):,} train, {len(i_val):,} val, '
              f'de {len(paths):,} tokens disponibles', flush=True)
    else:
        labels = np.array([labels_map.get(Path(p).stem, -1) for p in paths], dtype=np.int64)
        n = len(tokens)
        n_train = int(n * 0.8)
        dataset = TokenDataset(tokens, labels)
        train_ds, val_ds = random_split(dataset, [n_train, n - n_train],
                                        generator=torch.Generator().manual_seed(42))

    n = len(dataset)
    print(f'[{RAMA}{SUF}] mascara {MASCARA} | tokens {tokens.shape} | '
          f'clases {np.unique(labels[labels >= 0]).shape[0]}', flush=True)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                            num_workers=4, pin_memory=False)
    print(f'[{RAMA}] train {n_train:,} | val {n - n_train:,}', flush=True)

    model = MaskGITPlain(
        num_tokens=512, seq_len=1024, hidden_dim=256, n_layers=4, n_heads=4,
        ff_mult=4, dropout=0.1, learning_rate=1e-4,
    )

    # OUT_TAG separa corridas que comparten tokens pero cambian algo del
    # entrenamiento, para no pisar checkpoints anteriores. Lo usa el reentreno
    # con folds compartidos, que reusa tokens_full.npy pero parte distinto.
    tag = os.environ.get('OUT_TAG', '')
    ckpt_dir = f'results/enc_cmp{SUF}/{RAMA}/maskgit{VEN}{tag}/checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir, filename='maskgit-{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss', mode='min', save_top_k=1, save_last=True,
    )
    early_stop = EarlyStopping(monitor='val/loss', patience=5, mode='min')
    logger = WandbLogger(project='vqvae-supernovas',
                         name=f'maskgit-enc-{RAMA}{SUF}{VEN}{tag}',
                         save_dir=f'results/enc_cmp{SUF}/{RAMA}/maskgit{VEN}{tag}/logs')

    # Con 1.39M una epoca son ~53 min, asi que la corrida completa usa menos.
    trainer = L.Trainer(
        max_epochs=int(os.environ.get('MAX_EPOCHS', 40)),
        accelerator='gpu', devices=1, precision='16-mixed',
        gradient_clip_val=0.5, gradient_clip_algorithm='norm',
        callbacks=[checkpoint, early_stop], logger=logger,
        log_every_n_steps=100, val_check_interval=1.0, num_sanity_val_steps=2,
    )

    resume = f'{ckpt_dir}/last.ckpt'
    trainer.fit(model, train_loader, val_loader,
                ckpt_path=resume if os.path.exists(resume) else None)
    print(f'[{RAMA}] mejor checkpoint: {checkpoint.best_model_path}', flush=True)


if __name__ == '__main__':
    main()
