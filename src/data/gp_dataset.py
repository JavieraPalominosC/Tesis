"""Dataset de superficies GP (tiempo × ln λ) para el VQ-VAE.

Cada objeto es un array [H_λ, W_t] = [200, 300] (mean del GP). Se reescala a
image_size×image_size. Por defecto se usa el mean **tal como viene en el tar**
(ya dividido por scale_mean), solo con clip a [-1, 1] para el Tanh del decoder.

Por defecto un canal (mean). Si use_var=True, se agrega el std escalado del GP
como segundo canal.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L


def uid_to_stem(uid: str) -> str:
    return uid.replace("/", "__")


def encode_gp_surface(
    mean_scaled,
    scale_mean=None,
    enc="as_is",
    var_scaled=None,
    scale_var=None,
):
    """mean_scaled: (200, 300) ya dividido por scale_mean (como en el tar).

    enc:
      as_is   — default. Usa la superficie normalizada del paper; clip a [-1, 1].
      mad300  — opcional. Reescala a FLUXCAL y aplica mad300 (legado / ablación).

    Devuelve tensor float32 (C, 200, 300) en [-1, 1].
    """
    mean_scaled = mean_scaled.astype(np.float32)

    if enc == "as_is":
        x = np.clip(mean_scaled, -1.0, 1.0)
    elif enc.startswith("mad"):
        from scripts.enc_flujo import codificar

        if scale_mean is None:
            raise ValueError("mad* requiere scale_mean para volver a FLUXCAL")
        flux = mean_scaled * float(scale_mean)
        u = codificar(flux, flux, enc).astype(np.float32)
        x = u * 2.0 - 1.0
    else:
        raise ValueError(f"codificacion GP desconocida: {enc}")

    channels = [x]

    if var_scaled is not None:
        var_scaled = var_scaled.astype(np.float32)
        if enc == "as_is":
            # var también viene / scale_var; std escalado ≈ en [0, 1]
            std = np.sqrt(np.clip(var_scaled, 0.0, None))
            channels.append(np.clip(std, 0.0, 1.0) * 2.0 - 1.0)
        else:
            from scripts.enc_flujo import codificar

            var = var_scaled * float(scale_var)
            std = np.sqrt(np.clip(var, 0.0, None))
            u_std = codificar(std, std, enc).astype(np.float32)
            channels.append(u_std * 2.0 - 1.0)

    return torch.from_numpy(np.stack(channels, axis=0))


def resize_to_square(x, size=256):
    """x: (C, H, W) → (C, size, size)."""
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x.squeeze(0)


class GPSurfaceDataset(Dataset):
    """Carga superficies cacheadas como .npy (C, H, W) en [-1, 1]."""

    def __init__(self, paths, labels_map=None):
        self.paths = list(paths)
        self.labels_map = labels_map
        assert len(self.paths) > 0, "No se encontraron superficies GP"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        x = torch.from_numpy(np.load(path)).float()
        if self.labels_map is None:
            return x
        stem = Path(path).stem
        label = self.labels_map.get(stem, -1)
        return x, label


class GPSurfaceDataModule(L.LightningDataModule):
    def __init__(self, folds_path, fold=0, batch_size=32, num_workers=4, labels_path=None):
        super().__init__()
        self.folds_path = folds_path
        self.labels_path = labels_path
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.labels_map = None

    def setup(self, stage=None):
        if self.labels_path and Path(self.labels_path).exists():
            with open(self.labels_path) as f:
                self.labels_map = json.load(f)
            print(f"Labels GP: {len(self.labels_map):,} objetos")
        else:
            print("Sin labels — solo recon + vq loss")

        with open(self.folds_path) as f:
            folds = json.load(f)

        fold_data = folds[str(self.fold)]
        self.train_ds = GPSurfaceDataset(fold_data["train"], self.labels_map)
        self.val_ds = GPSurfaceDataset(fold_data["val"], self.labels_map)
        print(
            f"Fold {self.fold} GP — Train: {len(self.train_ds):,} | Val: {len(self.val_ds):,}"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
