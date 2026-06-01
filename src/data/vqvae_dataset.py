import os
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import lightning as L


class VQVAEDataset(Dataset):
    """
    Carga imágenes 2grid desde disco.
    Normaliza a [-1, 1] para compatibilidad con Tanh en el decoder.

    Si se pasa labels_map, devuelve (imagen, label_int).
    Si no, devuelve solo la imagen (comportamiento original).
    """
    def __init__(self, paths, image_size=256, labels_map=None):
        self.paths      = paths
        self.labels_map = labels_map
        assert len(self.paths) > 0, "No se encontraron imágenes"

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img  = Image.open(path).convert("RGB")
        x    = self.transform(img)

        if self.labels_map is None:
            return x

        snid  = Path(path).stem
        label = self.labels_map.get(snid, -1)
        return x, label


class VQVAEDataModule(L.LightningDataModule):
    def __init__(self, folds_path, fold=0, image_size=256,
                 batch_size=32, num_workers=4, labels_path=None):
        super().__init__()
        self.folds_path  = folds_path
        self.labels_path = labels_path
        self.fold        = fold
        self.image_size  = image_size
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.labels_map  = None

    def setup(self, stage=None):
        if self.labels_path and Path(self.labels_path).exists():
            with open(self.labels_path) as f:
                self.labels_map = json.load(f)
            print(f"Labels cargados: {len(self.labels_map):,} SNIDs")
        else:
            print("Sin labels — entrenamiento solo con recon + vq loss")

        with open(self.folds_path) as f:
            folds = json.load(f)

        fold_data   = folds[str(self.fold)]
        train_paths = fold_data['train']
        val_paths   = fold_data['val']

        self.train_ds = VQVAEDataset(train_paths, self.image_size, self.labels_map)
        self.val_ds   = VQVAEDataset(val_paths,   self.image_size, self.labels_map)

        label_str = "con labels" if self.labels_map else "sin labels"
        print(f"Fold {self.fold} ({label_str}) — Train: {len(train_paths):,} | Val: {len(val_paths):,}")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=False)
