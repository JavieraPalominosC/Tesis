"""
Dataset para cargar imágenes 2grid de curvas de luz de supernovas.

Las imágenes se generan al vuelo desde los datos crudos usando
la función create_2grid_images del repositorio VT_Model.

Uso:
    dataset = SupernovaImageDataset(
        data_dir='data/lightcurves/elasticc_1/raw',
        config=config,
        dataset_config=dataset_config,
        split='train'
    )
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path

# Importar función de creación de imágenes del repositorio VT_Model
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.processing.create_images import create_2grid_images


class SupernovaImageDataset(Dataset):
    """
    Dataset que genera imágenes 2grid desde curvas de luz de supernovas.
    
    Cada muestra es una imagen RGB con 6 bandas apiladas verticalmente:
        - Fila 0: banda u
        - Fila 1: banda g  
        - Fila 2: banda r
        - Fila 3: banda i
        - Fila 4: banda z
        - Fila 5: banda Y
    
    Args:
        data_dir:       Directorio con los parquet de datos crudos
        config:         Configuración del modelo (Hydra/OmegaConf)
        dataset_config: Configuración del dataset (bandas, columnas, etc.)
        split:          'train', 'val', o 'test'
        partition_path: Path al parquet de particiones (IDs por split)
        img_size:       Tamaño al que redimensionar las imágenes (H, W)
        save_dir:       Si se especifica, guarda las imágenes generadas en disco
        max_samples:    Límite de muestras (útil para debugging)
    """
    
    def __init__(
        self,
        data_dir,
        config,
        dataset_config,
        split='train',
        partition_path=None,
        img_size=(224, 224),
        save_dir=None,
        max_samples=None,
    ):
        self.data_dir = Path(data_dir)
        self.config = config
        self.dataset_config = dataset_config
        self.split = split
        self.img_size = img_size
        self.save_dir = Path(save_dir) if save_dir else None
        
        # Crear directorio de imágenes si se quiere guardar en disco
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Transformaciones: resize + normalizar a [0, 1]
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),  # convierte PIL → tensor y normaliza a [0,1]
        ])
        
        # Cargar datos y obtener lista de IDs para el split
        self.data, self.ids = self._load_data(partition_path, max_samples)
        
        print(f"[Dataset] Split '{split}': {len(self.ids)} supernovas cargadas.")

    def _load_data(self, partition_path, max_samples):
        """Carga los datos crudos y filtra por split."""
        
        # Cargar todos los chunks de datos
        chunks = sorted(self.data_dir.glob('chunk_*.parquet'))
        if len(chunks) == 0:
            raise FileNotFoundError(f"No se encontraron chunks en {self.data_dir}")
        
        print(f"[Dataset] Cargando {len(chunks)} chunks...")
        df = pd.concat([pd.read_parquet(c) for c in chunks], ignore_index=True)
        
        snid_col = self.dataset_config['dict_columns']['snid']
        
        # Filtrar por split si hay partición
        if partition_path and Path(partition_path).exists():
            partitions = pd.read_parquet(partition_path)
            split_ids = partitions[partitions['split'] == self.split][snid_col].tolist()
            df = df[df[snid_col].isin(split_ids)]
        
        all_ids = df[snid_col].unique().tolist()
        
        if max_samples:
            all_ids = all_ids[:max_samples]
            df = df[df[snid_col].isin(all_ids)]
        
        return df, all_ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        snid = self.ids[idx]
        snid_col = self.dataset_config['dict_columns']['snid']
        label_col = self.dataset_config['dict_columns'].get('label', None)
        
        # Datos de esta supernova
        obj_df = self.data[self.data[snid_col] == snid]
        
        # Obtener label si existe
        label = -1
        if label_col and label_col in obj_df.columns:
            label = obj_df[label_col].iloc[0]
        
        # Intentar cargar desde disco si ya fue guardada
        if self.save_dir:
            img_path = self.save_dir / f"{snid}.png"
            if img_path.exists():
                from PIL import Image
                image = Image.open(img_path).convert('RGB')
                return self.transform(image), label, snid
        
        # Generar imagen al vuelo
        image = create_2grid_images(obj_df, self.config, self.dataset_config)
        
        # Guardar en disco si se especificó
        if self.save_dir:
            image.save(self.save_dir / f"{snid}.png")
        
        return self.transform(image), label, snid


def get_dataloaders(data_dir, config, dataset_config, partition_path=None,
                   img_size=(224, 224), batch_size=32, num_workers=4,
                   save_dir=None, max_samples=None):
    """
    Crea DataLoaders para train, val y test.
    
    Returns:
        dict con keys 'train', 'val', 'test'
    """
    loaders = {}
    for split in ['train', 'val', 'test']:
        dataset = SupernovaImageDataset(
            data_dir=data_dir,
            config=config,
            dataset_config=dataset_config,
            split=split,
            partition_path=partition_path,
            img_size=img_size,
            save_dir=save_dir,
            max_samples=max_samples,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders
