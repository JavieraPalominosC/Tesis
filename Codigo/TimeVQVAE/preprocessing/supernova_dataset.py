import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

class SupernovaDataset(Dataset):
    def __init__(self, file_path, num_timesteps=50):
        """
        Carga un dataset de supernovas desde un archivo CSV y lo interpola a un grid uniforme.
        :param file_path: Ruta al archivo CSV
        :param num_timesteps: Número de puntos en el grid temporal
        """
        self.df = pd.read_csv(file_path)
        self.num_timesteps = num_timesteps
        self.data = self._process_data()

    def _process_data(self):
        """Interpolar y normalizar las series de tiempo"""
        t_min, t_max = self.df['time'].min(), self.df['time'].max()
        grid_times = np.linspace(t_min, t_max, self.num_timesteps)

        supernova_series = []
        for sn_id, group in self.df.groupby('id'):
            f = interp1d(group['time'], group['brightness'], kind='linear', fill_value="extrapolate")
            brightness_interp = f(grid_times)  # Interpolamos en el grid uniforme
            supernova_series.append(brightness_interp)

        X = np.array(supernova_series)  # (num_supernovas, num_timesteps)

        # Normalizar entre 0 y 1
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        # Convertir a Tensor PyTorch
        return torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (num_supernovas, num_timesteps, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # Devolvemos solo la serie
