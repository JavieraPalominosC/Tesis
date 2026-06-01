import sys
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, '.')
import os
import pandas as pd
from omegaconf import OmegaConf
from src.data.processing.create_images import create_2grid_images
from tqdm import tqdm

parquet_file = f'data/lightcurves/elasticc_1/raw/{sys.argv[1]}'
full_config = OmegaConf.to_container(OmegaConf.load('configs/online/run_config.yaml'), resolve=True)
config = full_config['ft_classification']
dataset_config = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml')['elasticc_1'])

output_dir = 'data/images/elasticc_1/2grid'
os.makedirs(output_dir, exist_ok=True)

print(f'Procesando {sys.argv[1]}...', flush=True)
df = pd.read_parquet(parquet_file)
snids = df['SNID'].unique()
print(f'Total SNIDs: {len(snids)}', flush=True)

for snid in tqdm(snids):
    out_path = f'{output_dir}/{snid}.png'
    if os.path.exists(out_path):
        continue
    obj_df = df[df['SNID'] == snid]
    img = create_2grid_images(obj_df, config, dataset_config)
    img.save(out_path)

print('Listo!', flush=True)
