import sys
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, '.')
import os
import glob
import pandas as pd
from omegaconf import OmegaConf
from src.data.processing.create_images import create_2grid_images
from tqdm import tqdm

full_config = OmegaConf.to_container(OmegaConf.load('configs/online/run_config.yaml'), resolve=True)
config = full_config['ft_classification']
dataset_config = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml')['elasticc_1'])

output_dir = 'data/images/elasticc_1/2grid'
os.makedirs(output_dir, exist_ok=True)

sn_files = [
    'lc_SNIa-SALT2.parquet', 'lc_SNIa-91bg.parquet', 'lc_SNIax.parquet',
    'lc_SNII-Templates.parquet', 'lc_SNII-NMF.parquet', 'lc_SNII+HostXT_V19.parquet',
    'lc_SNIb-Templates.parquet', 'lc_SNIb+HostXT_V19.parquet',
    'lc_SNIc-Templates.parquet', 'lc_SNIc+HostXT_V19.parquet',
    'lc_SNIcBL+HostXT_V19.parquet', 'lc_SNIIb+HostXT_V19.parquet',
    'lc_SNIIn-MOSFIT.parquet', 'lc_SNIIn+HostXT_V19.parquet',
    'lc_SLSN-I+host.parquet', 'lc_SLSN-I_no_host.parquet', 'lc_PISN.parquet'
]
parquet_files = [f'data/lightcurves/elasticc_1/raw/{f}' for f in sn_files]
print(f'Total archivos SN: {len(parquet_files)}', flush=True)

for chunk_file in parquet_files:
    print(f'Procesando {os.path.basename(chunk_file)}...', flush=True)
    df = pd.read_parquet(chunk_file)
    snids = df['SNID'].unique()
    for snid in tqdm(snids):
        out_path = f'{output_dir}/{snid}.png'
        if os.path.exists(out_path):
            continue
        obj_df = df[df['SNID'] == snid]
        img = create_2grid_images(obj_df, config, dataset_config)
        img.save(out_path)

print('Listo!', flush=True)
