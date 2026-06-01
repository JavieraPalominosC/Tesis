import sys
sys.path.insert(0, '.')
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

output_dir = 'results/eda'
os.makedirs(output_dir, exist_ok=True)

SN_FILES = [
    'lc_SNIa-SALT2.parquet', 'lc_SNIa-91bg.parquet', 'lc_SNIax.parquet',
    'lc_SNII-Templates.parquet', 'lc_SNII-NMF.parquet', 'lc_SNII+HostXT_V19.parquet',
    'lc_SNIb-Templates.parquet', 'lc_SNIb+HostXT_V19.parquet',
    'lc_SNIc-Templates.parquet', 'lc_SNIc+HostXT_V19.parquet',
    'lc_SNIcBL+HostXT_V19.parquet', 'lc_SNIIb+HostXT_V19.parquet',
    'lc_SNIIn-MOSFIT.parquet', 'lc_SNIIn+HostXT_V19.parquet',
    'lc_SLSN-I+host.parquet', 'lc_SLSN-I_no_host.parquet', 'lc_PISN.parquet'
]

stats = []
for fname in tqdm(SN_FILES):
    path = f'data/lightcurves/elasticc_1/raw/{fname}'
    df = pd.read_parquet(path)
    dict_col = df.columns.tolist()
    
    n_objects = df['SNID'].nunique()
    n_obs_total = len(df)
    n_obs_per_obj = df.groupby('SNID').size()
    
    # Detectar columnas
    mjd_col = [c for c in df.columns if 'MJD' in c.upper() or 'mjd' in c.lower()][0]
    flux_col = [c for c in df.columns if 'FLUX' in c.upper() and 'ERR' not in c.upper()][0]
    band_col = [c for c in df.columns if 'BAND' in c.upper() or 'band' in c.lower() or 'FILTER' in c.upper()][0]
    
    mjd_range = df[mjd_col].max() - df[mjd_col].min()
    bands_present = df[band_col].unique().tolist()
    
    stats.append({
        'tipo': fname.replace('lc_', '').replace('.parquet', ''),
        'n_objetos': n_objects,
        'n_obs_total': n_obs_total,
        'obs_por_objeto_mean': round(n_obs_per_obj.mean(), 1),
        'obs_por_objeto_median': round(n_obs_per_obj.median(), 1),
        'mjd_range': round(mjd_range, 1),
        'n_bandas': len(bands_present),
    })
    print(f"{fname}: {n_objects:,} objetos, {n_obs_total:,} observaciones", flush=True)

# Guardar estadísticas
df_stats = pd.DataFrame(stats)
df_stats.to_csv(f'{output_dir}/stats_por_tipo.csv', index=False)
print(f'\nTotal objetos: {df_stats["n_objetos"].sum():,}')
print(f'\nTabla guardada en {output_dir}/stats_por_tipo.csv')
print(df_stats.to_string())
