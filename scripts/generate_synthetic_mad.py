"""
Experimento sistematico de deteccion (pedido del profe):

  1. Elegir un punto existente al azar en una banda.
  2. MAD de esa banda = median(|f - median(f)|).
  3. Perturbar SOLO ese punto: f <- f + k * MAD, k = 1..10.
  4. El mismo punto se reusa para todos los k (solo cambia la amplitud).
  5. Hit rate vs k, una linea por banda.

La normalizacion del waterfall se CONGELA con los percentiles del control.
Si no, el p99 se mueve con el outlier y la perturbacion se diluye al pintar.

Representacion: waterfall (la que separa anomalías). Modelo: MaskGIT waterfall.

Uso: python scripts/generate_synthetic_mad.py [N_BASE]
"""
import sys
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os, io, json
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image

N_BASE = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEED = 42
BASE_FILE = 'lc_SNIa-SALT2.parquet'
RAW_DIR = 'data/lightcurves/elasticc_1/raw'
OUT_DIR = 'data/images/elasticc_1/synthetic_mad'
os.makedirs(OUT_DIR, exist_ok=True)

full_config = OmegaConf.to_container(OmegaConf.load('configs/online/run_config.yaml'), resolve=True)
config = full_config['ft_classification']
dataset_config = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml'), resolve=True)['elasticc_1']
dict_columns = dataset_config['dict_columns']
fig_params = config['imgs_params']['fig_params']
all_bands = dataset_config['all_bands']
col_mjd = dict_columns['mjd']
col_flux = dict_columns['flux']
col_band = dict_columns['band']
col_err = dict_columns.get('flux_err', 'FLUXCALERR')

N_BANDS = len(all_bands)
BAND_NAMES = list(all_bands.keys())
STEP, AMP_LAYOUT = 1.15, 1.0
MIN_OBS_BAND = 8
KS = list(range(1, 11))


def mad(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def band_stats(obj_df):
    stats = {}
    mjd_min = float(obj_df[col_mjd].min())
    mjd_max = float(obj_df[col_mjd].max())
    stats['mjd_min'] = mjd_min
    stats['mjd_max'] = mjd_max
    stats['mjd_range'] = mjd_max - mjd_min if mjd_max > mjd_min else 1.0
    for band in BAND_NAMES:
        bd = obj_df[obj_df[col_band] == band]
        if bd.empty:
            continue
        f = bd[col_flux].astype('float64')
        b_min = float(f.quantile(0.01))
        b_max = float(f.quantile(0.99))
        stats[band] = dict(p01=b_min, p99=b_max,
                           rng=(b_max - b_min) if b_max > b_min else 1.0)
    return stats


def render(obj_df, stats, out_path):
    fig = plt.figure(figsize=(fig_params['figsize']))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for band_key, j in all_bands.items():
        bd = obj_df[obj_df[col_band] == band_key]
        if bd.empty or band_key not in stats:
            continue
        bd = bd.sort_values(col_mjd)
        mjd_norm = (bd[col_mjd] - stats['mjd_min']) / stats['mjd_range']
        st = stats[band_key]
        flux_norm = ((bd[col_flux].astype('float64') - st['p01']) / st['rng']).clip(0.0, 1.0)
        y = flux_norm * AMP_LAYOUT + j * STEP
        ax.plot(mjd_norm, y, 'o-', color=fig_params['colors'][j],
                markersize=2, linewidth=1.0)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.2, (N_BANDS - 1) * STEP + AMP_LAYOUT + 0.2])
    ax.axis('off')
    rect = patches.Rectangle((0, 0), 1, 1, linewidth=1.5, edgecolor='black',
                             facecolor='none', transform=fig.transFigure)
    fig.add_artist(rect)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', pad_inches=0)
    buf.seek(0)
    Image.open(buf).convert('RGB').save(out_path)
    buf.close()
    plt.close(fig)


def pick_point(obj_df, band, rng):
    sub = obj_df[obj_df[col_band] == band]
    if len(sub) < MIN_OBS_BAND:
        return None
    f = sub[col_flux].astype('float64').values
    scale = mad(f)
    if not np.isfinite(scale) or scale <= 0:
        return None
    pos = int(rng.integers(0, len(sub)))
    row = sub.iloc[pos]
    err = float(row[col_err]) if col_err in sub.columns else None
    return dict(
        iloc=pos,
        index=int(sub.index[pos]),
        mjd=float(row[col_mjd]),
        flux0=float(row[col_flux]),
        err=err,
        mad=scale,
        n=int(len(sub)),
    )


def perturb(obj_df, rec, k):
    df = obj_df.copy()
    df[col_flux] = df[col_flux].astype('float64')
    df.loc[rec['index'], col_flux] = rec['flux0'] + k * rec['mad']
    return df


def main():
    rng = np.random.default_rng(SEED)
    print(f'Cargando {BASE_FILE}...', flush=True)
    df = pd.read_parquet(os.path.join(RAW_DIR, BASE_FILE))
    cand = []
    for s in df['SNID'].unique()[:40000]:
        sub = df[df['SNID'] == s]
        ok = True
        for b in BAND_NAMES:
            if (sub[col_band] == b).sum() < MIN_OBS_BAND:
                ok = False
                break
        if ok:
            cand.append(s)
        if len(cand) >= N_BASE * 3:
            break
    chosen = list(rng.choice(cand, size=min(N_BASE, len(cand)), replace=False))
    print(f'SNIDs ({len(chosen)}): {chosen[:8]}...', flush=True)

    manifest = []
    for snid in chosen:
        obj = df[df['SNID'] == snid].copy()
        stats = band_stats(obj)
        p_ctrl = f'{OUT_DIR}/{snid}__control.png'
        render(obj, stats, p_ctrl)
        manifest.append({'snid': str(snid), 'band': None, 'k': None,
                         'path': p_ctrl, 'tag': 'control'})
        print(f'\n  SNID {snid}', flush=True)
        for band in BAND_NAMES:
            rec = pick_point(obj, band, rng)
            if rec is None:
                print(f'    {band}: skip (MAD=0 o pocos pts)', flush=True)
                continue
            print(f'    {band}: n={rec["n"]} MAD={rec["mad"]:.3f} '
                  f'flux0={rec["flux0"]:.2f} mjd={rec["mjd"]:.3f}', flush=True)
            for k in KS:
                pert = perturb(obj, rec, k)
                tag = f'{band}_k{k}'
                path = f'{OUT_DIR}/{snid}__{tag}.png'
                render(pert, stats, path)
                manifest.append({
                    'snid': str(snid), 'band': band, 'k': k, 'path': path,
                    'tag': tag, 'mad': rec['mad'], 'flux0': rec['flux0'],
                    'flux1': rec['flux0'] + k * rec['mad'], 'mjd': rec['mjd'],
                    'flux_err': rec['err'], 'n_band': rec['n'],
                })

    man_path = 'data/synthetic_mad_manifest.json'
    os.makedirs('data', exist_ok=True)
    with open(man_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    n_img = len(manifest)
    print(f'\nListo: {n_img} imagenes en {OUT_DIR}/', flush=True)
    print(f'Manifest: {man_path}', flush=True)


if __name__ == '__main__':
    main()
