"""
Perturbacion MAD sobre PUNTOS EXISTENTES, con magnitud y duracion:

  df: veces MAD de la banda r
  dt: ventana en dias MJD centrada en un punto real
      todos los puntos EXISTENTES de r en [mjd_c - dt/2, mjd_c + dt/2]
      reciben f <- f + df * MAD. No se inserta fotometria.

El centro mjd_c se fija por objeto y se reusa en todo el grid (df, dt).
Normalizacion del waterfall CONGELADA en el control.

Uso: python scripts/generate_synthetic_mad_dt.py [N_BASE] [REP]
     REP in {waterfall, waterfall_global}  (default: waterfall_global)
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

REPS = {
    'waterfall': dict(
        step=1.15, amp=1.0, norm='band',
        out='data/images/elasticc_1/synthetic_mad_dt',
        man='data/synthetic_mad_dt_manifest.json',
    ),
    'waterfall_global': dict(
        step=2.0, amp=1.8, norm='global',
        out='data/images/elasticc_1/synthetic_mad_dt_wfg',
        man='data/synthetic_mad_dt_wfg_manifest.json',
    ),
}

N_BASE = 40
REP = 'waterfall_global'
for a in sys.argv[1:]:
    if a in REPS:
        REP = a
    else:
        N_BASE = int(a)

CFG = REPS[REP]
STEP, AMP_LAYOUT = CFG['step'], CFG['amp']
OUT_DIR = CFG['out']
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
BASE_FILE = 'lc_SNIa-SALT2.parquet'
RAW_DIR = 'data/lightcurves/elasticc_1/raw'
TARGET_BAND = 'r'
MIN_OBS_BAND = 8
DFS = [1, 2, 5, 10, 20, 50, 100, 200]
DTS = [1, 3, 7, 14, 30]

full_config = OmegaConf.to_container(OmegaConf.load('configs/online/run_config.yaml'), resolve=True)
config = full_config['ft_classification']
dataset_config = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml'), resolve=True)['elasticc_1']
dict_columns = dataset_config['dict_columns']
fig_params = config['imgs_params']['fig_params']
all_bands = dataset_config['all_bands']
col_mjd = dict_columns['mjd']
col_flux = dict_columns['flux']
col_band = dict_columns['band']

N_BANDS = len(all_bands)
BAND_NAMES = list(all_bands.keys())


def mad(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def freeze_stats(obj_df):
    stats = {}
    mjd_min = float(obj_df[col_mjd].min())
    mjd_max = float(obj_df[col_mjd].max())
    stats['mjd_min'] = mjd_min
    stats['mjd_max'] = mjd_max
    stats['mjd_range'] = mjd_max - mjd_min if mjd_max > mjd_min else 1.0
    if CFG['norm'] == 'global':
        f = obj_df[col_flux].astype('float64')
        p01, p99 = float(f.quantile(0.01)), float(f.quantile(0.99))
        stats['global'] = dict(p01=p01, p99=p99, rng=(p99 - p01) if p99 > p01 else 1.0)
    else:
        for band in BAND_NAMES:
            bd = obj_df[obj_df[col_band] == band]
            if bd.empty:
                continue
            f = bd[col_flux].astype('float64')
            p01, p99 = float(f.quantile(0.01)), float(f.quantile(0.99))
            stats[band] = dict(p01=p01, p99=p99, rng=(p99 - p01) if p99 > p01 else 1.0)
    return stats


def render(obj_df, stats, out_path):
    fig = plt.figure(figsize=(fig_params['figsize']))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for band_key, j in all_bands.items():
        bd = obj_df[obj_df[col_band] == band_key]
        if bd.empty:
            continue
        bd = bd.sort_values(col_mjd)
        mjd_norm = (bd[col_mjd] - stats['mjd_min']) / stats['mjd_range']
        if CFG['norm'] == 'global':
            st = stats['global']
        else:
            if band_key not in stats:
                continue
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


def pick_center(obj_df, rng):
    sub = obj_df[obj_df[col_band] == TARGET_BAND]
    if len(sub) < MIN_OBS_BAND:
        return None
    f = sub[col_flux].astype('float64').values
    scale = mad(f)
    if not np.isfinite(scale) or scale <= 0:
        return None
    pos = int(rng.integers(0, len(sub)))
    row = sub.iloc[pos]
    return dict(
        mjd_c=float(row[col_mjd]),
        mad=scale,
        n_band=int(len(sub)),
    )


def window_indices(obj_df, mjd_c, dt):
    half = dt / 2.0
    sub = obj_df[obj_df[col_band] == TARGET_BAND]
    mjd = sub[col_mjd].astype(float)
    hit = (mjd >= mjd_c - half) & (mjd <= mjd_c + half)
    return list(sub.index[hit])


def perturb(obj_df, indices, df_k, scale):
    out = obj_df.copy()
    out[col_flux] = out[col_flux].astype('float64')
    out.loc[indices, col_flux] = out.loc[indices, col_flux] + df_k * scale
    return out


def main():
    rng = np.random.default_rng(SEED)
    print(f'[{REP}] N_BASE={N_BASE}  band={TARGET_BAND}  '
          f'df={DFS}  dt={DTS}', flush=True)
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
        rec = pick_center(obj, rng)
        if rec is None:
            print(f'  skip {snid}: MAD=0 o pocos pts en r', flush=True)
            continue
        stats = freeze_stats(obj)
        p_ctrl = f'{OUT_DIR}/{snid}__control.png'
        if not os.path.exists(p_ctrl):
            render(obj, stats, p_ctrl)
        manifest.append({
            'snid': str(snid), 'df': None, 'dt': None, 'path': p_ctrl,
            'tag': 'control', 'rep': REP, 'band': TARGET_BAND,
            'mjd_c': rec['mjd_c'], 'mad': rec['mad'], 'n_pts': 0,
        })
        print(f'\n  SNID {snid}  mjd_c={rec["mjd_c"]:.3f}  MAD={rec["mad"]:.3f}',
              flush=True)
        for dt in DTS:
            idx = window_indices(obj, rec['mjd_c'], dt)
            print(f'    dt={dt:>2}d: {len(idx)} pts existentes', flush=True)
            for df_k in DFS:
                pert = perturb(obj, idx, df_k, rec['mad'])
                tag = f'df{df_k}_dt{dt}'
                path = f'{OUT_DIR}/{snid}__{tag}.png'
                if not os.path.exists(path):
                    render(pert, stats, path)
                manifest.append({
                    'snid': str(snid), 'df': df_k, 'dt': dt, 'path': path,
                    'tag': tag, 'rep': REP, 'band': TARGET_BAND,
                    'mjd_c': rec['mjd_c'], 'mad': rec['mad'],
                    'n_pts': int(len(idx)),
                })

    os.makedirs('data', exist_ok=True)
    with open(CFG['man'], 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nListo: {len(manifest)} imagenes en {OUT_DIR}/', flush=True)
    print(f'Manifest: {CFG["man"]}', flush=True)


if __name__ == '__main__':
    main()
