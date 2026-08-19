"""
Inserta un SEGUNDO peak en banda r, despues del maximo real.

Uso: python scripts/generate_synthetic_second_peak.py [N_BASE] [REP]
     REP in {waterfall, waterfall_global}  (default: waterfall)
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
        out='data/images/elasticc_1/synthetic_second_peak',
        man='data/synthetic_second_peak_manifest.json',
    ),
    'waterfall_global': dict(
        step=2.0, amp=1.8, norm='global',
        out='data/images/elasticc_1/synthetic_second_peak_wfg',
        man='data/synthetic_second_peak_wfg_manifest.json',
    ),
}

N_BASE = 8
REP = 'waterfall'
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

full_config = OmegaConf.to_container(OmegaConf.load('configs/online/run_config.yaml'), resolve=True)
config = full_config['ft_classification']
dataset_config = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml'), resolve=True)['elasticc_1']
dict_columns = dataset_config['dict_columns']
fig_params = config['imgs_params']['fig_params']
all_bands = dataset_config['all_bands']
col_mjd, col_flux, col_band = dict_columns['mjd'], dict_columns['flux'], dict_columns['band']

N_BANDS = len(all_bands)
TARGET_BAND = 'r'
MIN_OBS_BAND = 15
N_INSERT = 3
AMPLITUDES = [2.0, 5.0, 15.0]
DT_FRAC = 0.01
DELAY_FRAC = 0.40
MIN_SEP_FRAC = 0.25
MAX_SECOND_FRAC = 0.92
MAX_PEAK_FRAC = 0.55


def peak_and_second_frac(obj_df):
    sub = obj_df[obj_df[col_band] == TARGET_BAND]
    if len(sub) < 3:
        return None
    t = sub[col_mjd].values.astype(float)
    f = sub[col_flux].values.astype(float)
    t0, t1 = float(t.min()), float(t.max())
    span = t1 - t0 if t1 > t0 else 1.0
    t_peak = float(t[np.argmax(f)])
    peak_frac = (t_peak - t0) / span
    second_frac = peak_frac + DELAY_FRAC
    if peak_frac > MAX_PEAK_FRAC or second_frac > MAX_SECOND_FRAC:
        return None
    if (second_frac - peak_frac) < MIN_SEP_FRAC:
        return None
    return dict(t0=t0, t1=t1, span=span, t_peak=t_peak,
                peak_frac=float(peak_frac), second_frac=float(second_frac))


def insert_second_peak(obj_df, amp_factor, loc):
    df = obj_df.copy()
    df[col_flux] = df[col_flux].astype('float64')
    sub = df[df[col_band] == TARGET_BAND].sort_values(col_mjd)
    f0 = sub[col_flux].values.astype(float)
    rng_flux = np.percentile(f0, 99) - np.percentile(f0, 1)
    if rng_flux <= 0:
        rng_flux = abs(np.mean(f0)) + 1.0
    base = float(np.median(f0))
    A = amp_factor * rng_flux
    tc = loc['t0'] + loc['second_frac'] * loc['span']
    dt = DT_FRAC * loc['span']
    plantilla = sub.iloc[0].copy()
    nuevas = []
    offsets = np.linspace(-(N_INSERT // 2), N_INSERT // 2, N_INSERT)
    for off in offsets:
        fila = plantilla.copy()
        fila[col_mjd] = tc + off * dt
        peso = np.exp(-0.5 * (off / 0.8) ** 2)
        fila[col_flux] = base + A * peso
        nuevas.append(fila)
    df_new = pd.concat([df, pd.DataFrame(nuevas)], ignore_index=True)
    return df_new, N_INSERT, float(A / rng_flux)


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
        for band in all_bands:
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


def main():
    rng = np.random.default_rng(SEED)
    print(f'[{REP}] N_BASE={N_BASE}  norm={CFG["norm"]}  out={OUT_DIR}', flush=True)
    print(f'Cargando {BASE_FILE}...', flush=True)
    df = pd.read_parquet(os.path.join(RAW_DIR, BASE_FILE))
    cand, locs = [], {}
    for s in df['SNID'].unique()[:30000]:
        sub = df[df['SNID'] == s]
        if (sub[col_band] == TARGET_BAND).sum() < MIN_OBS_BAND:
            continue
        loc = peak_and_second_frac(sub)
        if loc is None:
            continue
        cand.append(s)
        locs[s] = loc
        if len(cand) >= N_BASE * 4:
            break
    chosen = list(rng.choice(cand, size=min(N_BASE, len(cand)), replace=False))
    print(f'SNIDs elegidos ({len(chosen)}): {chosen}', flush=True)

    manifest = []
    for snid in chosen:
        obj = df[df['SNID'] == snid]
        loc = locs[snid]
        stats = freeze_stats(obj)
        n_r = int((obj[col_band] == TARGET_BAND).sum())
        print(f'\n  SNID {snid}: {n_r} obs en r | peak_frac={loc["peak_frac"]:.2f} '
              f'| second_frac={loc["second_frac"]:.2f}', flush=True)
        p = f'{OUT_DIR}/{snid}__control.png'
        render(obj, stats, p)
        manifest.append({'snid': str(snid), 'amp': None, 'path': p, 'rep': REP,
                         'peak_frac': loc['peak_frac'], 'second_frac': loc['second_frac']})
        for a in AMPLITUDES:
            pert, n_ins, ratio = insert_second_peak(obj, a, loc)
            p = f'{OUT_DIR}/{snid}__amp{int(a)}.png'
            render(pert, stats, p)
            print(f'    amp={a:.0f}x: {n_ins} pts, altura={ratio:.1f}x rango banda r', flush=True)
            manifest.append({'snid': str(snid), 'amp': a, 'path': p, 'rep': REP,
                             'peak_frac': loc['peak_frac'], 'second_frac': loc['second_frac']})

    os.makedirs('data', exist_ok=True)
    with open(CFG['man'], 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nListo: {len(manifest)} imagenes en {OUT_DIR}/', flush=True)
    print(f'Manifest: {CFG["man"]}', flush=True)


if __name__ == '__main__':
    main()
