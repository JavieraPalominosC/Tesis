"""
MAD al FINAL de la curva, varios puntos existentes, escala SIN congelar.

Por que el df=200 anterior se veia chico: el p99 del waterfall estaba
congelado en el control, asi que el punto se recortaba al techo viejo
y el resto de la curva no se comprimía.

Aqui:
  - se toman TODOS los puntos EXISTENTES de r en el ultimo 30% del span
    (si hay menos de MIN_TAIL, se usan las ultimas MIN_TAIL obs de r)
  - f <- f + df * MAD
  - p01/p99 se calculan SOBRE la imagen perturbada
    => a df grande el resto de la curva se aplana

Uso: python scripts/generate_synthetic_mad_tail.py [N_BASE]
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

N_BASE = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SEED = 42
BASE_FILE = 'lc_SNIa-SALT2.parquet'
RAW_DIR = 'data/lightcurves/elasticc_1/raw'
OUT_DIR = 'data/images/elasticc_1/synthetic_mad_tail'
MAN_PATH = 'data/synthetic_mad_tail_manifest.json'
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_BAND = 'r'
MIN_OBS_BAND = 8
MIN_TAIL = 6
TAIL_FRAC = 0.30
DFS = list(range(10, 101, 10))
STEP, AMP_LAYOUT = 2.0, 1.8  # waterfall_global
PRIORITY = ['26824154', '12851454']

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
    return float(np.median(np.abs(x - np.median(x))))


def stats_of(obj_df):
    stats = {}
    mjd_min = float(obj_df[col_mjd].min())
    mjd_max = float(obj_df[col_mjd].max())
    stats['mjd_min'] = mjd_min
    stats['mjd_max'] = mjd_max
    stats['mjd_range'] = mjd_max - mjd_min if mjd_max > mjd_min else 1.0
    f = obj_df[col_flux].astype('float64')
    p01, p99 = float(f.quantile(0.01)), float(f.quantile(0.99))
    stats['global'] = dict(p01=p01, p99=p99, rng=(p99 - p01) if p99 > p01 else 1.0)
    return stats


def render(obj_df, stats, out_path):
    fig = plt.figure(figsize=(fig_params['figsize']))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    st = stats['global']
    for band_key, j in all_bands.items():
        bd = obj_df[obj_df[col_band] == band_key]
        if bd.empty:
            continue
        bd = bd.sort_values(col_mjd)
        mjd_norm = (bd[col_mjd] - stats['mjd_min']) / stats['mjd_range']
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


def tail_indices(obj_df):
    sub = obj_df[obj_df[col_band] == TARGET_BAND].sort_values(col_mjd)
    t0 = float(obj_df[col_mjd].min())
    t1 = float(obj_df[col_mjd].max())
    cut = t1 - TAIL_FRAC * max(t1 - t0, 1.0)
    hit = list(sub.index[sub[col_mjd] >= cut])
    if len(hit) < MIN_TAIL:
        hit = list(sub.index[-MIN_TAIL:])
    return hit


def perturb(obj_df, indices, df_k, scale):
    out = obj_df.copy()
    out[col_flux] = out[col_flux].astype('float64')
    out.loc[indices, col_flux] = out.loc[indices, col_flux] + df_k * scale
    return out


def main():
    rng = np.random.default_rng(SEED)
    print(f'tail unfrozen  N={N_BASE}  frac={TAIL_FRAC}  min_pts={MIN_TAIL}  df={DFS}',
          flush=True)
    df = pd.read_parquet(os.path.join(RAW_DIR, BASE_FILE))
    df['SNID'] = df['SNID'].astype(str)

    prev = []
    prev_path = 'data/synthetic_mad_dt_wfg_manifest.json'
    if os.path.exists(prev_path):
        with open(prev_path) as f:
            prev = [r['snid'] for r in json.load(f) if r.get('tag') == 'control']

    cand = []
    for s in df['SNID'].unique()[:40000]:
        sub = df[df['SNID'] == s]
        if all((sub[col_band] == b).sum() >= MIN_OBS_BAND for b in BAND_NAMES):
            cand.append(str(s))
        if len(cand) >= max(N_BASE * 3, 80):
            break

    chosen = [s for s in PRIORITY if s in cand or s in prev]
    rest = [s for s in prev if s not in chosen]
    for s in rest:
        if len(chosen) >= N_BASE:
            break
        chosen.append(s)
    if len(chosen) < N_BASE:
        extra = [s for s in cand if s not in chosen]
        n_need = min(N_BASE - len(chosen), len(extra))
        if n_need:
            chosen += list(rng.choice(extra, size=n_need, replace=False))
    chosen = chosen[:N_BASE]
    print(f'SNIDs ({len(chosen)}): {chosen}', flush=True)

    manifest = []
    for snid in chosen:
        obj = df[df['SNID'] == str(snid)].copy()
        f_r = obj.loc[obj[col_band] == TARGET_BAND, col_flux].astype('float64')
        scale = mad(f_r.values)
        if not np.isfinite(scale) or scale <= 0:
            print(f'  skip {snid}: MAD=0', flush=True)
            continue
        idx = tail_indices(obj)
        t0 = float(obj[col_mjd].min())
        t1 = float(obj[col_mjd].max())
        mjd_lo = float(obj.loc[idx, col_mjd].min())
        frac_lo = (mjd_lo - t0) / max(t1 - t0, 1.0)
        st_ctrl = stats_of(obj)
        p_ctrl = f'{OUT_DIR}/{snid}__control.png'
        if not os.path.exists(p_ctrl):
            render(obj, st_ctrl, p_ctrl)
        manifest.append({'snid': str(snid), 'df': None, 'tag': 'control',
                         'path': p_ctrl, 'n_pts': 0, 'mad': scale,
                         'frozen': None, 'frac_lo': frac_lo})
        print(f'\n  SNID {snid}  MAD={scale:.3f}  n_tail={len(idx)}  '
              f'frac_lo={frac_lo:.2f}', flush=True)
        for df_k in DFS:
            pert = perturb(obj, idx, df_k, scale)
            tag = f'df{df_k}_unfrozen'
            path = f'{OUT_DIR}/{snid}__{tag}.png'
            if not os.path.exists(path):
                render(pert, stats_of(pert), path)
            rec = {'snid': str(snid), 'df': df_k, 'tag': tag, 'path': path,
                   'n_pts': int(len(idx)), 'mad': scale, 'frozen': False,
                   'frac_lo': frac_lo}
            manifest.append(rec)
            if df_k == 200:
                tag_f = 'df200_frozen'
                path_f = f'{OUT_DIR}/{snid}__{tag_f}.png'
                if not os.path.exists(path_f):
                    render(pert, st_ctrl, path_f)
                manifest.append({**rec, 'tag': tag_f, 'path': path_f,
                                 'frozen': True, 'df': 200})
                print(f'    df={df_k}: unfrozen + frozen', flush=True)
            else:
                print(f'    df={df_k}: unfrozen', flush=True)

    os.makedirs('data', exist_ok=True)
    with open(MAN_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nListo: {len(manifest)} imagenes en {OUT_DIR}/', flush=True)


if __name__ == '__main__':
    main()
