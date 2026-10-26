"""Distribucion de la duracion de las curvas, para fijar la escala del eje temporal.

El renderizado actual normaliza el tiempo por objeto, `(mjd - mjd_min) / span`, de
modo que toda curva ocupa el ancho completo de la imagen y la duracion absoluta se
pierde. El diagnostico por rasgo mostro que esa es justamente la variable mas
discriminativa para las clases excluidas, asi que la variante con eje absoluto
necesita una escala global `T` comun a todos los objetos.

Esto mide contra que se esta eligiendo esa `T`: si el 10-90 de la distribucion cabe
en un factor chico, basta con un eje lineal; si abarca ordenes de magnitud, un eje
lineal aplasta a las curvas cortas y hay que comprimir en log.

Uso: python scripts/medir_span.py [out.json]
"""
import sys
sys.path.insert(0, '.')
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

RAW_DIR = 'data/lightcurves/elasticc_1/raw'
SALIDA = Path(sys.argv[1] if len(sys.argv) > 1 else 'results/enc_cmp_full/eval/spans.json')

# Mismo orden que SN_FILES en run_mad300_full.sh, que es el que fija las etiquetas.
ARCHIVOS = [
    'lc_SNIa-SALT2.parquet', 'lc_SNIa-91bg.parquet', 'lc_SNIax.parquet',
    'lc_SNII-Templates.parquet', 'lc_SNII-NMF.parquet', 'lc_SNII+HostXT_V19.parquet',
    'lc_SNIb-Templates.parquet', 'lc_SNIb+HostXT_V19.parquet',
    'lc_SNIc-Templates.parquet', 'lc_SNIc+HostXT_V19.parquet',
    'lc_SNIcBL+HostXT_V19.parquet', 'lc_SNIIb+HostXT_V19.parquet',
    'lc_SNIIn-MOSFIT.parquet', 'lc_SNIIn+HostXT_V19.parquet',
    'lc_SLSN-I+host.parquet', 'lc_SLSN-I_no_host.parquet', 'lc_PISN.parquet',
]
EXCLUIDAS = {'SNIa-91bg', 'SNIax', 'SNIIb+HostXT_V19',
             'SLSN-I+host', 'SLSN-I_no_host', 'PISN'}
QS = [1, 5, 10, 25, 50, 75, 90, 95, 99, 100]

ds = OmegaConf.to_container(OmegaConf.load('configs/datasets_config.yaml'), resolve=True)['elasticc_1']
col_mjd = ds['dict_columns']['mjd']


def cuantiles(v):
    v = np.asarray(v, dtype=float)
    return {str(q): round(float(np.percentile(v, q)), 1) for q in QS}


def main():
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    por_clase, todos = {}, []

    for fname in ARCHIVOS:
        clase = fname.replace('lc_', '').replace('.parquet', '')
        df = pd.read_parquet(os.path.join(RAW_DIR, fname), columns=['SNID', col_mjd])
        g = df.groupby('SNID')[col_mjd]
        span = (g.max() - g.min()).values
        span = span[np.isfinite(span) & (span > 0)]
        por_clase[clase] = {'n': int(len(span)), 'excluida': clase in EXCLUIDAS,
                            'q': cuantiles(span)}
        todos.append(span)
        print(f'{clase:<22} n={len(span):>7,}  mediana={np.median(span):>7.1f}  '
              f'p10={np.percentile(span, 10):>7.1f}  p99={np.percentile(span, 99):>7.1f}',
              flush=True)

    todos = np.concatenate(todos)
    glob = cuantiles(todos)
    # Cuanto se encoge la curva mas corta si el eje se vuelve lineal y comun.
    razon = {f'p{a}/p{b}': round(glob[str(a)] / max(glob[str(b)], 1e-9), 2)
             for a, b in ((99, 1), (95, 5), (90, 10))}

    print(f'\nglobal n={len(todos):,}')
    for q in QS:
        print(f'  p{q:<3} {glob[str(q)]:>8.1f} d')
    print(f'\nrazones de dispersion: {razon}')

    with open(SALIDA, 'w') as f:
        json.dump({'n': int(len(todos)), 'global': glob, 'razon': razon,
                   'por_clase': por_clase}, f, indent=2)
    print(f'\nListo: {SALIDA}', flush=True)


if __name__ == '__main__':
    main()
