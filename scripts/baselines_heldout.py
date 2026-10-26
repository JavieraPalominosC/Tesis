"""Baselines de deteccion de anomalias sobre las clases excluidas.

Cubre los puntos 2 y 5 de la reunion del 28 de septiembre, que comparten toda la
maquinaria: hay que puntuar las seis clases que quedaron fuera del entrenamiento
y comparar el prior contra alternativas mas simples.

Se evalua sobre clases reales y no sobre el barrido sintetico a proposito. La
perturbacion sintetica se define como sumar df veces el MAD de la banda r, de
modo que un estadistico que mida justamente la desviacion maxima en MAD es casi
un filtro adaptado a ella y ganaria por construccion. Las clases excluidas no
estan definidas en nuestras unidades, asi que la comparacion es limpia.

El baseline que se reporta es 'recon'. Es un modelo y no una receta de
estadisticos a mano, sale gratis porque el VQ-VAE ya esta entrenado, y sobre
todo es la ablacion mas limpia del metodo nuestro: mismo VQ-VAE, mismas
imagenes, mismo codebook, y lo unico que cambia es la pregunta que se hace.

  recon     error de reconstruccion del VQ-VAE. El decoder redibuja la imagen y
            se mide el error cuadratico medio. Alto error, objeto raro.
  prior     max NLL por token sobre R rondas de enmascarado. El metodo nuestro.

Los dos de abajo quedan como referencia secundaria, calculados por si hace falta
una comparacion que no sea de red neuronal. No necesitan GPU ni checkpoints.

  mad_max   desviacion maxima en MAD sobre la banda r. Una linea de codigo.
  iforest   Isolation Forest sobre estadisticos resumen de la curva, ajustado
            solo con objetos normales.

Uso: python scripts/baselines_heldout.py curva
     python scripts/baselines_heldout.py modelo
     python scripts/baselines_heldout.py resumen
"""
import os
import sys
import json
import random
import re
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts.eval_enc_cmp import (
    RAW_DIR, TARGET_BAND, col_band, col_flux, col_mjd, mad,
)
from scripts.make_folds_heldout import NOMBRES

# RAMA elige sobre que renderizado correr. 'mad300' es la rama principal, con el
# eje temporal normalizado por objeto; 'mad300t' es la ablacion con escala de
# tiempo absoluta. Los objetos muestreados son los mismos en las dos, porque los
# folds salen de ordenar los mismos nombres de archivo con la misma semilla, asi
# que la comparacion queda pareada.
RAMA = os.environ.get('RAMA', 'mad300')
ROOT = f'data/enc_cmp/{RAMA}'
OUT = Path('results/enc_cmp_heldout/eval' + ('' if RAMA == 'mad300' else f'_{RAMA}'))
N_NORM, N_ANOM, SEED = 2000, 500, 0


def muestrear():
    """Normales de la validacion vista, anomalos de cada clase excluida.

    Los normales salen de validacion y no de entrenamiento porque el Isolation
    Forest se ajusta con ellos y hay que poder puntuarlos sin ventaja.
    """
    rng = random.Random(SEED)
    folds = json.load(open(f'{ROOT}/folds_heldout.json'))['0']
    ev = json.load(open(f'{ROOT}/heldout_eval.json'))
    labels = json.load(open(f'{ROOT}/labels.json'))

    val = folds['val']
    normales = rng.sample(val, min(N_NORM, len(val)))

    grupos = {'normal': [(p, labels[Path(p).stem]) for p in normales]}
    for k, rutas in ev['excluidas'].items():
        sub = rng.sample(rutas, min(N_ANOM, len(rutas)))
        grupos[ev['nombres'][k]] = [(p, int(k)) for p in sub]
    return grupos


def rasgos(sub):
    """Estadisticos resumen de una curva en la banda objetivo.

    Todo lo que depende de escala va normalizado por el MAD del propio objeto,
    para que el detector no aprenda simplemente a separar por brillo aparente.
    """
    f = sub[col_flux].astype('float64').values
    t = sub[col_mjd].astype('float64').values
    med = np.median(f)
    s = mad(f) or 1.0
    z = (f - med) / s
    span = float(t.max() - t.min()) if len(t) > 1 else 0.0
    i_max = int(np.argmax(f))
    return {
        # Los 'abs_' quedan fuera del Isolation Forest y se usan solo en el
        # diagnostico por rasgo. Son la escala y la duracion absolutas, que es
        # justo lo que el renderizado de las imagenes borra al normalizar por
        # objeto y por span.
        'abs_log_pico': float(np.log10(max(f.max() - med, 1e-3))),
        'abs_log_mad': float(np.log10(max(s, 1e-3))),
        'abs_span': span,
        'n': float(len(f)),
        'span': span,
        'z_max': float(z.max()),
        'z_min': float(z.min()),
        'z_abs_max': float(np.abs(z).max()),
        'z_std': float(z.std()),
        'z_p95': float(np.percentile(z, 95)),
        'frac_sobre_3': float((np.abs(z) > 3).mean()),
        'frac_sobre_10': float((np.abs(z) > 10).mean()),
        'asimetria': float(((z - z.mean()) ** 3).mean()),
        'curtosis': float(((z - z.mean()) ** 4).mean()),
        'pos_max': float((t[i_max] - t.min()) / span) if span > 0 else 0.5,
        'mad_rel': float(s / (abs(med) + 1e-9)),
        'cadencia': float(span / len(f)) if len(f) else 0.0,
    }


def fase_curva():
    grupos = muestrear()
    por_archivo = {}
    for grupo, items in grupos.items():
        for p, lab in items:
            m = re.match(r'(\d+)', Path(p).stem)
            if m:
                por_archivo.setdefault(NOMBRES[lab], []).append((m.group(1), grupo))

    filas = []
    for nombre, items in sorted(por_archivo.items()):
        ruta = os.path.join(RAW_DIR, f'lc_{nombre}.parquet')
        if not os.path.exists(ruta):
            print(f'  falta {ruta}, se omite')
            continue
        df = pd.read_parquet(ruta)
        col_id = [c for c in df.columns if c.lower() in ('snid', 'objectid', 'id')][0]
        df[col_id] = df[col_id].astype(str)
        quiero = {s for s, _ in items}
        df = df[df[col_id].isin(quiero) & (df[col_band] == TARGET_BAND)]
        grupo_de = dict(items)
        n_ok = 0
        for snid, sub in df.groupby(col_id):
            if len(sub) < 6:
                continue
            r = rasgos(sub.sort_values(col_mjd))
            r.update(snid=snid, clase=nombre, grupo=grupo_de[snid])
            filas.append(r)
            n_ok += 1
        print(f'  {nombre:24s} {n_ok:5d} curvas')

    OUT.mkdir(parents=True, exist_ok=True)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(OUT / 'rasgos_curva.csv', index=False)
    print(f'\n{len(tabla):,} curvas en {OUT / "rasgos_curva.csv"}')
    print(tabla.groupby('grupo').size().to_string())
    return tabla


def fase_modelo():
    """Puntajes del VQ-VAE y del prior, sobre los mismos objetos de la fase curva.

    Se guardan tres puntajes. 'recon' es el error de reconstruccion, el baseline.
    Del prior se guardan dos, porque la eleccion no es obvia: 'prior_max' es el
    max NLL por token, que es lo que usamos con anomalias localizadas, y
    'prior_media' es la NLL promedio de la secuencia, que deberia ajustarse mejor
    cuando lo raro es el objeto entero y no un tramo.
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from scripts.eval_enc_cmp import mejor_ckpt, tf
    from scripts.eval_mascara_cruzada import mascara_ventana
    from src.models.maskgit import MaskGIT
    from src.models.vqvae.model import VQVAE

    N_RONDAS, RATIO, LOTE = 24, 0.5, 64
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def nll(t, forma, g):
        """NLL por token promediada sobre N_RONDAS de enmascarado."""
        B, T = t.shape
        n = int(T * RATIO)
        total = torch.zeros(B, T, device=dev)
        veces = torch.zeros(B, T, device=dev)
        for _ in range(N_RONDAS):
            if forma == 'aleatoria':
                mask = torch.zeros(B, T, dtype=torch.bool, device=dev)
                for i in range(B):
                    mask[i, torch.randperm(T, device=dev, generator=g)[:n]] = True
            else:
                mask = torch.cat([mascara_ventana(T, n, dev, g) for _ in range(B)])
            masked = t.clone()
            masked[mask] = mg.mask_token
            lp = F.log_softmax(mg(masked), dim=-1)
            total += (-lp.gather(2, t.unsqueeze(-1)).squeeze(-1)) * mask.float()
            veces += mask.float()
        return total / veces.clamp(min=1.0)

    tabla = pd.read_csv(OUT / 'rasgos_curva.csv', dtype={'snid': str})
    quiero = set(tabla['snid'])

    with open(f'{ROOT}/paths_heldout.txt') as f:
        paths = [l.strip() for l in f]
    tokens = np.load(f'{ROOT}/tokens_heldout.npy')

    idx, snids = [], []
    for i, p in enumerate(paths):
        m = re.match(r'(\d+)', Path(p).stem)
        if m and m.group(1) in quiero:
            idx.append(i)
            snids.append(m.group(1))
    print(f'{len(idx):,} de {len(quiero):,} objetos encontrados en los tokens')

    vq = VQVAE.load_from_checkpoint(
        mejor_ckpt(f'results/enc_cmp_heldout/{RAMA}/vqvae/checkpoints/**/*.ckpt'),
        map_location=dev).to(dev).eval()
    mg = MaskGIT.load_from_checkpoint(
        mejor_ckpt(f'results/enc_cmp_heldout/{RAMA}/maskgit/checkpoints/**/*.ckpt'),
        map_location=dev).to(dev).eval()

    recon = {}
    p = {f'prior_{m}{s}': {} for m in ('max', 'media') for s in ('', '_ven')}
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)
    with torch.no_grad():
        for a in range(0, len(idx), LOTE):
            lote = idx[a:a + LOTE]
            ids = snids[a:a + LOTE]

            x = torch.stack([tf(Image.open(paths[i]).convert('RGB')) for i in lote]).to(dev)
            z = vq.encoder(x)
            zq, *_ = vq.codebook(z)
            err = ((vq.decoder(zq) - x) ** 2).flatten(1).mean(1)
            for s, e in zip(ids, err.tolist()):
                recon[s] = e

            t = torch.from_numpy(tokens[lote]).long().to(dev)
            for forma, suf in (('aleatoria', ''), ('ventana', '_ven')):
                m = nll(t, forma, g)
                for s, mx, md in zip(ids, m.max(1).values.tolist(), m.mean(1).tolist()):
                    p[f'prior_max{suf}'][s] = mx
                    p[f'prior_media{suf}'][s] = md

            if a % (LOTE * 20) == 0:
                print(f'  {a:,}/{len(idx):,}', flush=True)

    for nombre, d in (('recon', recon), *p.items()):
        with open(OUT / f'puntajes_{nombre}.json', 'w') as f:
            json.dump(d, f)
        print(f'escrito {OUT / f"puntajes_{nombre}.json"}')


def fase_resumen():
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_auc_score

    tabla = pd.read_csv(OUT / 'rasgos_curva.csv')
    cols = [c for c in tabla.columns
            if c not in ('snid', 'clase', 'grupo') and not c.startswith('abs_')]
    es_norm = tabla['grupo'] == 'normal'

    X = tabla[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    mu, sd = X[es_norm].mean(0), X[es_norm].std(0) + 1e-9
    Xn = (X - mu) / sd

    iso = IsolationForest(n_estimators=300, random_state=SEED, contamination='auto')
    iso.fit(Xn[es_norm.values])

    puntajes = {
        'mad_max': tabla['z_abs_max'].values,
        'iforest': -iso.score_samples(Xn),
    }
    for extra in ('recon', 'prior_max', 'prior_media', 'prior_max_ven', 'prior_media_ven'):
        f = OUT / f'puntajes_{extra}.json'
        if f.exists():
            d = json.load(open(f))
            puntajes[extra] = np.array([d.get(str(s), np.nan) for s in tabla['snid']])

    y_norm = tabla.loc[es_norm]
    resultado = {}
    print(f'\n{"clase":24s} {"n":>5s} ' + ' '.join(f'{m:>9s}' for m in puntajes))
    for clase in sorted(tabla.loc[~es_norm, 'clase'].unique()):
        sel = tabla['clase'] == clase
        fila = {}
        for m, sc in puntajes.items():
            a, b = sc[es_norm.values], sc[sel.values]
            ok = np.isfinite(a), np.isfinite(b)
            if ok[1].sum() < 10:
                fila[m] = float('nan')
                continue
            y = np.r_[np.zeros(ok[0].sum()), np.ones(ok[1].sum())]
            fila[m] = float(roc_auc_score(y, np.r_[a[ok[0]], b[ok[1]]]))
        resultado[clase] = fila
        print(f'{clase:24s} {int(sel.sum()):5d} '
              + ' '.join(f'{fila[m]:9.3f}' for m in puntajes))

    prom = {m: float(np.nanmean([r[m] for r in resultado.values()])) for m in puntajes}
    print(f'\n{"promedio":24s} {len(y_norm):5d} ' + ' '.join(f'{prom[m]:9.3f}' for m in puntajes))

    # Diagnostico: cuanto separa cada rasgo por si solo. Interesa sobre todo si
    # los 'abs_' llevan la señal, porque esos son los que el renderizado borra.
    print('\n--- AUROC de cada rasgo por separado ---')
    diag = {}
    por_rasgo = {c: tabla[c].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
                 for c in tabla.columns if c not in ('snid', 'clase', 'grupo')}
    clases = sorted(tabla.loc[~es_norm, 'clase'].unique())
    print(f'{"rasgo":14s} ' + ' '.join(f'{c[:9]:>9s}' for c in clases) + f' {"|dif|med":>9s}')
    for c, sc in por_rasgo.items():
        fila = {}
        for cl in clases:
            sel = (tabla['clase'] == cl).values
            y = np.r_[np.zeros(es_norm.sum()), np.ones(sel.sum())]
            fila[cl] = float(roc_auc_score(y, np.r_[sc[es_norm.values], sc[sel]]))
        diag[c] = fila
        fuerza = float(np.mean([abs(v - 0.5) for v in fila.values()]))
        print(f'{c:14s} ' + ' '.join(f'{fila[cl]:9.3f}' for cl in clases) + f' {fuerza:9.3f}')

    with open(OUT / 'baselines_auroc.json', 'w') as f:
        json.dump({'por_clase': resultado, 'promedio': prom, 'por_rasgo': diag,
                   'n_normales': int(es_norm.sum())}, f, indent=1)
    print(f'\nescrito {OUT / "baselines_auroc.json"}')


if __name__ == '__main__':
    fase = sys.argv[1] if len(sys.argv) > 1 else 'curva'
    if fase == 'curva':
        fase_curva()
    elif fase == 'modelo':
        fase_modelo()
    elif fase == 'resumen':
        fase_resumen()
    else:
        raise SystemExit(f'fase desconocida: {fase}. Use curva | modelo | resumen')
