"""F1 del clasificador auxiliar del VQ-VAE sobre las 17 clases de supernova.

Hasta ahora de este clasificador solo reportabamos accuracy, y con 17 clases muy
desbalanceadas eso engana: un modelo puede sacar 40% acertando solo las clases
abundantes y sin tocar nunca las raras. El macro-F1 pesa a todas las clases por
igual, asi que es el que delata ese comportamiento.

Sobre la comparacion entre codificaciones. Cada rama del subconjunto se entreno
con SUS propias imagenes, asi que lo que se mide aca no es "cual codificacion
clasifica mejor" en abstracto, sino cuanta informacion de clase conserva cada
codificacion en el latente z_q del VQ-VAE. Por eso la interseccion se toma por
SNID y no por ruta: los mismos objetos, cada modelo viendolos en su propia
codificacion.

Dos conjuntos de evaluacion, porque no hay uno solo que sirva para los tres:

  comun3   SNIDs en validacion de las tres ramas a la vez. Es el unico limpio
           para el modelo completo, pero queda chico y las clases raras casi no
           aparecen; el support por clase va en la tabla para que se vea.
  comun2   SNIDs en validacion de las dos ramas del subconjunto. Mucho mas
           grande, sirve para comparar actual contra mad300 con soporte decente.
           No se usa para el modelo completo porque los vio en entrenamiento.
  natural  Muestra de la validacion del modelo completo, con la proporcion de
           clases REAL del dataset. Los dos conjuntos de arriba heredan del
           subconjunto, que se renderizo tomando N objetos por clase y por lo
           tanto quedo casi balanceado; ahi el macro-F1 no puede delatar el
           desbalance porque no hay desbalance que delatar. Este es el unico
           que responde de verdad si el modelo abandona a las clases raras.

Uso: python scripts/f1_clasificador_vqvae.py
"""
import sys
sys.path.insert(0, '.')
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.vqvae_dataset import VQVAEDataset
from src.models.vqvae.model import VQVAE

# El indice de clase es la posicion en SN_FILES de run_*.sh: render_subset_enc.py
# recibe ese indice como label_id y lo escribe tal cual en labels.json.
CLASES = [
    'SNIa-SALT2', 'SNIa-91bg', 'SNIax', 'SNII-Templates', 'SNII-NMF',
    'SNII+HostXT_V19', 'SNIb-Templates', 'SNIb+HostXT_V19', 'SNIc-Templates',
    'SNIc+HostXT_V19', 'SNIcBL+HostXT_V19', 'SNIIb+HostXT_V19', 'SNIIn-MOSFIT',
    'SNIIn+HostXT_V19', 'SLSN-I+host', 'SLSN-I_no_host', 'PISN',
]
CORTO = [c.replace('+HostXT_V19', '+HXT').replace('-Templates', '-Tpl')
         .replace('_no_host', '_nh') for c in CLASES]

MODELOS = {
    'actual (subconj.)': {
        'cfg': 'configs/vqvae_enc_actual.yaml',
        'ckpt': 'results/enc_cmp/actual/vqvae/checkpoints',
        'imgs': 'data/enc_cmp/actual/images',
    },
    'mad300 (subconj.)': {
        'cfg': 'configs/vqvae_enc_mad300.yaml',
        'ckpt': 'results/enc_cmp/mad300/vqvae/checkpoints',
        'imgs': 'data/enc_cmp/mad300/images',
    },
    'mad300 (1.39M)': {
        'cfg': 'configs/vqvae_enc_mad300_full.yaml',
        'ckpt': 'results/enc_cmp_full/mad300/vqvae/checkpoints',
        'imgs': 'data/enc_cmp/mad300/images',
    },
}
SUBCONJUNTO = ['actual (subconj.)', 'mad300 (subconj.)']
COMPLETO = 'mad300 (1.39M)'
OUT = Path('results/enc_cmp_full/eval')
MAX_COMUN2, MAX_NATURAL, SEED = 12000, 12000, 0


def loss_de(p):
    m = re.search(r'loss=([0-9.]+)\.ckpt$', p)
    return float(m.group(1)) if m else float('inf')


def mejor_ckpt(d):
    c = [p for p in glob.glob(f'{d}/**/*.ckpt', recursive=True)
         if 'last' not in Path(p).name]
    if not c:
        raise SystemExit(f'sin checkpoints en {d}')
    return min(c, key=loss_de)


def stems(paths):
    return {Path(p).stem for p in paths}


def metricas(y, yhat, n_clases=len(CLASES)):
    """Precision, recall, F1 y support por clase, mas los agregados."""
    cm = np.zeros((n_clases, n_clases), dtype=int)
    for a, b in zip(y, yhat):
        cm[a, b] += 1
    tp = np.diag(cm).astype(float)
    sup = cm.sum(axis=1).astype(float)
    pred = cm.sum(axis=0).astype(float)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    rec = np.divide(tp, sup, out=np.zeros_like(tp), where=sup > 0)
    den = prec + rec
    f1 = np.divide(2 * prec * rec, den, out=np.zeros_like(tp), where=den > 0)

    presentes = sup > 0
    peso = sup / sup.sum() if sup.sum() else sup
    return {
        'cm': cm,
        'precision': prec, 'recall': rec, 'f1': f1, 'support': sup.astype(int),
        'accuracy': float(tp.sum() / sup.sum()) if sup.sum() else 0.0,
        'macro_f1': float(f1[presentes].mean()) if presentes.any() else 0.0,
        'weighted_f1': float((f1 * peso).sum()),
        'clases_presentes': int(presentes.sum()),
        'clases_f1_cero': [CLASES[i] for i in range(n_clases)
                           if presentes[i] and f1[i] == 0.0],
    }


@torch.no_grad()
def evaluar(nombre, info, snids, dev):
    cfg = OmegaConf.load(info['cfg']).vqvae
    with open(cfg.labels_path) as f:
        lbl = json.load(f)

    rutas, y = [], []
    for s in snids:
        if s in lbl and lbl[s] >= 0:
            p = f"{info['imgs']}/{s}.png"
            rutas.append(p)
            y.append(int(lbl[s]))
    if not rutas:
        raise SystemExit(f'{nombre}: sin imagenes etiquetadas')

    ckpt = mejor_ckpt(info['ckpt'])
    model = VQVAE.load_from_checkpoint(ckpt, map_location=dev).to(dev).eval()
    if model.classifier is None:
        raise SystemExit(f'{nombre}: el modelo no tiene clasificador')

    loader = DataLoader(VQVAEDataset(rutas, image_size=cfg.image_size),
                        batch_size=64, shuffle=False, num_workers=4)
    pred = []
    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(dev)
        *_, z_q = model(x)
        pred.append(model.classifier(z_q).argmax(1).cpu().numpy())
    pred = np.concatenate(pred)

    m = metricas(np.array(y), pred)
    m['n'] = len(y)
    m['ckpt'] = ckpt
    m['epoca'] = int(re.search(r'epoch=(\d+)', ckpt).group(1)) if 'epoch=' in ckpt else -1
    del model
    torch.cuda.empty_cache()
    return m


def tabla(nombre, m):
    print(f'\n=== {nombre}   (epoca {m["epoca"]}, n={m["n"]}) ===')
    print(f'  accuracy {m["accuracy"] * 100:5.1f}%   macro-F1 {m["macro_f1"]:.3f}   '
          f'weighted-F1 {m["weighted_f1"]:.3f}')
    print(f'  {"clase":<20}{"prec":>7}{"recall":>8}{"F1":>7}{"support":>9}')
    for i, c in enumerate(CLASES):
        if m['support'][i] == 0:
            continue
        marca = '   <- F1 cero' if m['f1'][i] == 0 else ''
        print(f'  {c:<20}{m["precision"][i]:7.3f}{m["recall"][i]:8.3f}'
              f'{m["f1"][i]:7.3f}{m["support"][i]:9d}{marca}')
    if m['clases_f1_cero']:
        print(f'  clases con F1 = 0 ({len(m["clases_f1_cero"])}): '
              f'{", ".join(m["clases_f1_cero"])}')


def figura(res, conjunto, archivo):
    nombres = [n for n in res]
    fig, axes = plt.subplots(1, len(nombres), figsize=(7.2 * len(nombres), 7.4))
    if len(nombres) == 1:
        axes = [axes]
    for ax, n in zip(axes, nombres):
        m = res[n]
        cm = m['cm'].astype(float)
        fila = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, fila, out=np.full_like(cm, np.nan), where=fila > 0)
        im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(CLASES)))
        ax.set_yticks(range(len(CLASES)))
        ax.set_xticklabels(CORTO, rotation=90, fontsize=7)
        ax.set_yticklabels([f'{c}  ({s})' for c, s in zip(CORTO, m['support'])],
                           fontsize=7)
        ax.set_xlabel('prediccion', fontsize=9)
        ax.set_ylabel('clase verdadera  (support)', fontsize=9)
        ax.set_title(f'{n}\nacc {m["accuracy"] * 100:.1f}%   '
                     f'macro-F1 {m["macro_f1"]:.3f}   '
                     f'weighted-F1 {m["weighted_f1"]:.3f}', fontsize=10)
        for i in range(len(CLASES)):
            for j in range(len(CLASES)):
                if m['cm'][i, j] and norm[i, j] >= 0.10:
                    ax.text(j, i, f'{norm[i, j]:.2f}'.lstrip('0'), ha='center',
                            va='center', fontsize=5.5,
                            color='white' if norm[i, j] > 0.6 else '#22303c')
    fig.colorbar(im, ax=axes, label='fraccion de la fila (recall por clase)',
                 fraction=0.02, pad=0.015)
    fig.suptitle('Clasificador auxiliar del VQ-VAE, 17 clases   '
                 f'({conjunto})', fontsize=11)
    # y negativo para que no choque con la etiqueta del eje x: bbox_inches='tight'
    # expande el lienzo para incluirlo.
    fig.text(0.5, -0.02, 'Matriz normalizada por fila: la diagonal es el recall de '
             'cada clase. Solo se anotan celdas sobre 0.10.',
             ha='center', fontsize=8, style='italic', color='#555555')
    fig.savefig(OUT / archivo, dpi=140, bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'dispositivo: {dev}', flush=True)

    vals = {}
    for n, info in MODELOS.items():
        cfg = OmegaConf.load(info['cfg']).vqvae
        with open(cfg.folds_path) as f:
            vals[n] = stems(json.load(f)['0']['val'])
        print(f'{n}: {len(vals[n])} en validacion  ({cfg.folds_path})', flush=True)

    comun3 = sorted(set.intersection(*vals.values()))
    comun2 = sorted(set.intersection(*(vals[n] for n in SUBCONJUNTO)))
    print(f'\ncomun3 (las tres ramas): {len(comun3)}', flush=True)
    print(f'comun2 (solo subconjunto): {len(comun2)}', flush=True)

    if len(comun2) > MAX_COMUN2:
        rng = np.random.default_rng(SEED)
        comun2 = sorted(rng.choice(comun2, MAX_COMUN2, replace=False).tolist())
        print(f'comun2 submuestreado a {len(comun2)} por tiempo de computo', flush=True)

    salida = {'clases': CLASES, 'conjuntos': {}}

    print(f'\n{"=" * 70}\nCONJUNTO comun3: {len(comun3)} objetos, limpio para los tres')
    res3 = {}
    for n, info in MODELOS.items():
        res3[n] = evaluar(n, info, comun3, dev)
        tabla(n, res3[n])
    figura(res3, f'validacion comun a las tres ramas, n={len(comun3)}',
           'confusion_clasificador.png')

    print(f'\n{"=" * 70}\nCONJUNTO comun2: {len(comun2)} objetos, limpio para las dos '
          'ramas del subconjunto')
    res2 = {}
    for n in SUBCONJUNTO:
        res2[n] = evaluar(n, MODELOS[n], comun2, dev)
        tabla(n, res2[n])
    figura(res2, f'validacion comun al subconjunto, n={len(comun2)}',
           'confusion_clasificador_subconjunto.png')

    # Distribucion natural: sin el balanceo que impuso el renderizado por clase.
    rng = np.random.default_rng(SEED)
    natural = sorted(vals[COMPLETO])
    if len(natural) > MAX_NATURAL:
        natural = sorted(rng.choice(natural, MAX_NATURAL, replace=False).tolist())
    print(f'\n{"=" * 70}\nCONJUNTO natural: {len(natural)} objetos de la validacion '
          f'del modelo completo, proporciones reales')
    res1 = {COMPLETO: evaluar(COMPLETO, MODELOS[COMPLETO], natural, dev)}
    tabla(COMPLETO, res1[COMPLETO])
    sup = res1[COMPLETO]['support']
    print(f'  desbalance: clase mas comun {sup.max()} objetos, menos comun '
          f'{sup[sup > 0].min()}, razon {sup.max() / max(sup[sup > 0].min(), 1):.0f}:1')
    figura(res1, f'validacion del modelo completo, proporciones reales, '
                 f'n={len(natural)}', 'confusion_clasificador_natural.png')

    for etiqueta, res, n_obj in (('comun3', res3, len(comun3)),
                                 ('comun2', res2, len(comun2)),
                                 ('natural', res1, len(natural))):
        salida['conjuntos'][etiqueta] = {
            'n_objetos': n_obj,
            'modelos': {
                n: {
                    'n': m['n'], 'epoca': m['epoca'], 'ckpt': m['ckpt'],
                    'accuracy': m['accuracy'], 'macro_f1': m['macro_f1'],
                    'weighted_f1': m['weighted_f1'],
                    'clases_presentes': m['clases_presentes'],
                    'clases_f1_cero': m['clases_f1_cero'],
                    'por_clase': {
                        CLASES[i]: {
                            'precision': float(m['precision'][i]),
                            'recall': float(m['recall'][i]),
                            'f1': float(m['f1'][i]),
                            'support': int(m['support'][i]),
                        } for i in range(len(CLASES))
                    },
                    'matriz_confusion': m['cm'].tolist(),
                } for n, m in res.items()
            },
        }

    print(f'\n{"=" * 70}\nRESUMEN')
    print(f'  {"modelo":<20}{"conjunto":>10}{"n":>7}{"acc":>8}{"macro-F1":>10}'
          f'{"weighted-F1":>13}{"F1=0":>7}')
    for etiqueta, res in (('comun3', res3), ('comun2', res2), ('natural', res1)):
        for n, m in res.items():
            print(f'  {n:<20}{etiqueta:>10}{m["n"]:>7}{m["accuracy"] * 100:7.1f}%'
                  f'{m["macro_f1"]:10.3f}{m["weighted_f1"]:13.3f}'
                  f'{len(m["clases_f1_cero"]):7d}')

    with open(OUT / 'f1_clasificador.json', 'w') as f:
        json.dump(salida, f, indent=2)
    print(f'\nGuardado en {OUT}/f1_clasificador.json', flush=True)


if __name__ == '__main__':
    main()
