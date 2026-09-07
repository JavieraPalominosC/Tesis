"""Compara el VQ-VAE mad300 del subconjunto contra el del dataset completo.

Motivo: en el entrenamiento completo la perdida total bajo (0.198 -> ~0.10) pero
la recon_loss que muestra la barra de progreso SUBIO a lo largo de las epocas
(~0.0014 en la epoca 3 hasta ~0.0030 en la 23). Si eso es real y no ruido de un
batch suelto, significa que el modelo cambio fidelidad de reconstruccion por
exactitud del clasificador auxiliar, y que elegir el checkpoint por train/loss
nos dejo justo con uno de los peores en reconstruccion. Para deteccion de outliers
lo que importa es la reconstruccion y el uso del codebook, no el clasificador.

Para que la comparacion sea limpia los dos modelos se evaluan sobre la MISMA
interseccion de imagenes: las que quedaron en validacion para ambos, o sea que
ninguno de los dos vio en entrenamiento.

Uso: python scripts/desglosar_vqvae_full.py
"""
import sys
sys.path.insert(0, '.')
import glob
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.vqvae_dataset import VQVAEDataset
from src.models.vqvae.model import VQVAE

MODELOS = {
    'mad300 subconjunto': ('configs/vqvae_enc_mad300.yaml',
                           'results/enc_cmp/mad300/vqvae/checkpoints'),
    'mad300 completo': ('configs/vqvae_enc_mad300_full.yaml',
                        'results/enc_cmp_full/mad300/vqvae/checkpoints'),
}
N_VAL = 4096


def loss_de(p):
    m = re.search(r'loss=([0-9.]+)\.ckpt$', p)
    return float(m.group(1)) if m else float('inf')


def mejor_ckpt(d):
    c = [p for p in glob.glob(f'{d}/**/*.ckpt', recursive=True) if 'last' not in Path(p).name]
    return min(c, key=loss_de) if c else None


def val_de(cfg_path):
    cfg = OmegaConf.load(cfg_path).vqvae
    with open(cfg.folds_path) as f:
        return cfg, set(json.load(f)['0']['val'])


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfgs, vals = {}, {}
    for nombre, (cfg_path, _) in MODELOS.items():
        cfgs[nombre], vals[nombre] = val_de(cfg_path)
        print(f'{nombre}: {len(vals[nombre])} en validacion', flush=True)

    comun = sorted(set.intersection(*vals.values()))
    print(f'interseccion (ninguno la vio en entrenamiento): {len(comun)}', flush=True)
    if not comun:
        raise SystemExit('sin interseccion, no hay comparacion limpia posible')
    rutas = comun[:N_VAL]

    cfg0 = next(iter(cfgs.values()))
    with open(cfg0.labels_path) as f:
        lbl = json.load(f)
    labels = torch.tensor([lbl.get(Path(p).stem, -1) for p in rutas], dtype=torch.long)

    filas = []
    for nombre, (_, ckpt_dir) in MODELOS.items():
        ckpt = mejor_ckpt(ckpt_dir)
        if ckpt is None:
            print(f'[{nombre}] sin checkpoint en {ckpt_dir}', flush=True)
            continue
        epoca = re.search(r'epoch=(\d+)', ckpt)
        model = VQVAE.load_from_checkpoint(ckpt, map_location=dev).to(dev).eval()
        loader = DataLoader(VQVAEDataset(rutas, image_size=cfg0.image_size),
                            batch_size=64, shuffle=False, num_workers=4)

        rec = vq = cls = acc = 0.0
        perp, usados = [], set()
        n = off = 0
        with torch.no_grad():
            for batch in loader:
                x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(dev)
                y = labels[off:off + x.size(0)].to(dev)
                off += x.size(0)
                b = x.size(0)

                x_hat, vq_loss, idx, perplexity, _, _, z_q = model(x)
                rec += F.mse_loss(x_hat, x).item() * b
                vq += float(vq_loss) * b
                perp.append(float(perplexity))
                usados.update(torch.unique(idx).cpu().tolist())

                if model.classifier is not None:
                    v = y >= 0
                    if v.any():
                        logits = model.classifier(z_q[v])
                        cls += F.cross_entropy(logits, y[v]).item() * b
                        acc += (logits.argmax(1) == y[v]).float().mean().item() * b
                n += b

        filas.append({
            'modelo': nombre, 'epoca': int(epoca.group(1)) if epoca else -1,
            'ckpt_total': loss_de(ckpt), 'n': n,
            'recon': rec / n, 'vq': vq / n, 'cls': cls / n, 'acc': acc / n,
            'perplejidad': float(np.mean(perp)), 'codigos_usados': len(usados),
        })
        print(f'[{nombre}] listo, epoca {filas[-1]["epoca"]}', flush=True)
        del model
        torch.cuda.empty_cache()

    print(f'\nSobre {filas[0]["n"]} imagenes que ninguno vio en entrenamiento:\n')
    print('  modelo               epoca   recon(MSE)   vq_loss   acc_cls   perplej.   codigos')
    for f in filas:
        print(f"  {f['modelo']:<20} {f['epoca']:5d}   {f['recon']:10.6f}  {f['vq']:8.5f}  "
              f"{f['acc'] * 100:6.1f}%   {f['perplejidad']:8.1f}   {f['codigos_usados']:4d}/512")

    with open('results/enc_cmp/desglose_vqvae_full.json', 'w') as f:
        json.dump(filas, f, indent=2)
    print('\nGuardado en results/enc_cmp/desglose_vqvae_full.json')


if __name__ == '__main__':
    main()
