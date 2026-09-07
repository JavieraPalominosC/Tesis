"""Desglosa la loss del VQ-VAE de cada rama en sus tres componentes.

El numero que guarda el nombre del checkpoint es la suma:

    loss = recon_loss + vq_loss + 0.1 * cls_loss

que mezcla que tan bien se reconstruye la imagen, que tan bien encaja el
codebook y que tan bien clasifica el clasificador colgado. Compararlas entre
ramas sin separar no dice gran cosa, asi que aca se calcula cada parte por
separado sobre el conjunto de validacion de cada rama.

Ojo igual al interpretar: cada rama se evalua sobre SUS propias imagenes, y las
codificaciones mad producen curvas mas planas, que son intrinsecamente mas
faciles de reconstruir. Un MSE mas bajo puede significar "imagen mas facil" y no
"mejor representacion".
"""
import sys
sys.path.insert(0, '.')
import glob
import json
import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.models.vqvae.model import VQVAE
from src.data.vqvae_dataset import VQVAEDataset

RAMAS = ('actual', 'mad100', 'mad300')
N_VAL = 4096


def loss_de(p):
    m = re.search(r'loss=([0-9.]+)\.ckpt$', p)
    return float(m.group(1)) if m else float('inf')


def mejor_ckpt(rama):
    c = glob.glob(f'results/enc_cmp/{rama}/vqvae/checkpoints/**/*.ckpt', recursive=True)
    return min(c, key=loss_de) if c else None


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filas = []

    for rama in RAMAS:
        ckpt = mejor_ckpt(rama)
        if ckpt is None:
            print(f'[{rama}] sin checkpoint', flush=True)
            continue
        cfg = OmegaConf.load(f'configs/vqvae_enc_{rama}.yaml').vqvae

        with open(cfg.folds_path) as f:
            val_paths = json.load(f)['0']['val'][:N_VAL]
        with open(cfg.labels_path) as f:
            lbl = json.load(f)
        from pathlib import Path
        labels = torch.tensor([lbl.get(Path(p).stem, -1) for p in val_paths], dtype=torch.long)

        model = VQVAE.load_from_checkpoint(ckpt).to(device)
        model.eval()

        loader = DataLoader(VQVAEDataset(val_paths, image_size=cfg.image_size),
                            batch_size=128, shuffle=False, num_workers=4)

        rec = vq = cls = acc = 0.0
        perp = []
        n = off = 0
        with torch.no_grad():
            for batch in loader:
                x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
                y = labels[off:off + x.size(0)].to(device)
                off += x.size(0)

                x_hat, vq_loss, _, perplexity, _, _, z_q = model(x)
                b = x.size(0)
                rec += F.mse_loss(x_hat, x).item() * b
                vq += float(vq_loss) * b
                perp.append(float(perplexity))

                if model.classifier is not None:
                    v = y >= 0
                    if v.any():
                        logits = model.classifier(z_q[v])
                        cls += F.cross_entropy(logits, y[v]).item() * b
                        acc += (logits.argmax(1) == y[v]).float().mean().item() * b
                n += b

        filas.append({
            'rama': rama, 'n': n, 'recon': rec / n, 'vq': vq / n,
            'cls': cls / n, 'acc': acc / n, 'perplejidad': float(np.mean(perp)),
            'total': rec / n + vq / n + 0.1 * (cls / n),
            'ckpt_total': loss_de(ckpt),
        })
        print(f'[{rama}] listo sobre {n} imagenes', flush=True)
        del model
        torch.cuda.empty_cache()

    print('\n  rama      recon(MSE)   vq_loss   cls_loss   acc_cls   perplejidad   total')
    for f in filas:
        print(f"  {f['rama']:<8}  {f['recon']:9.5f}  {f['vq']:8.5f}  {f['cls']:8.4f}  "
              f"{f['acc'] * 100:6.1f}%   {f['perplejidad']:9.1f}   {f['total']:7.4f}")

    with open('results/enc_cmp/desglose_loss.json', 'w') as f:
        json.dump(filas, f, indent=2)
    print('\nGuardado en results/enc_cmp/desglose_loss.json')


if __name__ == '__main__':
    main()
