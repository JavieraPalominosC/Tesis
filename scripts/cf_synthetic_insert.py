"""
Contrafactual de las imagenes con pulso insertado.
Si el modelo detecta el pulso, el contrafactual deberia borrarlo.

Uso: python scripts/cf_synthetic_insert.py
"""
import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

from scripts.train_maskgit_cls import MaskGITWithViz
from src.models.vqvae.model import VQVAE

VQVAE_CKPT   = 'results/vqvae_waterfall_full/checkpoints/vqvae-epoch=26-train/loss=0.0808.ckpt'
MASKGIT_CKPT = 'results/maskgit_full/waterfall/checkpoints/maskgit-epoch=31-val/loss=3.1783.ckpt'
IMG_DIR      = 'data/images/elasticc_1/synthetic_insert'
OUT          = Path('results/synthetic_eval/insert_cf'); OUT.mkdir(parents=True, exist_ok=True)

GRID = 32
QUANTILE = 0.85
N_STEPS = 12


@torch.no_grad()
def nll_per_token(model, tokens, n_rounds=5, mask_ratio=0.5):
    B, T = tokens.shape
    total = torch.zeros(B, T, device=tokens.device)
    counts = torch.zeros(B, T, device=tokens.device)
    for _ in range(n_rounds):
        n_mask = int(T*mask_ratio)
        mask = torch.zeros(B, T, dtype=torch.bool, device=tokens.device)
        for i in range(B):
            perm = torch.randperm(T, device=tokens.device)
            mask[i, perm[:n_mask]] = True
        masked = tokens.clone(); masked[mask] = model.mask_token
        lp = F.log_softmax(model(masked), dim=-1)
        tlp = lp.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
        total += (-tlp)*mask.float(); counts += mask.float()
    return total/counts.clamp(min=1.0)


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vq = VQVAE.load_from_checkpoint(VQVAE_CKPT, map_location=dev); vq.eval(); vq.to(dev)
    mg = MaskGITWithViz.load_from_checkpoint(MASKGIT_CKPT, map_location=dev, strict=False)
    mg.eval(); mg.to(dev)
    tf = T.Compose([T.Resize((256,256)), T.ToTensor(), T.Normalize([0.5]*3,[0.5]*3)])
    print("modelos cargados", flush=True)

    # tomar los que tienen amp15 (el pulso mas evidente) y su control
    amp_files = sorted(Path(IMG_DIR).glob('*__amp15.png'))
    for pa in amp_files:
        snid = pa.stem.split('__')[0]
        pc = Path(IMG_DIR)/f'{snid}__control.png'
        if not pc.exists(): continue

        # tokens de la perturbada
        x = tf(Image.open(pa).convert('RGB')).unsqueeze(0).to(dev)
        with torch.no_grad():
            z = vq.encoder(x); _,_,idxs,*_ = vq.codebook(z)
        tok = idxs.view(1,-1).long().to(dev)

        nll = nll_per_token(mg, tok, n_rounds=5)
        thr = torch.quantile(nll, QUANTILE)
        amask = nll > thr
        n_mod = int(amask.sum().item())
        cf_tok = mg.generate_counterfactual(tok, amask, n_steps=N_STEPS)

        with torch.no_grad():
            x_rec = vq.decode_from_indices(tok.view(1,GRID,GRID))
            x_cf  = vq.decode_from_indices(cf_tok.view(1,GRID,GRID))
        rec = x_rec[0].cpu().permute(1,2,0).numpy().clip(0,1)
        cf  = x_cf[0].cpu().permute(1,2,0).numpy().clip(0,1)
        ctrl = np.asarray(Image.open(pc).convert('RGB').resize((256,256)), dtype=np.float32)/255.0
        pert = np.asarray(Image.open(pa).convert('RGB').resize((256,256)), dtype=np.float32)/255.0

        diff = np.abs(rec-cf).mean(axis=2)
        vmax = np.percentile(diff, 99.5); vmin = np.percentile(diff, 60)

        fig, ax = plt.subplots(1, 5, figsize=(22, 5))
        fig.suptitle(f'{snid} | pulso amp15 en banda r | tokens mod={n_mod}/1024', fontsize=12)
        for a, img, ttl in zip(ax[:4],
                               [ctrl, pert, rec, cf],
                               ['Control (sin pulso)', 'Con pulso insertado',
                                'Reconstruccion VQ-VAE', 'Contrafactual']):
            a.imshow(img); a.set_title(ttl); a.axis('off')
        im = ax[4].imshow(diff, cmap='inferno', vmin=vmin, vmax=vmax)
        ax[4].set_title('|Recon - CF|'); ax[4].axis('off')
        plt.colorbar(im, ax=ax[4], fraction=0.046)
        plt.tight_layout()
        plt.savefig(OUT/f'cf_insert_{snid}.png', dpi=120, bbox_inches='tight')
        plt.close()
        print(f'  {snid}: {n_mod} tokens modificados', flush=True)

    print(f'Listo en {OUT}/', flush=True)


if __name__ == '__main__':
    main()
