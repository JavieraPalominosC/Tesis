"""Mide en CPU el costo de un paso de entrenamiento del VQ-VAE GP y del MaskGIT.

Uso: python scripts/bench_cpu_gp.py [configs/vqvae_gp_pilot_tr.yaml]
"""
import sys
sys.path.insert(0, ".")
import json
import os
import time

import torch
from omegaconf import OmegaConf

from src.models.vqvae.model_gp import VQVAEGP
from src.models.maskgit import MaskGIT


def medir(fn, n_warm=2, n=5):
    for _ in range(n_warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/vqvae_gp_pilot_tr.yaml"
    cfg = OmegaConf.load(cfg_path).vqvae
    with open(cfg.folds_path) as f:
        folds = json.load(f)["0"]
    n_train, n_val = len(folds["train"]), len(folds["val"])
    with open(os.path.join(os.path.dirname(cfg.folds_path), "classes.json")) as f:
        n_cls = len(json.load(f)["classes"])

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
    print(f"hilos: {torch.get_num_threads()} | train {n_train} | val {n_val}", flush=True)

    bs = cfg.batch_size
    vq = VQVAEGP(in_channels=1, hidden_channels=list(cfg.hidden_channels),
                 embedding_dim=cfg.embedding_dim, num_embeddings=cfg.num_embeddings,
                 commitment_cost=cfg.commitment_cost, num_classes=n_cls,
                 cls_weight=cfg.cls_weight)
    opt = torch.optim.Adam(vq.parameters(), lr=cfg.learning_rate)
    x = torch.randn(bs, 1, cfg.image_size, cfg.image_size).clamp(-1, 1)

    def paso_vq():
        opt.zero_grad()
        x_hat, vq_loss, *_ = vq(x)
        loss = torch.nn.functional.mse_loss(x_hat, x) + vq_loss
        loss.backward()
        opt.step()

    def eval_vq():
        with torch.no_grad():
            vq(x)

    t_vq = medir(paso_vq)
    t_vq_eval = medir(eval_vq)
    epoca_vq = (n_train / bs) * t_vq + (n_val / bs) * t_vq_eval

    mg = MaskGIT(num_tokens=512, seq_len=1024, hidden_dim=256, n_layers=4,
                 n_heads=4, ff_mult=4, dropout=0.1, learning_rate=1e-4)
    opt_mg = torch.optim.Adam(mg.parameters(), lr=1e-4)
    tok = torch.randint(0, 512, (32, 1024))

    def paso_mg():
        opt_mg.zero_grad()
        masked, mask = mg._mask_tokens(tok)
        logits = mg(masked)
        loss = torch.nn.functional.cross_entropy(logits[mask], tok[mask])
        loss.backward()
        opt_mg.step()

    t_mg = medir(paso_mg)
    epoca_mg = (n_train / 32) * t_mg * 1.25

    print(f"VQ-VAE: {t_vq:.2f} s/paso train (batch {bs}), {t_vq_eval:.2f} s/paso val", flush=True)
    print(f"VQ-VAE: {epoca_vq / 60:.1f} min/epoca -> 25 epocas = {25 * epoca_vq / 3600:.1f} h", flush=True)
    print(f"MaskGIT: {t_mg:.2f} s/paso, {epoca_mg / 60:.1f} min/epoca -> 30 epocas = "
          f"{30 * epoca_mg / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
