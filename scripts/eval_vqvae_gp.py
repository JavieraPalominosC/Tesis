"""Evalúa un VQ-VAE GP sobre el split de validación completo.

Las métricas que Lightning muestra en la barra son del último batch
(on_step=True, on_epoch=False, val sin barajar), no del set entero. Este
script recorre todo el split y agrega.

Uso:
    python scripts/eval_vqvae_gp.py [cfg.yaml] [--split val|train] [--ckpt RUTA]

Salida: {checkpoint_dir}/../eval_{split}.json y eval_{split}_confusion.csv
"""
import sys
sys.path.insert(0, ".")
import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

from src.models.vqvae.model_gp import VQVAEGP
from src.data.gp_dataset import GPSurfaceDataset


def loss_de(p):
    m = re.search(r"loss=([0-9.]+)\.ckpt$", p)
    return float(m.group(1)) if m else float("inf")


def perplexity_de(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", nargs="?", default="configs/vqvae_gp_pilot_tr.yaml")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg).vqvae
    root = Path(cfg.folds_path).parent

    ckpt = args.ckpt
    if ckpt is None:
        cands = glob.glob(f"{cfg.checkpoint_dir}/**/*.ckpt", recursive=True)
        if not cands:
            raise SystemExit(f"sin checkpoints en {cfg.checkpoint_dir}")
        ckpt = min(cands, key=loss_de)
    print(f"VQ-VAE GP: {ckpt}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
    model = VQVAEGP.load_from_checkpoint(ckpt, map_location=device).to(device)
    model.eval()
    model.freeze()

    with open(cfg.folds_path) as f:
        paths = json.load(f)["0"][args.split]
    with open(cfg.labels_path) as f:
        labels_map = json.load(f)
    with open(root / "classes.json") as f:
        classes = json.load(f)["classes"]
    meta = pd.read_csv(root / "meta.csv").set_index("stem")

    loader = DataLoader(
        GPSurfaceDataset(paths, labels_map),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=False,
    )

    K = model.codebook.num_embeddings
    code_counts = np.zeros(K, dtype=np.int64)
    recon_obj, preds, labels = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            x_hat, _, indices, _, _, _, z_q = model(x)
            recon_obj.append(F.mse_loss(x_hat, x, reduction="none").mean(dim=[1, 2, 3]).cpu().numpy())
            code_counts += np.bincount(indices.view(-1).cpu().numpy(), minlength=K)
            if model.classifier is not None:
                preds.append(model.classifier(z_q).argmax(1).cpu().numpy())
            labels.append(y.numpy())

    recon_obj = np.concatenate(recon_obj)
    labels = np.concatenate(labels)
    stems = [Path(p).stem for p in paths]
    df = pd.DataFrame({"stem": stems, "label": labels, "recon": recon_obj})
    df["survey"] = meta.loc[stems, "survey"].values

    out = {
        "checkpoint": ckpt,
        "split": args.split,
        "n": int(len(df)),
        "recon_mse": float(recon_obj.mean()),
        "codebook_perplexity": perplexity_de(code_counts),
        "codes_used": int((code_counts > 0).sum()),
        "codebook_size": int(K),
        "recon_mse_por_survey": df.groupby("survey")["recon"].mean().to_dict(),
    }

    if preds:
        df["pred"] = np.concatenate(preds)
        df["ok"] = df["pred"] == df["label"]
        acc_clase = df.groupby("label")["ok"].mean()
        out["acc"] = float(df["ok"].mean())
        out["balanced_acc"] = float(acc_clase.mean())
        out["chance"] = 1.0 / len(classes)
        out["acc_por_survey"] = df.groupby("survey")["ok"].mean().to_dict()
        out["acc_por_clase"] = {classes[int(k)]: float(v) for k, v in acc_clase.items()}
        conf = pd.crosstab(
            df["label"].map(lambda i: classes[i]),
            df["pred"].map(lambda i: classes[i]),
            rownames=["real"], colnames=["pred"],
        ).reindex(index=classes, columns=classes, fill_value=0)
        conf.to_csv(Path(cfg.checkpoint_dir).parent / f"eval_{args.split}_confusion.csv")

    out_path = Path(cfg.checkpoint_dir).parent / f"eval_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{args.split}: n={out['n']}")
    print(f"  recon MSE            {out['recon_mse']:.5f}  {out['recon_mse_por_survey']}")
    print(f"  perplejidad codebook {out['codebook_perplexity']:.1f}  ({out['codes_used']}/{K} códigos usados)")
    if preds:
        print(f"  acc                  {out['acc']:.3f}  (azar {out['chance']:.3f})")
        print(f"  acc por survey       {out['acc_por_survey']}")
        for c, v in out["acc_por_clase"].items():
            print(f"    {c:<24s} {v:.3f}")
    print(f"→ {out_path}", flush=True)


if __name__ == "__main__":
    main()
