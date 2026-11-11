"""Folds GP de transientes con clases excluidas, para usarlas como anomalías reales.

Mismo criterio que scripts/make_folds_heldout.py en ELAsTiCC-1: las clases
excluidas no aparecen ni en train ni en val, para que la selección del
checkpoint tampoco las vea. Se excluyen las equivalentes a las de ese
experimento; SNIIb+HostXT_V19 no tiene par en el sample del paper.

Sobre el cv completo de transientes (sin submuestrear), con los folds del
paper: fold 0 = val, folds 1-4 = train. Los .npy apuntan al cache de
data/gp_pilot/images.

Escribe en --out:
  folds.json         train/val solo con las clases vistas
  labels.json        etiquetas 0..K-1 de las vistas, en orden alfabético
  classes.json       nombres de las vistas
  heldout_eval.json  rutas de las excluidas, agrupadas por clase
  meta.csv           meta de todo lo usado, con columna heldout

Uso: python scripts/make_gp_heldout.py [--out data/gp_tr_ho]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXCLUIDAS = [
    "SNIa-91bg-elastic",
    "SNIax-elastic",
    "SLSN-I",
    "PISN-MOSFIT",
    "PISN-STELLA-HECORE",
    "PISN-STELLA-HYDROGENIC",
]
N_TRANSIENTES = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/gp_pilot")
    ap.add_argument("--out", default="data/gp_tr_ho")
    ap.add_argument("--val-fold", type=int, default=0)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if out.resolve() == src.resolve():
        raise SystemExit("no puede escribirse encima del piloto completo")

    with open(src / "prepare_args.json") as f:
        prep = json.load(f)
    if prep.get("enc") != "as_is" or prep.get("use_var"):
        raise SystemExit(f"el cache no es el as-is de 1 canal: {prep}")

    meta = pd.read_csv(src / "meta.csv")
    meta = meta[meta["kind"] == "transient"].copy()
    todas = sorted(meta["class"].unique())
    if len(todas) != N_TRANSIENTES:
        raise SystemExit(f"se esperaban {N_TRANSIENTES} transientes, hay {len(todas)}")
    faltan = set(EXCLUIDAS) - set(todas)
    if faltan:
        raise SystemExit(f"clases excluidas que no están en el sample: {faltan}")

    meta["heldout"] = meta["class"].isin(EXCLUIDAS)
    vistas = meta[~meta["heldout"]]
    fuera = meta[meta["heldout"]]

    clases = sorted(vistas["class"].unique())
    idx = {c: i for i, c in enumerate(clases)}

    # en el cv el paper balancea por clase (800); DDF/WFD y los folds varían
    # unos pocos objetos alrededor de 400 y 160
    cuentas = vistas.groupby("class").size()
    if cuentas.nunique() != 1:
        raise SystemExit(f"las vistas no están balanceadas:\n{cuentas}")
    if vistas["fold"].nunique() != 5 or vistas["survey"].nunique() != 2:
        raise SystemExit("faltan folds o surveys en las vistas")

    for p in meta["path"].head(20):
        x = np.load(p)
        if x.shape != (1, 256, 256):
            raise SystemExit(f"shape inesperada {x.shape} en {p}")

    es_val = vistas["fold"] == args.val_fold
    folds = {"0": {
        "train": sorted(vistas.loc[~es_val, "path"]),
        "val": sorted(vistas.loc[es_val, "path"]),
    }}
    labels = {s: idx[c] for s, c in zip(vistas["stem"], vistas["class"])}
    heldout = {c: sorted(fuera.loc[fuera["class"] == c, "path"]) for c in EXCLUIDAS}

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "folds.json", "w") as f:
        json.dump(folds, f)
    with open(out / "labels.json", "w") as f:
        json.dump(labels, f)
    with open(out / "classes.json", "w") as f:
        json.dump({"classes": clases}, f, indent=2)
    with open(out / "heldout_eval.json", "w") as f:
        json.dump({"excluidas": heldout}, f)
    meta.to_csv(out / "meta.csv", index=False)
    with open(out / "heldout_args.json", "w") as f:
        json.dump({**vars(args), "excluidas": EXCLUIDAS}, f, indent=2)

    print(f"vistas: {len(clases)} clases, train {len(folds['0']['train']):,} | "
          f"val {len(folds['0']['val']):,}")
    for c in clases:
        n_val = int(((vistas["class"] == c) & es_val).sum())
        print(f"  {idx[c]:2d}  {c:24s} {int((vistas['class'] == c).sum()):5d}  (val {n_val})")
    print(f"excluidas: {len(EXCLUIDAS)} clases, {len(fuera):,} objetos")
    for c in EXCLUIDAS:
        print(f"      {c:24s} {len(heldout[c]):5d}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
