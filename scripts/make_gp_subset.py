"""Subset del piloto GP sin re-encodear y sin pisar data/gp_pilot.

Lee el meta del cv completo (as-is, 26 clases) y escribe folds/labels en otro
directorio, apuntando a los .npy ya cacheados.

Se conserva:
  - split cv (el test del paper queda afuera)
  - el fold original: fold 0 = val, folds 1-4 = train
  - DDF y WFD
  - balance: el mismo N en cada (clase, survey, fold)

Con --kind se filtra a transient o nontransient; las etiquetas se reindexan
a 0..K-1 en orden alfabético sobre las clases que quedan.

Uso:
  python scripts/make_gp_subset.py
  python scripts/make_gp_subset.py --kind transient --out data/gp_pilot_tr
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

CLASES_ESPERADAS = {None: 26, "transient": 15, "nontransient": 11}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/gp_pilot")
    ap.add_argument("--out", default="data/gp_pilot_s80")
    ap.add_argument("--per-fold", type=int, default=16)
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--kind", default=None, choices=["transient", "nontransient"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if out.resolve() == src.resolve():
        raise SystemExit("el subset no puede escribirse encima del piloto completo")

    meta = pd.read_csv(src / "meta.csv")
    with open(src / "prepare_args.json") as f:
        prep = json.load(f)
    if prep.get("enc") != "as_is" or prep.get("use_var"):
        raise SystemExit(f"el cache no es el as-is de 1 canal: {prep}")

    if args.kind:
        meta = meta[meta["kind"] == args.kind]

    classes = sorted(meta["class"].unique().tolist())
    esperadas = CLASES_ESPERADAS[args.kind]
    if len(classes) != esperadas:
        raise SystemExit(f"se esperaban {esperadas} clases, hay {len(classes)}")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    parts = []
    for _, g in meta.groupby(["class", "survey", "fold"], sort=False):
        if len(g) < args.per_fold:
            raise SystemExit(
                f"grupo con {len(g)} < {args.per_fold}: "
                f"{g.iloc[0][['class', 'survey', 'fold']].to_dict()}"
            )
        parts.append(g.sample(n=args.per_fold, random_state=args.seed))
    sub = pd.concat(parts, ignore_index=True)

    missing = [p for p in sub["path"] if not Path(p).exists()]
    if missing:
        raise SystemExit(f"faltan {len(missing)} npy, ej. {missing[0]}")
    if (sub["path"].map(lambda p: Path(p).stem) != sub["stem"]).any():
        raise SystemExit("stem no coincide con el nombre del .npy")

    labels = {
        stem: int(class_to_idx[klass])
        for stem, klass in zip(sub["stem"], sub["class"])
    }

    train = sub.loc[sub["fold"] != args.val_fold, "path"].tolist()
    val = sub.loc[sub["fold"] == args.val_fold, "path"].tolist()

    counts = sub.groupby(["class", "survey", "fold"]).size()
    if counts.min() != args.per_fold or counts.max() != args.per_fold:
        raise SystemExit("el subset quedó desbalanceado")
    if set(sub["class"]) != set(classes):
        raise SystemExit("faltan clases en el subset")
    if set(sub["survey"]) != {"DDF", "WFD"}:
        raise SystemExit("faltan surveys")
    kinds = {args.kind} if args.kind else {"transient", "nontransient"}
    if set(sub["kind"]) != kinds:
        raise SystemExit(f"kinds inesperados: {set(sub['kind'])}")
    if sub["fold"].nunique() != 5:
        raise SystemExit("no están los 5 folds del paper")

    spot = np.load(sub["path"].iloc[0])
    if spot.shape != (1, 256, 256):
        raise SystemExit(f"forma inesperada: {spot.shape}")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "labels.json", "w") as f:
        json.dump(labels, f)
    with open(out / "classes.json", "w") as f:
        json.dump({"classes": classes, "class_to_idx": class_to_idx}, f, indent=2)
    with open(out / "folds.json", "w") as f:
        json.dump({str(args.val_fold): {"train": train, "val": val}}, f)
    sub.to_csv(out / "meta.csv", index=False)
    with open(out / "subset_args.json", "w") as f:
        json.dump(
            {
                "src": str(src),
                "kind": args.kind,
                "per_fold": args.per_fold,
                "val_fold": args.val_fold,
                "seed": args.seed,
                "enc": "as_is",
                "n": int(len(sub)),
                "n_train": len(train),
                "n_val": len(val),
                "n_classes": len(classes),
            },
            f,
            indent=2,
        )

    print(
        f"Subset: {len(sub):,} objetos | train {len(train):,} | val {len(val):,} | "
        f"{args.per_fold} por clase×survey×fold | {len(classes)} clases "
        f"({args.kind or 'todas'}) | as-is",
        flush=True,
    )
    print(f"Salida: {out} (imágenes siguen en {src / 'images'})", flush=True)


if __name__ == "__main__":
    main()
