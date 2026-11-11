"""Prepara el piloto GP a partir de lsst_paper_sample_v2(_gp).

Lee sample + gp_index + arrays mean/var, cachea cada superficie como
data/gp_pilot/images/<uid_stem>.npy con forma (C, 256, 256) en [-1, 1],
y escribe folds.json / labels.json / classes.json.

Por defecto:
  - las 26 clases balanceadas (DDF+WFD, transient+nontransient)
  - split=cv (test afuera, como el paper)
  - fold 0 leave-one-fold
  - superficie GP as-is (ya normalizada por scale_mean en el tar)

Uso:
  python scripts/prepare_gp_pilot.py
  python scripts/prepare_gp_pilot.py --force
  python scripts/prepare_gp_pilot.py --max-per-class 80
"""
import argparse
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, ".")
from src.data.gp_dataset import encode_gp_surface, resize_to_square, uid_to_stem


ROOT = Path("data/gp_pilot")
TAR_SAMPLE = Path("lsst_paper_sample_v2.tar")
TAR_GP = Path("lsst_paper_sample_v2_gp.tar")
EXTRACT_SAMPLE = Path("data/lsst_paper_sample_v2")
EXTRACT_GP = Path("data/lsst_paper_sample_v2_gp")


def ensure_extracted():
    if not (EXTRACT_SAMPLE / "sample.csv").exists():
        if not TAR_SAMPLE.exists():
            raise SystemExit(f"Falta {TAR_SAMPLE}")
        print(f"Extrayendo {TAR_SAMPLE} → data/ ...", flush=True)
        Path("data").mkdir(parents=True, exist_ok=True)
        with tarfile.open(TAR_SAMPLE) as tf:
            tf.extractall("data")
        assert (EXTRACT_SAMPLE / "sample.csv").exists()

    if not (EXTRACT_GP / "gp_index.csv").exists():
        if not TAR_GP.exists():
            raise SystemExit(f"Falta {TAR_GP}")
        print(f"Extrayendo {TAR_GP} → data/ ...", flush=True)
        Path("data").mkdir(parents=True, exist_ok=True)
        with tarfile.open(TAR_GP) as tf:
            tf.extractall("data")
        assert (EXTRACT_GP / "gp_index.csv").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--enc",
        default="as_is",
        help="as_is (default, superficie del paper) o mad300 (ablación)",
    )
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--survey", default=None, help="DDF | WFD | None=ambas")
    ap.add_argument("--kind", default=None, help="transient | nontransient | None=ambos")
    ap.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Tope por (class, survey) sobre el cv. Ej: 80 ≈ tarde corta.",
    )
    ap.add_argument("--use-var", action="store_true", help="2 canales: mean + std")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Reescribe .npy cacheados (necesario si cambió --enc)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()


    ensure_extracted()

    sample = pd.read_csv(EXTRACT_SAMPLE / "sample.csv", low_memory=False)
    gp_ix = pd.read_csv(EXTRACT_GP / "gp_index.csv")
    df = sample.merge(gp_ix, on="uid", how="inner")
    df = df[df["split"] == "cv"].copy()
    if args.survey:
        df = df[df["survey"] == args.survey]
    if args.kind:
        df = df[df["kind"] == args.kind]

    if args.max_per_class is not None:
        parts = []
        for _, g in df.groupby(["class", "survey"], sort=False):
            if len(g) > args.max_per_class:
                parts.append(g.sample(n=args.max_per_class, random_state=args.seed))
            else:
                parts.append(g)
        df = pd.concat(parts, ignore_index=True)

    classes = sorted(df["class"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    img_dir = ROOT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    mean_cache = {}
    var_cache = {}

    def get_mean(set_name):
        if set_name not in mean_cache:
            mean_cache[set_name] = np.load(
                EXTRACT_GP / "gp" / f"{set_name}_mean.npy", mmap_mode="r"
            )
        return mean_cache[set_name]

    def get_var(set_name):
        if set_name not in var_cache:
            var_cache[set_name] = np.load(
                EXTRACT_GP / "gp" / f"{set_name}_var.npy", mmap_mode="r"
            )
        return var_cache[set_name]

    labels = {}
    train_paths, val_paths = [], []
    meta_rows = []

    print(
        f"Prepare GP: enc={args.enc} | clases={len(classes)} | "
        f"objetos cv={len(df):,} | force={args.force}",
        flush=True,
    )

    for _, row in tqdm(df.iterrows(), total=len(df), desc="cache GP"):
        stem = uid_to_stem(row["uid"])
        out = img_dir / f"{stem}.npy"
        if args.force or not out.exists():
            mean = np.asarray(get_mean(row["set"])[int(row["row"])])
            var = (
                np.asarray(get_var(row["set"])[int(row["row"])])
                if args.use_var
                else None
            )
            x = encode_gp_surface(
                mean,
                row["scale_mean"],
                enc=args.enc,
                var_scaled=var,
                scale_var=row["scale_var"] if args.use_var else None,
            )
            x = resize_to_square(x, args.image_size)
            np.save(out, x.numpy().astype(np.float16))


        path = str(out)
        labels[stem] = int(class_to_idx[row["class"]])
        meta_rows.append(
            {
                "uid": row["uid"],
                "stem": stem,
                "path": path,
                "fold": int(row["fold"]),
                "class": row["class"],
                "survey": row["survey"],
                "kind": row["kind"],
            }
        )
        if int(row["fold"]) == args.fold:
            val_paths.append(path)
        else:
            train_paths.append(path)

    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "labels.json", "w") as f:
        json.dump(labels, f)
    with open(ROOT / "classes.json", "w") as f:
        json.dump({"classes": classes, "class_to_idx": class_to_idx}, f, indent=2)
    with open(ROOT / "folds.json", "w") as f:
        json.dump({str(args.fold): {"train": train_paths, "val": val_paths}}, f)
    pd.DataFrame(meta_rows).to_csv(ROOT / "meta.csv", index=False)

    with open(ROOT / "prepare_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    channels = 2 if args.use_var else 1
    print(
        f"Listo: {len(labels):,} objetos | train {len(train_paths):,} | "
        f"val {len(val_paths):,} | clases {len(classes)} | canales {channels}",
        flush=True,
    )
    print(f"Salida: {ROOT}", flush=True)


if __name__ == "__main__":
    main()
