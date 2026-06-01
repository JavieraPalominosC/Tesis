"""
Genera data/folds/labels.json con el mapeo SNID → clase (int).
"""
import sys
sys.path.insert(0, '.')
import json
import pandas as pd
from pathlib import Path

SN_FILES = [
    ("lc_SNIa-SALT2.parquet",        "SNIa-SALT2"),
    ("lc_SNIa-91bg.parquet",         "SNIa-91bg"),
    ("lc_SNIax.parquet",             "SNIax"),
    ("lc_SNII-Templates.parquet",    "SNII-Templates"),
    ("lc_SNII-NMF.parquet",          "SNII-NMF"),
    ("lc_SNII+HostXT_V19.parquet",   "SNII+HostXT_V19"),
    ("lc_SNIb-Templates.parquet",    "SNIb-Templates"),
    ("lc_SNIb+HostXT_V19.parquet",   "SNIb+HostXT_V19"),
    ("lc_SNIc-Templates.parquet",    "SNIc-Templates"),
    ("lc_SNIc+HostXT_V19.parquet",   "SNIc+HostXT_V19"),
    ("lc_SNIcBL+HostXT_V19.parquet", "SNIcBL+HostXT_V19"),
    ("lc_SNIIb+HostXT_V19.parquet",  "SNIIb+HostXT_V19"),
    ("lc_SNIIn-MOSFIT.parquet",      "SNIIn-MOSFIT"),
    ("lc_SNIIn+HostXT_V19.parquet",  "SNIIn+HostXT_V19"),
    ("lc_SLSN-I+host.parquet",       "SLSN-I+host"),
    ("lc_SLSN-I_no_host.parquet",    "SLSN-I_no_host"),
    ("lc_PISN.parquet",              "PISN"),
]

RAW_DIR = Path("data/lightcurves/elasticc_1/raw")
OUT_DIR = Path("data/folds")
OUT_DIR.mkdir(parents=True, exist_ok=True)

snid_to_class = {}
class_names   = {}

for class_id, (fname, typename) in enumerate(SN_FILES):
    fpath = RAW_DIR / fname
    if not fpath.exists():
        print(f"  [SKIP] {fname} no encontrado")
        continue
    print(f"[{class_id:2d}] {typename}")
    class_names[class_id] = typename
    df = pd.read_parquet(fpath, columns=["SNID"])
    for snid in df["SNID"].unique():
        snid_to_class[str(snid)] = class_id
    print(f"      {len(df['SNID'].unique()):,} SNIDs")

print(f"\nTotal: {len(snid_to_class):,} SNIDs, {len(class_names)} clases")

with open(OUT_DIR / "labels.json", "w") as f:
    json.dump(snid_to_class, f)
with open(OUT_DIR / "class_names.json", "w") as f:
    json.dump({str(k): v for k, v in class_names.items()}, f, indent=2)

print("Guardado: data/folds/labels.json")
print("Guardado: data/folds/class_names.json")
