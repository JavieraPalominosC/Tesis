#!/bin/bash

SN_FILES=(
    "lc_SNIa-SALT2.parquet"
    "lc_SNIa-91bg.parquet"
    "lc_SNIax.parquet"
    "lc_SNII-Templates.parquet"
    "lc_SNII-NMF.parquet"
    "lc_SNII+HostXT_V19.parquet"
    "lc_SNIb-Templates.parquet"
    "lc_SNIb+HostXT_V19.parquet"
    "lc_SNIc-Templates.parquet"
    "lc_SNIc+HostXT_V19.parquet"
    "lc_SNIcBL+HostXT_V19.parquet"
    "lc_SNIIb+HostXT_V19.parquet"
    "lc_SNIIn-MOSFIT.parquet"
    "lc_SNIIn+HostXT_V19.parquet"
    "lc_SLSN-I+host.parquet"
    "lc_SLSN-I_no_host.parquet"
    "lc_PISN.parquet"
)

for f in "${SN_FILES[@]}"; do
    sbatch --partition=mi210 \
           --gres=gpu:1 \
           --job-name=gen_${f%.parquet} \
           --output=logs/gen_${f%.parquet}_%j.log \
           --cpus-per-task=8 \
           --mem=32G \
           --time=12:00:00 \
           --wrap="cd /home/jpalominos/VT_Model_for_LightCurves_Classification && source venv/bin/activate && python scripts/generate_one.py $f"
    echo "Lanzado: $f"
done
