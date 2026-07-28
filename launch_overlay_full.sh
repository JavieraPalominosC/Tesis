#!/bin/bash
declare -a JOBS=(
  "lc_SNIa-SALT2.parquet 0"
  "lc_SNIa-91bg.parquet 1"
  "lc_SNIax.parquet 2"
  "lc_SNII-Templates.parquet 3"
  "lc_SNII-NMF.parquet 4"
  "lc_SNII+HostXT_V19.parquet 5"
  "lc_SNIb-Templates.parquet 6"
  "lc_SNIb+HostXT_V19.parquet 7"
  "lc_SNIc-Templates.parquet 8"
  "lc_SNIc+HostXT_V19.parquet 9"
  "lc_SNIcBL+HostXT_V19.parquet 10"
  "lc_SNIIb+HostXT_V19.parquet 11"
  "lc_SNIIn-MOSFIT.parquet 12"
  "lc_SNIIn+HostXT_V19.parquet 13"
  "lc_SLSN-I+host.parquet 14"
  "lc_SLSN-I_no_host.parquet 15"
  "lc_PISN.parquet 16"
)
for job in "${JOBS[@]}"; do
  set -- $job
  fname=$1; label=$2
  part="${fname#lc_}"; part="${part%.parquet}"
  sbatch --partition=mi210 --gres=gpu:1 --job-name=ov_${part} \
    --output=logs/ov_full_${part}_%j.log --cpus-per-task=4 --mem=16G \
    --time=12:00:00 \
    --wrap="cd /home/jpalominos/VT_Model_for_LightCurves_Classification && source venv/bin/activate && python scripts/generate_overlay_full.py $fname $label"
  echo "Lanzado: $part"
done
