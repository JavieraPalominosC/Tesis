#!/bin/bash
# Segundo peak sobre waterfall_global (imagenes + VQ-VAE + MaskGIT de esa rep).
set -euo pipefail
cd /home/jpalominos/VT_Model_for_LightCurves_Classification
mkdir -p logs
ROCM_LIB=/home/modules/spack/opt/spack/linux-rocky9-zen4/aocc-5.0.0/hip-6.3.1-zrxiwdkzpkvfportqftvhr5ex23xipct/lib
WRAP="cd /home/jpalominos/VT_Model_for_LightCurves_Classification && source venv/bin/activate"

GEN=$(sbatch --parsable --partition=main --job-name=spk_wfg_gen \
  --output=logs/spk_wfg_gen_%j.log --cpus-per-task=4 --mem=16G --time=00:40:00 \
  --wrap="${WRAP} && python scripts/generate_synthetic_second_peak.py 8 waterfall_global")
echo "generacion: job ${GEN} (main)"

EVAL=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=spk_wfg_eval \
  --dependency=afterok:${GEN} \
  --output=logs/spk_wfg_eval_%j.log --cpus-per-task=4 --mem=32G --time=01:00:00 \
  --wrap="${WRAP} && export LD_LIBRARY_PATH=${ROCM_LIB}:\$LD_LIBRARY_PATH && python scripts/eval_synthetic_second_peak.py waterfall_global && python scripts/cf_synthetic_second_peak.py waterfall_global")
echo "eval+cf:    job ${EVAL} (mi210, afterok ${GEN})"
