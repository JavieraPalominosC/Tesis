#!/bin/bash
# Genera imagenes (CPU, particion main) y despues eval + CF (GPU mi210).
# Correr desde el root del repo en Leftraru:
#   bash run_synthetic_second_peak.sh
set -euo pipefail
cd /home/jpalominos/VT_Model_for_LightCurves_Classification
mkdir -p logs

ROCM_LIB=/home/modules/spack/opt/spack/linux-rocky9-zen4/aocc-5.0.0/hip-6.3.1-zrxiwdkzpkvfportqftvhr5ex23xipct/lib
WRAP_ENV="cd /home/jpalominos/VT_Model_for_LightCurves_Classification && source venv/bin/activate"

GEN=$(sbatch --parsable --partition=main --job-name=syn2pk_gen \
  --output=logs/syn2pk_gen_%j.log --cpus-per-task=4 --mem=16G --time=00:40:00 \
  --wrap="${WRAP_ENV} && python scripts/generate_synthetic_second_peak.py 8")
echo "generacion: job ${GEN} (main)"

EVAL=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=syn2pk_eval \
  --dependency=afterok:${GEN} \
  --output=logs/syn2pk_eval_%j.log --cpus-per-task=4 --mem=32G --time=01:00:00 \
  --wrap="${WRAP_ENV} && export LD_LIBRARY_PATH=${ROCM_LIB}:\$LD_LIBRARY_PATH && python scripts/eval_synthetic_second_peak.py && python scripts/cf_synthetic_second_peak.py")
echo "eval+cf:    job ${EVAL} (mi210, afterok ${GEN})"
echo "logs: logs/syn2pk_gen_${GEN}.log  logs/syn2pk_eval_${EVAL}.log"
