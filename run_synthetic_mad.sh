#!/bin/bash
# Genera imagenes MAD (CPU, main) y eval hit-rate (GPU, mi210).
set -euo pipefail
cd /home/jpalominos/VT_Model_for_LightCurves_Classification
mkdir -p logs
ROCM_LIB=/home/modules/spack/opt/spack/linux-rocky9-zen4/aocc-5.0.0/hip-6.3.1-zrxiwdkzpkvfportqftvhr5ex23xipct/lib
WRAP="cd /home/jpalominos/VT_Model_for_LightCurves_Classification && source venv/bin/activate"

GEN=$(sbatch --parsable --partition=main --job-name=mad_gen \
  --output=logs/mad_gen_%j.log --cpus-per-task=4 --mem=16G --time=01:30:00 \
  --wrap="${WRAP} && python scripts/generate_synthetic_mad.py 40")
echo "generacion: job ${GEN} (main)"

EVAL=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=mad_eval \
  --dependency=afterok:${GEN} \
  --output=logs/mad_eval_%j.log --cpus-per-task=4 --mem=32G --time=02:00:00 \
  --wrap="${WRAP} && export LD_LIBRARY_PATH=${ROCM_LIB}:\$LD_LIBRARY_PATH && python scripts/eval_synthetic_mad.py 0.3")
echo "eval:       job ${EVAL} (mi210, afterok ${GEN})"
