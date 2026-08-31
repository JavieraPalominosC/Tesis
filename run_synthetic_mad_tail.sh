#!/bin/bash
# Cola MAD sin congelar: 40 SNIa + F1 del maximo NLL.
set -euo pipefail
cd /home/jpalominos/VT_Model_for_LightCurves_Classification
mkdir -p logs
ROCM_LIB=/home/modules/spack/opt/spack/linux-rocky9-zen4/aocc-5.0.0/hip-6.3.1-zrxiwdkzpkvfportqftvhr5ex23xipct/lib
WRAP="cd /home/jpalominos/VT_Model_for_LightCurves_Classification && source venv/bin/activate"

GEN=$(sbatch --parsable --partition=main --job-name=mad_tail_gen \
  --output=logs/mad_tail_gen_%j.log --cpus-per-task=4 --mem=16G --time=01:00:00 \
  --wrap="${WRAP} && python scripts/generate_synthetic_mad_tail.py 40")
# df=10 y 50 unfrozen ya existen; se generan 20,30,40,60,...,100.
echo "generacion: job ${GEN} (main)"

EVAL=$(sbatch --parsable --partition=mi210 --gres=gpu:1 --job-name=mad_tail_eval \
  --dependency=afterok:${GEN} \
  --output=logs/mad_tail_eval_%j.log --cpus-per-task=4 --mem=32G --time=02:00:00 \
  --wrap="${WRAP} && export LD_LIBRARY_PATH=${ROCM_LIB}:\$LD_LIBRARY_PATH && python scripts/eval_synthetic_mad_tail.py")
echo "eval:       job ${EVAL} (mi210, afterok ${GEN})"
echo "logs: logs/mad_tail_gen_${GEN}.log  logs/mad_tail_eval_${EVAL}.log"
