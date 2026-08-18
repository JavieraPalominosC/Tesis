#!/bin/bash
# Eval de anomaly scores para waterfall_global (~5h en mi210).
set -euo pipefail
cd /home/jpalominos/VT_Model_for_LightCurves_Classification
mkdir -p logs
ROCM_LIB=/home/modules/spack/opt/spack/linux-rocky9-zen4/aocc-5.0.0/hip-6.3.1-zrxiwdkzpkvfportqftvhr5ex23xipct/lib
sbatch --partition=mi210 --gres=gpu:1 --job-name=eval_wfg \
  --output=logs/eval_wfg_%j.log --cpus-per-task=8 --mem=64G --time=08:00:00 \
  --wrap="cd /home/jpalominos/VT_Model_for_LightCurves_Classification && source venv/bin/activate && export LD_LIBRARY_PATH=${ROCM_LIB}:\$LD_LIBRARY_PATH && python scripts/eval_anomaly_scores_full.py waterfall_global && python scripts/compare_representations.py"
