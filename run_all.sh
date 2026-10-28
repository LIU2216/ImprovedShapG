#!/bin/zsh
# Full benchmark on seed 42: library-default threads, serial, each method timed standalone from an empty
# coalition cache. Per pair: retraining benchmark (10 methods), then the no-retraining run (Original,
# Improved ShapG fixed / adaptive, Exact; SHAP from the retraining run).
# Order: small to large, crimedata last. Logs in logs/. Re-running skips finished pairs.
# Activate your Python environment first (see requirements.txt), or set CONDA_ENV to a conda env name.
if [[ -n "$CONDA_ENV" ]]; then source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV"; fi
export RANK_DESC=0   # also set inside the entry scripts (paper setting)
cd "$(dirname "$0")"
mkdir -p logs
for ds in admission abalone Churn adult ucs_scm diabetes life H1N1 MIC crimedata; do
  for m in lgb mlp; do
    echo "$(date +%H:%M:%S) start $ds $m"
    (cd retrain && python run_benchmark.py --dataset $ds --model $m > ../logs/retrain_${ds}_${m}.log 2>&1)
    if [[ -f noretrain/results/${ds}_${m}.json ]]; then echo "  noretrain done, skipped"
    else (cd noretrain && python run_noretrain.py --datasets $ds --models $m > ../logs/noretrain_${ds}_${m}.log 2>&1); fi
    echo "$(date +%H:%M:%S) done  $ds $m"
  done
done
echo ALL DONE
