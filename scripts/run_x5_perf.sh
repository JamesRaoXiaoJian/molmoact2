#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:-both}
SCHEME=${2:-baseline}
case "$MODE" in rgb|mae|both) ;; *) exit 2;; esac
cd "$PROJECT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export HF_HOME="$PROJECT/data/hf-cache" HF_HUB_CACHE="$PROJECT/data/hf-cache/hub"
export PYTHONPATH="$PROJECT/experiments:$PROJECT/experiments/lerobot/src:$PROJECT/deployment"
export TOKENIZERS_PARALLELISM=false
exec 9>/tmp/rxj_inference_deploy_gpu.lock
flock 9
MODES=("$MODE")
if [[ "$MODE" == both ]]; then MODES=(rgb mae); fi
for mode in "${MODES[@]}"; do
  .venv/bin/python -u deployment/benchmark_x5_native.py --mode "$mode" --scheme "$SCHEME" --profile \
    2>&1 | tee "deployment/perf_20261012/${mode}_${SCHEME}.log"
done
