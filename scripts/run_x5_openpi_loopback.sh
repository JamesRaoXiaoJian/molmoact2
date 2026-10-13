#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:-both}
case "$MODE" in rgb|mae|both) ;; *) echo 'Expected rgb, mae, or both' >&2; exit 2;; esac
if (( $# > 0 )); then shift; fi
cd "$PROJECT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export HF_HOME="$PROJECT/data/hf-cache" HF_HUB_CACHE="$PROJECT/data/hf-cache/hub"
export PYTHONPATH="$PROJECT/experiments:$PROJECT/experiments/lerobot/src"
export TOKENIZERS_PARALLELISM=false
exec 9>/tmp/rxj_inference_deploy_gpu.lock
echo 'Waiting for shared GPU lock; recorded samples and fake sink only.'
flock 9
MODES=("$MODE")
if [[ "$MODE" == both ]]; then MODES=(rgb mae); fi
mkdir -p outputs/inference/openpi_bridge
for mode in "${MODES[@]}"; do
  "$PROJECT/.venv/bin/python" -u deployment/openpi_loopback_smoke.py --mode "$mode" "$@" \
    2>&1 | tee "outputs/inference/openpi_bridge/$mode.log"
done
