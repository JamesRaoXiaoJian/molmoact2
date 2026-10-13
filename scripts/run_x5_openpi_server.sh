#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:?Select rgb or mae}
case "$MODE" in rgb|mae) ;; *) exit 2;; esac
PORT=${2:-8001}
if (( $# >= 2 )); then shift 2; else shift; fi
cd "$PROJECT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export HF_HOME="$PROJECT/data/hf-cache" HF_HUB_CACHE="$PROJECT/data/hf-cache/hub"
export PYTHONPATH="$PROJECT/experiments:$PROJECT/experiments/lerobot/src"
export TOKENIZERS_PARALLELISM=false
exec 9>/tmp/rxj_inference_deploy_gpu.lock
echo 'Waiting for shared GPU lock; loopback candidate service, no actuation path.'
flock 9
exec "$PROJECT/.venv/bin/python" experiments/scripts/serve_x5_openpi.py --mode "$MODE" --port "$PORT" "$@"
