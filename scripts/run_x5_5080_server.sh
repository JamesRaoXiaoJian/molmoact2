#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:?Expected rgb or mae}
case "$MODE" in rgb) KEYS=x5;; mae) KEYS=x5_tactile;; *) exit 2;; esac
PORT=${2:-8000}
cd "$PROJECT"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export HF_HOME="$PROJECT/data/hf-cache" HF_HUB_CACHE="$PROJECT/data/hf-cache/hub"
export PYTHONPATH="$PROJECT/experiments:$PROJECT/experiments/lerobot/src"
export TOKENIZERS_PARALLELISM=false
exec 9>/tmp/rxj_inference_deploy_gpu.lock
echo 'Waiting for shared GPU deployment lock...'
flock 9
exec "$PROJECT/.venv/bin/python" experiments/scripts/serve_policy.py \
  --checkpoint "$PROJECT/outputs/runs/molmoact2-x5-cleaned-$MODE-fft20k-b64-s42-20261008/checkpoints/step20000-unsharded" \
  --norm_tag "arx_x5_cleaned_$MODE" --image_keys "$KEYS" --device cuda:0 \
  --parameter_dtype bfloat16 --tokenizer_dir "$PROJECT/data/hf-cache/hub" \
  --disable_inference_cuda_graph --num_steps 10 --host 127.0.0.1 --port "$PORT"
