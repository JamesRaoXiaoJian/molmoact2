#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ROUTE=${1:-all}
OUTPUT=${2:-"$PROJECT/outputs/inference/latency_official_$(date +%Y%m%d)"}
case "$ROUTE" in all) ROUTES=(rgb mae official);; rgb|mae|official) ROUTES=("$ROUTE");; *) exit 2;; esac
cd "$PROJECT"
mkdir -p "$OUTPUT"
exec 9>/tmp/rxj_inference_deploy_gpu.lock
if ! flock -n 9; then
  echo 'The shared inference GPU lock is busy. This benchmark will not stop an existing service.' >&2
  exit 3
fi
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$PROJECT/data/hf-cache" HF_HUB_CACHE="$PROJECT/data/hf-cache/hub"
export PYTHONPATH="$PROJECT/experiments:$PROJECT/experiments/lerobot/src"
if [[ -f "$PROJECT/artifacts/ffmpeg-libs/env.sh" ]]; then source "$PROJECT/artifacts/ffmpeg-libs/env.sh"; fi
CUDA_WHEEL_LIBS=$(find "$PROJECT/.venv/lib/python3.12/site-packages/nvidia" -maxdepth 2 -type d -name lib -print | paste -sd: -)
export LD_LIBRARY_PATH="$PROJECT/.venv/lib/python3.12/site-packages/torch/lib:$CUDA_WHEEL_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EXTRA=()
if [[ -f "$PROJECT/data/offline_samples/manifest.json" ]]; then EXTRA+=(--samples-dir "$PROJECT/data/offline_samples"); fi
if [[ -n "${X5_OFFICIAL_MODEL:-}" ]]; then EXTRA+=(--official-model "$X5_OFFICIAL_MODEL"); fi
for ITEM in "${ROUTES[@]}"; do
  "$PROJECT/.venv/bin/python" experiments/scripts/benchmark_x5_inference_paths.py \
    --route "$ITEM" --output "$OUTPUT" "${EXTRA[@]}" > "$OUTPUT/$ITEM.log" 2>&1
done
