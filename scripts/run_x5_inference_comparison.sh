#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT=${1:-"$PROJECT/outputs/inference/x5_cleaned_20261010"}
mkdir -p "$OUTPUT"
cd "$PROJECT"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PROJECT/data/hf-cache" HF_HUB_CACHE="$PROJECT/data/hf-cache/hub"
export TRANSFORMERS_CACHE="$PROJECT/data/hf-cache/hub"
export PYTHONPATH="$PROJECT/experiments:$PROJECT/experiments/lerobot/src"
source "$PROJECT/artifacts/ffmpeg-libs/env.sh"
CUDA_WHEEL_LIBS=$(find "$PROJECT/.venv/lib/python3.12/site-packages/nvidia" -maxdepth 2 -type d -name lib -print | paste -sd: -)
export LD_LIBRARY_PATH="$PROJECT/.venv/lib/python3.12/site-packages/torch/lib:$CUDA_WHEEL_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for MODE in rgb mae; do
  "$PROJECT/.venv/bin/python" experiments/scripts/run_x5_inference_comparison.py \
    --mode "$MODE" --output "$OUTPUT" > "$OUTPUT/$MODE.log" 2>&1
done
