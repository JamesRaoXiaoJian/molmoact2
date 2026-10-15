#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT=${1:-"$PROJECT/outputs/inference/adaptation_eval_20261015"}
PHASE=${2:-all}
case "$PHASE" in all|global|grasp|report) ;; *) exit 2;; esac
mkdir -p "$OUTPUT"
cd "$PROJECT"
exec 9>"$OUTPUT/.evaluation.lock"
flock -n 9 || { echo 'This evaluation already has an active writer.' >&2; exit 1; }
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export HF_HOME="$PROJECT/data/hf-cache" HF_HUB_CACHE="$PROJECT/data/hf-cache/hub"
export PYTHONPATH="${OPENPI_ROOT:-/root/openpi-arx/openpi}/packages/openpi-client/src:$PROJECT/experiments:$PROJECT/experiments/lerobot/src"
source "$PROJECT/artifacts/ffmpeg-libs/env.sh"
CUDA_WHEEL_LIBS=$("$PROJECT/.venv/bin/python" -c 'from pathlib import Path; print(":".join(str(p.resolve()) for p in Path(".venv/lib/python3.12/site-packages/nvidia").glob("*/lib")))')
export LD_LIBRARY_PATH="$PROJECT/.venv/lib/python3.12/site-packages/torch/lib:$CUDA_WHEEL_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CHECKPOINTS="$PROJECT/outputs/runs/molmoact2-x5-v2-rgb-warmstart5000-b64-s42-20261014-r1/checkpoints"
if [[ "$PHASE" == all || "$PHASE" == global ]]; then
  for STEP in 5000 2000; do
    CHECKPOINT="$CHECKPOINTS/step$STEP"
    [[ "$STEP" != 5000 ]] || CHECKPOINT+="-unsharded"
    mkdir -p "$OUTPUT/step$STEP"
    if [[ ! -f "$OUTPUT/step$STEP/rgb/summary.json" ]]; then
      "$PROJECT/.venv/bin/python" deployment/evaluate_x5_actions.py --mode rgb \
        --checkpoint "$CHECKPOINT" --norm-tag arx_x5_cleaned_v2_rgb \
        --output "$OUTPUT/step$STEP" --rgb-geometry=native --input-contract=aligned \
        > "$OUTPUT/step$STEP/global.log" 2>&1
    fi
  done
fi
if [[ "$PHASE" == all || "$PHASE" == grasp ]]; then
  for VARIANT in step5000 step2000 legacy aligned; do
    mkdir -p "$OUTPUT/$VARIANT"
    if [[ -f "$OUTPUT/$VARIANT/grasp_summary.json" ]]; then continue; fi
    ARGS=()
    case "$VARIANT" in
      step5000) ARGS+=(--checkpoint "$CHECKPOINTS/step5000-unsharded");;
      step2000) ARGS+=(--checkpoint "$CHECKPOINTS/step2000");;
    esac
    "$PROJECT/.venv/bin/python" deployment/evaluate_x5_grasp.py --variant "$VARIANT" \
      --output "$OUTPUT" "${ARGS[@]}" > "$OUTPUT/$VARIANT/grasp.log" 2>&1
  done
fi
if [[ "$PHASE" == all || "$PHASE" == report ]]; then
  "$PROJECT/.venv/bin/python" deployment/summarize_x5_adaptation.py --output "$OUTPUT"
fi
