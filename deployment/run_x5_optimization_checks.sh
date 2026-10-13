#!/usr/bin/env bash
# Recorded observations and hardware mocks only. Caller owns the GPU lock.
set -Eeuo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT"
RESULT_DIR=${1:?Pass an isolated result directory}
REPLAY_SAMPLES=${2:?Pass the directory containing the 24-observation replay manifest}
mkdir -p "$RESULT_DIR"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=2
export HF_HOME="$PROJECT/data/hf-cache" HF_HUB_CACHE="$PROJECT/data/hf-cache/hub"
OPENPI_ROOT=${OPENPI_ROOT:-$HOME/lwm_ws/openpi-x5-arx5-beta/openpi}
export PYTHONPATH="$PROJECT/experiments:$PROJECT/experiments/lerobot/src:$OPENPI_ROOT/packages/openpi-client/src:$PROJECT/deployment"
.venv/bin/python deployment/capture_x5_device_state.py > "$RESULT_DIR/device_start.json"
for mode in rgb mae; do
  for phase in eager graph; do
    .venv/bin/python -u deployment/benchmark_x5_incremental.py --mode "$mode" --phase "$phase" \
      --iterations 60 --compile-blocks --output "$RESULT_DIR" > "$RESULT_DIR/${mode}_${phase}.log" 2>&1
  done
  .venv/bin/python -u deployment/evaluate_x5_compile.py --mode "$mode" \
    --samples "$REPLAY_SAMPLES" --output "$RESULT_DIR" > "$RESULT_DIR/${mode}_quality.log" 2>&1
  .venv/bin/python -u deployment/benchmark_x5_native.py --mode "$mode" --scheme graph --profile \
    --output "$RESULT_DIR/service" > "$RESULT_DIR/${mode}_service.log" 2>&1
  .venv/bin/python -u deployment/openpi_main_smoke.py --mode "$mode" \
    --openpi-root "$OPENPI_ROOT" > "$RESULT_DIR/${mode}_control_mock.log" 2>&1
done
.venv/bin/python -u deployment/graph_safety_final.py > "$RESULT_DIR/cuda_safety.log" 2>&1
.venv/bin/python deployment/capture_x5_device_state.py > "$RESULT_DIR/device_end.json"
