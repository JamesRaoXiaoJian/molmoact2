#!/usr/bin/env bash
set -Eeuo pipefail
# Shared credentials are loaded by common entrypoint:
# source /root/RXJ/dlc_shared/wandb_env.sh "$RUN_DIR"
export MOLMOACT2_TRAINING_CODE=/root/RXJ/molmoact2/outputs/dlc/x5_cleaned_ablation_20261008/code-mae-workerfix
exec /root/RXJ/molmoact2/scripts/run_dlc_x5_cleaned_common.sh mae
