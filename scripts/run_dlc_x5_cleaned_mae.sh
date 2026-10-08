#!/usr/bin/env bash
set -Eeuo pipefail
# Shared credentials are loaded by common entrypoint:
# source /root/RXJ/dlc_shared/wandb_env.sh "$RUN_DIR"
exec /root/RXJ/molmoact2/scripts/run_dlc_x5_cleaned_common.sh mae
