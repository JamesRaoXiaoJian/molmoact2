#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
usage() {
  cat <<'EOF'
MODEL_SELECT=0 bash scripts/run_x5_molmo.sh server|client
0 = MolmoAct2 RGB cleaned step20000; 1 = MolmoAct2 Tactile-MAE cleaned step20000.
Hardware defaults: configs/x5_molmo.env, shared OpenPI configs/site.env.
Policies and prompts: configs/x5_molmo_policies.json, configs/x5_molmo_tasks.json.
client initializes real hardware and enables policy motion like the original launcher.
DRY_RUN=1 disables policy motion/reset; SDK hold/gain/damping still occur.
Legacy server|client rgb|mae [arguments] remains supported.
MOLMO_EXECUTION_MODE=rtc selects asynchronous RTC on the client (default: sync).
RTC_QUEUE_THRESHOLD=18 RTC_EXECUTION_HORIZON=20 RTC_MAX_GUIDANCE_WEIGHT=10.
Joint step limit: control.max_step_delta in the registry (deployed default 0.1 rad).
Legacy/direct clients also default to 0.1 rad; --max-step-delta overrides it.
Client Python logs and per-action traces are saved in logs/molmo-*-client-*.log.
MOLMO_CLIENT_LOG_DIR overrides the log directory; motion parameters are unchanged.
EOF
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
ROLE=${1:-}
case "$ROLE" in server|client) ;; *) usage; exit 2;; esac
shift
cd "$PROJECT"
LEGACY=0
if [[ ${1:-} == rgb || ${1:-} == mae ]]; then
  LEGACY=1
  case "$1" in rgb) MODEL_SELECT=0;; mae) MODEL_SELECT=1;; esac
  shift
fi
# Default policy number: 0=RGB20k, 1=MAE20k; add later IDs in the registry.
MODEL_SELECT=${MODEL_SELECT:-0}
[[ "$MODEL_SELECT" =~ ^[0-9]+$ ]] || { echo "MODEL_SELECT must be a registered nonnegative integer" >&2; exit 2; }
source "$PROJECT/configs/x5_molmo.env"
export PYTHONDONTWRITEBYTECODE=1
if [[ "$LEGACY" == 1 && "$ROLE" == client ]]; then
  # Preserve explicit long-command prompt/authorization behavior.
  MODE=rgb
  [[ "$MODEL_SELECT" == 1 ]] && MODE=mae
  MODEL_ARGS=(--control-hz=30 --action-start-index=0 --execute-steps=30)
else
  PROFILE="$("$PROJECT/.venv/bin/python" "$PROJECT/experiments/scripts/x5_molmo_profile.py" --role "$ROLE" --model-select "$MODEL_SELECT")"
  # Generated assignments/argv use shlex.quote for every data value.
  eval "$PROFILE"
  echo "MODEL_SELECT=$MODEL_SELECT: $MODEL_DESCRIPTION"
fi
if [[ "$ROLE" == server ]]; then
  PORT=${1:-$PORT}
  exec bash "$PROJECT/scripts/run_x5_openpi_server.sh" "$MODE" "$PORT" "${SERVER_ARGS[@]}"
fi
CLIENT_PYTHON=${MOLMO_CLIENT_PYTHON:-$OPENPI_ROOT/.venv/bin/python}
[[ -x "$CLIENT_PYTHON" ]] || { echo "Missing existing OpenPI Python: $CLIENT_PYTHON" >&2; exit 1; }
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$X5_PYTHON_DIR:$OPENPI_ROOT:$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:$OPENPI_ROOT/examples/x5_client${PYTHONPATH:+:$PYTHONPATH}"
EXTRA_ARGS=()
case "${MOLMO_EXECUTION_MODE:-sync}" in
  sync) ;;
  rtc) EXTRA_ARGS+=(--rtc --rtc-queue-threshold="${RTC_QUEUE_THRESHOLD:-18}"
                   --rtc-execution-horizon="${RTC_EXECUTION_HORIZON:-20}"
                   --rtc-max-guidance-weight="${RTC_MAX_GUIDANCE_WEIGHT:-10}");;
  *) echo "MOLMO_EXECUTION_MODE must be sync or rtc" >&2; exit 2;;
esac
if [[ "$MODE" == mae ]]; then
  EXTRA_ARGS+=(--tactile-serials "$LEFT_GRIPPER_LEFT_TACTILE" "$LEFT_GRIPPER_RIGHT_TACTILE" "$RIGHT_GRIPPER_LEFT_TACTILE" "$RIGHT_GRIPPER_RIGHT_TACTILE")
fi
if [[ "$LEGACY" == 0 ]]; then
  echo "Data task $MODEL_TASK_INDEX: ${MODEL_ARGS[0]#--prompt=}"
  if [[ ${CTRL_C_EXITS:-1} == 1 ]]; then EXTRA_ARGS+=(--exit-on-ctrl-c); fi
  if [[ ${ENABLE_MOTION:-1} == 1 && ${DRY_RUN:-0} != 1 ]]; then
    EXTRA_ARGS+=(--enable-motion --confirm-motion=MOVE_X5 --auto-reset-on-startup)
  fi
fi
exec "$CLIENT_PYTHON" "$PROJECT/experiments/scripts/run_x5_openpi_client.py" \
  --mode "$MODE" --openpi-root "$OPENPI_ROOT" \
  --policy-transport=websocket --host="$HOST" --port="$PORT" --max-chunks=0 \
  --gravity-compensation-setup --episode-control="$EPISODE_CONTROL" \
  --pedal-device-path="$PEDAL_DEVICE_PATH" \
  --left-can="$LEFT_CAN" --right-can="$RIGHT_CAN" \
  --left-camera="$LEFT_CAMERA" --right-camera="$RIGHT_CAMERA" \
  --head-camera-serial="$ORBBEC_HEAD_SERIAL" \
  --left-gripper-open-readout="$LEFT_GRIPPER_OPEN_READOUT" \
  --right-gripper-open-readout="$RIGHT_GRIPPER_OPEN_READOUT" \
  --camera-width="$CAMERA_WIDTH" --camera-height="$CAMERA_HEIGHT" --camera-fps="$CAMERA_FPS" \
  --left-gripper-close-offset="${LEFT_GRIPPER_CLOSE_OFFSET:-0.35}" \
  --right-gripper-close-offset="${RIGHT_GRIPPER_CLOSE_OFFSET:-0.2}" \
  --left-gripper-close-threshold="${LEFT_GRIPPER_CLOSE_THRESHOLD:--2.75}" \
  --right-gripper-close-threshold="${RIGHT_GRIPPER_CLOSE_THRESHOLD:--0.75}" \
  --no-auto-reset-on-startup --no-auto-reset-each-episode \
  "${MODEL_ARGS[@]}" "${EXTRA_ARGS[@]}" "$@"
