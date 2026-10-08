#!/usr/bin/env bash
set -Eeuo pipefail

MODE=${1:?Expected rgb or mae}
case "$MODE" in rgb|mae) ;; *) exit 2 ;; esac
PROJECT=/root/RXJ/molmoact2
COHORT="$PROJECT/outputs/dlc/x5_cleaned_ablation_20261008"
RUN_NAME="molmoact2-x5-cleaned-${MODE}-fft20k-b64-s42-20261008"
RUN_DIR="$PROJECT/outputs/runs/$RUN_NAME"
CODE="${MOLMOACT2_TRAINING_CODE:-$COHORT/code}"
PYTHON="$PROJECT/.venv/bin/python"
MODEL="$PROJECT/artifacts/models/MolmoAct2"
DATA_SHARED="$COHORT/dataset"
LOCAL_ROOT=${MOLMOACT2_NODE_CACHE:-/dev/shm/molmoact2_x5_cleaned}
mkdir -p "$RUN_DIR" "$LOCAL_ROOT"
if [[ -e "$RUN_DIR/training_started.json" ]]; then
  echo "This run was already started; use a new run name or an explicit resume recipe." >&2
  exit 1
fi
exec > >(tee -a "$RUN_DIR/launcher.log") 2>&1
export PATH="$PROJECT/.venv/bin:$PATH"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export DLC_WANDB_PROJECT=molmoact2-x5-cleaned-tactile-ablation
source /root/RXJ/dlc_shared/wandb_env.sh "$RUN_DIR"
source "$PROJECT/artifacts/ffmpeg-libs/env.sh"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PROJECT/data/hf-cache" HF_MODULES_CACHE="$RUN_DIR/cache/huggingface/modules"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$CODE/experiments:$CODE/experiments/lerobot/src"
export MOLMO_DATA_DIR="$PROJECT/data/molmo"
export OLMO_SHARED_FS=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=1
export LEROBOT_VIDEO_BACKEND=pyav
export MPLCONFIGDIR="$RUN_DIR/cache/matplotlib"
export CUDA_DEVICE_MAX_CONNECTIONS=1
mkdir -p "$HF_HOME" "$HF_MODULES_CACHE" "$MPLCONFIGDIR"

"$PYTHON" "$CODE/experiments/scripts/prepare_x5_cleaned_comparison.py" \
  --mode "$MODE" --cohort "$COHORT" --run-dir "$RUN_DIR" --local-root "$LOCAL_ROOT" --stage
export LEROBOT_DATA_ROOT="$LOCAL_ROOT/data"
export LEROBOT_PACKED_IMAGE_ROOT="$LOCAL_ROOT/packed"

"$PYTHON" - "$RUN_DIR" <<'PY'
import json,sys,torch
from pathlib import Path
if torch.cuda.device_count()!=8:
    raise SystemExit(f"Expected 8 GPUs, found {torch.cuda.device_count()}")
devices=[{'index':i,'name':torch.cuda.get_device_name(i),'memory_gib':torch.cuda.get_device_properties(i).total_memory/2**30} for i in range(8)]
if any('A800' not in d['name'] or d['memory_gib']<74 for d in devices):
    raise SystemExit(f"Expected 8 A800 80GB GPUs: {devices}")
Path(sys.argv[1],'devices.json').write_text(json.dumps(devices,indent=2))
print(devices,flush=True)
PY

COMMON=(
  "$CODE/experiments/scripts/train_x5_cleaned_comparison.py" "$MODE" "$MODEL"
  --device_batch_size=2 --global_batch_size=64 --n_obs_steps=1 --max_action_dim=32
  --num_workers=4 --prefetch_factor=2 --pin_memory=true --data.timeout=900
  --packing=false --dynamic_seq_len=true --separate_vlm_dataloader=false
  --crop_mode=resize --img_aug=full --random_camera_order=none
  --action_format=continuous --state_format=discrete --norm_mode=q01_q99
  --add_action_expert=true --ft_vlm=true --ft_action_expert=true --ft_embedding=lm_head
  --lora_enable=false --action_expert_detach_vlm=false
  --llm_learning_rate=1e-5 --vit_learning_rate=5e-6 --connector_learning_rate=5e-6
  --action_expert_learning_rate=5e-5 --num_flow_timesteps=4 --mask_action_dim_padding=true
  --use_annotated_task=false --sample_annotated_task=false
  --enable_depth_reasoning=false --style_robot_action=1.0 --style_robot_depth=0.0 --style_robot_depth_action=0.0
  --frame_loading_backend=av --seed=42 --data.seed=42
  --model.llm.tokenizer.tokenizer_dir=/root/RXJ/molmoact2/data/hf-cache/hub
  --scheduler.connector_t_warmup=500 --scheduler.vit_t_warmup=500
  --scheduler.llm_t_warmup=500 --scheduler.action_expert_t_warmup=500
  --scheduler.alpha_f=0.1 --max_grad_norm=1.0 --activation_checkpointing=true
  --compile_loss=false --log_interval=10 --allow_resume=false --save_overwrite=false
)
if [[ "$MODE" == mae ]]; then
  COMMON+=(--tactile_backend=tactile_mae --tactile_num_tokens=32 --tactile_history_stride=2
    --tactile_encoder_path=/root/RXJ/pretrained_models/huggingface/AnyTouch-ViT-L-16
    --freeze_tactile_encoder=false --tactile_backbone_learning_rate=5e-6 --tactile_adapter_learning_rate=5e-5)
else
  COMMON+=(--tactile_backend=as_image)
fi

if [[ ! -f "$RUN_DIR/smoke_ok.json" ]]; then
  echo "Starting the matched 8-GPU, batch-64 training smoke (2 updates)."
  torchrun --standalone --nnodes=1 --nproc-per-node=8 "${COMMON[@]}" \
    --run_name="${RUN_NAME}-smoke" --wandb=null \
    --max_duration=2 --save_interval=2000 --save_folder="$RUN_DIR/smoke" \
    --save_final_optim=false --save_final_unsharded_checkpoint=false \
    2>&1 | tee "$RUN_DIR/smoke.log"
  "$PYTHON" - "$RUN_DIR/smoke_ok.json" <<'PY'
import json,sys,datetime
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'status':'passed','updates':2,'gpu_count':8,'global_batch':64,'time':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
PY
fi

"$PYTHON" - "$RUN_DIR/training_started.json" "$MODE" <<'PY'
import json,sys,datetime
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'mode':sys.argv[2],'steps':20000,'batch_size':64,'effective_epochs':20000*64/382422,'seed':42,'time':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
PY
echo "Starting formal training: $RUN_NAME, 20000 updates, global batch 64, seed 42."
torchrun --standalone --nnodes=1 --nproc-per-node=8 "${COMMON[@]}" \
  --run_name="$RUN_NAME" --wandb.name="$RUN_NAME" --wandb.entity="$WANDB_ENTITY" --wandb.project="$WANDB_PROJECT" \
  --max_duration=20000 --save_interval=2000 --save_num_checkpoints_to_keep=10 \
  --save_final_unsharded_checkpoint=true --save_final_optim=true \
  --save_folder="$RUN_DIR/checkpoints" 2>&1 | tee "$RUN_DIR/train.log"
printf '{"status":"complete"}\n' > "$RUN_DIR/training_complete.json"
