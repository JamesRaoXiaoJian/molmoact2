#!/usr/bin/env bash
set -Eeuo pipefail
INIT=${1:?Expected warmstart or base}
MODE=${2:-rgb}
case "$INIT" in warmstart) STEPS=5000; LLM_LR=3e-6; VIT_LR=1.5e-6; ACTION_LR=1.5e-5;; base) STEPS=20000; LLM_LR=1e-5; VIT_LR=5e-6; ACTION_LR=5e-5;; *) exit 2;; esac
case "$MODE" in rgb|mae) ;; *) exit 2;; esac
PROJECT=/root/RXJ/molmoact2
COHORT="$PROJECT/outputs/dlc/x5_contract_v2_20261014"
CODE="$COHORT/code"
PYTHON="$PROJECT/.venv/bin/python"
RUN_NAME="molmoact2-x5-v2-${MODE}-${INIT}${STEPS}-b64-s42-20261014"
RUN_DIR="$PROJECT/outputs/runs/$RUN_NAME"
MODEL="$PROJECT/artifacts/models/MolmoAct2"
if [[ "$INIT" == warmstart ]]; then
  MODEL="$PROJECT/outputs/runs/molmoact2-x5-cleaned-${MODE}-fft20k-b64-s42-20261008/checkpoints/step20000-unsharded"
fi
[[ -f "$COHORT/code_revision.json" && -f "$CODE/experiments/scripts/train_x5_contract_v2.py" ]]
mkdir -p "$RUN_DIR"
[[ ! -e "$RUN_DIR/training_started.json" ]] || { echo "Run already started; use explicit resume or a new run." >&2; exit 1; }
exec > >(tee -a "$RUN_DIR/launcher.log") 2>&1
export PATH="$PROJECT/.venv/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export DLC_WANDB_PROJECT=molmoact2-x5-input-contract-v2
source /root/RXJ/dlc_shared/wandb_env.sh "$RUN_DIR"
source "$PROJECT/artifacts/ffmpeg-libs/env.sh"
CUDA_WHEEL_LIBS=$("$PYTHON" -c 'from pathlib import Path; print(":".join(str(p) for p in Path("/root/RXJ/molmoact2/.venv/lib/python3.12/site-packages/nvidia").glob("*/lib")))')
export LD_LIBRARY_PATH="$PROJECT/.venv/lib/python3.12/site-packages/torch/lib:$CUDA_WHEEL_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PROJECT/data/hf-cache" HF_HUB_CACHE="$PROJECT/data/hf-cache/hub"
export HF_MODULES_CACHE="$RUN_DIR/cache/huggingface/modules" MPLCONFIGDIR="$RUN_DIR/cache/matplotlib"
export PYTHONPATH="$CODE/experiments:$CODE/experiments/lerobot/src"
export MOLMO_DATA_DIR="$PROJECT/data/molmo" OLMO_SHARED_FS=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=1 LEROBOT_VIDEO_BACKEND=pyav CUDA_DEVICE_MAX_CONNECTIONS=1
export X5_INITIALIZATION="$INIT"
mkdir -p "$HF_MODULES_CACHE" "$MPLCONFIGDIR"
LOCAL_ROOT=/dev/shm/molmoact2_x5_contract_v2
"$PYTHON" "$CODE/experiments/scripts/prepare_x5_cleaned_comparison.py" --mode "$MODE" \
  --cohort "$COHORT" --run-dir "$RUN_DIR" --local-root "$LOCAL_ROOT" --stage
export LEROBOT_DATA_ROOT="$LOCAL_ROOT/data" LEROBOT_PACKED_IMAGE_ROOT="$LOCAL_ROOT/packed"
"$PYTHON" - "$RUN_DIR" "$COHORT" "$INIT" "$MODEL" "$STEPS" <<'PY'
import json,sys,torch
from pathlib import Path
run,cohort,init,checkpoint,steps=sys.argv[1:]
if torch.cuda.device_count()!=8:
    raise SystemExit('Expected eight A800 GPUs')
devices=[{'index':i,'name':torch.cuda.get_device_name(i),'memory_gib':torch.cuda.get_device_properties(i).total_memory/2**30} for i in range(8)]
if any('A800' not in d['name'] or d['memory_gib']<74 for d in devices):
    raise SystemExit(f'Unexpected GPUs: {devices}')
Path(run,'devices.json').write_text(json.dumps(devices,indent=2))
receipt={'initialization':init,'checkpoint':checkpoint,'steps':int(steps),'global_batch':64,'seed':42,
         'cohort':str(cohort),'code':json.loads(Path(cohort,'code_revision.json').read_text()),
         'validation_seen_by_initialization':init=='warmstart'}
Path(run,'recipe.json').write_text(json.dumps(receipt,indent=2))
print(receipt,flush=True)
PY
COMMON=("$CODE/experiments/scripts/train_x5_contract_v2.py" "$MODE" "$MODEL" "$COHORT/splits.json"
  --device_batch_size=2 --global_batch_size=64 --n_obs_steps=1 --max_action_dim=32
  --num_workers=4 --prefetch_factor=2 --pin_memory=true --data.timeout=900
  --packing=false --dynamic_seq_len=true --separate_vlm_dataloader=false
  --crop_mode=resize --img_aug=full --random_camera_order=none
  --action_format=continuous --state_format=discrete --norm_mode=min_max
  --add_action_expert=true --ft_vlm=true --ft_action_expert=true --ft_embedding=lm_head
  --lora_enable=false --action_expert_detach_vlm=false
  --llm_learning_rate="$LLM_LR" --vit_learning_rate="$VIT_LR" --connector_learning_rate="$VIT_LR"
  --action_expert_learning_rate="$ACTION_LR" --num_flow_timesteps=4 --mask_action_dim_padding=true
  --use_annotated_task=false --sample_annotated_task=false
  --enable_depth_reasoning=false --style_robot_action=1.0 --style_robot_depth=0.0 --style_robot_depth_action=0.0
  --frame_loading_backend=av --seed=42 --data.seed=42
  --model.llm.tokenizer.tokenizer_dir="$PROJECT/data/hf-cache/hub"
  --scheduler.connector_t_warmup=200 --scheduler.vit_t_warmup=200
  --scheduler.llm_t_warmup=200 --scheduler.action_expert_t_warmup=200
  --scheduler.alpha_f=0.1 --max_grad_norm=1.0 --activation_checkpointing=true
  --compile_loss=false --log_interval=10 --allow_resume=false --save_overwrite=false)
if [[ "$MODE" == mae ]]; then
  COMMON+=(--tactile_backend=tactile_mae --tactile_num_tokens=32 --tactile_history_stride=2
    --tactile_encoder_path=/root/RXJ/pretrained_models/huggingface/AnyTouch-ViT-L-16
    --freeze_tactile_encoder=false --tactile_backbone_learning_rate="$VIT_LR" --tactile_adapter_learning_rate="$ACTION_LR")
else
  COMMON+=(--tactile_backend=as_image)
fi
torchrun --standalone --nnodes=1 --nproc-per-node=8 "${COMMON[@]}" \
  --run_name="${RUN_NAME}-smoke" --wandb=null --max_duration=2 --save_interval=1000 \
  --save_folder="$RUN_DIR/smoke" --save_final_optim=false --save_final_unsharded_checkpoint=false \
  2>&1 | tee "$RUN_DIR/smoke.log"
"$PYTHON" - "$RUN_DIR" <<'PY'
from pathlib import Path
import datetime,json,sys
Path(sys.argv[1],'training_started.json').write_text(json.dumps({'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'smoke_updates':2,'smoke_validation':True}))
PY
torchrun --standalone --nnodes=1 --nproc-per-node=8 "${COMMON[@]}" \
  --run_name="$RUN_NAME" --wandb.name="$RUN_NAME" --wandb.group=x5-native-v2 \
  --wandb.entity="$WANDB_ENTITY" --wandb.project="$WANDB_PROJECT" \
  --max_duration="$STEPS" --save_interval=1000 --save_num_checkpoints_to_keep=6 \
  --save_final_unsharded_checkpoint=true --save_final_optim=true \
  --save_folder="$RUN_DIR/checkpoints" 2>&1 | tee "$RUN_DIR/train.log"
printf '{"status":"complete"}\n' > "$RUN_DIR/training_complete.json"
