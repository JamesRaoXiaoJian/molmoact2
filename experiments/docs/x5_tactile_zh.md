# MolmoAct2 接入 ARX X5 触觉

本分支参考 `/root/openpi-arx/openpi` 接入四种触觉编码方式。输入均为 DM SDK
`getInferImg()` 输出的三通道 RGB 图像。`as_image` 复用 MolmoAct2 视觉编码器；
独立方案只将 RGB 送入原视觉编码器，触觉共享独立骨干，加入传感器身份 embedding
并投影为 VLM 前缀 token，再通过原 KV 条件进入 action expert。没有额外 gate 或阶段预测。

| `--tactile_backend` | 骨干 | 每个观测的触觉输入 | 每路 token | 默认初始权重目录 |
|---|---|---|---:|---|
| `as_image` | MolmoAct2 原视觉编码器 | 当前帧 | 原模型的图像 token 数 | 无额外权重 |
| `anytouch2` | AnyTouch2 ViT-B/16 | `t-6,t-4,t-2,t` 四帧 | 32 | `AnyTouch2-Model` |
| `sparsh_vjepa` | Sparsh V-JEPA ViT-S/16 | 同上，骨干内反转为当前到历史 | 32 | `Sparsh-VJEPA-Small` |
| `tactile_mae` | AnyTouch stage-1 ViT-L/14 | 当前帧 | 32 | `AnyTouch-ViT-L-16` |

权重目录前缀为 `/root/RXJ/pretrained_models/huggingface/`。
首次训练通过 `--tactile_encoder_path` 覆盖；骨干检查所有实际使用的权重及形状，
不把缺失权重替换成随机参数。重建头不参与策略训练。AnyTouch2／V-JEPA 使用可学习
注意力汇聚，MAE 将 query 注入骨干并取对应输出。默认联合微调骨干，
`--freeze_tactile_encoder true` 可冻结骨干，queries、汇聚、投影继续训练。

## 观测与布局

三路 RGB 固定为 `left_d405`、`right_d405`、`head_camera`，随后追加触觉传感器。
LeRobot 数据字段在这些名称前加 `observation.images.`。

| mixture | 触觉路数 | 触觉顺序 | normalization tag |
|---|---:|---|---|
| `x5` | 0 | — | `arx_x5` |
| `x5_tactile` / `x5_tactile_two` | 2 | `left_gripper_left_tactile`, `right_gripper_right_tactile` | `arx_x5_tactile_two` |
| `x5_tactile_four` | 4 | `left_gripper_left_tactile`, `left_gripper_right_tactile`, `right_gripper_left_tactile`, `right_gripper_right_tactile` | `arx_x5_tactile_four` |

默认两路沿用参考仓库的传感器身份 `[0, 3]`。训练相机随机排序只作用于 RGB，
触觉顺序固定。外层历史采用当前 MolmoAct2/LeRobot 的连续历史帧；各外层时刻
再读取其独立触觉窗口，均不读取未来或跨 episode，episode 开头由 LeRobot 重复首帧。
图像路径按相机／传感器优先、时间从旧到新排列；独立触觉前缀按外层时间、传感器、query 排列。
推理使用同样顺序。
触觉图像不使用随机颜色、裁剪、旋转增强，不做背景扣除。
`as_image` 分辨率和归一化沿用 MolmoAct2 视觉骨干。独立路径统一 224×224：
AnyTouch2 用 CLIP mean/std；V-JEPA 横图顺时针旋转后缩放并保留 `[0,1]`；
MAE 右侧／底部补零成正方形后缩放、使用 ImageNet mean/std，并保留参考单帧 Conv3d 路径。

`camera_keys`、`tactile_keys`、`tactile_backend`、`tactile_layout` 随
`robot_processor.metadata_by_tag` 保存到原生 checkpoint。训练自动扩大 `max_images`
以容纳所有历史图像；错误配置导致截断时会明确报错。缺失触觉不会补成零图像。
独立方案的完整配置、骨干、queries、投影及传感器身份参数一起保存；完整恢复／推理
不再访问初始权重目录。独立前缀目前使用未打包的纯机器人 batches，不能开启 packing
或混入 VLM 数据。三个独立骨干的 Transformer 层支持激活重计算和 FSDP2 分层分片。

X5 state/action 为原生 14 维：左侧 6 关节＋夹爪，再右侧 6 关节＋夹爪。
沿用数据集的原生单位，不套用 Pi0.5 的夹爪转换。默认预测 70 帧，执行 5 帧，
动作间隔仍为 MolmoAct2 的连续帧；没有复制参考 StreamPI 的 `action_gap=3` 或五个稀疏外层观测。

## 数据与训练

默认 repo id 为 `local/hot_stamp_bag_merged_0912`，可通过
`MOLMOACT2_X5_REPO_ID` 修改。`LEROBOT_DATA_ROOT` 采用原有 repo 分层根目录，
实际数据位于 `${LEROBOT_DATA_ROOT}/${MOLMOACT2_X5_REPO_ID}`。可用软链接映射已有目录。

先验证真实数据及预处理，不加载模型权重：

```bash
cd /root/RXJ/molmoact2
.venv/bin/python experiments/scripts/smoke_x5_tactile.py \
  --dataset-root /mnt/tacumi_data/realman_data/hot_stamp_bag_merged_0912 \
  --layout four --n-obs-steps 2 --backend anytouch2 \
  --checkpoint /root/RXJ/molmoact2/artifacts/models/MolmoAct2
```

训练复用现有入口，例如单卡启动两路 AnyTouch2 微调：

```bash
mkdir -p data/x5_tactile/local
ln -s /mnt/tacumi_data/realman_data/hot_stamp_bag_merged_0912 \
  data/x5_tactile/local/hot_stamp_bag_merged_0912
export LEROBOT_DATA_ROOT="$PWD/data/x5_tactile"
export LEROBOT_VIDEO_BACKEND=pyav
export WANDB_PROJECT=molmoact2_x5_tactile
export WANDB_ENTITY="your_wandb_account_or_team"
PYTHONPATH=experiments:experiments/lerobot/src \
.venv/bin/torchrun --standalone --nproc_per_node=1 \
  experiments/launch_scripts/train_lerobot.py \
  /root/RXJ/molmoact2/artifacts/models/MolmoAct2 x5_tactile \
  --tactile_backend anytouch2 --tactile_num_tokens 32 \
  --frame_loading_backend av --global_batch_size 1 --device_batch_size 1 \
  --num_workers 2 --n_obs_steps 1 --action_format continuous \
  --state_format continuous --img_aug photometric \
  --add_setup_tokens false --add_control_tokens false \
  --save_folder=outputs/x5_tactile_two_anytouch2 --max_duration=10000
```

将 mixture 替换为 `x5_tactile_four` 即使用四路，替换 `--tactile_backend` 即切换编码器。
`--tactile_history_stride` 默认 2；`--tactile_sensor_id` 默认 -1；`--tactile_num_tokens` 默认 32。
正式训练的 batch、显存、训练步数和
W&B 设置沿用项目入口并按设备调整。本次接入不会自动启动训练。

## HTTP 推理

使用微调得到的**原生** checkpoint（包含 `config.yaml` 及模型状态）：

```bash
.venv/bin/python experiments/scripts/serve_policy.py \
  --checkpoint /path/to/native/checkpoint \
  --image_keys x5_tactile --norm_tag arx_x5_tactile_two \
  --host 127.0.0.1 --port 8000
```

触觉 checkpoint 按保存的元数据读取输入，四路模型自动要求四路触觉。
`POST /act` 可使用参考 X5 客户端的短字段名或完整 LeRobot 字段名：

```python
payload = {
    "left_d405": left_rgb,
    "right_d405": right_rgb,
    "head_camera": head_rgb,
    "left_gripper_left_tactile": left_touch,
    "right_gripper_right_tactile": right_touch,
    "state": state14,
    "instruction": "Stamp the bag.",
    "session_id": "episode-1",
    "single_action": True,
}
```

也可以把图像放进 `images` 字典，或按 checkpoint 顺序传入恰好 5／7 张图像的列表。
AnyTouch2／V-JEPA 的每个触觉字段必须提供 `[4,H,W,3]` 历史窗口，按旧到新排列，
间隔与 checkpoint 的 `tactile_history_stride` 一致。`/health` 会返回 backend、keys、
frames 和 stride。不能将动作块请求间隔当成连续传感器帧，也不会用四份当前帧伪造窗口；
客户端应按数据集 FPS（默认 30 Hz）持续采样并按需组装窗口。
图像沿用服务端的 `json_numpy`、RGB 数组或 base64 编码支持。episode 开始调用
`POST /reset` 清空该会话历史，单步请求沿用原策略的历史缓冲。
参考仓库客户端使用 WebSocket，需要将请求桥接到这里的 HTTP `/act`；本次不修改硬件采集程序。

## 验证

```bash
PYTHONPATH=experiments:experiments/lerobot/src .venv/bin/python -m pytest -q \
  experiments/tests/test_tactile_integration.py \
  experiments/tests/test_tactile_encoders.py \
  experiments/tests/data/test_lerobot_camera_keys_alternative.py \
  experiments/tests/test_molmoact2_eval_parity.py
```

测试覆盖传感器／历史顺序、RGB 排序隔离、checkpoint 元数据恢复、图像数量上限、
训练／推理触觉预处理一致性、HTTP 输入与缺失传感器错误。
独立小模型还验证骨干梯度、优化器更新、触觉改变动作输出、完整保存／恢复，以及
恢复时无需初始权重、部分 checkpoint 缺权重时报错。真实数据 smoke 单独检查视频解码与完整预处理。

三个真实预训练骨干与参考 PyTorch 的数值对照：

```bash
.venv/bin/python experiments/scripts/verify_tactile_encoders.py anytouch2
.venv/bin/python experiments/scripts/verify_tactile_encoders.py sparsh_vjepa
.venv/bin/python experiments/scripts/verify_tactile_encoders.py tactile_mae
```

原生 checkpoint 可训练及通过 HTTP 推理；此次没有扩展 HF 导出模型的自定义触觉结构。
训练效果和实机任务成功率需要微调后评估。编码器来源及许可证见
`olmo/nn/tactile_licenses/THIRD_PARTY.md`。
