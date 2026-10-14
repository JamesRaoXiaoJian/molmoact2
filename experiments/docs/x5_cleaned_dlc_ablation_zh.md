# 0918–20 cleaned：RGB 与 Tactile-MAE 对照

## 2026-10-14：输入规范 v2 修复实验

旧 20k checkpoint 的夹爪状态没有归一化，却进入 `[-1,1]` 的 256 档离散编码；
关节 q01/q99 硬裁剪也丢失了极值目标。v2 使用独立的状态/动作归一化声明：
全部 14 维按训练轨迹的 min/max 缩放；状态在离散编码边界裁剪，连续动作缩放与
反变换不裁剪。统计范围不替代机器人限位。所有物理量仍是原生角度，动作仍是绝对目标。
没有 `normalization` 声明的旧 native checkpoint 继续使用原规则。

训练与推理共享 `olmo/data/robot_text.py` 的任务清理逻辑；客户端发送原始 RGB，
由 checkpoint 的 SigLIP 执行 resize。episode 末尾复制帧通过 `action_is_pad` 排除监督。
该输入对齐按 checkpoint 的 v2 metadata 自动启用。旧模型默认保持历史输入；
服务端显式 `MOLMO_INPUT_CONTRACT=aligned` 可作对照。1011 个观测的 RGB 回放显示
关节 MAE 从 0.888° 降到 0.675°，但右夹爪 MAE 从 0.0254 升到 0.0450 rad，
因此不能把输入修复单独认定为旧模型抓取性能升级，必须配套训练与分项验收。

新 cohort：`outputs/dlc/x5_contract_v2_20261014`。使用数据已有的整 episode 时间顺序划分：
训练 269 条 / 299,228 帧，验证 33 条 / 41,116 帧，测试 35 条 / 42,078 帧。
只读取既有 split 的 episode ID，不使用 subtask/phase 标签。验证与测试均为 task 1，
属于后段轨迹评估，不能代表两个任务的均衡泛化。统计量只由训练分区拟合。
原 20k 权重见过全部 337 条，因此其验证分区指标明确命名为 `regression_seen_val`。

首轮各 8×A800、global batch 64，先 RGB 比较两种初始化：

| 初始化 | 更新步 | LLM / ViT / connector / action expert LR | 验证指标 |
|---|---:|---|---|
| 旧 RGB step20000 适配 | 5,000 | 3e-6 / 1.5e-6 / 1.5e-6 / 1.5e-5 | 已见数据回归 |
| 官方 MolmoAct2 基座 | 20,000 | 1e-5 / 5e-6 / 5e-6 / 5e-5 | 未参与本次训练的验证轨迹 |

两者都保持连续动作、256 档状态、30 帧 horizon、4 个训练 flow 时间样本、全参数微调，
warmup 200、每 1,000 步验证和保存 checkpoint。2 步 8 卡 smoke 同时验证数据、反传和
验证循环，成功后才启动正式 W&B run。5000/20000 是筛选预算，不保证最佳收敛点。
后续 RGB/MAE 正式对照应采用相同的新规范和划分，按回放与实机结果选择初始化。

```bash
# 数据准备仅执行一次，不覆盖既有 cohort。
python experiments/scripts/prepare_x5_contract_v2.py \
  --previous outputs/dlc/x5_cleaned_ablation_20261008 \
  --output outputs/dlc/x5_contract_v2_20261014
python experiments/scripts/validate_x5_contract_v2.py \
  --cohort outputs/dlc/x5_contract_v2_20261014 \
  --output outputs/dlc/x5_contract_v2_20261014/preflight.json
# DLC 使用已提交代码的不可变快照 code/；code_revision.json 记录 SHA。
bash scripts/run_dlc_x5_contract_v2.sh warmstart rgb
bash scripts/run_dlc_x5_contract_v2.sh base rgb
```

W&B 项目 `molmoact2-x5-input-contract-v2`。训练 loss 首屏，真实验证/回归 loss 产生后
才添加图表，再排列进度、LR/梯度、速度/显存。不展示 gate、phase 或零值语言损失。
不同归一化下的 loss 不能直接与旧 run 数值横比；部署决策以原生单位误差和抓取成功率为准。

以下为 2026-10-08 初版对照的历史记录。

两个作业各使用一台 8×A800 80GB 节点，从同一份本地
`allenai/MolmoAct2` 基座继续全参数微调，不混入 VLM 数据，不添加 gate／subtask。
数据为全部 337 条轨迹、382,422 帧：
`/mnt/tacumi_data/realman_data/hot_stamp_bag_merged_0918-20_cleaned`。

| 项目 | RGB | Tactile-MAE |
|---|---|---|
| 图像输入 | 左腕、右腕、头部 RGB | 相同 RGB＋左夹爪左指、右夹爪右指触觉 |
| 触觉编码器 | 无 | AnyTouch stage-1 ViT-L/14，32 queries／路 |
| 训练步数／全局 batch | 20,000／64 | 20,000／64 |
| GPU microbatch／梯度累积 | 每卡 2／4 次 | 每卡 2／4 次 |
| 数据遍历量 | 1,280,000 样本，约 3.35 epochs | 相同 |
| seed／动作块／观测历史 | 42／30 帧／1 帧 | 相同 |
| 动作空间 | 原生 14 维绝对关节目标，padding 到基座 32 维并 mask | 相同 |
| 微调模式 | VLM、视觉塔、连接器、动作专家全参数训练 | 同左，另训练 MAE、queries、投影和身份参数 |
| LLM／ViT／连接器／动作专家 LR | `1e-5`／`5e-6`／`5e-6`／`5e-5` | 相同；MAE 骨干 `5e-6`、适配参数 `5e-5` |
| 优化器与调度 | AdamW，warmup 500 步，cosine 到峰值 0.1 倍，grad norm 1 | 相同 |
| flow timesteps／精度 | 4／FP32 master＋BF16 AMP，FSDP2 | 相同 |
| 增强 | 原生 RGB full 增强 | RGB 相同；触觉无颜色、裁剪或旋转增强 |
| state/action 归一化 | 当前全部 cleaned 数据重算 q01／q99 | 同一份统计 |
| checkpoint | 每 2,000 步保留一份，共 10 份；最终另存原生 unsharded 模型 | 相同 |

## 依据与边界

[官方自定义数据指南](https://github.com/allenai/molmoact2/blob/main/experiments/README.md)
给出的 full fine-tuning 起点是 batch 64 和上述四组学习率，并建议新任务关闭 packing、
使用动态长度。其 50,000 步是示例预算。
[LeRobot 的 MolmoAct2 指南](https://huggingface.co/docs/lerobot/main/en/molmoact2)
说明较小数据可从 batch 16–32／LoRA 开始，较大数据倾向 full fine-tuning。

[Poke & Wiggle 的公开实机对照](https://pokeandwiggle.com/blog/introducing-paw-10)
包含约 300 条示范的数据档，MolmoAct2 使用全参数、batch 64、30 帧动作块和相同分组 LR，
并依据多任务实验选择约 3 epochs。这里据此选择 20,000 步，约 3.35 次数据遍历，
作为首轮公平对照；这不是针对热压袋任务已验证的最优预算。
该社区的相对 TCP 动作和 RTC 不复制到本实验，本实验保留数据集的绝对关节动作。

旧 Pi0.5 任务的训练配置后来延长到 150,000 步、batch 32，但模型、动作采样和优化器不同。
这里只参考其共享环境、缓存审核与本节点 RAM staging，不将其训练预算原样移植。
两组都使用全部轨迹，因此当前没有独立验证集；训练 loss 不能替代实机成功率评估。

## 数据与启动

现有 packed JPEG 缓存为原始分辨率（RGB 480×640，触觉 288×384），无合成补帧。
准备程序重新核验当前 1,685 个训练视频的完整 SHA-256、每个 JPEG shard 的帧数和 offset，
再将 meta/data 拷贝到本项目共享目录，重算 native state/action 统计，保留原始数据和旧缓存。
作业只将所需 3／5 路缓存拷入节点 `/dev/shm`，通过严格的 JPEG reader 读取，
不为 RGB 基线解码触觉，也不依赖视频随机 seek。

共享准备目录：`outputs/dlc/x5_cleaned_ablation_20261008/`。
日志和模型目录：`outputs/runs/molmoact2-x5-cleaned-{rgb,mae}-fft20k-b64-s42-20261008/`。
源码在提交前冻结到共享准备目录的 `code/`。
MAE 启动兼容修复单独冻结到 `code-mae-workerfix/`，修复 worker 序列化、CUDA 库路径
和触觉参数组学习率调度；RGB 正式训练继续使用已启动的冻结版本，未为这些修复重启。
实际 JobId、代码版本和替代作业记录保存在 `active_jobs.json`。

```bash
python3 /root/.codex/skills/dlc/scripts/submit.py \
  --command-file /root/RXJ/molmoact2/scripts/run_dlc_x5_cleaned_rgb.sh \
  --name molmoact2-x5-cleaned-rgb-fft20k-b64-s42 \
  --record-dir /root/RXJ/molmoact2/outputs/dlc/x5_cleaned_ablation_20261008/rgb \
  --gpu a800 --gpu-count 8 --mount tacumi --submit

python3 /root/.codex/skills/dlc/scripts/submit.py \
  --command-file /root/RXJ/molmoact2/scripts/run_dlc_x5_cleaned_mae.sh \
  --name molmoact2-x5-cleaned-mae-fft20k-b64-s42 \
  --record-dir /root/RXJ/molmoact2/outputs/dlc/x5_cleaned_ablation_20261008/mae \
  --gpu a800 --gpu-count 8 --mount tacumi --submit
```

每个作业完成缓存 staging 后，先以相同 8 GPU、batch 64 和正式参数运行 2 步 smoke，
成功后从预训练权重和 seed 42 重新开始正式训练。W&B 凭据仅由共享的受限文件加载；
正式训练默认在线，项目名为 `molmoact2-x5-cleaned-tactile-ablation`。

## W&B 对照面板

训练真实主损失为 `train/action_flow_loss`，每 10 个 optimizer steps 上传。
`configure_x5_wandb_workspace.py` 从已上传 history 检查有限 loss 和递增 step，
再保存有序面板：动作损失、训练进度、学习率与梯度、训练速度与显存。
首屏同时保留原始损失、带原始曲线的平滑趋势和最新数值。
RGB / MAE 使用蓝色 / 橙色；MAE 专属指标仅在实际上传后加入，避免空面板。
GPU 利用率来自独立系统采集流，因此以已用时间为横轴。
没有独立验证集，本面板不显示验证 loss 或成功率。

布局修改只使用 W&B 服务端 API，不连接或重启训练进程，也不另启同一 run 的 logger。
修改前备份 workspace，修改后回读检查首屏及 section 顺序；重复执行更新同一 saved view。
需要 `wandb-workspaces`，建议使用独立工具环境，不修改运行中训练所用的依赖：

```bash
source /root/RXJ/dlc_shared/wandb_env.sh /root/RXJ/molmoact2/outputs/dlc/x5_cleaned_ablation_20261008/observability
/root/RXJ/dlc_shared/.venv-wandb-workspaces/bin/python \
  experiments/scripts/configure_x5_wandb_workspace.py \
  --entity jamesraoxiaojian-shenzhen-university \
  --project molmoact2-x5-cleaned-tactile-ablation \
  --record-dir outputs/dlc/x5_cleaned_ablation_20261008/observability \
  --update-personal
```

`--update-personal` 同时整理当前 API 账号的默认 workspace。
两个 workspace 均关闭自动追加与字母排序，只保留实际有记录的必要指标白名单；
原始布局保存在操作前备份中，可用于恢复。
saved view 的直达链接和数据核验结果保存在 `observability/workspace_verified.json`。

## 最终 checkpoint 离线推理检查

`run_x5_inference_comparison.py` 使用两组 `step20000-unsharded` 原生模型，
在相同 8 条轨迹的早、中、晚三个位置回放，共 24 个真实观测。
RGB 仅输入左腕、右腕和头部相机；MAE 额外输入两路真实触觉。
两组使用同一目标动作块、同一随机种子和 10 次 flow 积分，生成完整 `30×14` 动作块。
另外检查相同种子的可重复性、ASGI HTTP 请求与直接调用的一致性、
默认 5 步动作队列、会话重置、缺失触觉输入拒绝，以及置空／交换触觉的固定噪声对照。

主回放使用原生 FP32 推理。另在 4 个相同观测上测量 BF16 autocast 并再次检查接口一致性；
BF16 是本测试中的性能测量，不自动改变生产服务精度。
已修复反归一化时对 `mask=False` 维度的误裁剪：
未归一化夹爪值保留原值，归一化关节仍按 `[-1,1]` 裁剪后反变换。
此修复不改变训练权重。

```bash
bash scripts/run_x5_inference_comparison.sh outputs/inference/x5_cleaned_20261010
.venv/bin/python experiments/scripts/summarize_x5_inference_comparison.py \
  outputs/inference/x5_cleaned_20261010
```

产物包括两组的 `summary.json`、逐样本误差、动作数组、触觉对照、BF16 profile，
以及总表 `comparison.json`、配对误差 CSV、14 维动作曲线与误差图（PNG/PDF）。
`policy_space_*` 指训练动作尺度：12 个关节按 q01/q99 归一化，2 个夹爪保持原单位；
另外单独报告关节和夹爪误差。
所有轨迹都参与了训练，因此这属于训练数据回放，不能据此报告独立验证结果或实机成功率。
HTTP 检查使用内存中的 ASGI transport，不启动监听端口，不发送机械臂控制指令。
实际服务入口 `serve_policy.py` 默认仅绑定 `127.0.0.1`。

2026-10-10 在共享 A100 80GB 上的首轮结果：

| 项目 | RGB | Tactile-MAE |
|---|---:|---:|
| 24 个观测的完整动作块 MSE（训练动作尺度） | 0.001753 | 0.001883 |
| 前 5 步 MSE（训练动作尺度） | 0.001337 | 0.001335 |
| FP32 中位推理时间 | 1.234 秒 | 1.336 秒 |
| BF16 中位推理时间（4 观测参考测量） | 0.909 秒 | 0.959 秒 |

两组直接调用、HTTP 完整动作块、5 步队列、reset、相同 seed 重复生成均通过。
修复前 RGB 的夹爪 raw MAE 为 1.0124，修复后为 0.07131；测试输入和权重相同。
两处触觉置空对照改变动作的平均绝对量约 `1.1e-4`，交换传感器约 `6e-6`
（训练动作尺度），这两个观测的触觉响应较弱，不能据此证明触觉带来收益。
当前延迟高于 5 步／30 Hz 对应的约 0.167 秒重规划间隔，需优化推理后再评估此执行配置。

## 2026-10-12 官方推理路径延迟对照

`benchmark_x5_inference_paths.py` 在同一台 A100 80GB、batch 1、BF16 参数与 autocast 下，
使用 2 条轨迹的 6 个真实观测，重复 3 轮；模型生成 30 帧动作，每次计时都等待 GPU 完成。
计时包含预处理、设备拷贝、模型和后处理，不包含网络及机器人执行。
官方对照使用现有 `artifacts/models/MolmoAct2` 的原始 HF 权重和 `predict_action`。
权重不修改；仅在单独复制的评估目录中添加同一份 X5 RGB 归一化元数据，
并应用官方机器人 server 同样使用的 BF16→NumPy 兼容修复。
因此这是同输入的延迟对照，不是官方原始模型在 X5 上的任务效果评估。

| 路径 | 4 次 flow 积分 | 10 次 flow 积分 | 30 次 flow 积分 |
|---|---:|---:|---:|
| 当前原生 RGB | 420 ms | 914 ms | 2,540 ms |
| 当前原生 MAE | 438 ms | 919 ms | 2,427 ms |
| 官方 HF，关闭 CUDA Graph | 262 ms | 505 ms | 1,329 ms |
| 官方 HF，开启 CUDA Graph，固定输入形状稳定测量 | 142 ms | 210 ms | 434 ms |

前三行是 18 次混合观测的中位数，最后一行是预热后相同输入形状 8 次测量的中位数。
图模式固定形状的 10 次积分 P95 为 212 ms；混合输入形状的 P95 为 1,128 ms，
因为当前官方 graph manager 只缓存一个形状，切换 prompt 长度会重新捕获图。
这两个统计不能混用。

10 次积分的独立 profiling 显示：原生 RGB 的动作专家阶段约 795 ms，
其中 KV 准备累计约 183 ms、调用 10 次；官方关闭 graph 时动作专家约 391 ms，
KV 准备约 10 ms、调用 1 次。MAE 触觉编码约 12 ms。
CUDA event 分段计时包含其间的 CPU dispatch/同步等待；嵌套分段不能相加。

返回 5／15／30 帧时，原生 RGB 分别约 907／897／916 ms，MAE 约 869／871／884 ms。
生成的动作 horizon 始终为 30；返回长度只是切片，不会减少 flow 模型计算。
官方 graph 路径分别约 211／210／209 ms。

[官方公开数字](https://allenai.org/blog/molmoact2) 为单张 H100、LIBERO 环境下
基础模型约 180 ms，深度推理模型约 790 ms，属于另一种硬件与输入条件。
本轮没有可用的 5080 Laptop SSH 连接，以上实测不能标为 5080 Laptop 120W 实测。

报告目录：`outputs/inference/latency_official_20261012/`，包括各路径 JSON、
总表 `comparison.json`、`latency.csv`、PNG/PDF 对照图和完整日志。
官方第一次尝试的计时矩阵已成功完成，后续 profiling hook 名称错误导致退出；
第二次复用那些原始计时记录并补齐固定形状测试及 profiling，原始记录留在
`official_attempt0_profile_hook.log`。

```bash
# 使用已完成部署的两个小样本，无需完整数据集或 packed cache。
.venv/bin/python experiments/scripts/benchmark_x5_inference_paths.py \
  --route rgb --samples-dir data/offline_samples --output outputs/inference/latency
# 同样可测 --route mae。
# 官方原始 HF 模型需预先存在；路径可由 --official-model 指定。
.venv/bin/python experiments/scripts/benchmark_x5_inference_paths.py \
  --route official --samples-dir data/offline_samples \
  --official-model /path/to/official/MolmoAct2 --output outputs/inference/latency
```
