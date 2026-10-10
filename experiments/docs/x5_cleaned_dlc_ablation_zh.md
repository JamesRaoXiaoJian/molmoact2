# 0918–20 cleaned：RGB 与 Tactile-MAE 对照

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
