# X5 / RTX 5080 Laptop 推理部署评审

日期：2026-10-13。适用：RGB、Tactile-MAE 的 cleaned step20000 权重，
BF16，3 路 RGB，14 维绝对关节动作，动作块 30 帧，flow 积分 10 次，控制频率 30 Hz。
本轮没有连接机器人硬件，也没有评估真机任务成功率。

## 部署决策

保留原生 PyTorch 模型和现有权重，分离每个观测仅需计算一次的条件准备与重复的
flow 计算。当前可发布路径为静态计算复用 + 原生动作头 CUDA Graph；关闭 Graph
时使用同一套准备逻辑。无需为这些优化重新训练，也无需转换成 JAX。

需要改变的执行架构是客户端的同步等待：目前 OpenPI X5 主循环执行
`采集 → client.infer → 执行 K 帧 → 再采集`，`supports_rtc=False`。
“使用 OpenPI 的 WebSocket 协议”不代表已经使用 JAX 或 RTC。
增加执行帧数只能降低等待占比，不能消除每块之间的等待。

控制端应采用独立的 30 Hz 执行器、单个后台推理工作线程、有时间基准的有界动作队列，
并在生成新块时用即将执行的旧动作约束它。后者需要在 Molmo 的 flow 采样器中实现
RTC 条件；仅打开客户端 `--rtc` 不成立。RTC 原论文也指出，直接切换两个独立预测块，
以及简单平均不同策略的轨迹，都可能产生不合理的过渡。
[RTC 原论文](https://arxiv.org/html/2506.07339v2)

本页的性能数据对应 `40fa7b1`，当时仅完成推理计算路径优化。
后续已在 `feat/x5-rtc` 接入异步执行器和 RTC 采样，默认仍为同步模式，
实现、启用方式与验证边界见 [X5 RTC 文档](x5_rtc_zh.md)。

## 该参考哪些数值

| 指标 | 本项目的解释与验收方式 |
| --- | --- |
| 模型延迟 p50 / p95 | 同一设备、同一功耗条件、同一权重、观测、随机种子、精度和 flow 步数比较；冷启动单列 |
| 端到端延迟 p95 / p99 | 从观测采集时间到可执行动作就绪，包括预处理、网络、排队、GPU、反归一化；loopback 不代表实机网络 |
| 控制间隔 | 30 Hz 对应 33.33 ms；统计间隔分布与超期次数，不能拿模型调用频率冒充控制频率 |
| 动作队列 | 记录剩余帧、观测年龄、丢弃的过期动作、队列耗尽、跨 episode 的迟到响应 |
| 轨迹连续性 | 分关节统计块边界的位置、速度、加速度变化；夹爪独立统计，不盲目平均 |
| 数值与任务质量 | 固定种子输出差值、归一化动作误差、前 5 帧误差、真机成功率分开报告 |
| 显存与设备状态 | 同时报 allocated/reserved、温度、时钟和功耗；Laptop 的结果不能套用桌面 5080 |
| Graph / 编译 | 记录首次捕获/编译耗时、缓存命中、重捕获次数及失败行为；稳态统计排除冷启动但不隐藏它 |

官方发布的约 **180 ms** 是 MolmoAct2 base 在 **单张 H100、LIBERO** 条件下的动作调用，
约 **790 ms** 对应带深度推理的版本。它们可作量级参考，不能作为 120 W 5080 Laptop、
三相机加触觉模型的合格线。[Ai2 发布说明](https://allenai.org/blog/molmoact2)

官方服务支持 BF16 和可选动作头 CUDA Graph；其约 2 倍收益、额外约 2 GB 显存的描述
也依赖具体路径和配置。本项目应使用实测增量，不能照搬这些比例。
[官方部署仓库](https://github.com/allenai/molmoact2)

## 30 Hz 下的时间预算

5 / 15 / 30 帧分别覆盖 166.7 / 500 / 1000 ms。同步模式下，假设模型等待为 `L` 秒，
执行 K 帧的平均命令频率近似为 `K / (L + K / 30)`，等待占比为
`L / (L + K / 30)`；真实系统还要加采集和通信开销。因此模型即便降到 300 ms，
同步执行 5 帧仍只有约 10.7 次命令/秒的周期平均频率，块内部才接近 30 Hz。

异步时定义 `d = ceil(端到端延迟预算 × 30)`。补充请求启动时队列至少要覆盖 d 帧，
另加实测抖动裕量。新块以请求观测时刻为基准，返回时前 d 帧已经过期，不能从第 0 帧重播。
在单工作线程、H=30、每次新预测覆盖未来 H 帧的简化调度下，持续工作还需要
`H >= 2d + 裕量`：当前块要覆盖一次推理，新返回的剩余块还要覆盖下一次推理。
这只是可行性预算，不保证动作质量。

例如端到端 p95=400 ms 时 d=12，若暂取 2 帧裕量，队列触发阈值的可行区间约为
14–18 帧；p95=500 ms 时 d=15，30 帧预测已经没有这 2 帧裕量。
“阈值 0.5”不应直接作为默认答案。LeRobot 官方也要求结合延迟、帧率与队列大小调参。
[LeRobot 异步部署指南](https://huggingface.co/docs/lerobot/async)

## 本轮保留与淘汰的方案

同一临时 performance 档位、每组 60 次、固定两帧录制观测的配对结果如下。
这组在启用 Dynamic Boost 之前完成，GPU 通常受 90 W 预算约束。数值为 p50 / p95，单位 ms；
不是原装电源标称功率，也不是客户实机端到端控制延迟。

| 模型 | 原无 Graph | 等价优化无 Graph | 编译无 Graph | 原 Graph | 等价优化 Graph | 编译 + Graph |
| --- | --- | --- | --- | --- | --- | --- |
| RGB | 488 / 492 | 436 / 439 | 310 / 325 | 367 / 372 | 341 / 349 | 300 / 317 |
| MAE | 549 / 557 | 489 / 494 | 365 / 374 | 417 / 423 | 392 / 400 | 353 / 366 |

等价优化在两观测所有测量调用、以及 24 个跨 episode 录制观测上与原 BF16 路径逐位一致。
编译候选的 24 观测回放结果：

| 模型 | 原 policy-space MSE | 编译 MSE | 相对变化 | 关节最大输出差 / rad | 夹爪最大输出差 / rad |
| --- | --- | --- | --- | --- | --- |
| RGB | 0.00154445 | 0.00154873 | +0.28% | 0.00665 | 0.01611 |
| MAE | 0.00190540 | 0.00191200 | +0.35% | 0.00646 | 0.01758 |

这里只能说明录制数据上的误差变化很小；接触任务的成功率、物体受力和闭环稳定性还未验证。
新默认使用等价优化；编译选项保持显式开启。

修复并启用 Dynamic Boost 后，每组 40 次的复测 p50 / p95：

| 模型 | 等价优化无 Graph | 编译无 Graph | 等价优化 Graph | 编译 + Graph |
| --- | --- | --- | --- | --- |
| RGB | 436 / 439 | 304 / 319 | 341 / 352 | 294 / 312 |
| MAE | 470 / 475 | 358 / 373 | 389 / 402 | 349 / 371 |

编译路径峰值 allocated 显存：RGB 无 Graph 11.13 GiB、Graph 11.32 GiB；
MAE 分别 11.71 / 11.91 GiB。以上功耗由驱动动态分配，未锁定为 105 W；
开启 Dynamic Boost 本身不是本轮主要加速来源。每组单独保存了设备状态和全部延迟样本。

77 项唯一 CPU 回归用例通过；未编译/编译两种模式的真实 CUDA 数值安全、缓存失败恢复、
输出存储独立性检查通过；RGB/MAE 在两种模式下的完整控制 mock 均通过，涵盖正常、
暂停后迟到响应、推理异常。真实机器人初始化和动作发送均为 0。

- **保留：** 每次观测只做一次 cross-attention K 归一化；按形状缓存动作 RoPE；
  主干收集 KV 时复用 attention 内部已经准备的 KV；共享注意力 bias 的精度转换；
  用一次张量归约替代逐个 CUDA 标量转 Python 布尔值。训练/梯度路径保持原逻辑。
- **淘汰：** 单独融合 finite 检查的 Triton 候选。无 Graph 路径实测更慢，Graph 下收益不稳定。
- **淘汰：** 整段主干 decoder CUDA Graph。两模型的 20 次试验没有稳定收益，显存反而增加约 0.4 GiB。
- **实验评估：** action block 的 `torch.compile`，明确关闭其内部 CUDA Graph 后单独测量。
  它改变了浮点执行次序；评估输出误差与录制动作误差，不因少量样本 MSE 相近就宣称真机等价。
  PyTorch 官方也不保证不同浮点计算路径逐位一致。
  [数值精度说明](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
- **暂不迁移：** TensorRT、INT8/FP8、减少相机输入/flow 步数。它们分别涉及算子支持与部署维护、
  数值精度、模型输入或采样质量的改变；必须单列质量对照。动态形状还需要明确的构建 profile。
  [Torch-TensorRT 动态形状](https://docs.pytorch.org/TensorRT/user_guide/dynamic_shapes.html)

编译参数以安装的 PyTorch 2.10 为准；`reduce-overhead` 等模式可能启用 CUDA Graph，
本项目的无 Graph 对照显式设置 `triton.cudagraphs=False`。
[PyTorch 2.10 compile 参数](https://docs.pytorch.org/docs/2.10/generated/torch.compile.html)

正式入口默认保持未编译的等价路径。编译候选通过 `MOLMO_COMPILE_ACTION_EXPERT=1`
显式选择，服务 metadata 会回报 `native_action_blocks_compiled`。在同一启动命令前设置
`MOLMO_NATIVE_OPTIMIZATION=modcache` 即关闭原生 Graph，设为 `graph` 则启用它。
取消编译环境变量即可恢复默认。编译的首次请求存在额外耗时；这些开关不改变训练权重。

```bash
# 等价优化、关闭 Graph
MODEL_SELECT=0 MOLMO_NATIVE_OPTIMIZATION=modcache bash scripts/run_x5_molmo.sh server
# 编译候选、关闭 Graph（需要按下面的误差和轨迹验证边界评估）
MODEL_SELECT=0 MOLMO_NATIVE_OPTIMIZATION=modcache MOLMO_COMPILE_ACTION_EXPERT=1 bash scripts/run_x5_molmo.sh server
# 编译候选 + 原生 Graph；MODEL_SELECT=1 切换为 MAE
MODEL_SELECT=1 MOLMO_COMPILE_ACTION_EXPERT=1 bash scripts/run_x5_molmo.sh server
```

## 部署边界

本轮长测发现设备状态并不恒定：power-profiles-daemon 报告 `power-saver`，
ACPI platform profile 报告 `balanced`；NVIDIA 的 GPU-scope 功耗上限在 55 W 和 90 W
之间变化，默认 90 W、报告最大 105 W，还出现功耗限制与软件温度限频计数。
因此“120 W”暂时只能视为用户提供的设备描述，不能视为本轮测试的受控条件。
普通 `nvidia-smi --query-gpu=power.limit` 在该机返回 N/A，应读 `nvidia-smi -q -x`
中的 `gpu_power_readings`；`deployment/capture_x5_device_state.py` 已保存这一检查。
出现限电变化的长测须保留并标注，不能与稳定档位的结果混算为代码收益。

用户确认原装电源连接并授权提高功耗后，已将 profile 设为 `performance`，安装并启用
驱动包自带的 `nvidia-powerd.service`。显卡拒绝 `nvidia-smi --power-limit=105 --scope=0`
的固定限额设置，因此使用 BIOS 支持的 Dynamic Boost 动态分配，不能宣称固定解锁了 120 W。
NVIDIA 明确说明 Dynamic Boost 受整机 BIOS 热/电预算约束。
[NVIDIA 580 Dynamic Boost 文档](https://download.nvidia.com/XFree86/Linux-x86_64/580.126.18/README/dynamicboost.html)

另修复两项服务初始化问题：

1. D-Bus 原先拒绝拥有 `nvidia.powerd.server`。添加仅允许 root 拥有这个精确名称的本地 policy。
2. 此机 `lscpu -p` 给出 20 个逻辑 socket ID，NVIDIA 据此报 `AutoFL: >1 CPU package`。
   内核 `physical_package_id` 实际全部为 0。只在该服务的私有 PATH 下使用
   `lscpu --physical`，逐个核对 20 个 CPU 的 core/package 值与 sysfs 一致；
   没有改写拓扑数值或替换全局 lscpu。服务重启后 D-Bus 成功，多 package 错误未再出现。
   [util-linux 对物理编号的定义](https://raw.githubusercontent.com/util-linux/util-linux/master/sys-utils/lscpu.1.adoc)

对应系统文件为 `/etc/systemd/system/nvidia-powerd.service`、
`/etc/systemd/system/nvidia-powerd.service.d/10-physical-topology.conf`、
`/usr/local/libexec/nvidia-powerd-physical-topology/lscpu`、
`/etc/dbus-1/system.d/nvidia-powerd-local.conf`。源码模板在 `deployment/nvidia-powerd-*`。
需要撤销时先停用 `nvidia-powerd`，删除本次创建的 drop-in、私有 wrapper 和 D-Bus policy，
重新加载 systemd / D-Bus 配置，再通过 `powerprofilesctl` 选择所需电源档位。
内核、显卡驱动、风扇控制与温度保护未修改；无需重启操作系统。

Graph 复用要求输入结构稳定、设备地址有效，并在 replay 前更新当前观测。输出需要拥有独立
存储，不能被下一次 replay 覆盖。参数改变、输入形状改变须失效缓存；捕获失败不能发布半成品。
[PyTorch 2.10 CUDA Graph 语义](https://docs.pytorch.org/docs/2.10/notes/cuda.html#cuda-graphs)

正式服务保持一个模型实例和串行的 GPU 推理临界区。避免在同一 16 GB 显卡上并发装入 RGB/MAE，
或同时运行其他 GPU 程序后引用空闲设备上的基准值。发布前要用实际任务文字和相机形状完成预热，
将冷启动和任务切换捕获耗时纳入就绪判断。当前入口的首次请求仍可能支付这些成本。

原有状态/动作维度、左右顺序、归一化统计、夹爪原生角度、时间基准和 operator motion gate
都是部署契约。CPU 回归与硬件 mock 能验证软件行为，不能替代真实碰撞、接触、夹取和任务成功率测试。
所有本轮测试均使用录制数据或硬件替身；未初始化 SDK、相机、CAN，也未发送实际动作。

实验原始数据位于 `outputs/inference/optimization_20261013/`；远端隔离实验位于
`/home/dm/rxj_ws/molmoact2_perf_20261013/deployment/perf_20261013/`。
数据回放使用训练集的 8 条 episode、24 个观测，不能称为验证集或泛化评估。
精简结果也随代码保存为 `deployment/x5_5080_optimization_results_20261013.json`。
复测入口为 `deployment/run_x5_optimization_checks.sh RESULT_DIR REPLAY_SAMPLES`，
由调用者持有共享 GPU lock；它只运行录制观测和硬件 mock。
