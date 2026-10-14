# X5 原生 MolmoAct2 异步 RTC

本实现支持现有 cleaned step20000 的 RGB、Tactile-MAE 原生权重。无需转换 checkpoint
或重新训练。默认仍为同步执行；客户端显式选择 RTC 后，30 Hz 控制循环消费动作队列，
后台线程同时请求下一块，并将剩余的原始模型动作作为 flow 采样约束。

## 启动

服务端启动方式保持不变，默认使用原生 CUDA Graph：

```bash
MODEL_SELECT=0 bash scripts/run_x5_molmo.sh server
```

操作者按原有相机、踏板和运动确认流程启动客户端，增加一个环境变量：

```bash
MODEL_SELECT=0 MOLMO_EXECUTION_MODE=rtc bash scripts/run_x5_molmo.sh client
```

`MODEL_SELECT=1` 选择 MAE，服务端与客户端须一致。显式设置
`MOLMO_EXECUTION_MODE=sync` 可恢复同步执行。直接 Python 客户端也支持 `--rtc`。
实际客户端会初始化硬件并遵循原有运动开关；离线验证使用下面的 mock 入口。

2026-10-14 按操作者要求，RGB/MAE 部署配置的 `control.max_step_delta` 均改为
`0.1` rad，同步和 RTC 共用此关节单步限幅。旧式 `client rgb/mae` 和直接 Python
入口在未显式传参时也使用 `0.1`。可用 `--max-step-delta=<值>` 显式覆盖；
它不改变复位限幅、首动作拒绝阈值、夹爪参数或模型的 q01/q99 归一化边界。

可选的编译设置在**服务端**启用：

```bash
MODEL_SELECT=1 MOLMO_COMPILE_ACTION_EXPERT=1 bash scripts/run_x5_molmo.sh server
```

编译不是 RTC 的必需条件。它会带来独立的浮点输出变化，沿用先前的显式选择方式。
`MOLMO_NATIVE_OPTIMIZATION=modcache` 可关闭 Graph，用于回退和对照；较慢的路径
可能无法满足 H=30、30 Hz 的持续执行时间预算，运行时会明确停止。

## 数据流与时序

客户端每次启动自动创建 `logs/molmo-<rgb|mae>-client-<时间>-<PID>.log`，终端会显示
绝对路径；可用 `MOLMO_CLIENT_LOG_DIR` 指定其他目录。日志保存实际启动参数、服务端
metadata、控制参数、RTC 请求耗时/队列、原控制循环的开始/结束和 Python 异常。
2026-10-14 以前的入口没有客户端自动文件日志；服务端 `debug.log` 不能替代实机控制日志。

额外的 `control_trace` 仅写文件，逐帧区分 `candidate`（限幅前后与实测状态）和
`sent`（原 SDK 调用成功返回、调用耗时、发送间隔）。候选目标不代表已经发送；
SDK 返回成功也不代表机械臂已到位，跟踪误差要与后续实测状态比较。`adjusted_target`
已经包含夹爪闭合补偿。分析间隔须按原日志中的 episode 开始/结束分段，排除人工等待。
文件写入在独立线程中执行，队列有界，队列满时丢日志而不阻塞控制；退出末行记录
`dropped_records`。原生 SDK 直接写到 stdout/stderr 的内容不包含在此 Python 日志中；
强制杀进程或断电可能丢失队列尾部记录。该日志功能不改变限幅、轨迹、RTC 参数或踏板行为。

```text
相机/状态 → 单个后台 RPC → VLM/触觉编码 → 准备 KV → RTC flow → 30×14 原始动作
                ↑                                         ↓
         未消费的原始动作 ← 有界动作队列 → 夹爪校准后的动作 → 原 OpenPI 控制循环
                                                           30 Hz、限幅、运动确认
```

- 复用原 OpenPI X5 的硬件控制主体，仅在本仓库加载的模块中替换 RTC worker，保留原有
  episode reset hook；没有修改 OpenPI 仓库或 SDK。
- 每个客户端至多一个 RPC 在途。推理不会占住动作消费线程。
- 前缀始终使用原始模型关节角。队列另存夹爪偏移后的执行目标，避免偏移被反馈并重复叠加。
- 服务端按 checkpoint 的同一套统计量归一化前缀，保持两维夹爪的 `normalize_gripper=false`，
  再补齐模型内部动作维度。RGB、MAE 的观测编码路径保持原样。
- 后续响应根据从观测提交到处理完成的单调时钟耗时，丢弃 `ceil(latency × 30)` 个过期帧；
  消费端还按经过时间跳过已过期帧。不会把新预测重新从第 0 帧回放。
- 推理延迟预算采用最近最多 64 个有效请求的 p95，再加一帧余量。预算放不进当前前缀时停止，
  正常执行中队列耗尽也停止，交还原控制程序的异常/重力补偿处理。
- 暂停和关闭递增 generation、清空队列。在途响应只有在 generation 仍匹配时才能原子发布；
  新 episode 的 reset 不会被旧 RPC 的完成错误地确认。
- 首轮请求超过 500 ms 时只作为预热，下一控制周期重新取观测。连续三次仍超过此预算则停止；
  首轮 RPC 超过 30 秒也会超时。预热响应不会成为电机目标。

默认 `RTC_QUEUE_THRESHOLD=18`、`RTC_EXECUTION_HORIZON=20`、
`RTC_MAX_GUIDANCE_WEIGHT=10`。RTC 固定 30 Hz、生成 30 帧、起始索引 0；
同步模式中的 `EXECUTE_STEPS=5/15/30` 不决定 RTC 的消费长度。
这些参数适用于本轮测量范围，持续更高的延迟不能单靠加大队列阈值弥补。

## 采样算法与来源

参考并适配以下已有实现（Apache-2.0）：

- [官方 LeRobot MolmoAct2 RTC 接口，24f15b92](https://github.com/huggingface/lerobot/blob/24f15b92b556cb861ca65e989462e23505a701bd/src/lerobot/policies/molmoact2/modeling_molmoact2.py)
- [cijerezg 的轻量 RTC，6617d375](https://github.com/cijerezg/lerobot/blob/6617d375684c44f214e664de27bb80553ac95084/src/lerobot/policies/rtc/modeling_rtc.py)
- [cijerezg 的异步运行时](https://github.com/cijerezg/lerobot/blob/6617d375684c44f214e664de27bb80553ac95084/src/lerobot/rl/rtc_actor_runtime.py)
- 当前 OpenPI X5 `src/openpi/rtc_client.py` 的 worker/queue 协议和 generation 处理。

使用 identity-Jacobian 近似，省去反向计算；不宣称是完整 Jacobian 引导。
Molmo 的 flow 时间从噪声 0 到动作 1。每一步以 `x + (1-t)v` 估计终点，
用已知前缀的误差修正 Euler 更新。LINEAR 权重在承诺前缀上为 1，在软衔接区逐渐下降，
未来未知部分为 0。guidance 限制在 `(0, 10]`。

基础 BF16 Euler 项保持原来的运算顺序，修正项用 FP32。零权重位置直接选择基础结果，
因此空前缀与原同步采样器在相同随机种子下逐位一致。固定尺寸的前缀/权重缓冲区可以随
Graph replay 更新；前缀长度、延迟和内容变化不触发重捕获。形状、权重、flow 步数或
guidance 上限变化仍会失效缓存。保留非有限值检查、捕获失败恢复与独立输出存储。

## 协议

服务 metadata 增加 `supports_rtc=true`、`rtc_protocol=molmo-rtc-v1`、
`rtc_guidance=identity_jacobian`。普通同步 observation 仍走原采样路径。
RTC observation 额外包含：

```python
"actions": float32_array_of_shape_30_by_14,  # 原始模型关节角，未叠加客户端夹爪偏移
"_rtc": {
    "prefix_valid_steps": 18,
    "inference_delay": 12,
    "execution_horizon": 20,
    "max_guidance_weight": 10.0,
}
```

字段、维度、dtype 和所有前缀值均校验，包括权重为零的未使用位置。
目前限于 native backend、单观测、连续 action-only 模式与完整动作块。

## 验证与复现

测试入口均为录制观测或硬件 mock，不启动真实机器人：

- `experiments/tests/test_x5_rtc.py`：符号/时间方向、零前缀、带约束采样、时钟对齐、
  前缀/校准分离、观测快照、暂停竞态、启动预热、异常和队列耗尽。
- `deployment/rtc_cuda_safety.py`：真实 CUDA 的 RTC eager/Graph 一致性、动态前缀复用、
  非有限值拒绝与恢复、输出存储独立性；支持 `--compile-blocks`。
- `deployment/benchmark_x5_rtc.py`：两种真实权重的空前缀一致性、前缀敏感性、
  夹爪前缀保持、eager/Graph 延迟与捕获计数。
- `deployment/rtc_control_mock.py --real-model`：真实模型经过 WebSocket 与原 OpenPI
  RTC 主循环，SDK、相机、踏板、CAN 初始化/动作边界均预先替换或拦截。

输出保存于 `outputs/inference/rtc_20261013/`，隔离远端为
`/home/dm/rxj_ws/molmoact2_rtc_20261013/deployment/rtc_results/`。
软件验证不等于熨烫任务成功率验证；实际接触轨迹与机械臂表现仍须由操作者在真机上验证。

2026-10-13，RTX 5080 Laptop 上两段录制观测、每组 20 次的结果：

| 模型 | 编译 | RTC 无 Graph p50 | RTC Graph p50 / p95 | 控制 mock 间隔 p95 |
| --- | --- | --- | --- | --- |
| RGB | 关闭（默认） | 429 ms | 331 / 364 ms | 34.6 ms |
| MAE | 关闭（默认） | 465 ms | 387 / 404 ms | 33.9 ms |
| RGB | 开启 | 300 ms | 291 / 311 ms | 33.7 ms |
| MAE | 开启 | 357 ms | 347 / 365 ms | 33.9 ms |

推理计时包括模型输入准备与 GPU 同步，不含真实相机采集和实机网络；控制间隔来自
原控制主体 + 真实模型 + loopback + 假硬件，正常案例每组两个 episode、70 个候选动作，
不是长期或真实机械臂控制频率保证。正常案例均无队列耗尽/预算失败，推理与执行重叠。
每组 Graph 仅捕获一次，空前缀、有效前缀及变化前缀共用捕获；eager/Graph 结果按原始
float32 位模式一致。空前缀与各自的未启用 RTC 路径逐位一致；编译与未编译的比较仍遵循
先前的浮点差异说明。

91 项唯一 CPU 回归用例通过；编译/未编译的真实 CUDA 检查通过；两模型、两编译模式下
正常/迟到暂停/错误/耗尽共 16 个完整控制 mock 案例通过。慢首轮在正常案例中被丢弃一次，
随后用新观测填充队列。真实机器人初始化与动作发送均为 0。
精简机器可读结果为 `deployment/x5_rtc_validation_20261013.json`。
