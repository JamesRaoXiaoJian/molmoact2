# 日常启动（当前有效入口）

5080 上两个终端分别执行：

```bash
cd ~/rxj_ws/molmoact2
bash scripts/run_x5_molmo.sh server
```

```bash
cd ~/rxj_ws/molmoact2
bash scripts/run_x5_molmo.sh client
```

默认编号在 `scripts/run_x5_molmo.sh` 的 `MODEL_SELECT=${MODEL_SELECT:-0}` 一处选择：
0=RGB20k，1=MAE20k；修改后重启两个进程。也可两边都使用
`MODEL_SELECT=1 bash scripts/run_x5_molmo.sh server/client` 的相应子命令。
Molmo 编号独立于 OpenPI；两进程须使用相同编号。

共享硬件默认来自 `configs/x5_molmo.env` 及原 OpenPI `configs/site.env`。
固定 prompt 来自 `configs/x5_molmo_tasks.json` 的 task_index=1，其原始来源为
A100 `outputs/dlc/x5_cleaned_ablation_20261008/dataset/meta/tasks.parquet`；它对应当前
OpenPI RTC 启动器固定任务，采用数据原文，不随机。无需手写 prompt 或硬件参数。

client 沿用原启动器内部 `MOVE_X5` 确认、启动保存姿态复位和脚踏控制；
脚踏 1 开始/结束当前轮，脚踏 3 等待状态退出，默认 Ctrl+C 直接清理退出。
原故障/首动作差检查与当前较严格的限幅仍生效。真实 SDK 会发送保持、增益、damping；
`DRY_RUN=1` 关闭策略动作与自动复位，但不会消除这些 SDK 行为。现场由操作者执行。

新增同结构模型：在 `configs/x5_molmo_policies.json` 追加独立 id（例如2），填
`backend=molmoact2`、`mode=rgb|mae`、`description`、`checkpoint`、`norm_tag`、
`tokenizer_dir`、`tasks_file`、`prompt_selection=fixed_data_task`、`task_index` 和 `control`。
checkpoint 指向含 `model.pt`、`config.yaml` 的 native 目录；归一化 tag 必须存在于该
checkpoint，tokenizer 资产须与训练配置匹配；任务文本来自对应数据元信息。
当前适配面为相同 X5 三路 RGB（MAE两路单帧触觉）、14维绝对关节动作、30动作 horizon、
native 默认 n_action_steps=5、normalize_gripper=false。不同结构/动作空间需要额外适配。
无需为同结构的新 id 改脚本分支，仅将脚本默认 MODEL_SELECT 或环境变量改成该 id。

控制默认 30Hz、消费5步、同步是本地 X5 数据/checkpoint 与已验证接口的兼容起点，
不是官方最优或模型原理限制；`EXECUTE_STEPS` 可配置1–30且窗口不能超出horizon。
当前同步桥接存在推理间隙；官方支持其他异步/RTC路径，本次未升级或引入那些实现。

本次最小 CPU/mock 检查通过：0/1 两角色参数、固定数据 prompt、共享硬件默认、
授权/复位与 DRY_RUN、后续注册 id、权重/tag/tokenizer 到 loader 的接线。
本轮真实模型加载、GPU测试、真实硬件初始化和动作下发均为0，未commit/push。

以下为先前协议与 mock 验证记录，长命令继续兼容，日常以本节短命令为准。

---

# MolmoAct2 接入现有 OpenPI X5 推理与控制

新增入口只在 `127.0.0.1` 监听，使用 OpenPI 的二进制 NumPy/MessagePack 格式：
连接先收到 metadata，`WebsocketClientPolicy.infer()` 返回 `actions: float32(30,14)`。
原 Molmo HTTP 服务、模型、RGB/MAE 预处理、归一化与反归一化保持原逻辑。
本次改动只保留本地 diff，没有提交或 push。

## 安全复测

```bash
ssh 5080
cd ~/rxj_ws/molmoact2
bash scripts/run_x5_openpi_loopback.sh both
# 可单独选择 rgb 或 mae；共用 /tmp/rxj_inference_deploy_gpu.lock。
```

复测使用现有 OpenPI `WebsocketClientPolicy`、`ActionChunkBroker` 和从原客户端
源码读取的已审查纯函数（观测构建、动作窗口校验、夹爪 preload、限幅与时长）。
录制样本替代相机，进程内 fake sink 只保存候选；禁止硬件模块导入、CAN socket、
传感器设备访问和非 loopback 网络连接。该协议复测不导入硬件 main；下面新增的完整控制复测会在硬件导入/初始化边界全部预先 mock 后执行原 main。
结果位于 `outputs/inference/openpi_bridge/{rgb,mae}.json`、日志、原始动作 `*_actions.npz`（2×30×14）与限幅候选 `*_candidates.npz`（12×14）；
`actual_action_sends` 和 `hardware_initializations` 必须为 0。

仅启动候选策略服务：

```bash
bash scripts/run_x5_openpi_server.sh rgb 8001
# 退出 RGB 后再选择 MAE：
bash scripts/run_x5_openpi_server.sh mae 8001
```

服务没有 SDK、CAN、ROS 或实际动作分支。metadata 的 `dry_run=true` 只描述本服务，
不会替外部硬件客户端禁用电机。候选服务验收不能证明真实机器人性能。
OpenPI 原硬件客户端的构造/退出会发送保持、增益或 damping，即使 predict-only、
不传 enable-motion 或 RTC 的 DRY_RUN，也不能保证零硬件命令；本次没有运行那些入口。

## 观测与动作约定

输入 state 严格为有限 `float32(14,)`，顺序为左 6 关节＋左夹爪＋右 6 关节＋右夹爪；
prompt 为非空字符串。`left_image/right_image/head_image` 对应
`left_d405/right_d405/head_camera`，三路都必需，HWC RGB uint8。
现有 OpenPI 观测函数先将 RGB resize-with-pad 到 224×224；适配层不再缩放，
Molmo 原模型预处理继续处理收到的图像。与原始高分辨率回放的像素输入不同，
一致性验收比较的是同一份客户端图像在直接 Molmo 与网络适配路径上的输出。

MAE 必需 `left_gripper_left_tactile/right_gripper_right_tactile`，保留原始 HWC 尺寸；
最终模型只有 1 帧触觉，THWC 多帧、其他触觉路由、缺失触觉均显式拒绝。
RGB 拒绝触觉输入；额外 base camera、RTC、未知字段也拒绝，不吞掉未使用数据。

最终 checkpoint 声明 `absolute joint pose`、`normalize_gripper=false`：输出关节 rad，
夹爪为校准后的原生电机角 rad（0 闭合、负值打开），左右顺序保持不变。
适配层不套 OpenPI 的 `-1/(2π)` 状态或 `*-0.5` 输出转换，也不做角度→米转换。
源训练数据为 30 Hz；Molmo 的动作索引 0 对应当前训练帧，后续连续帧，不添加 gap/跳帧。

输出 horizon=30，推荐消费前 5 步，保持 Molmo 的 n_action_steps=5。
原 X5 控制函数要求 `action_start_index+execute_steps<=30`，首动作关节差超限会拒绝，
随后使用原限幅、夹爪范围/preload 和 `duration_scale/control_hz`，适配层不改变它们。
假执行沿用原控制默认值；录制样本中一部分夹爪超出默认软范围，仍由原控制限幅处理，
不修改模型返回的原始动作。

`supports_rtc=false` 明确关闭 RTC；同步模型推理发生在 5 步窗口之间，存在推理间隙，
30 Hz 只是窗口内假执行节拍，不能保证持续 30 Hz 控制或真实机器人性能。

OpenPI 的 `WebsocketClientPolicy.reset()` 是 no-op；`ActionChunkBroker.reset()` 清本地队列。
适配服务支持观测 `_reset:true` 清该连接的 Molmo 会话，连接之间隔离，断连后清理。
原 main 仅对 pistream/pi05_tactile 自动加 _reset，不会自动给 Molmo 加。新增 EpisodePolicy 保持 model_type=molmoact2，每轮开始/恢复显式发送 _reset；不会伪装成其他模型。
原客户端不会自动重连，断连/非法请求后需新建客户端，重新握手；复测覆盖这些行为。

## 范围与证据

每模型用两个先前训练数据回放观测进行真实最终权重 BF16 推理，比较直接/native 与网络
结果，检查握手、左右/夹爪哨兵、动作形状/有限性、原客户端 5 步 broker、reset、
断连/重连、异常输入及 MAE 缺触觉。也复测原 HTTP 单动作队列 `[1,0,0,0,0,1]`。
原客户端完整控制循环的模拟资源生命周期另见下面测试；没有真实 SDK 生命周期、留出集或实机性能评估。
仅新增 websockets/msgpack/dm-tree 到 Molmo 隔离环境；OpenPI 项目及原 dirty 文件不改动。

实际机器人夹爪 readout 是否与源训练数据采用同一标定尚未验证；接口单位一致不代表实机标定一致。
日志中的 HTTP Missing-required-tactile 错误来自故意缺输入的负例；应以最终 JSON status=passed 和零下发计数为准。

## 本次实际验收

- rgb: 2 个观测，网络中位耗时 0.578 秒，峰值分配显存 11.14 GiB；直接/网络逐维最大差 0.0，真实下发 0，fake sink 保存 12 个候选。
- mae: 2 个观测，网络中位耗时 0.643 秒，峰值分配显存 11.72 GiB；直接/网络逐维最大差 0.0，真实下发 0，fake sink 保存 12 个候选。

原 12 项推理回归＋18 项适配测试＋5 项控制 glue 测试共 35 项通过，原 30 项全部保留；新文件 lint、格式、编译、shell 语法检查通过。
双连接 reset/断连隔离、显式重连、原控制窗口/非有限动作/维度/首动作差限负例均通过。
OpenPI 原 11 个 dirty 文件内容和状态未改变；141 个 Molmo 环境包兼容性检查通过。
本地未提交补丁：`deployment/openpi-bridge.patch`；汇总：`deployment/openpi-bridge-summary.json`。

## 原 main 完整控制循环复测

```bash
ssh 5080
cd ~/rxj_ws/molmoact2
bash scripts/run_x5_openpi_control.sh both
# 可单独选 rgb 或 mae；与其他测试共用 GPU 锁。
```

新增 `experiments/scripts/x5_openpi_control.py` 是薄控制 glue：读取并导入 OpenPI
原 `examples/x5_client/main.py`，对原 main 只在已核实的每轮触觉 history reset
位置之前插入一次 `client.reset()`，其余 main 控制体保持原代码。
该 mock 入口要求同步 WebSocket、`action_start_index=0/execute_steps=5`，拒绝 37 步、
RTC、enable-motion、自动姿态 reset 和 DAgger。若原每轮 hook 结构改变则拒绝运行。

`deployment/openpi_main_smoke.py` 在导入原 main **之前**注入硬件模块 trap；
SDK loader、CAN 检查、双臂对象、RGB/触觉相机、脚踏和终端输入在构造前替换为
带标记的模拟对象。真实电机/夹爪方法一旦被调用即报错。原观察构建、startup gate、
episode 循环、动作校验/窗口/限幅/时长、异常处理及 finally 资源清理实际执行；
重力补偿和 close 也只落在模拟对象上。没有真实相机、脚踏、CAN/ROS 或 SDK 初始化。

RGB、MAE 各自用最终权重真实 GPU 推理通过以下三条路径：

- normal：第一轮消费 2 个候选后暂停，剩余 3 个丢弃；第二轮发送 reset，
  用另一个观测生成新块，消费 5 个候选后正常退出，共 7 个候选。
- late_pause：推理仍在阻塞时模拟暂停，返回的旧块在首步前丢弃；恢复后发送 reset，
  消费新块 5 个候选再退出。
- error：第一块消费 5 个候选，下一次故意发送 15 维 state；网络显式报错，
  原 main 进入模拟 gravity fallback 和错误退出，脚踏/所有相机停止，模拟双臂 close。

两模型三条路径的网络动作与相同客户端观测、相同 seed 的直接 Molmo 输出逐维
最大差均为 0.0；每模型共记录 17 个候选事件，真实下发和真实硬件初始化均为 0。
逐案例 reset、候选所属 episode、推理耗时、错误和清理记录见
`outputs/inference/openpi_bridge/{rgb,mae}_main.json`，完整日志为 `*_main.log`。
日志出现 state 维度错误及重力补偿/电机提示是预期负例或原 main 的模拟运行日志；
以最终报告 status=passed、零下发/零初始化以及清理断言为准。

另外对原 Molmo HTTP 单动作模式做实际内容隔离检查：两个会话使用不同录制观测和 seed，
对完整 PolicySessionState 的数组/张量/图像内容取指纹；另一会话推理、消费或 reset
不得改变该会话内容。首次消费后剩余队列恰好 4 行，每行 `(1,14)`，其数值对应
native 参考的第 1～4 行，历史恰好 1 帧；继续消费不会新增推理，reset 一个会话
后另一个的队列及后续输出保持一致。WebSocket 全块路径不靠服务器单动作队列，
连接 reset/close 隔离及显式重连由原协议复测覆盖。

本次完整测试覆盖的是**提前 mock 后的原 main 控制流程和资源生命周期**，不是实际
硬件 SDK 的构造/关闭行为；同步推理间隙、真实夹爪标定和实机表现仍需单独评估。

## 当前编号入口与离线性能更新（2026-10-12）

两个终端均在 `~/rxj_ws/molmoact2`。`client` 会初始化真实 SDK/相机/脚踏，
默认使能动作并启动复位，由现场操作员执行；离线验收没有运行真实 client。

终端 1：

```bash
bash scripts/run_x5_molmo.sh server
```

终端 2：

```bash
bash scripts/run_x5_molmo.sh client
```

默认编号只在 `scripts/run_x5_molmo.sh` 的 `MODEL_SELECT=${MODEL_SELECT:-0}`；
0 为 RGB20k，1 为 MAE20k。临时选择 MAE 时两个终端均在命令前加 `MODEL_SELECT=1`。
checkpoint/norm_tag/tokenizer/task/control 关系在 `configs/x5_molmo_policies.json`；
该注册表独立于 OpenPI 原编号。固定 prompt 来自数据 task_index=1，逐轮复用，
没有随机选任务。硬件配置读取 `configs/x5_molmo.env` 和现有共享 site.env。

短命令默认 30 Hz、窗口 `[0:5]`、同步推理；`EXECUTE_STEPS`、`CONTROL_HZ`、
`ACTION_START_INDEX` 环境覆盖或末尾显式 CLI 参数仍可使用，窗口不能超出 30。
30 步是 checkpoint action_horizon，5 步是当前消费默认值。
`Ctrl+C` 默认退出，脚踏流程和每轮 reset 保持原控制主循环语义。
故障检查、首步关节差 0.25 rad、每步关节 0.02 rad、夹爪 0.1 rad 等限制保留。
`DRY_RUN=1` 关闭策略动作与自动复位；真实 SDK 构造/关闭仍有保持、增益、damping。

原显式 `server|client rgb|mae` 长命令兼容入口保留，其默认/末尾覆盖与短命令不同，
日常使用上面编号短命令。`run_x5_openpi_control.sh` 始终是全硬件 mock 验证。

正式 server 默认启用 `MOLMO_NATIVE_OPTIMIZATION=graph`，优化原生模型专家 flow。
prompt、相机、action_horizon=30、10 步 solver、bf16、控制频率及动作窗口均未因优化改变。
两个录制观测、20 次重复和完整控制 mock 的最终验收见
`deployment/perf_20261012/RESULTS_zh.md`。第一次请求有预热/捕获开销。
同步消费仍会在每块前等待推理；本次没有接入异步执行或 RTC，也未验证实机质量。

正常结束自己的 Molmo server 后，可用同一入口按 baseline 重启回退：

```bash
MOLMO_NATIVE_OPTIMIZATION=baseline bash scripts/run_x5_molmo.sh server
```

完整源码回退仅恢复三份性能源码，先验证当前文件和备份哈希：

```bash
.venv/bin/python deployment/perf_20261012/restore_baseline.py --check
.venv/bin/python deployment/perf_20261012/restore_baseline.py
```

回退不会改模型、控制配置、环境或 OpenPI；检测到后来编辑则拒绝全部写入。
