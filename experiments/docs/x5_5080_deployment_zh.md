# X5 cleaned RGB / MAE 在 RTX 5080 上离线部署

目标项目为 `~/rxj_ws/molmoact2`，隔离环境为该项目 `.venv`。
原始代码来自 `JamesRaoXiaoJian/molmoact2` 的 `feat/tactile`，部署基线提交为
`76036895de5e2b80ca9031b09cf5d60a172e05cf`。
训练冻结版本为 RGB `c3c0a2e7f58cdbab7360098016b5792985e1fde8`，
MAE worker fix `5c8982bdc0ee74c1296109490cf95054fd8a9484`。
源 `/root/molmoact2` 是 `/root/RXJ/molmoact2` 的符号链接。
源仓库其他未提交数据集和训练改动没有迁入。

## 资源与精度

两份最终目录只包含原始 `model.pt` 和 `config.yaml`，路径保持为：

```text
outputs/runs/molmoact2-x5-cleaned-rgb-fft20k-b64-s42-20261008/checkpoints/step20000-unsharded/
outputs/runs/molmoact2-x5-cleaned-mae-fft20k-b64-s42-20261008/checkpoints/step20000-unsharded/
```

RGB 原始 model.pt 为 21,962,673,375 字节；MAE 为 23,193,643,844 字节。
权重保留 FP32 原文件；运行时通过 `--parameter_dtype bfloat16` 在 meta 上先转换，
再分配 CUDA 内存，避免先在 16 GB GPU 上分配完整 FP32 权重。
低精度连续动作路径仅计算最后一个 token 的未使用语言 logits，保留全序列 KV conditioning，
避免分配大尺寸的“序列长度 × 词表”张量。CPU 回归检查 KV 与固定噪声动作完全一致。
原生推理调用同时启用对应 autocast；默认未传该参数的行为保留。
关闭 inference CUDA graph 以减少部署显存需求。
这与 A100 上的 FP32 基线精度不同，测试报告记录实际数值偏差。

两份 config.yaml 完全保留原文和源 SHA-256。
`--tokenizer_dir` 只在运行时覆盖源机器绝对缓存路径。
仅迁入 Qwen/Qwen3-4B tokenizer 的四个小文件及必要缓存布局，
源 tokenizer revision 为 `1cfa9a7208912126459214e8b04321603b3df60c`。
最终 MAE checkpoint 已包含 399 个触觉编码器 tensor，推理不需要原始 AnyTouch 预训练文件。

离线样本 `data/offline_samples` 仅包含两个先前基线回放中的真实观测（索引 0、12），
对应 episode 0 / 192；其图像、14 维状态、30×14 目标动作和 A100 参考输出来自
`outputs/dlc/x5_cleaned_ablation_20261008/dataset` 及既有回放报告。
该目录约 2.6 MB，没有复制训练数据集或 packed cache。
所有观测来自训练数据，测试不代表独立验证或实机性能。

## 使用

```bash
ssh 5080
cd ~/rxj_ws/molmoact2
# 两个模型依次测试；脚本自动等待与 OpenPI 共用的 GPU 锁。
bash scripts/run_x5_5080_offline.sh both
# 也可以单独测试。
bash scripts/run_x5_5080_offline.sh rgb
bash scripts/run_x5_5080_offline.sh mae
```

输出为 `outputs/inference/5080_deployment/{rgb,mae}.json`、日志和动作 NPZ。
测试检查最终模型真实推理、(1,30,14) 输出、有限数值、固定 seed 重复性、
内存 ASGI 完整动作接口一致性、5 步队列、reset；MAE 还检查缺少触觉拒绝与置空/交换对照。
测试不启动网络监听，不连接机器人/CAN/ROS，不发送执行指令。

如需仅在本机提供软件 API，以下启动命令只绑定 `127.0.0.1`，
并持有共享 GPU 锁；使用完通过该服务终端 Ctrl-C 退出以释放锁。
部署测试没有运行此长期服务或连接任何执行硬件。

```bash
bash scripts/run_x5_5080_server.sh rgb 8000
# 退出 RGB 后，才能启动 MAE 或 OpenPI GPU 工作。
bash scripts/run_x5_5080_server.sh mae 8000
```

输入 RGB camera keys 为 `left_d405`、`right_d405`、`head_camera`；
MAE 额外要求 `left_gripper_left_tactile`、`right_gripper_right_tactile`。
14 维状态/输出顺序为左侧 6 关节+夹爪、右侧 6 关节+夹爪。
默认动作队列为 5 步，完整输出块为 30 步；归一化 tag 必须与模式匹配。

## 环境重建与证据

Python 和 uv 位于本项目 `.python`、`.tools`；没有复制 A100 虚拟环境或系统库，
没有升级驱动。Torch CUDA wheel 来自官方 `download.pytorch.org`，其余依赖来自 PyPI。
Python 3.12.15，Torch 2.10.0+cu128，torchvision 0.25.0+cu128，transformers 5.18.0，
tokenizers 0.23.2，NumPy 2.2.6。完整已安装版本记录在 `deployment/environment-freeze.txt`。
`deployment/requirements-inference.txt` 是最小推理依赖入口；重建时先安装官方 Torch：

```bash
export UV_CACHE_DIR="$PWD/.uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.python" UV_PYTHON_BIN_DIR="$PWD/.tools"
.tools/uv venv --python 3.12 .venv
.tools/uv --no-config pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0
.tools/uv --no-config pip install --python .venv/bin/python \
  --index-url https://pypi.org/simple -r deployment/requirements-inference.txt
```

`--no-config` 避免根目录其他服务依赖和 Torch extra index 干扰最小环境解析。
重建命令应在旧环境已确认可替换或已自行备份时使用。

源哈希记录为 `deployment/source_sha256.txt`；最终核验结果为 `deployment/target_sha256.txt`。
逐模型运行结果和显存/耗时以离线 JSON 报告为准，只有报告成功生成才算真实推理通过。
A100 的 24 观测基线不能替代 5080 实测。
部署过程传输路线为目标已有 `xiaojian_A100` SSH 直连 A100；使用已验证主机密钥，
不复制凭据，不关闭主机验证。传输断点/区间记录在 `deployment`，不属于最终推理资源。
