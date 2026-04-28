# deepdive_kai0 项目完整指南

> 生成于 2026-04-25，涵盖项目架构、数据采集、构建、训练、部署、调优、常见坑与当前实验汇总。

---

## 目录

1. [项目概览](#1-项目概览)
2. [硬件拓扑](#2-硬件拓扑)
3. [目录结构详解](#3-目录结构详解)
4. [数据采集与来源](#4-数据采集与来源)
5. [数据构建流水线](#5-数据构建流水线)
6. [训练系统](#6-训练系统)
7. [动态数据集训练](#7-动态数据集训练)
8. [Checkpoint 结构与传输](#8-checkpoint-结构与传输)
9. [真机部署与测试](#9-真机部署与测试)
10. [监测与过拟合检测](#10-监测与过拟合检测)
11. [当前实验汇总](#11-当前实验汇总)
12. [常见坑与规则](#12-常见坑与规则)
13. [命令速查](#13-命令速查)

---

## 1. 项目概览

**目标**：复现 + 部署 kai0（χ₀）机器人操控框架，基于 Physical Intelligence 的 π₀/π₀.₅ 模型，任务包括：
- **Task_A** — T-shirt 铺平折叠（flatten & fold），双臂 Agilex Piper
- **Task_E** — 扶起倒下的盒子（stand up fallen box）
- **Task_P** — 拾取并放入盒子（pick and place in box）

**核心技术模块**（均在 `kai0/` 内）：
1. **Model Arithmetic** — 权重空间合并（4 split → greedy/inverse_loss/...）
2. **Stage Advantage** — 阶段感知 advantage 估计 → AWBC (Advantage-Weighted BC)
3. **Train-Deploy Alignment** — 数据增强 + DAgger + RTC 推理

**模型变体**：pi0 / pi0_fast / pi0_rtc，基于 PaliGemma (vision) + Action Expert。

---

## 2. 硬件拓扑

| 机器 | 硬件 | 角色 | 访问 |
|---|---|---|---|
| **sim01** (本地) | 4× RTX 5090 32GB | 推理部署 + IPC + 真机 | 直接本地 |
| **gf0** | 8× A100 80GB | 离线训练 | `ssh -p 55555 -R 29290:localhost:29290 tim@14.103.44.161` |
| **gf1** | 8× A100 80GB | 离线训练 | `ssh -p 11111 -R 29290:localhost:29290 tim@14.103.44.161` |

- 外网访问：gf0/gf1 通过 SSH 反向隧道（端口 29290，sim01 aurora-slim 代理）
- gf0/gf1 共用 `/vePFS/` (GPFS 50T)，sim01 本地 `/data1/` (NVMe 7T)
- TOS (Volcano 对象存储) 作为跨机大文件中转：`transfer-shanghai` bucket

---

## 3. 目录结构详解

### 仓库根布局（`/data1/tim/workspace/deepdive_kai0/` 或 `/vePFS/tim/workspace/deepdive_kai0/`）

```
deepdive_kai0/
├── kai0/                   核心 kai0 repo (fork of openpi)
│   ├── src/openpi/         模型源码
│   │   ├── models/         JAX 模型 (pi0, pi0_fast, pi0_rtc, gemma, siglip)
│   │   ├── models_pytorch/ PyTorch 复现 (advantage estimator)
│   │   ├── policies/       机器人策略包装 (agilex_policy.py)
│   │   ├── training/       训练循环 + config.py (核心)
│   │   ├── serving/        WebSocket policy server
│   │   └── transforms.py   归一化/图像 resize/tokenize
│   ├── model_arithmetic/   权重合并 (jax + torch 双版)
│   ├── stage_advantage/    advantage pipeline
│   ├── train_deploy_alignment/  aug + DAgger + RTC
│   ├── scripts/
│   │   ├── train.py              ← 主训练入口
│   │   ├── compute_norm_states_fast.py  ← 归一化统计
│   │   ├── serve_policy.py       ← WebSocket 推理服务
│   │   └── train_pytorch.py      ← PyTorch advantage 训练
│   ├── data/               ← 训练数据根目录
│   └── checkpoints/        ← 训练产物（各 config/exp 的 step 子目录）
│
├── train_scripts/          训练侧脚本（不属 kai0 repo 本体）
│   ├── launch/             启动器 .sh
│   │   ├── run_taska_mixed_gf0.sh
│   │   ├── run_taska_mixed_gf1.sh
│   │   ├── run_visrobot01_only_gf1.sh
│   │   ├── dynamic_dataset_train.sh    ← 动态数据 watcher
│   │   └── ...
│   ├── data/               数据构建 + 修复
│   │   ├── build_task_a_mixed.py       ← 混合 3 源
│   │   ├── build_task_a_visrobot01_only.py
│   │   ├── prepare_task_e_splits.py
│   │   ├── prepare_task_p_splits.py
│   │   ├── generate_episodes_stats.py  ← v2.1 per-ep stats
│   │   ├── compute_delta_norm_stats_fast.py
│   │   ├── to_tos_file.py              ← TOS 上传
│   │   └── from_tos_file.py            ← TOS 下载
│   ├── eval/               离线评估
│   │   ├── eval_val_action_mse.py      ← 主 MAE eval 工具
│   │   ├── eval_awbc_compare.py
│   │   └── print_mae.py
│   └── monitor/            进度/健康
│       ├── check_progress.py
│       └── overfit_watcher.py          ← MAE 反弹检测
│
├── start_scripts/          sim01 部署脚本
│   ├── start_autonomy.sh   主入口（相机+CAN+ROS2+policy node）
│   ├── start_autonomy_temp.sh  快速切换 ckpt 的 scratch
│   ├── start_policy_node.sh
│   ├── start_teleop.sh
│   ├── start_data_collect.sh   遥操+脚踏采集
│   ├── toggle_execute.sh   execute_mode 切换
│   ├── rtc_apply.sh        RTC 参数运行时调整
│   └── test_{hardware,cameras,integration_ros2,inference_parity}.py
│
├── ros2_ws/                ROS2 workspace
│   └── src/piper/
│       ├── scripts/policy_inference_node.py  ← 核心推理节点
│       └── launch/autonomy_launch.py         ← launch 配置
│
├── piper_tools/            Piper 机械臂 CAN 工具
├── web/                    data_manager 前端 + 后端 (遥操 UI)
├── config/                 硬件配置
│   ├── pipers.yml          CAN 端口 + 反馈 Hz
│   ├── cameras.yml         RealSense 序列号 + 分辨率
│   └── calibration.yml     手眼标定矩阵
├── calib/                  标定数据
└── docs/
    ├── deployment/
    └── training/
        ├── dynamic_dataset_workflow.md
        ├── task_p_unfreeze_8k_20k_analysis.md
        └── project_complete_guide.md       ← 本文档
```

---

## 4. 数据采集与来源

### 4.1 LeRobot v2.1 数据格式

```
<dataset>/
├── data/chunk-000/episode_000000.parquet   obs.state [N,14], action [N,14], timestamp, frame_index, episode_index, index, task_index
├── videos/chunk-000/
│   ├── observation.images.top_head/episode_000000.mp4   480×640 AV1 30fps
│   ├── observation.images.hand_left/
│   ├── observation.images.hand_right/
│   └── (optional) *_depth/ 深度 zarr — 训练时 strip
└── meta/
    ├── info.json              总 ep/frame 数 + features schema
    ├── episodes.jsonl         {"episode_index":N, "tasks":[prompt], "length":L}
    ├── tasks.jsonl            {"task_index":0, "task":"<prompt>"}
    └── episodes_stats.jsonl   per-ep state/action min/max/q01/q99 (v2.1 required)
```

### 4.2 当前数据源（Task_A）

| 来源 | 路径 | 规模 | 性质 |
|---|---|---|---|
| **visrobot01 raw** | `/vePFS/visrobot01/KAI0/Task_A/2026-MM-DD/` | 210+ 完整 ep（持续增长）| 当前采集数据，按日期子目录 |
| **kai0 历史 base** | `/vePFS/.../data/Task_A/base/` | 3055 ep, 3.36M frames | kai0 官方训练集 |
| **kai0 历史 dagger** | `/vePFS/.../data/Task_A/dagger/` | 3457 ep, 2.42M frames | kai0 DAgger 补充 |
| **kai0 advantage** | `/vePFS/.../data/Task_A/advantage/` | 3055 ep (with labels) | AWBC 用 |

### 4.3 采集流程（sim01 端）

```bash
# 启动遥操 + 脚踏采集
./start_scripts/start_data_collect.sh

# 采集触发: 脚踏 USB button (USB foot-pedal)
# 录制内容: 3 相机 mp4 + parquet (state/action/timestamp)
# 自动上传到 visrobot01: data_manager 前端 "auto session sync"
# 目录按日期分: visrobot01/KAI0/Task_A/<YYYY-MM-DD>/base|dagger/
```

**命名差异**（需 build 时统一）：
- visrobot01: `top_head/`, `hand_left/`, `hand_right/`
- 历史数据: `observation.images.top_head/`, ...
- episodes.jsonl: visrobot01 用 `episode_id`，历史用 `episode_index`

---

## 5. 数据构建流水线

### 5.1 三种常用 build 脚本

| 脚本 | 功能 | 输出 |
|---|---|---|
| `build_task_a_mixed.py` | 混合 visrobot01 + old base + old dagger，各取 N | `Task_A_mixed_<hostname>/` |
| `build_task_a_visrobot01_only.py` | 仅 visrobot01，合并所有日期 | `Task_A_visrobot01_only/` |
| `prepare_task_e_splits.py` / `prepare_task_p_splits.py` | 单源按 seed 切 train/val | `Task_E/{base,val}` 等 |

### 5.2 `build_task_a_mixed.py` 用法

```bash
# Dry-run 查看计划
python train_scripts/data/build_task_a_mixed.py --dry-run

# 正式构建（含 val 分层抽样，每源 --val-size/3 ep）
python train_scripts/data/build_task_a_mixed.py --val-size 21 --force

# 输出位置 (默认)
# /vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A_mixed_gf1/
# ├── base/            N_train×3 eps
# ├── val/             N_val×3 eps
# └── manifest.json    可追溯（seed, 各源 ep_id list）
```

**核心逻辑**：
1. 扫 visrobot01 各日期子目录，过滤 "parquet + 3 cam 齐全" 的 ep
2. 随机从 old base/dagger 各抽 N（seed=42）
3. 重编 `episode_index` 连续 0..3N-1
4. 重编 parquet 的 `episode_index` / `index` / `timestamp` 列
5. video 做符号链接到 `observation.images.<cam>/` 命名
6. 写统一 meta 文件

### 5.3 三步必备后处理

```bash
# 1. Generate episodes_stats (v2.1 required)
python train_scripts/data/generate_episodes_stats.py \
    /vePFS/.../data/<dataset>/base
python train_scripts/data/generate_episodes_stats.py \
    /vePFS/.../data/<dataset>/val

# 2. Compute norm_stats (state/action 的 mean/std/q01/q99)
cd kai0
source ../setup_env.sh
.venv/bin/python scripts/compute_norm_states_fast.py \
    --config-name <your_config_name>
# 输出到 <dataset>/base/norm_stats.json
```

### 5.4 数据集层次关系

```
[visrobot01 raw]     [kai0 historical]
       │                    │
       └──── mixed ─────────┘       →  Task_A_mixed_<host>/
              │
       └──── visrobot01 only         →  Task_A_visrobot01_only/
```

---

## 6. 训练系统

### 6.1 Config 中心（`kai0/src/openpi/training/config.py`）

所有训练配置集中此处。核心 TrainConfig 字段：

```python
TrainConfig(
    name="<唯一配置名>",                        # CLI 第一参数
    model=pi0_config.Pi0Config(pi05=True),     # 模型变体
    data=LerobotAgilexDataConfig(
        repo_id=f"{_KAI0_DATA_ROOT}/data/<dataset>/base",
        default_prompt="<任务 prompt>",
        use_delta_joint_actions=False,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        f"{_KAI0_DATA_ROOT}/checkpoints/<init>/params"
    ),
    # LR schedule (cosine decay with warmup)
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=500,
        peak_lr=1.5e-5,
        decay_steps=12_000,     # ≥ num_train_steps 否则后段 LR=decay_lr 恒定
        decay_lr=1.5e-6,
    ),
    # EMA (Exponential Moving Average)
    ema_decay=0.999,            # 半衰期 ≈ 700 步；None 则不用 EMA
    # Training length
    num_train_steps=12_000,
    # Checkpoint 控制
    keep_period=1_000,          # step % keep_period == 0 永不删
    save_interval=1_000,
    # Data loader
    num_workers=8,
    batch_size=128,
    fsdp_devices=8,             # 8 卡 FSDP
    # Freeze control (默认 nnx.Nothing = 全参数解冻)
    # freeze_filter=nnx.All(nnx_utils.PathRegex(".*PaliGemma.*"),
    #                       nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*"))),
    # Inline eval (训练中自动算 MAE@1/10/25/50)
    inline_eval_val_root=f"{_KAI0_DATA_ROOT}/data/<dataset>/val",
    inline_eval_n_frames=200,
    inline_eval_every=1,         # 每 save 都 eval
)
```

### 6.2 关键参数解读

| 参数 | 含义 | 实践值 |
|---|---|---|
| `peak_lr` | LR 峰值 | 1.25e-5 (保守) ~ 2.5e-5 (激进，易过拟合)；**推荐 1.5e-5** |
| `decay_steps` | cosine 总跨度 | **必须 ≥ num_train_steps**（否则后段 LR 恒定）|
| `warmup_steps` | 线性 warmup 步数 | 一般 `num_train_steps / 30`，至少 200 |
| `ema_decay` | EMA 衰减 | **0.999**（半衰期 700 步）；**不要用 0.9999**（对短训 EMA 稀释严重）|
| `num_train_steps` | 总步数 | 根据 dataset 大小定，**2-5 epoch 足够**用 mixed_1 init |
| `save_interval` | 每 N 步存 ckpt + eval | 500-2000；**10 个 eval 点够看曲线** |
| `keep_period` | 永久保留间隔 | ≤ save_interval 确保每 save 不删 |
| `batch_size` × `fsdp_devices` | 有效 batch | 128 × 8 = 1024 effective；bs=128 per-GPU=16 |
| `freeze_filter` | 冻结模块 | 默认全解冻；可冻 vision `PathRegex(".*PaliGemma.img.*")` 节省显存 |

### 6.3 标准启动器模板

```bash
#!/bin/bash
# run_<xxx>.sh
set -euo pipefail

export PATH=/home/tim/miniconda3/bin:/home/tim/.local/bin:$PATH
export PYTHONUNBUFFERED=1
export KAI0_DATA_ROOT=/vePFS/tim/workspace/deepdive_kai0/kai0
export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=offline
export LD_LIBRARY_PATH=/home/tim/miniconda3/lib:/home/tim/.cuda_compat:/usr/local/cuda-12.8/targets/x86_64-linux/lib
for d in /home/tim/.kai0_venv/lib/python3.11/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH=$d:$LD_LIBRARY_PATH
done

cd /vePFS/tim/workspace/deepdive_kai0/kai0

# ⚠️ 永远 --resume (不使用 --overwrite, 它会 rmtree 整个 exp 目录)
echo "[train] === START $(date) ==="
.venv/bin/python scripts/train.py <config_name> \
  --exp_name=<experiment_name> \
  --resume 2>&1
echo "[train] === END $(date) ==="
```

启动方式：
```bash
nohup bash train_scripts/launch/run_<xxx>.sh > /tmp/train_<xxx>.log 2>&1 &
```

### 6.4 训练输出结构

```
{KAI0_DATA_ROOT}/checkpoints/<config_name>/<exp_name>/
├── 500/                           step 子目录
│   ├── params/                    12 GB, EMA 权重 (部署用)
│   ├── train_state/               31 GB, 全状态 (resume 用)
│   ├── assets/                    空
│   └── _CHECKPOINT_METADATA
├── 1000/...
├── 2000/...
├── norm_stats.json                exp 级（启动时复制）
└── wandb_id.txt
```

---

## 7. 动态数据集训练

### 7.1 需求场景

visrobot01 持续采集 → 希望训练中途吸收新 ep 而不从零开始。

### 7.2 方案 A（采纳）：外部 watcher + `--resume`

不修改 openpi 源码，用 `dynamic_dataset_train.sh` 监控 visrobot01 新 ep 数。满足触发条件时：

```
[trigger] +N new eps, last rebuild >15 min ago
  ↓
step 1/5: pkill train → 10s
step 2/5: build_task_a_mixed.py --force (rebuild 数据)
step 3/5: generate_episodes_stats.py base + val
step 4/5: compute_norm_states_fast.py (重算 norm)
step 5/5: train.py ... --resume (自动从最新 ckpt 续训)
```

**核心原理**：
- `--resume` 加载 `train_state/`（step 计数器、optimizer m/v、EMA、LR schedule 位置）
- dataloader 在启动时重读数据目录 → 自动吸收新 ep
- norm_stats 从数据目录 asset_id=repo_id 绝对路径读取 → 与新分布对齐
- LR schedule 是 step 的函数 → 跨 resume 连续

### 7.3 启动方式

```bash
# 1. 初始训练 (--resume 语义安全，无 ckpt 时 fallback 到 weight_loader init)
nohup bash train_scripts/launch/run_<xxx>.sh > /tmp/train_<xxx>.log 2>&1 &

# 2. 启动 watcher (一次性)
nohup /tmp/dynamic_dataset_train.sh > /tmp/dyn_train.log 2>&1 &
```

### 7.4 参数（脚本顶部）

```bash
POLL_SEC=30                  # 检查间隔
MIN_NEW_EPS=3                # 至少 3 新 ep 才触发
MIN_REBUILD_INTERVAL=900     # 两次 rebuild 间 ≥15 min
VAL_SIZE=21                  # 每源 val ep 数 (÷3)
```

### 7.5 失败模式

1. **build 失败** → 脚本记 `[ERROR]`，训练保持停止（不自动启 错误 build）
2. **norm_stats 失败** → 同上
3. **watcher 崩溃** → 训练继续，人工重启 watcher 即可
4. **空 ckpt 触发** → `--resume` 优雅 fallback 到 weight_loader（openpi `checkpoints.py:62-64`）

**详细设计文档**：`docs/training/dynamic_dataset_workflow.md`

---

## 8. Checkpoint 结构与传输

### 8.1 Ckpt 目录内容

| 子目录/文件 | 大小 | 用途 | 部署需要？ |
|---|---|---|---|
| `params/` | ~12 GB | **EMA 参数**（openpi 保存 EMA 而非 live 当 `ema_decay!=None`）| ✅ |
| `train_state/` | ~31 GB | 完整 train state (live params + optimizer + EMA) | ❌（只 resume 用）|
| `assets/` | 0 | 预留 CallbackHandler | ✅（空目录也保留）|
| `_CHECKPOINT_METADATA` | <1 KB | orbax metadata | ✅ |

**重要**：openpi 的 `params/` 保存的是 **EMA** 权重（见 `checkpoints.py:_split_params`）。inline_eval 和部署都用这份。对短训（<5k steps），EMA 可能被 init 权重稀释 → `ema_decay=0.999` 比 `0.9999` 更合适。

### 8.2 跨机传输（gf1 → sim01）

**不用 scp**（环境中慢），走 TOS：

```bash
# 1. gf1 侧：打包 + 上传
cd /vePFS/.../checkpoints/<config>/<exp>/<step>
tar -cf /vePFS/tim/workspace/deepdive_kai0_tmp/data/<name>.tar \
    params _CHECKPOINT_METADATA assets

VOLC_TOS_AK=... VOLC_TOS_SK=... \
python train_scripts/data/to_tos_file.py \
    --file <name>.tar \
    --object_key kai0/checkpoints/<name>.tar \
    --task_num 16 --part_size_mb 64

# 2. sim01 侧：下载 + 解压
cd /data1/DATA_IMP/KAI0/ckpt_downloads
unset all_proxy ALL_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY socks_proxy
python3 /data1/DATA_IMP/KAI0/from_tos_file.py \
    --object_key kai0/checkpoints/<name>.tar \
    --file <name>.tar

cd /data1/tim/workspace/deepdive_kai0/kai0/checkpoints/<config>/<exp>/
tar -xf /data1/DATA_IMP/KAI0/ckpt_downloads/<name>.tar
mkdir -p <step> && mv params assets _CHECKPOINT_METADATA <step>/
```

### 8.3 TOS 挂载提示

gf 机器挂载 `/transfer-shanghai/`，但**只 root 可写**（用户 tim 读可以写不行）。上传必须用 `to_tos_file.py` API。sim01 没有挂载，读写都走 API。

---

## 9. 真机部署与测试

### 9.1 部署 ckpt 所需

- **sim01 config.py** 里有对应 config name
- **sim01 ckpt 目录** 含 `params/` + `assets/` + `_CHECKPOINT_METADATA`
- **norm_stats.json** 在数据目录（asset_id 解析为绝对路径）
- 对应 `kai0/data/<dataset>/base/norm_stats.json` 存在

### 9.2 启动真机

```bash
# start_autonomy.sh 负责:
# 1. 清理残留进程
# 2. USB 相机 reset
# 3. CAN bus 激活
# 4. colcon build (如需)
# 5. 启动 ros2 launch (相机+双臂+policy 节点)
#
# 标准调用
./start_scripts/start_autonomy.sh --execute \
    config_name:=<config_name> \
    checkpoint_dir:=/data1/.../checkpoints/<config>/<exp>/<step> \
    prompt:='<任务 prompt>'
```

**参数**：
- `--execute` / `execute_mode:=true`：机械臂实际动作（不加则 OBSERVE 模式）
- `config_name`：决定模型架构 + data_transforms
- `checkpoint_dir`：要加载的 step 子目录
- `prompt`：传给 policy 的文本 prompt

### 9.3 策略推理节点参数（运行时可调）

`policy_inference_node.py` 内参数（ros2 param set 动态修改）：

| 参数 | 默认 | 作用 |
|---|---|---|
| `execute_mode` | false | 机械臂是否执行 |
| `chunk_size` | 50 | 每次推理生成的动作序列长度 |
| `inference_rate` | 3.0 Hz | 策略查询频率 |
| `latency_k` | 8 | 新 chunk 替换旧 chunk 前 N 步 |
| `min_smooth_steps` | 8 | 新旧 chunk 混合窗口最小 |
| `decay_alpha` | 0.25 | 混合衰减系数 |
| `enable_rtc` | true | RTC (Real-Time Chunking) 开关 |
| `rtc_execute_horizon` | 16 | RTC guidance window |
| `rtc_max_guidance_weight` | 0.5 | RTC guidance 权重上限 |

**用 `rtc_apply.sh` 快捷切换**：

```bash
./start_scripts/rtc_apply.sh show      # 看当前
./start_scripts/rtc_apply.sh off       # RTC OFF (纯 smoothing)
./start_scripts/rtc_apply.sh on        # 默认 RTC ON
./start_scripts/rtc_apply.sh rtc_tight # 高频 replan (每 3 步) + 短 guidance
./start_scripts/rtc_apply.sh rtc5      # 每 5 步 replan
./start_scripts/rtc_apply.sh rtc_long  # 全 50 步 guidance (A/B 对照)
```

### 9.4 离线评估（不真机）

```bash
# 单 ckpt 算 MAE@1/10/25/50
cd kai0
.venv/bin/python ../train_scripts/eval/eval_val_action_mse.py \
    --config <config_name> \
    --ckpt checkpoints/<config>/<exp>/<step> \
    --val /data1/.../data/<dataset>/val \
    --n-sample-frames 200

# N-sample ensemble (降方差)
... --flow-samples 8
```

输出 `<ckpt>/eval_val.json`。

---

## 10. 监测与过拟合检测

### 10.1 Monitor 事件流

Claude Code 用 Monitor 工具监听 log 文件：

```bash
# MAE 事件 (inline-eval 触发时)
ssh gf1 'tail -F /tmp/train_<xxx>.log' | \
  grep --line-buffered -E "inline-eval.*MAE|Traceback|\[train\] ==="

# Step 进度 (每 100 步)
grep --line-buffered -E "^Step [0-9]+:"

# 错误信号
grep --line-buffered -E "Traceback|OOM|Killed|FileExistsError"
```

### 10.2 Overfit 检测脚本

`overfit_watcher.py` 跟踪 MAE@1 历史最低，超过 +2% 回升即触发 `[OVERFIT]`：

```bash
# 部署到 gf 机器
scp overfit_watcher.py gf1:/tmp/
ssh gf1 "nohup python3 /tmp/overfit_watcher.py > /tmp/overfit_watcher.log 2>&1 &"

# 监听
tail -F /tmp/overfit_watcher.log | grep -E "new best|OVERFIT"
```

### 10.3 典型过拟合模式

1. **Val MAE 反弹** — train loss 继续降但 val 不降（+1-3% 即警示）
2. **train/val gap 扩大** — 比如 0.0009 vs 0.0219 (24×)
3. **Gradient norm 继续降但 val 不动** — 已在过拟合 minimum 震荡

详见 `docs/training/task_p_unfreeze_8k_20k_analysis.md`。

---

## 11. 当前实验汇总

### 11.1 Task_A mixed (gf0, 13k steps)

| step | MAE@1 | MAE@10 | MAE@25 | MAE@50 |
|---|---|---|---|---|
| 1000 | 0.0153 | 0.0352 | 0.0647 | 0.1020 |
| 4000 | 0.0134 | 0.0306 | 0.0540 | 0.0816 |
| 9000 | **0.0129** ⭐ | 0.0296 | 0.0521 | 0.0786 |
| 12999 | **0.0129** | 0.0296 | 0.0520 | 0.0785 |

**Config**：`pi05_flatten_fold_mixed_gf0` (519 train = 173×3, val 30)
**时长**：11h 24m
**状态**：完成，tar 已打包
- `/vePFS/tim/workspace/deepdive_kai0_tmp/data/mixed_gf0_best_at_4k.tar`
- `/vePFS/tim/workspace/deepdive_kai0_tmp/data/mixed_gf0_step12999_final.tar`

### 11.2 Task_A visrobot01-only (gf1, 12k 计划，已跑 ~6k)

| step | MAE@1 | vs mixed 同 step |
|---|---|---|
| 1000 | 0.0241 | +58% |
| 2000 | 0.0203 | +45% |
| 4000 | 0.0185 | +38% |
| 6000 | 0.0181 | +37% |

**Config**：`pi05_flatten_fold_visrobot01_only` (193 train, 17 val)
**结论**：visrobot01-only 比 mixed 差 ~40%，**mixed 胜出**
- tar 已 pack：`/vePFS/tim/workspace/deepdive_kai0_tmp/data/visrobot01_only_best_step6000.tar`

### 11.3 Task_P unfreeze_8k / 20k (历史完成)

- 最佳：20k run @ step 4000 MAE@1=0.0195
- 详细分析：`docs/training/task_p_unfreeze_8k_20k_analysis.md`

### 11.4 Task_E unfreeze_2k (已部署)

- MAE@1=0.0396 (EMA 稀释)
- 真机失败模式：抓取瞬间偏

---

## 12. 常见坑与规则

### 12.1 训练

1. **永远不用 `--overwrite`** — 会 rmtree 整个 exp 目录（删所有 step ckpt）
2. **`--resume` 在空 ckpt 时 fallback 到 `weight_loader`** — 安全无害
3. **`decay_steps >= num_train_steps`** — 否则 LR 提早到底 decay_lr 恒定
4. **`ema_decay=0.999`** 比 `0.9999` 适合短训（< 5k steps）
5. **`save_interval=500~1000`** 给 10+ eval 点，看曲线选 best
6. **`keep_period <= save_interval`** 确保每个 eval 对应的 ckpt 都保留
7. **不同实验用不同 `--exp_name`**，不要混在一个 exp 下

### 12.2 数据

1. **build 后必须 gen stats + compute norm** — 缺任一 dataloader 会炸
2. **cam 命名**：训练 loader 要 `observation.images.<cam>/`，raw 是 `<cam>/`，build 时 symlink 转名
3. **episodes.jsonl 格式**：v2.1 要 `episode_index`，老脚本可能用 `episode_id` — build 时统一
4. **完整性过滤**：raw visrobot01 可能 parquet 早于 video，build 要过滤不齐的 ep
5. **`info.json total_episodes` 可能与 `episodes.jsonl` 行数不一致** — build 要 reconcile

### 12.3 部署

1. **sim01 config.py 必须有对应 config name**（与 gf 上一致）
2. **ckpt 目录必须有 `assets/` 子目录**（即便空）— orbax 校验
3. **`norm_stats.json` 不在 ckpt 里，在数据目录**（asset_id 解析）
4. **RTC 默认 ON** — 做 A/B 测试记得切 off

### 12.4 传输

1. **TOS 上传慢（2-3 MB/s）** — 不要重复打包上传
2. **sim01 下载快（100 MB/s）** — 瓶颈在上传
3. **`to_tos_file.py` 需 `VOLC_TOS_AK/SK` env var**（gf 版本 hardcoded 已移除）
4. **sim01 `from_tos_file.py` 可能被 SOCKS_PROXY 干扰** — 跑前 `unset *proxy*`

---

## 13. 命令速查

### 13.1 SSH

```bash
alias gf0="ssh -p 55555 -R 29290:localhost:29290 tim@14.103.44.161"
alias gf1="ssh -p 11111 -R 29290:localhost:29290 tim@14.103.44.161"
```

### 13.2 常见操作

```bash
# 看训练进度
ssh gf1 "grep 'Progress on' /tmp/train_<xxx>.log | tail -1"
ssh gf1 "grep 'inline-eval' /tmp/train_<xxx>.log"

# GPU 状态
ssh gf1 "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | head -8"

# 算当前 visrobot01 完整 ep
ssh gf1 'bash -c "..."'  # (用 docs/training/dynamic_dataset_workflow.md 里的脚本)

# 启动训练
ssh gf1 "nohup bash /vePFS/.../train_scripts/launch/run_<xxx>.sh > /tmp/train_<xxx>.log 2>&1 & echo pid=\$!"

# kill 训练
ssh gf1 "pkill -f 'train.py <config_name>'"

# ckpt tar + 上传
ssh gf1 "cd <ckpt>/<step> && tar -cf /tmp/x.tar params _CHECKPOINT_METADATA assets"
ssh gf1 "VOLC_TOS_AK=... VOLC_TOS_SK=... python to_tos_file.py --file /tmp/x.tar --object_key kai0/checkpoints/x.tar"

# sim01 下载
unset all_proxy; python3 /data1/DATA_IMP/KAI0/from_tos_file.py --object_key kai0/checkpoints/x.tar --file x.tar
```

### 13.3 真机测试切换

```bash
# 修改 start_autonomy_temp.sh 里的 config_name + checkpoint_dir，然后
./start_scripts/start_autonomy_temp.sh

# 第二终端 RTC 切换
source ros2_ws/install/setup.bash
./start_scripts/rtc_apply.sh rtc5
```

---

## 参考

- `docs/training/dynamic_dataset_workflow.md` — 动态数据集方案
- `docs/training/task_p_unfreeze_8k_20k_analysis.md` — Task_P 过拟合分析
- `docs/training/task_a_master_plan.md` — Task_A 训练路线
- `docs/training/training_cli_notes.md` — 命令注记
- `docs/deployment/sim01_deployment.md` — sim01 部署细节
- kai0 官方论文 Model Arithmetic / Stage Advantage / Train-Deploy Alignment 三模块

---

_Document version: 1.0 (2026-04-25)_
_Authors: tim (user) + Claude Code_
