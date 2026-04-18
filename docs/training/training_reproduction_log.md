# kai0 Task A 训练复现日志

> 日期: 2026-03-28
> 目标: 在 gf0/gf1 (各 8×A100 80GB) 上复现 kai0 Task A (T恤展平&折叠) 全流程训练

---

## 一、基础设施

### 1.1 机器配置

| 机器 | 规格 | 角色 | 登录方式 |
|------|------|------|---------|
| **sim01** | 2×RTX 5090 32GB | 工控机+推理 | tim@sim01 (本机) |
| **gf0** | 8×A100 80GB | 主训练 | `ssh -p 55555 tim@14.103.44.161` |
| **gf1** | 8×A100 80GB | 并行训练 | `ssh -p 11111 tim@14.103.44.161` |

### 1.2 存储架构

```
gf0/gf1 容器环境:
  /home/tim/          ← overlay 40GB (gf0 原仅 11GB free, miniconda 占 30GB)
  /vePFS/tim/         ← 22TB 共享 PFS, gf0/gf1 都能访问 (关键!)

解决方案:
  1. miniconda 迁移到 vePFS → home 释放 ~30GB
  2. kai0 repo + data + checkpoints 放 vePFS (共享, 下载一次)
  3. .venv 放本地盘 (uv 硬链接不能跨文件系统)
  4. uv cache 迁移到 vePFS (共享加速 gf1 安装)

最终目录结构:
  /vePFS/tim/workspace/
  ├── deepdive_kai0/kai0/     ← 代码 + 数据 + checkpoints (共享)
  ├── lerobot/                 ← lerobot 依赖 (本地 clone)
  ├── dlimp/                   ← dlimp 依赖 (本地 clone)
  ├── miniconda3_gf0/          ← gf0 miniconda (迁移自 home)
  ├── miniconda3_gf1/          ← gf1 miniconda (迁移自 home)
  ├── uv_cache/                ← 共享 uv 缓存
  ├── hf_cache/                ← HuggingFace 缓存
  └── openpi_cache/            ← openpi 模型缓存 (pi0.5 base 下载位置)

  ~/workspace/deepdive_kai0 → /vePFS/tim/workspace/deepdive_kai0  (软链接)
  ~/.kai0_venv               → 本地 overlay (各机器独立)
  kai0/.venv                 → /home/tim/.kai0_venv (软链接)
  ~/.cache/uv                → /vePFS/tim/workspace/uv_cache (软链接)
```

### 1.3 网络与代理

gf0/gf1 在国内机房, 外网访问情况:

| 目标 | 直连 | 代理 (8888) | 最优方案 |
|------|------|------------|---------|
| **PyPI** | 200, 66KB/s | 可用 | **阿里云镜像直连** (3.4MB/s) |
| **GitHub** | 不通 | 200 | 代理 `http_proxy=http://127.0.0.1:8888` |
| **HuggingFace** | 不通 | 200, 15KB/s (极慢) | **ModelScope 直连** (~8MB/s) |
| **GCS** (gs://openpi-assets) | 不通 | 待测 | 代理 或 sim01 中转 |
| **sim01 ↔ gf0/gf1** | SSH 直连 | — | `scp -P 55555` (不走代理) |

**关键经验**: 不要给 `uv sync` / `pip install` 设 http_proxy, 用国内镜像直连才快。

---

## 二、环境搭建

### 2.1 步骤概览

```
Step 1: vePFS 工作区 + 软链接          (~1 min)
Step 2: miniconda 迁移到 vePFS         (~5 min, 30GB mv)
Step 3: uv 安装                        (~1 min)
Step 4: clone kai0 repo 到 vePFS       (~2 min)
Step 5: clone lerobot + dlimp 依赖     (~10 min, lerobot 大)
Step 6: 修改 pyproject.toml 本地依赖   (~1 min)
Step 7: uv sync (阿里云镜像)           (~10 min)
Step 8: 验证 JAX 8×A100                (~1 min)
```

### 2.2 详细命令

**Step 1-2: 工作区 + miniconda 迁移**

```bash
# gf0 上执行:
mkdir -p /vePFS/tim/workspace/deepdive_kai0
ln -sfn /vePFS/tim/workspace/deepdive_kai0 ~/workspace/deepdive_kai0

# 迁移 miniconda 释放 home 空间 (30GB → vePFS)
mv /home/tim/miniconda3 /vePFS/tim/workspace/miniconda3_gf0
ln -sfn /vePFS/tim/workspace/miniconda3_gf0 /home/tim/miniconda3
```

**Step 3-4: uv + clone**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH

cd /vePFS/tim/workspace/deepdive_kai0
export http_proxy=http://127.0.0.1:8888  # GitHub 需要代理
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/OpenDriveLab/kai0.git
```

**Step 5-6: 依赖 clone + pyproject.toml 修改**

kai0 的 `pyproject.toml` 中 lerobot 和 dlimp 是 git 依赖, 通过代理 clone 大 repo 不稳定。
解决: 在 sim01 (有直接外网) clone 后 scp 到 vePFS。

```bash
# sim01 上:
cd /tmp
GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout https://github.com/huggingface/lerobot.git
cd lerobot && git fetch origin 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
git checkout 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5

# scp 到 gf0 vePFS (不走代理!)
scp -P 55555 -r /tmp/lerobot/ tim@14.103.44.161:/vePFS/tim/workspace/lerobot

# dlimp 小得多, 直接在 gf0 clone:
# gf0:
export http_proxy=http://127.0.0.1:8888
git clone https://github.com/kvablack/dlimp.git /vePFS/tim/workspace/dlimp
cd /vePFS/tim/workspace/dlimp
git checkout ad72ce3a9b414db2185bc0b38461d4101a65477a
```

修改 `pyproject.toml`, 将 git URL 改为本地路径:

```toml
# 原始:
# lerobot = { git = "https://github.com/huggingface/lerobot", rev = "0cf8648..." }
# dlimp = { git = "https://github.com/kvablack/dlimp", rev = "ad72ce3..." }

# 修改为:
lerobot = { path = "/vePFS/tim/workspace/lerobot" }
dlimp = { path = "/vePFS/tim/workspace/dlimp" }
```

**Step 7: uv sync (关键: 不走代理, 用阿里云镜像)**

```bash
# gf0:
export PATH=$HOME/.local/bin:$PATH
unset http_proxy https_proxy  # 不走代理!

# .venv 必须在本地盘 (uv 硬链接不能跨文件系统)
mkdir -p /home/tim/.kai0_venv
ln -sfn /home/tim/.kai0_venv kai0/.venv

cd ~/workspace/deepdive_kai0/kai0
UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ GIT_LFS_SKIP_SMUDGE=1 uv sync
```

**Step 8: 验证**

```bash
uv run python -c "import jax; print('JAX devices:', jax.devices())"
# 输出: JAX devices: [CudaDevice(id=0), ..., CudaDevice(id=7)]
```

**gf1 环境**: vePFS 共享代码, 只需创建本地 .venv + 软链接 + uv sync:

```bash
# gf1:
mkdir -p /home/tim/.kai0_venv
ln -sfn /home/tim/.kai0_venv ~/workspace/deepdive_kai0/kai0/.venv
ln -sfn /vePFS/tim/workspace/uv_cache /home/tim/.cache/uv  # 共享缓存, 秒装
cd ~/workspace/deepdive_kai0/kai0
UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ uv sync  # ~1 min (缓存命中)
```

---

## 三、数据准备

### 3.1 数据源

kai0 数据托管在 HuggingFace (`OpenDriveLab-org/Kai0`) 和 ModelScope (`OpenDriveLab/Kai0`)。

**注意**: HF repo 的目录名是 `Task_A/Task_B/Task_C`, 但 `download_dataset.py` 脚本的 `--tasks` 参数用的是 `FlattenFold/TeeShirtSort/HangCloth` — **名称不匹配, 脚本有 bug**, 直接用 API 下载。

### 3.2 下载方式选择

| 方式 | 速度 | 可用性 | 推荐 |
|------|------|--------|------|
| HuggingFace (代理) | ~15KB/s | 可用但极慢 | 不推荐 |
| hf-mirror.com | ~640KB/s | 不稳定 | 备选 |
| **ModelScope CLI** | **~8MB/s** | **稳定** | **首选** |

### 3.3 下载命令

```bash
# gf0 上, 不走代理:
export PATH=$HOME/.local/bin:$PATH
unset http_proxy https_proxy

cd ~/workspace/deepdive_kai0/kai0

# 安装 modelscope
UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ uv pip install modelscope

# 下载 Task_A 数据 (base + advantage + dagger, ~115GB, ~3.5h)
uv run modelscope download \
  --dataset OpenDriveLab/Kai0 \
  --include 'Task_A/**' \
  --local_dir /vePFS/tim/workspace/deepdive_kai0/kai0/data

# 下载 Task_A checkpoint (~5.7GB, ~15min)
uv run modelscope download \
  --model OpenDriveLab/Kai0 \
  --include 'Task_A/**' \
  --local_dir /vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints
```

### 3.4 下载结果验证

```
data/Task_A/
├── base/       46GB  3,055 parquet + 视频  (基础演示)
├── advantage/  31GB  3,055 parquet + 6,427 mp4 (含优势标签)
├── dagger/     39GB  3,457 parquet + 10,371 mp4 (DAgger 纠正)
└── Task_A/     211MB (元数据)
总计: ~115GB

checkpoints/Task_A/
└── mixed_1/    5.7GB (kai0 best model: params + norm_stats + metadata)
```

数据量与论文描述一致:
- base: 3,055 episodes (~42h) ✅
- dagger: 3,457 episodes (~13h) ✅
- advantage: 3,055 episodes (base 的带标签版) ✅

### 3.5 config.py 路径配置

```python
# src/openpi/training/config.py

# Normal fine-tune:
TrainConfig(
    name="pi05_flatten_fold_normal",
    data=LerobotAgilexDataConfig(
        repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/base",
        ...
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"  # openpi pi0.5 预训练权重
    ),
    batch_size=256,
    num_train_steps=100_000,
)

# AWBC:
TrainConfig(
    name="pi05_flatten_fold_awbc",
    data=LerobotAgilexDataConfig(
        repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/advantage",
        base_config=DataConfig(prompt_from_task=True),  # prompt 从 task_index 决定
        ...
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    batch_size=256,
    num_train_steps=100_000,
)
```

### 3.6 Norm Stats 计算

```bash
# gf0:
export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_normal
```

输出 `data/Task_A/base/norm_stats.json`, 包含:
- `state`: mean/std/q01/q99 (14D, 双臂关节)
- `actions`: mean/std/q01/q99 (14D)

**用途**: 训练时将 state/action 归一化到统一尺度, 消除不同关节角度范围差异。

---

## 四、kai0 训练方案

### 4.1 完整 Pipeline

kai0 解决三类分布不一致性 (P_train, Q_model, P_test), 对应三个模块:

```
Phase 1: Model Arithmetic (解决 P_train 覆盖不足)
  ├── 全量 normal fine-tune → ckpt_normal
  ├── 数据切 N 份 → 分别训练 N 个子集模型
  └── 合并: greedy/inverse_loss/GD 等 6 种方法 → ckpt_merged

Phase 2: Stage Advantage (解决 Q_model 缺乏阶段感知)
  ├── 人工标注 stage_progress_gt (已在 advantage 数据中提供)
  ├── 训练 Advantage 估计器 (PyTorch, 基于 Pi0 架构)
  ├── 预测 advantage → 离散化为 binary task_index
  └── AWBC 训练 (prompt 编码 advantage: positive/negative)

Phase 3: Train-Deploy Alignment (解决 P_test 部署差距)
  ├── 数据增强: 时间缩放 + 空间镜像
  ├── DAgger: 策略在线执行 → 人工纠正 → 收集恢复行为
  └── 推理: 时序平滑 / 时序集成 / RTC
```

### 4.2 训练配置详情

| 配置名 | 框架 | 数据 | batch | steps | GPU 需求 |
|--------|------|------|-------|-------|---------|
| `pi05_flatten_fold_normal` | JAX | base (3055ep) | 256 | 100K | 1×A100 80GB |
| `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD` | PyTorch DDP | advantage (3055ep) | 16 (1卡) / 144 (8卡) | 100K | 1~8×A100 |
| `pi05_flatten_fold_awbc` | JAX | advantage (3055ep) | 256 | 100K | 1×A100 80GB |
| `pi05_rtc_flatten_fold_inference` | JAX | — | — | — | 推理专用 |

**关键训练参数** (所有 flatten_fold 配置共享):
- `fsdp_devices=1`: 单卡训练 (A100 80GB 足够)
- `keep_period=5000`: 每 5000 步保留一个 checkpoint
- `save_interval=1000`: 每 1000 步存一次 (只保留最近的)
- `num_workers=8`: 数据加载线程
- `lr_schedule=CosineDecaySchedule`: 余弦衰减学习率
- `optimizer=AdamW`: 默认 lr=2.5e-5
- `ema_decay=0.99`: 指数移动平均
- `seed=42`

### 4.3 训练命令

```bash
# Normal fine-tune (JAX, 单卡 A100):
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_flatten_fold_normal \
  --exp_name=normal_v1

# Advantage 估计器 (PyTorch, 多卡 DDP):
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD \
  --exp_name=advantage_v1 --save_interval 10000

# AWBC (JAX, 单卡 A100):
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_flatten_fold_awbc \
  --exp_name=awbc_v1
```

---

## 五、并行复现方案

### 5.1 gf0/gf1 详细训练计划

#### 甘特图

```
时间    │ 0h        5h        10h       15h       20h       25h       30h
────────┼─────────────────────────────────────────────────────────────────
        │
gf0     │ ███████████████████████████  Day 1: normal fine-tune (100K, ~10h)
        │                              ██ Day 2: Model Arithmetic (~1h)
        │                                ██████████████████████████  Day 2-3: AWBC (100K, ~10h)
        │
gf1     │ ██████████████████████████  Day 1: 子集模型×4 (每个20K, 串行~8h)
        │                              ██████████████  Day 2: Advantage 估计器 (DDP 8卡, ~5h)
        │                                              ██ Day 2: Advantage 预测+离散化 (~1h)
        │
sim01   │                                                              Day 3+: 部署推理
        │                                                              Day 3+: DAgger 采集
────────┼─────────────────────────────────────────────────────────────────
依赖    │ gf0 normal ─┐
        │ gf1 子集×4 ─┴→ Model Arithmetic ─→ AWBC
        │ gf1 估计器 ─→ (可选: 标注自己数据)
        │ AWBC ckpt ─→ sim01 部署
```

#### 详细步骤表

| # | 机器 | 任务 | 配置/命令 | 前置依赖 | 预计时长 | 输出 |
|---|------|------|----------|---------|---------|------|
| 1a | **gf0** | Normal fine-tune | `train.py pi05_flatten_fold_normal --exp_name=normal_v1` | norm stats ✅ | ~10h | `checkpoints/pi05_flatten_fold_normal/normal_v1/100000/` |
| 1b | **gf1** | 数据切分 | `split_data.py --split_num 4` | 数据 ✅ | ~5min | `data/Task_A/splits/split_{0,1,2,3}/` |
| 1c | **gf1** | 子集模型 ×4 | `train.py pi05_flatten_fold_normal --exp_name=subset_{i}` (修改 repo_id 指向 split_i) | 1b | ~8h (串行) | 4 个 checkpoint |
| 2a | **gf0** | 导出验证数据 | `dump_data.py --dataset pi05_flatten_fold_normal --output val.pkl` | 1a | ~5min | `flatfold_val.pkl` |
| 2b | **gf0** | Model Arithmetic | `arithmetic.py --optimize_method greedy --checkpoints 1a+1c` | 1a, 1c, 2a | ~1h | `checkpoints/merged_v1/` |
| 2c | **gf1** | Advantage 估计器 | `torchrun --nproc_per_node=8 train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD` | 数据 ✅ | ~5h | PyTorch ckpt |
| 2d | **gf1** | Advantage 预测 | `stage_advantage/annotation/eval.py Task-A KAI0` | 2c | ~1h | advantage 标签 |
| 2e | **gf1** | Advantage 离散化 | `discretize_advantage.py --threshold 30` | 2d | ~10min | task_index 标签 |
| 3a | **gf0** | AWBC 训练 | `train.py pi05_flatten_fold_awbc --exp_name=awbc_v1` | 2b (或直接用开源 advantage 数据) | ~10h | `checkpoints/.../awbc_v1/100000/` |
| 4a | **sim01** | 部署推理 | `serve_policy.py --policy.dir=awbc_v1/100000` | 3a | 持续 | 推理服务 :8000 |
| 4b | **sim01** | DAgger 采集 | `agilex_openpi_dagger_collect.py` | 4a + 真机 | 持续 | HDF5 episodes |
| 4c | **gf0** | 数据增强+重训 | merge + time_scaling + space_mirroring → train | 4b | ~12h | 新 checkpoint |

#### 快速路径 (跳过 Phase 1)

kai0 已开源 `mixed_1` (best model), 可以跳过 1a/1b/1c/2a/2b 直接做 AWBC:

| # | 机器 | 任务 | 前置依赖 | 预计时长 |
|---|------|------|---------|---------|
| Q1 | **gf0** | AWBC 训练 (用开源 advantage 数据 + mixed_1 做 weight_loader) | 数据 ✅, ckpt ✅ | ~10h |
| Q2 | **sim01** | 部署 + 验证 | Q1 | 即时 |

```bash
# 快速路径: 直接 AWBC
# 先修改 pi05_flatten_fold_awbc 的 weight_loader 指向 mixed_1:
# weight_loader=weight_loaders.CheckpointWeightLoader(
#   "/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/Task_A/mixed_1"
# )
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_flatten_fold_awbc --exp_name=awbc_from_mixed1
```

#### gf0/gf1 环境变量模板

```bash
# === gf0 训练启动模板 ===
export PATH=$HOME/.local/bin:$PATH
unset http_proxy https_proxy  # PyPI 不走代理
export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
cd ~/workspace/deepdive_kai0/kai0

# JAX 训练:
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py <config_name> --exp_name=<exp_name>

# === gf1 训练启动模板 (同上, 只是不同实验) ===
# PyTorch DDP:
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_pytorch.py <config_name> --exp_name=<exp_name>
```

### 5.2 为什么这样分

**核心瓶颈**: Model Arithmetic 需要多个子集模型 checkpoint

- 单机串行: normal 10h + 4 子集 8h = **18h**
- 双机并行: max(10h, 8h) = **10h** (节省 8h, 44% 加速)

**数据共享**: vePFS 上数据两台机器都能读, 无需复制

**独立 .venv**: 各机器本地盘, 避免 vePFS 上 I/O 冲突

### 5.3 具体执行计划

```bash
# === gf0: Normal fine-tune ===
export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_flatten_fold_normal \
  --exp_name=normal_v1

# === gf1: 数据切分 + 子集训练 ===
# 切分数据为 4 份:
uv run python model_arithmetic/split_data.py \
  --source_path /vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/base \
  --dst_path /vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/splits \
  --split_num 4

# 训练 4 个子集模型 (串行, 每个 ~20K steps):
for i in 0 1 2 3; do
  # 修改 config.py 的 repo_id 指向 splits/split_${i}
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py pi05_flatten_fold_normal \
    --exp_name=subset_${i}
done

# === gf0: Model Arithmetic (两台训练完成后) ===
# 导出验证数据:
uv run python model_arithmetic/dump_data.py \
  --dataset pi05_flatten_fold_normal \
  --output flatfold_val.pkl

# 合并 (greedy 方法, 论文中表现最好):
CUDA_VISIBLE_DEVICES=0 uv run python model_arithmetic/arithmetic.py \
  --config pi05_flatten_fold_normal \
  --data-path flatfold_val.pkl \
  --checkpoints ckpt_normal/100000 ckpt_subset0/20000 ckpt_subset1/20000 ckpt_subset2/20000 ckpt_subset3/20000 \
  --output /vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/merged_v1 \
  --optimize_method greedy \
  --use_gpu --gpu_ids "0"

# === gf1: Advantage 估计器 (并行) ===
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD \
  --exp_name=advantage_v1 --save_interval 10000

# === gf0: AWBC 训练 ===
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_flatten_fold_awbc \
  --exp_name=awbc_v1
```

### 5.4 捷径: 直接用开源 advantage 数据

kai0 已开源 `Task_A/advantage/` 数据 (含 relative_advantage, absolute_advantage, task_index 标签), 可以跳过 Phase 2 的估计器训练:

```
advantage/data 中每个 parquet 已包含:
  - task_index: 0=negative (83%), 1=positive (17%)
  - relative_advantage, absolute_advantage, absolute_value
  - progress_gt, stage_progress_gt

tasks.jsonl:
  {"task_index": 0, "task": "fold the cloth, Advantage: negative"}
  {"task_index": 1, "task": "fold the cloth, Advantage: positive"}
```

直接 AWBC 训练, 推理时用 `prompt="fold the cloth, Advantage: positive"`。

### 5.5 Phase 3: Train-Deploy Alignment 复现

Phase 3 不是纯训练 — 是**部署→采集→增强→重训**的迭代循环。

#### 5.5.1 数据增强 (无需真机)

kai0 提供两种数据增强, 可直接在 gf0/gf1 上对已有数据做:

**时间缩放** — 每 N 帧取一帧, 模拟更快的动作执行:

```bash
# gf0 或 gf1:
uv run python train_deploy_alignment/data_augment/time_scaling.py \
  --src_path /vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/base \
  --tgt_path /vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/base_ts2 \
  --repo_id base_time_scaled \
  --extraction_factor 2
```

**空间镜像** — 水平翻转视频 + 交换左右臂, 数据量翻倍:

```bash
uv run python train_deploy_alignment/data_augment/space_mirroring.py full \
  --src-path /vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/base \
  --mirror-path /vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/base_mirror \
  --merge-path /vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/base_merged \
  --repo-id base_merged
```

**合并 base + dagger + 增强数据**:

```bash
uv run python scripts/merge_lerobot.py \
  --src-paths \
    /vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/base_merged \
    /vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/dagger \
  --tgt-path /vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/all_merged \
  --repo-id all_merged
```

#### 5.5.2 DAgger 数据采集 (需真机, sim01)

```bash
# sim01 终端 1: 启动推理服务
CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_flatten_fold_normal \
  --policy.dir=checkpoints/Task_A/mixed_1 \
  --port=8000

# sim01 终端 2: DAgger 采集
conda activate kai0_inference
cd train_deploy_alignment/dagger/agilex
python agilex_openpi_dagger_collect.py \
  --host localhost --port 8000 \
  --ctrl_type joint --use_temporal_smoothing --chunk_size 50 \
  --dataset_name my_dagger_v1

# 键盘操作:
#   d     → 进入 DAgger 模式 (人工接管)
#   Space → 开始录制
#   s     → 保存 episode
#   r     → 恢复自动推理
```

#### 5.5.3 DAgger 数据转换 + 训练

```bash
# HDF5 → LeRobot 格式
cd train_deploy_alignment/data_augment/utils
export PYTHONPATH="${PYTHONPATH}:$(pwd)/mini_lerobot"
python convert_h5_lerobot.py \
  /path/to/my_dagger_v1 /path/to/output my_dagger_v1 \
  --prompt "fold the cloth" --max-workers 8

# 空间镜像 → 数据翻倍
python train_deploy_alignment/data_augment/space_mirroring.py full \
  --src-path /path/to/output --mirror-path /path/to/mirrored \
  --merge-path /path/to/merged --repo-id my_dagger_merged

# rsync 到 gf0 → 合并数据 → 重新训练
```

#### 5.5.4 推理模式 (部署时选择)

| 模式 | 命令 | 适用场景 |
|------|------|---------|
| 时序平滑 | `agilex_inference_openpi_temporal_smoothing.py` | **推荐**, 降低抖动 |
| 时序集成 | `agilex_inference_openpi_temporal_ensembling.py` | 多次推理取平均 |
| RTC | `agilex_inference_openpi_rtc.py` | 最先进, 利用执行前缀 |
| 同步 | `agilex_inference_openpi.py` | 最简单, 每步一次推理 |

推理参数:
```bash
python agilex_inference_openpi_temporal_smoothing.py \
  --host localhost --port 8000 \
  --ctrl_type joint --chunk_size 50 \
  --publish_rate 30 --inference_rate 3.0 \
  --use_temporal_smoothing \
  --latency_k 8 --min_smooth_steps 8 --exp_decay_alpha 0.25
```

#### 5.5.5 Phase 3 迭代流程图

```
                    ┌─────────────────────────┐
                    │  训练好的 checkpoint     │
                    │  (Phase 1/2 产出)        │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  sim01: serve_policy     │
              ┌─────│  + 时序平滑推理          │
              │     └───────────┬─────────────┘
              │                 │
              │     ┌───────────▼─────────────┐
              │     │  真机测试, 观察失败模式   │
     效果满意  │     │  成功率: __%             │
     → 完成   │     └───────────┬─────────────┘
              │                 │ 效果不够好
              │     ┌───────────▼─────────────┐
              │     │  DAgger 采集纠正数据     │
              │     │  ~50-200 episodes        │
              │     └───────────┬─────────────┘
              │                 │
              │     ┌───────────▼─────────────┐
              │     │  HDF5→LeRobot + 增强    │
              │     │  时间缩放 + 空间镜像     │
              │     └───────────┬─────────────┘
              │                 │ rsync 到 gf0
              │     ┌───────────▼─────────────┐
              │     │  gf0/gf1: 合并+重训     │
              │     │  → 新 checkpoint         │
              │     └───────────┬─────────────┘
              │                 │
              └─────────────────┘ (循环)
```

---

## 六、复现结果验证与对比方案

### 6.1 kai0 论文报告的 Task A 性能

**论文 (arXiv:2602.09021) 消融实验** (Task A: 展平&折叠):

| 配置 | 成功率趋势 | 吞吐量趋势 | 说明 |
|------|-----------|-----------|------|
| π₀.₅ Base | 基线 | 基线 | openpi 预训练模型直接微调 |
| +MA (Model Arithmetic) | ↑ | ↑ | 子集训练+合并, 覆盖更多分布 |
| +SA (Stage Advantage) | ↑↑ | ↑↑ | 吞吐量提升最大 |
| +TDA (Train-Deploy Alignment) | ↑ | ↑ (retry cost ↑) | 成功率提升但重试开销增加 |
| **Full χ₀ (MA+SA+TDA)** | **↑↑↑ (~250%)** | **↑↑↑** | **最优组合** |

**其他基线**: X-VLA, GO-1, UniVLA, OpenVLA 成功率接近 0% (论文原文: "negligible")

**关键指标**:
- 总体成功率提升: ~250% vs π₀.₅ baseline
- 训练资源: 20h 数据 + 88 A100 GPU-hours
- 鲁棒性验证: 24 小时连续运行, 任意初始状态

> 注: 论文中每个消融组合的精确百分比数值需参考 arXiv:2602.09021 Table 1。
> 代码仓库未包含自动化评估脚本, 成功率通过真机实验人工判定。

### 6.2 各阶段复现验证方案

#### Stage 0: 基础微调验证 (pi05_flatten_fold_normal)

**训练指标** (wandb 监控):

| 指标 | 预期范围 | 异常标志 |
|------|---------|---------|
| `train/loss` | 初始 ~0.5, 收敛 ~0.05-0.1 | 不下降或 NaN |
| `train/grad_norm` | < 10 | 持续 > 100 = 梯度爆炸 |
| learning rate | cosine decay from 2.5e-5 → 0 | — |
| 训练时长 | ~10h (100K steps, batch=256, 1×A100) | — |

**离线验证** (无需真机):

```bash
# 1. 用训练好的 checkpoint 启动推理服务
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_flatten_fold_normal \
  --policy.dir=checkpoints/normal_v1/100000 \
  --port=8000

# 2. 发送测试数据, 检查 action 输出是否合理
uv run python -c "
from openpi_client import websocket_client_policy
import numpy as np

policy = websocket_client_policy.WebsocketClientPolicy('localhost', 8000)
obs = {
    'state': np.zeros(14, dtype=np.float32),  # 14D 关节状态
    'images': {
        'top_head': np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
        'hand_left': np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
        'hand_right': np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
    },
    'prompt': 'fold the cloth',
}
result = policy.infer(obs)
actions = result['actions']
print(f'Action shape: {actions.shape}')  # 预期: (50, 14)
print(f'Action range: [{actions.min():.3f}, {actions.max():.3f}]')  # 预期: 合理关节角范围
print(f'Action std: {actions.std():.4f}')  # 预期: > 0 (非全零)
"
```

**与官方 checkpoint 对比**:

```bash
# 用官方 mixed_1 和自训练 checkpoint 分别推理, 对同一组输入比较:
# 1. 加载官方 checkpoint
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_flatten_fold_normal \
  --policy.dir=checkpoints/Task_A/mixed_1 \
  --port=8000

# 2. 收集 N 组 (state, images) 输入的 action 输出 → actions_official.npy

# 3. 切换到自训练 checkpoint, 同样输入 → actions_ours.npy

# 4. 对比:
# python -c "
# import numpy as np
# a_off = np.load('actions_official.npy')
# a_ours = np.load('actions_ours.npy')
# print(f'MSE: {((a_off - a_ours)**2).mean():.6f}')
# print(f'Cosine sim: {np.mean(np.sum(a_off*a_ours, axis=-1) / (np.linalg.norm(a_off,axis=-1)*np.linalg.norm(a_ours,axis=-1))):.4f}')
# "
```

#### Stage 1: Model Arithmetic 验证

**合并质量指标**:

```bash
# arithmetic.py 输出的 validation loss 是关键指标:
CUDA_VISIBLE_DEVICES=0 uv run python model_arithmetic/arithmetic.py \
  --config pi05_flatten_fold_normal \
  --data-path flatfold_val.pkl \
  --checkpoints ckpt1 ckpt2 ckpt3 ckpt4 \
  --output merged_ckpt \
  --optimize_method greedy \
  --use_gpu --gpu_ids "0"

# 输出会打印:
#   各 checkpoint 的 validation loss
#   最优权重组合
#   合并后的 validation loss (应 < 最好的单 checkpoint)
```

**预期**: 合并后 validation loss < 任何单个子集模型 (论文核心发现)

**关键发现 (论文)**: 用 DAgger 数据做 OOD 验证集效果远好于 in-domain 验证集

#### Stage 2: Stage Advantage (AWBC) 验证

**Advantage 估计器质量** (如果自己训练):

```bash
# 评估估计器预测准确性:
uv run python stage_advantage/annotation/eval.py Task-A KAI0 \
  /vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/advantage

# 输出: 预测的 absolute_advantage 与 stage_progress_gt 的相关性
# 预期: 高相关性 (advantage 正确反映进度)
```

**AWBC 训练验证**:

| 指标 | 预期 | 说明 |
|------|------|------|
| `train/loss` (positive prompt) | 低 | 模型学会了好的动作 |
| `train/loss` (negative prompt) | 较高 | 模型区分了好坏动作 |
| 推理时只用 positive prompt | 动作质量 > normal | AWBC 的核心价值 |

```bash
# AWBC 推理 (注意 prompt 包含 Advantage 标签):
# serve_policy.py 用 pi05_flatten_fold_awbc config
# 推理时 prompt = "fold the cloth, Advantage: positive"
```

#### Stage 3: Train-Deploy Alignment 验证

**真机成功率** (需要 Piper 机器人):

| 测试项 | 方法 | 预期 |
|--------|------|------|
| 基础折叠 | 标准 T 恤, 平整初始状态 | 高成功率 |
| 随机初始 | 皱褶/半折叠起始 | 需 DAgger 改善 |
| 连续运行 | 连续 N 次, 不人工重置 | 检验鲁棒性 |
| 时序平滑 vs 无平滑 | 对比抖动程度 | 平滑后明显更流畅 |

**推理延迟** (sim01 离线可测):

```bash
# 测量单次推理延迟:
uv run python -c "
import time
from openpi_client import websocket_client_policy
import numpy as np

policy = websocket_client_policy.WebsocketClientPolicy('localhost', 8000)
obs = {
    'state': np.zeros(14, dtype=np.float32),
    'images': {
        'top_head': np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
        'hand_left': np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
        'hand_right': np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
    },
    'prompt': 'fold the cloth',
}
# warmup
for _ in range(3): policy.infer(obs)

# benchmark
times = []
for _ in range(20):
    t0 = time.time()
    policy.infer(obs)
    times.append(time.time() - t0)

print(f'Inference latency: {np.mean(times)*1000:.1f}ms ± {np.std(times)*1000:.1f}ms')
print(f'Throughput: {1/np.mean(times):.1f} Hz')
# 预期: 5090 上 <200ms, >5Hz (kai0 原版 4090 ~4Hz)
"
```

### 6.3 复现结果记录模板

每个阶段训练完成后, 填写以下表格:

```markdown
## 复现结果 (待填写)

### Normal Fine-tune
| 指标 | 官方 | 复现 | 差异 |
|------|------|------|------|
| 最终 train loss | — | — | — |
| 训练时长 (100K steps) | ~10h (1×A100) | — | — |
| checkpoint 大小 | — | — | — |
| 推理 action shape | (50, 14) | — | — |

### Model Arithmetic
| 指标 | 官方 | 复现 | 差异 |
|------|------|------|------|
| 最佳合并方法 | greedy | — | — |
| 合并后 val loss | — | — | — |
| val loss 降幅 (vs 单模型) | — | — | — |

### AWBC
| 指标 | 官方 | 复现 | 差异 |
|------|------|------|------|
| 最终 train loss | — | — | — |
| positive vs negative loss 差 | — | — | — |
| 推理用 positive prompt | ✅ | — | — |

### 真机测试 (需 Piper 机器人)
| 指标 | 官方 | 复现 | 差异 |
|------|------|------|------|
| 标准折叠成功率 (N=20) | ~250% vs base | — | — |
| 推理延迟 | ~250ms (4090) | — | — |
| 24h 连续运行 | ✅ | — | — |
```

### 6.4 快速验证路径 (不训练)

如果只想验证 pipeline 能跑通, 不需要从头训练:

```
1. 用官方 mixed_1 checkpoint 直接推理 → 验证 serve_policy + 推理链路
2. 用官方 advantage 数据 + mixed_1 做 AWBC 训练 (短: 10K steps) → 验证训练能跑
3. 真机测试 mixed_1 → 验证部署链路
```

```bash
# 快速验证: 10K steps AWBC (约 1 小时)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_flatten_fold_awbc \
  --exp_name=awbc_quick_test \
  --num_train_steps=10000
```

---

## 七、遇到的问题与解决 (Troubleshooting)

### 6.1 存储空间不足

**问题**: gf0/gf1 的 `$HOME` 是 40GB overlay, miniconda 占 30GB, 仅剩 11GB。uv cache + .venv 需要 ~12GB, 空间不够。

**解决**:
1. miniconda 迁移到 vePFS: `mv ~/miniconda3 /vePFS/tim/workspace/miniconda3_gf0 && ln -sfn ...`
2. uv cache 迁移到 vePFS: `mv ~/.cache/uv /vePFS/tim/workspace/uv_cache && ln -sfn ...`
3. 释放 ~34GB, 足够 .venv (~8GB)

### 6.2 uv sync 无法跨文件系统硬链接

**问题**: 将 `.venv` 放在 vePFS 上, uv 从本地 cache 创建硬链接到 vePFS 失败 (跨文件系统), .venv 始终为空 (92K)。

**解决**: `.venv` 必须在本地盘:
```bash
mkdir -p /home/tim/.kai0_venv
ln -sfn /home/tim/.kai0_venv kai0/.venv
```

### 6.3 lerobot git 依赖下载失败

**问题**: `uv sync` 需要从 GitHub clone lerobot (大 repo), 通过 8888 代理 git fetch 报 TLS 错误:
```
error: RPC failed; curl 56 GnuTLS recv error (-9): Error decoding the received TLS packet.
fetch-pack: unexpected disconnect while reading sideband packet
```

**解决**: 在 sim01 (有直接外网) clone 后 scp 到 gf0 vePFS, 然后修改 pyproject.toml 为本地路径:
```toml
lerobot = { path = "/vePFS/tim/workspace/lerobot" }
dlimp = { path = "/vePFS/tim/workspace/dlimp" }
```

### 6.4 uv sync 通过代理下载 PyPI 极慢

**问题**: `.bashrc` 设了 `http_proxy=http://127.0.0.1:8888`, uv sync 下载 PyPI 包走代理, 速度仅 ~66KB/s。JAX + CUDA 等大包 (~5GB) 需要数小时。

**发现**: PyPI (pypi.org, files.pythonhosted.org) 可以直连! 阿里云镜像更快 (3.4MB/s)。

**解决**:
```bash
unset http_proxy https_proxy
UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ uv sync
```
从数小时 → **~10 分钟**。

### 6.5 HuggingFace 下载极慢

**问题**: 通过 8888 代理下载 HuggingFace 数据, 速度仅 ~15KB/s。Task_A 数据 115GB 需要数天。

**解决**: 改用 ModelScope (国内镜像) 直连, ~8MB/s:
```bash
unset http_proxy https_proxy
uv run modelscope download --dataset OpenDriveLab/Kai0 --include 'Task_A/**' --local_dir ./data
```
从数天 → **~3.5 小时**。

### 6.6 download_dataset.py 任务名不匹配

**问题**: 脚本 `--tasks` 参数接受 `FlattenFold`, 但 HF repo 实际目录名是 `Task_A`。下载只得到 README.md。

**解决**: 不用脚本, 直接调用 `modelscope download` 或 `snapshot_download` 并指定正确的 `allow_patterns=["Task_A/**"]`。

### 6.7 DataConfig episodes 字段重复

**问题**: `config.py` 中 `DataConfig` 类定义了两次 `episodes` 字段 (line 94 和 100), 导致 `dataclasses.replace()` 报错:
```
TypeError: DataConfig.__init__() got an unexpected keyword argument 'episodes'
```

**解决**: 删除重复定义, 保留一个。

### 6.8 缺少 chex 模块

**问题**: `compute_norm_states_fast.py` 依赖 `chex` (通过 `fsq_tokenizer.py`), 但 `pyproject.toml` 未声明。

**解决**: `uv pip install chex`。注意: chex 会升级 numpy 到 2.x, 需确认不影响其他依赖。

---

## 八、关键文件索引

| 文件 | 用途 |
|------|------|
| `src/openpi/training/config.py` | 所有训练配置 (修改路径在这里) |
| `scripts/train.py` | JAX 训练入口 (normal, AWBC) |
| `scripts/train_pytorch.py` | PyTorch DDP 训练入口 (Advantage 估计器) |
| `scripts/compute_norm_states_fast.py` | 归一化统计计算 |
| `scripts/serve_policy.py` | 推理服务器 |
| `model_arithmetic/split_data.py` | 数据切分 (用于 MA) |
| `model_arithmetic/arithmetic.py` | JAX checkpoint 合并 |
| `model_arithmetic/dump_data.py` | 导出 MA 验证数据 |
| `stage_advantage/annotation/eval.py` | Advantage 预测 |
| `stage_advantage/annotation/discretize_advantage.py` | Advantage 离散化 |

---

## 九、代码差异审计 (2026-03-28)

> 对比 `origin/main` (commit `9d93078`) 与本地工作副本，排查可能影响复现的不一致。

### 9.1 差异总览

本地共有 5 个文件被修改 (`git diff --stat`)：

| 文件 | 变更类型 | 风险等级 | 说明 |
|------|---------|---------|------|
| `src/openpi/training/config.py` | 路径替换 + 新增字段 | **高** | weight_loader 指向 mixed_1 而非 π₀.₅ base; episodes 字段重复 |
| `packages/openpi-client/.../websocket_client_policy.py` | 超时参数 | 无 | ping/close/open_timeout=300 |
| `src/openpi/serving/websocket_policy_server.py` | 超时参数 | 无 | ping/close_timeout=300 |
| `pyproject.toml` | 依赖覆盖 | 低-中 | 新增 av==13.1.0, mujoco>=3.0.0 |
| `uv.lock` | 依赖锁更新 | 低 | 4607 行变更, 核心训练依赖版本不变 |

### 9.2 详细分析

#### 差异 #1: `DataConfig` 基类 `episodes` 字段重复定义 — 风险: 中

**文件:** `src/openpi/training/config.py` line 93-100

原版 `DataConfig` 基类无 `episodes` 字段（仅在子类 `LerobotAgilexDataConfig`/`LerobotARXDataConfig` 中定义）。
本地在基类中插入了 **两次** 相同定义：

```python
# Line 93-94 (新增 #1)
episodes: list[int] | None = None

prompt_from_task: bool = False   # 原有字段

# Line 99-100 (新增 #2, 与 #1 重复)
episodes: list[int] | None = None
```

**影响:** Python dataclass 静默接受重复字段（后者覆盖前者），运行时不报错。但：
- 改变了 `DataConfig` 字段顺序
- 子类已有 `episodes`，基类重复定义是冗余的
- 可能影响序列化/反序列化行为

**修复:** 删除基类中的两个 `episodes` 定义，保留子类中的原始定义。

#### 差异 #2: `weight_loader` 指向 `mixed_1` 而非 π₀.₅ base — 风险: **高**

**文件:** `src/openpi/training/config.py` line 1190

```python
# 原版 (占位符):
weight_loader=weight_loaders.CheckpointWeightLoader("<path/to/pi05_base/checkpoint>")

# 本地:
weight_loader=weight_loaders.CheckpointWeightLoader(
    "/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/Task_A/mixed_1"
)
```

README 明确说明 `pi05_flatten_fold_normal` 用于 **正常 π₀.₅ 微调** (Step 3)，weight_loader 应为 π₀.₅ base checkpoint。
`mixed_1` 是 Model Arithmetic 的输出（22GB），用它作为起点会导致训练轨迹完全不同于论文。

**训练复现日志 §3.5 中的计划配置也确认应使用:**
```
gs://openpi-assets/checkpoints/pi05_base/params
```

**本地可用的 π₀.₅ base checkpoint:**
- `openpi_cache/pi05_base_gsutil/params/` (3.3GB, gsutil 下载)
- `openpi_cache/pi05_base_direct/` (609MB, 可能不完整)
- `openpi_cache/openpi-assets/checkpoints/pi05_base/params.partial/` (1.8GB, 未完成下载)

**修复:** sim01 上改为 `openpi_cache/pi05_base_gsutil/params/`；gf0/gf1 上使用 `gs://openpi-assets/checkpoints/pi05_base/params`（openpi 自动缓存到 `$OPENPI_DATA_HOME`）。

#### 差异 #3: 数据路径 `FlattenFold` → `Task_A` — 风险: 无

```python
# 原版: repo_id="<path_to_repo_root>/data/FlattenFold/base"
# 本地: repo_id="/data1/tim/workspace/deepdive_kai0/kai0/data/Task_A/base"
```

**正确修改。** HuggingFace 数据集实际目录为 `Task_A`，README 也指导使用此名称。原版 `FlattenFold` 是旧名称占位符。

#### 差异 #4: WebSocket 超时参数 — 风险: 无

客户端和服务端新增 `ping_timeout=300, close_timeout=300, open_timeout=300`。
仅影响推理通信稳定性，不影响训练。保留不动。

#### 差异 #5: `pyproject.toml` 新增依赖覆盖 — 风险: 低-中

```toml
# 原版:
override-dependencies = ["ml-dtypes==0.4.1", "tensorstore==0.1.74"]
# 本地:
override-dependencies = ["ml-dtypes==0.4.1", "tensorstore==0.1.74", "av==13.1.0", "mujoco>=3.0.0"]
```

- `av` 14.4.0 → 13.1.0 (降级): PyAV 用于视频解码，降级可能有兼容性原因
- `mujoco` 2.3.7 → 3.6.0 (大版本升级): kai0 训练不直接依赖 MuJoCo

**核心训练依赖版本未变** (已验证)：
JAX 0.5.3, Flax 0.10.2, PyTorch 2.7.1, Transformers 4.53.2, LeRobot 0.1.0, numpy 1.26.4, orbax-checkpoint 0.11.13, optax 0.2.4, websockets 15.0.1

暂不修改，但需关注 `av` 降级对视频数据加载的影响。

#### 差异 #6: AWBC 配置中的内部路径未替换 — 风险: 高 (仅 AWBC 阶段)

**文件:** `src/openpi/training/config.py` line 1342, 1374

```python
weight_loader=weight_loaders.CheckpointWeightLoader("/cpfs01/shared/checkpoint/pi05_base/params")
```

这是 kai0 作者内部集群路径。AWBC 阶段需替换为本地 π₀.₅ base checkpoint 路径。
当前只跑 normal 微调，AWBC 后续再修复。

### 9.3 未跟踪文件 (新增，未提交)

```
install/                                        ← 安装脚本
log/                                            ← 训练日志
train_deploy_alignment/dagger/agilex/*_ros2.py  ← ROS2 适配
train_deploy_alignment/inference/agilex/*_ros2.py ← ROS2 推理适配
```

这些是本地新增文件，不影响原版代码行为。

### 9.4 修复记录

| # | 修复项 | 操作 | 验证 |
|---|--------|------|------|
| 1 | `DataConfig.episodes` 重复 | 删除基类中的两处 `episodes` 定义 (line 93-94, 99-100)，子类定义不变 | `grep -n episodes config.py` 仅显示子类 (L373, L471) |
| 2 | `weight_loader` 指向 mixed_1 | 改为 `gs://openpi-assets/checkpoints/pi05_base/params`，openpi 自动下载并缓存到 `$OPENPI_DATA_HOME` | `git diff config.py` 仅剩 repo_id + weight_loader 两处路径替换 |
| 3 | AWBC 内部路径 `/cpfs01/...` | 暂未修复，当前只跑 normal，AWBC 阶段再处理 | — |

**修复后 `config.py` diff 与原版的唯一差异:**
```diff
- repo_id="<path_to_repo_root>/data/FlattenFold/base"
+ repo_id="/data1/tim/workspace/deepdive_kai0/kai0/data/Task_A/base"

- weight_loader=weight_loaders.CheckpointWeightLoader("<path/to/pi05_base/checkpoint>")
+ weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")
```

**后续 action items:**
1. norm_stats 可能需要重新计算（如果之前是用 mixed_1 权重算的，norm_stats 本身基于数据而非模型权重，可能不受影响，但建议确认）
2. gf0/gf1 上的 config.py 也需同步修复（repo_id 改为 `/vePFS/...` 路径）
3. 训练启动前确认 `$OPENPI_DATA_HOME` 已正确设置，确保 π₀.₅ base 能自动下载

---

## 十、当前状态 (更新于 2026-03-29)

### 代码审计与修复

| 项目 | 状态 | 备注 |
|------|------|------|
| kai0 原版代码审计 | ✅ | §9: config.py 差异 6 项, 修复 2 项 |
| IPC/推理服务审计 | ✅ | §11: ROS2 迁移 6 项差异, 修复 3 项 (DAgger drift + QoS) |
| 推理管线端到端审计 | ✅ | §12: prompt 三方不一致 (已记录), 其余一致 |
| policy_inference_node.py | ✅ **已修复+验证** | §13: 7 项对齐修复, 5 项测试全部通过 |
| sim01 venv | ✅ **已升级** | Python 3.11 → 3.12, 与 ROS2 Jazzy 统一 |
| inference_full_launch.py | ✅ **已修复** | topic/ckpt/prompt 对齐, camera_f namespace 修复 |

### 验证结果

| 测试 | 输入 | 结果 |
|------|------|------|
| Test 1 合成图像管线 | 随机 640x480 | **PASS** — 像素一致 |
| Test 2 合成模型推理 | 固定 RNG | **PASS** — EXACT MATCH |
| Test 3 真实相机单帧 | 3× RealSense + Piper | **PASS** — EXACT MATCH |
| Test 4 实时连续推理 | mode=both 10s | **PASS** — 6.2Hz, p50=101ms |
| Test 5 双模式独立对比 | ros2 vs websocket 各 10s | **PASS** — max diff 0.50° |

### 训练与部署

| 项目 | 状态 | 备注 |
|------|------|------|
| gf0 环境 | ✅ | JAX 8×A100, uv sync 完成 |
| gf1 环境 | ✅ | JAX 8×A100, uv sync 完成 |
| Task_A 数据 | ✅ | 115GB, base/advantage/dagger 完整 |
| Task_A checkpoint | ✅ | mixed_1 (kai0 best, 22GB) + pi05_base (GCS 自动缓存) |
| config.py (sim01) | ✅ | weight_loader → pi05 base (GCS); repo_id → 本地路径; episodes 已修正 |
| config.py (gf0/gf1) | ⚠️ 待同步 | 需替换 repo_id 为 /vePFS/... 路径 |
| norm stats | ✅ 可复用 | norm_stats 基于数据统计, 不依赖模型权重 |
| sim01 推理 | ✅ **可用** | mode=ros2 / websocket / both 均已验证 |
| **训练** | **待启动** | gf0: normal, gf1: 子集训练 |

---

## 十一、IPC 与推理服务差异审计 (2026-03-28)

> 对比 `origin/main` (commit `9d93078`) 与本地工作副本中 IPC、推理服务、数据采集相关代码。
> 涉及: WebSocket 通信层、ROS1→ROS2 迁移的 3 个新文件。

### 11.1 差异总览

| 文件 | 变更类型 | 风险等级 | 说明 |
|------|---------|---------|------|
| `packages/.../websocket_client_policy.py` | 超时参数 (tracked) | 无 | ping/close/open_timeout=300s |
| `src/openpi/serving/websocket_policy_server.py` | 超时参数 (tracked) | 无 | ping/close_timeout=300s |
| `src/openpi/policies/` | 无改动 | 无 | 所有 policy 文件与原版一致 |
| `.../inference/agilex_inference_openpi_temporal_smoothing_ros2.py` | ROS2 迁移 (untracked) | **低** | 迁移质量好, rate.sleep 一致 |
| `.../dagger/agilex_openpi_dagger_collect_ros2.py` | ROS2 迁移 (untracked) | **高** | 多处 time.sleep 替代 rate.sleep, 累积漂移 |
| `.../dagger/collect_data_ros2.py` | ROS2 迁移 (untracked) | **高** | BEST_EFFORT QoS 可能丢帧 |

### 11.2 WebSocket 通信层 (tracked 修改)

#### 已修改 #1: 客户端超时 — 风险: 无

```python
# websocket_client_policy.py — 新增 3 个参数
conn = websockets.sync.client.connect(
    self._uri, compression=None, max_size=None, additional_headers=headers,
    ping_timeout=300, close_timeout=300, open_timeout=300,  # ← 新增
)
```

#### 已修改 #2: 服务端超时 — 风险: 无

```python
# websocket_policy_server.py — 新增 2 个参数
async with _server.serve(
    ..., compression=None, max_size=None,
    ping_timeout=300, close_timeout=300,  # ← 新增
    process_request=_health_check,
)
```

**结论:** 默认超时仅 20s, 大模型推理可能超过此值导致连接断开。300s 是合理的操作改进，不影响推理结果。
`open_timeout` 仅客户端参数, 服务端无此选项, 不存在不对称问题。

### 11.3 推理 Temporal Smoothing ROS2 迁移 — 风险: 低

**文件:** `agilex_inference_openpi_temporal_smoothing_ros2.py` (1250 行, 基于 ROS1 版 1247 行)

逐项对比结果:

| 组件 | 一致性 | 说明 |
|------|--------|------|
| `StreamActionBuffer` | ✅ 完全一致 | 时序平滑逻辑逐行相同, 含 threading.Lock |
| `get_frame()` | ✅ 语义一致 | 时间戳同步逻辑相同, `.to_sec()` → `_stamp_to_sec()` |
| Rate/Sleep | ✅ 一致 | 全部使用 `create_rate()` + `rate.sleep()`, 无漂移 |
| QoS | ✅ 等价 | 使用默认 QoS (RELIABLE, depth=1000), 等价 ROS1 TCP |
| 线程安全 | ✅ 等价 | ROS2 显式 spin 线程 + deque 原子操作, 与 ROS1 等价 |

**唯一差异:** 异常处理从 `rospy.ROSInterruptException` 改为 `KeyboardInterrupt`，语义正确。

### 11.4 DAgger 数据采集 ROS2 迁移 — 风险: **高**

**文件:** `agilex_openpi_dagger_collect_ros2.py` (2395 行, 基于 ROS1 版 2397 行)

#### 问题 #1: `time.sleep(1.0/hz)` 替代 `rate.sleep()` — **累积漂移**

ROS1 的 `rospy.Rate.sleep()` 是**补偿式休眠** — 自动扣除循环体执行时间, 保证恒定频率。
`time.sleep(1.0/hz)` 是**固定休眠** — 不考虑循环体耗时, 实际周期 = 循环体耗时 + 1/hz。

受影响的位置:

| 位置 | 函数 | ROS1 | ROS2 (当前) | 影响 |
|------|------|------|-------------|------|
| L567 | `get_ros_observation()` | `rate.sleep()` | `time.sleep(1.0/publish_rate)` | 观测等待循环漂移 |
| L1182 | `puppet_arm_publish_continuous()` | `rate.sleep()` | `time.sleep(1.0/publish_rate)` | **机械臂控制频率漂移** |
| L1226 | `puppet_arm_publish_continuous()` | `rate.sleep()` | `time.sleep(1.0/publish_rate)` | **同上** |
| L1240 | `puppet_arm_publish_linear()` | `rate.sleep()` | `time.sleep(1.0/200.0)` | **200Hz 插值控制漂移** |
| L1269 | `puppet_arm_publish_linear()` | `rate.sleep()` | `time.sleep(1.0/200.0)` | **同上** |
| L1968 | 数据采集主循环 | `rate.sleep()` | `time.sleep(1.0/frame_rate)` | **采集帧率不稳** |
| L2020 | 数据采集主循环 | `rate.sleep()` | `time.sleep(1.0/frame_rate)` | **同上** |

**预期影响:**
- 50Hz 控制循环, 若循环体耗时 5ms, 实际频率降为 ~40Hz (偏差 20%)
- 200Hz 插值循环偏差更大
- 数据采集帧率不稳 → 训练数据时间轴不均匀

**注:** 推理线程和主控制循环 (L431, L719) 正确使用了 `create_rate()`, 不受影响。

#### 问题 #2: Master 臂初始化循环 — 风险: 低

```python
# L1655, L1728: master arm enable/move 循环
time.sleep(0.1)  # ROS1 用 rate = rospy.Rate(10); rate.sleep()
```

仅影响一次性初始化操作 (master 臂使能/归位), 不影响持续控制和数据质量。

#### 其他迁移: 一致

| 组件 | 一致性 |
|------|--------|
| `StreamActionBuffer` | ✅ 完全一致 (复用同一实现) |
| HDF5 保存格式 | ✅ 一致 (observations/images + qpos + actions) |
| MP4 视频编码 | ✅ 一致 (libx264, quality=23) |
| 键盘控制 (DAgger/Save/Resume) | ✅ 一致 |
| QoS | ✅ 默认 RELIABLE (depth=1000) |

### 11.5 数据录制脚本 ROS2 迁移 — 风险: **高**

**文件:** `collect_data_ros2.py` (412 行, 基于 ROS1 版 390 行)

#### 问题 #3: BEST_EFFORT QoS — **可能丢帧**

```python
# collect_data_ros2.py L214-218
qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,  # ← 问题所在
    history=HistoryPolicy.KEEP_LAST,
    depth=1000,
)
# 所有订阅 (3 相机 + 2 关节 + 里程计) 都用此 QoS
```

```python
# collect_data.py (ROS1) L208-219
rospy.Subscriber(..., queue_size=1000, tcp_nodelay=True)  # 可靠 TCP
```

**差异:**
- ROS1: 可靠 TCP 传输, `tcp_nodelay=True` 禁用 Nagle 算法 → 低延迟且**不丢包**
- ROS2: `BEST_EFFORT` = UDP 语义 → 当订阅者处理慢时**静默丢弃消息**

**影响:**
- 图像消息 (~480x640x3 ≈ 900KB) 可能在高负载时被丢弃
- 丢弃的帧不会报错, 但导致训练数据中时间步不连续
- 对于数据采集 (teleop recording), 数据完整性至关重要

**修复:** 改为 `ReliabilityPolicy.RELIABLE`。

### 11.6 通信架构总图 (确认一致)

```
Robot (ROS2 Cameras + Arms)
    │ ROS2 Topics (Image, JointState)
    ▼
RosOperator (Node)
    │ get_frame() → observation dict
    ▼
WebsocketClientPolicy
    │ msgpack_numpy.pack(obs) → ws.send()
    ▼ ──── TCP/WebSocket ws://host:8000 ────
WebsocketPolicyServer
    │ policy.infer(obs) → actions
    │ msgpack_numpy.pack(actions) → ws.send()
    ▼
WebsocketClientPolicy.infer()
    │ actions[chunk_size, 14]
    ▼
StreamActionBuffer (temporal smoothing)
    │ pop_next_action() → single action
    ▼
RosOperator.puppet_arm_publish()
    │ JointState msgs → /master/joint_*
    ▼
Robot Arms
```

消息格式、序列化协议 (msgpack-numpy)、action chunk broker 均与原版一致, 无风险。

### 11.7 修复记录

| # | 修复项 | 文件 | 操作 | 验证 |
|---|--------|------|------|------|
| 1 | DAgger `get_ros_observation()` 漂移 | `agilex_openpi_dagger_collect_ros2.py` L537 | 新增 `rate = ros_operator.create_rate(...)`, L567 `time.sleep` → `rate.sleep()` | `grep 'time.sleep(1.0 / args'` 无结果 |
| 2 | DAgger `puppet_arm_publish_continuous()` 漂移 | 同上 L1174, L1228 | 新增 `rate = self.create_rate(...)`, 两处 `time.sleep` → `rate.sleep()` | 同上 |
| 3 | DAgger `puppet_arm_publish_linear()` 200Hz 漂移 | 同上 L1231, L1243, L1272 | 新增 `rate_200 = self.create_rate(200)`, 两处 `time.sleep(1.0/200.0)` → `rate_200.sleep()` | 同上 |
| 4 | DAgger 数据采集主循环漂移 | 同上 L1957, L1971, L2023 | 新增 `rate = self.create_rate(...)`, 两处 `time.sleep` → `rate.sleep()` | 同上 |
| 5 | Collect Data BEST_EFFORT QoS | `collect_data_ros2.py` L215 | `BEST_EFFORT` → `RELIABLE` | `grep BEST_EFFORT collect_data_ros2.py` 无结果 |

**修复后状态:**
- `agilex_openpi_dagger_collect_ros2.py`: 所有控制/采集循环均使用 `create_rate()` + `rate.sleep()`, 与 ROS1 版行为一致
- `collect_data_ros2.py`: QoS 改为 RELIABLE, 等价 ROS1 的 TCP 可靠传输
- `agilex_inference_openpi_temporal_smoothing_ros2.py`: 无需修复, 本身就正确
- WebSocket 超时参数: 保留不动, 合理的操作改进

**未修复 (低优先级):**
- DAgger master 臂初始化中的 `time.sleep(0.1)` (L1658, L1731) — 一次性操作, 不影响持续控制
- 异常捕获从 `rospy.ROSInterruptException` → `KeyboardInterrupt` — 语义正确, 无需改

---

## 十二、推理流水线端到端深度审计 (2026-03-28)

> 对端到端推理路径做深度审计：ROS2 观测采集 → WebSocket 客户端 → 服务端策略推理 → 动作返回 → 机器人执行。
> 重点关注训练-推理一致性，以及各推理模式之间的行为差异。

### 12.1 审计范围

```
Robot (ROS2)                              GPU Host
┌─────────────────┐                  ┌──────────────────────────────────┐
│ RosOperator     │    WebSocket     │ serve_policy.py                  │
│  get_frame()    │◄──────────────►  │  WebsocketPolicyServer           │
│  observation    │  msgpack-numpy   │    ↓                             │
│  construction   │                  │  Policy.infer(obs)               │
│  action         │                  │    ├ AgilexInputs (π clamp, pad) │
│  execution      │                  │    ├ ResizeImages(224,224)       │
│                 │                  │    ├ Normalize (z-score)         │
│ StreamAction    │                  │    ├ Model.sample_actions()      │
│  Buffer         │                  │    ├ Unnormalize                 │
│  (temporal      │                  │    └ AgilexOutputs ([:14])       │
│   smoothing)    │                  │                                  │
└─────────────────┘                  └──────────────────────────────────┘
```

### 12.2 发现 #1: 推理 Prompt 三方不一致 — 风险: **高**

三处各用不同的 prompt:

| 来源 | Prompt | 位置 |
|------|--------|------|
| **训练数据集** | `"flat the cloth"` | `data/Task_A/base/meta/tasks.jsonl` |
| **训练配置** | `"Flatten and fold the cloth."` | `config.py` L1181 `default_prompt` |
| **推理脚本** | `"fold the sleeve"` | 所有 `agilex_inference_*.py` 的 `lang_embeddings` |

**机制分析:**

训练时:
- `prompt_from_task=False` (pi05_flatten_fold_normal 默认值)
- `InjectDefaultPrompt("Flatten and fold the cloth.")` 注入 (因为数据集 sample 中无 "prompt" 键)
- 模型学到的 prompt → `"Flatten and fold the cloth."`

推理时:
- 客户端发送 `payload = {"prompt": "fold the sleeve", ...}`
- 服务端 `InjectDefaultPrompt` 检查: `if "prompt" not in data` → 已存在, 跳过
- 模型实际收到的 prompt → `"fold the sleeve"` (客户端覆盖)

**影响:** VLA 模型的行为受 prompt 影响。训练时用 "Flatten and fold the cloth."，推理时用 "fold the sleeve"，可能导致:
- 模型未见过 "fold the sleeve" 这个 prompt → 行为不可预测
- 尤其在 AWBC (prompt_from_task=True) 模型中，prompt 变化更敏感

**README 明确警告 (inference/agilex/README.md L210):**
> "Ensure the prompt in the script matches training"

**注:** 此问题在原版 ROS1 代码中也存在 (所有推理脚本都硬编码 "fold the sleeve")。可能是作者的测试配置遗留。

### 12.3 发现 #2: Gripper Offset 在不同推理模式间不一致 — 风险: **中** (原版问题)

`RIGHT_OFFSET = 0.003` 的应用方式:

| 推理模式 | 左臂 gripper [6] | 右臂 gripper [6] |
|---------|------------------|------------------|
| **sync** | ❌ 不减 | ✅ `-= 0.003` |
| **RTC (非 smoothing 路径)** | ❌ 不减 | ✅ `-= 0.003` |
| **RTC (smoothing 路径)** | ✅ `-= 0.003` | ✅ `-= 0.003` |
| **temporal_smoothing (ROS1)** | ✅ `-= 0.003` | ✅ `-= 0.003` |
| **temporal_smoothing (ROS2)** | ✅ `-= 0.003` | ✅ `-= 0.003` |
| **temporal_ensembling** | ✅ `-= 0.003` | ✅ `-= 0.003` |

**结论:** ROS2 版 (temporal_smoothing) 与 ROS1 temporal_smoothing **完全一致**。
sync 模式只对右臂做 offset 是原版代码行为 (可能是早期版本遗留), 与 ROS2 迁移无关。

如果使用推荐的 temporal smoothing 模式部署, **此差异不影响复现**。

### 12.4 发现 #3: serve_policy.py 无 kai0 专属环境 — 风险: 低

`serve_policy.py` 的 `EnvMode` 只有 ALOHA, ALOHA_SIM, DROID, LIBERO, 无 "AGILEX" 或 "KAI0"。

**正确用法:**
```bash
uv run scripts/serve_policy.py \
    --policy.config pi05_flatten_fold_normal \
    --policy.dir <checkpoint_path> \
    --port 8000
```
直接指定 `policy:checkpoint` 模式, 绕过 EnvMode。README 已有示例。

### 12.5 端到端数据流一致性验证

#### 观测构建 (客户端) — ✅ ROS1/ROS2 完全一致

```python
# 1. 帧同步: get_frame() 基于 min(timestamps) 对齐多传感器
# 2. JPEG 对齐 (与训练时 LeRobot 数据加载一致):
img = cv2.imencode(".jpg", img)[1].tobytes()
img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
# 3. BGR → RGB:
imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
# 4. Resize with padding → 224x224:
imgs = image_tools.resize_with_pad(np.array(imgs), 224, 224)
# 5. HWC → CHW:
imgs[i].transpose(2, 0, 1)
```

#### 状态提取 — ✅ 一致

```python
qpos = np.concatenate((puppet_arm_left.position, puppet_arm_right.position))  # [14]
```

#### 服务端推理流水线 — ✅ 无本地修改 (除超时参数)

```
InjectDefaultPrompt → ResizeImages(224,224) → TokenizePrompt → PadStatesAndActions
→ Normalize (z-score) → uint8→[-1,1] → Model.sample_actions(JIT) → Unnormalize
→ AgilexOutputs[:14]
```

关键 transform:
- `AgilexInputs`: 状态值超出 [-π, π] 的置零 (异常检测)
- `Normalize`: z-score, epsilon=1e-6
- `Observation.from_dict`: uint8 → float32: `img / 255.0 * 2.0 - 1.0` → [-1, 1]
- `AgilexOutputs`: 取 actions[:, :14] (丢弃 padding 维度)

#### 动作执行 — ✅ ROS1/ROS2 temporal_smoothing 一致

```python
act = stream_buffer.pop_next_action()         # 从时序平滑 buffer 取动作
left_action = act[:7].copy()
right_action = act[7:14].copy()
left_action[6] = max(0.0, left_action[6] - 0.003)   # gripper offset
right_action[6] = max(0.0, right_action[6] - 0.003)
ros_operator.puppet_arm_publish(left_action, right_action)
```

#### msgpack 序列化 — ✅ 无修改

numpy array → `{__ndarray__: True, data: tobytes(), dtype: str, shape: tuple}`

### 12.6 ROS2 迁移质量总结

| 维度 | 评估 | 详情 |
|------|------|------|
| 图像预处理 | ✅ 一致 | JPEG, resize, transpose 完全相同 |
| 状态提取 | ✅ 一致 | 关节拼接方式相同 |
| 推理触发 | ✅ 一致 | 非阻塞线程 + inference_rate 限速 |
| 时序平滑 | ✅ 一致 | StreamActionBuffer 逐行相同, 含 threading.Lock |
| 动作执行 | ✅ 一致 | gripper offset 双臂、publish 格式相同 |
| Rate/Sleep | ✅ 一致 | 全部使用 create_rate + rate.sleep |
| QoS | ✅ 等价 | 默认 RELIABLE, depth=1000 |
| 线程模型 | ✅ 等价 | ROS2 显式 spin 线程 ≈ ROS1 内置回调线程 |
| 信号处理 | ✅ 语义等价 | rclpy.shutdown() ≈ rospy.signal_shutdown() |
| 参数默认值 | ✅ 一致 | publish_rate=30, chunk_size=50, latency_k=8 等 |

### 12.7 待修复: 推理 Prompt 对齐

推理脚本中的 `lang_embeddings` 需要与训练配置的 `default_prompt` 对齐:

```python
# 当前 (所有推理脚本):
lang_embeddings = "fold the sleeve"

# 应改为 (匹配训练配置 pi05_flatten_fold_normal):
lang_embeddings = "Flatten and fold the cloth."
```

**注:** 此修改需在实际部署推理时执行。如果 kai0 作者实际部署时确实使用 "fold the sleeve", 则需要确认该 prompt 在训练数据中出现过, 或者模型对 prompt 变体有足够泛化能力。

### 12.8 ROS2 客户端 vs WebSocket 服务端数据流逐步对齐审计

> 逐步追踪一次完整推理请求中每个字节的变换路径, 排查客户端-服务端之间是否存在格式不匹配、双重处理或遗漏。

#### 12.8.1 图像流水线对比

**客户端 (ROS2 temporal_smoothing):**

```
ROS2 Camera (BGR uint8 [480,640,3])
  ↓ JPEG encode/decode (与训练 MP4 压缩对齐)
BGR uint8 [480,640,3]
  ↓ cv2.cvtColor(BGR→RGB)
RGB uint8 [480,640,3]
  ↓ image_tools.resize_with_pad(224,224)  ← openpi_client 包
RGB uint8 [224,224,3]
  ↓ transpose(2,0,1) → CHW
RGB uint8 [3,224,224]
  ↓ msgpack_numpy.pack → WebSocket 发送
```

**服务端 (`Policy.infer` transform 链):**

```
接收 → msgpack_numpy.unpack
payload["images"]["top_head"] = RGB uint8 [3,224,224]

  ↓ AgilexInputs.__call__():
    img.dtype = uint8 → 跳过 float→uint8 转换
    img.shape[0] == 3 → np.transpose(1,2,0) → HWC
    → 重命名: top_head → base_0_rgb
RGB uint8 [224,224,3]

  ↓ Normalize():
    → 仅对 state 做 z-score, 图像不变

  ↓ ResizeImages(224,224):
    images.shape[-3:-1] == (224,224) → True → 直接返回 (NO-OP)
RGB uint8 [224,224,3]

  ↓ Observation.from_dict():
    → img.astype(float32) / 255.0 * 2.0 - 1.0
float32 [-1, 1] [224,224,3]

  ↓ 加 batch 维 → JAX jnp.asarray → Model.sample_actions()
```

**训练时图像路径 (对比):**

```
LeRobot 加载 MP4 视频帧 → torch float32 [C,H,W] (约 [0,1])

  ↓ RepackTransform (key 映射)
  ↓ AgilexInputs.__call__():
    isinstance(img, Tensor) → img.cpu().numpy()
    np.floating → img = (255 * img).astype(uint8)  → uint8 [C,H,W]
    shape[0]==3 → transpose → uint8 [H,W,C]
    → 此时图像仍为原始分辨率 (480x640)

  ↓ Normalize(): 仅 state
  ↓ ResizeImages(224,224):
    (480,640) ≠ (224,224) → 实际 resize
    → 使用同一个 image_tools.resize_with_pad() 函数
uint8 [224,224,3]

  ↓ Observation.from_dict(): uint8 → float32 [-1,1]
```

**结论:**

| 维度 | 训练 | 推理 | 一致? |
|------|------|------|-------|
| 色彩空间 | RGB (LeRobot) | RGB (cvtColor) | ✅ |
| Resize 函数 | `openpi_client.image_tools.resize_with_pad` | 同左 (客户端调用) | ✅ |
| Resize 时机 | 服务端 transform 链中 | 客户端预处理; 服务端 no-op | ✅ 等价 |
| JPEG/视频压缩 | MP4 解码 (有损) | JPEG encode/decode (有损, 对齐用) | ✅ 设计如此 |
| dtype 到 model | uint8 → float32 [-1,1] | uint8 → float32 [-1,1] | ✅ |
| 双重 resize | 无 | 无 (服务端 early return) | ✅ |
| 双重 normalize | 无 | 无 (客户端不做 normalize) | ✅ |

**无不一致风险。**

#### 12.8.2 状态 (State) 流水线对比

**客户端:**
```python
qpos = np.concatenate((puppet_arm_left.position, puppet_arm_right.position))  # float64 [14]
payload["state"] = qpos
```

**服务端:**
```python
# AgilexInputs:
state = pad_to_dim(data["state"], action_dim)  # [14] → [action_dim], 零填充
state = state.squeeze()
state = np.where(state > np.pi, 0, state)      # 异常值置零
state = np.where(state < -np.pi, 0, state)

# Normalize:
state = (state - mean) / (std + 1e-6)           # z-score

# PadStatesAndActions:
state = np.pad(state, ...)                       # 再次 pad (已 pad 则 no-op)
```

**训练时:**
```
LeRobot → "observation.state" → RepackTransform → "state"
  → 相同的 AgilexInputs / Normalize / PadStatesAndActions 链
```

| 维度 | 训练 | 推理 | 一致? |
|------|------|------|-------|
| 原始 dtype | float32 (parquet) | float64 (ROS JointState) | ⚠️ 精度略高, 无影响 |
| π clamp | ✅ | ✅ | ✅ |
| z-score 归一化 | 服务端 | 服务端 | ✅ 同一 norm_stats |
| padding | 服务端 | 服务端 | ✅ |

**无不一致风险。** float64→float32 的精度损失在 z-score 归一化后可忽略。

#### 12.8.3 Prompt 流水线对比 — **存在不一致**

**完整 Prompt 注入链 (服务端 transform 顺序):**

```
1. InjectDefaultPrompt(CLI --default_prompt)  ← 通常 None
2. AgilexInputs: if "prompt" in data → passthrough
3. Normalize: 不处理 prompt
4. InjectDefaultPrompt("Flatten and fold the cloth.")  ← 来自训练配置
5. TokenizePrompt → 分词
```

**三种场景下模型实际收到的 prompt:**

| 场景 | 客户端发送 | 步骤 1 | 步骤 4 | 模型收到 |
|------|-----------|--------|--------|---------|
| A: 当前代码 | `"fold the sleeve"` | 跳过 (已有) | 跳过 (已有) | `"fold the sleeve"` |
| B: 不发 prompt | (无 prompt key) | 跳过 (default=None) | **注入** | `"Flatten and fold the cloth."` |
| C: CLI 指定 | (无 prompt key) | **注入** CLI 值 | 跳过 (已有) | CLI 指定值 |

**当前状态:** 场景 A — 客户端硬编码 `"fold the sleeve"` 覆盖了训练 prompt。

**训练时模型学到的 prompt:**
- `pi05_flatten_fold_normal`: `prompt_from_task=False`, `default_prompt="Flatten and fold the cloth."` → 模型学到 `"Flatten and fold the cloth."`
- 数据集 tasks.jsonl 中的 task: `"flat the cloth"` (未被使用, 因为 prompt_from_task=False)

**三方 prompt 值:**

| 来源 | 值 |
|------|---|
| 数据集 tasks.jsonl | `"flat the cloth"` |
| 训练配置 default_prompt | `"Flatten and fold the cloth."` |
| 推理脚本 lang_embeddings | `"fold the sleeve"` |

**这是原版代码中就存在的问题, 不是 ROS2 迁移引入的。** 但对复现有直接影响: 使用不匹配的 prompt 可能降低成功率。

#### 12.8.4 动作输出流水线

**服务端返回:**
```python
# Model.sample_actions() → actions [1, chunk_size, action_dim]
# 去 batch: [chunk_size, action_dim]
# Unnormalize: 逆 z-score
# AgilexOutputs: actions[:, :14]  → [chunk_size, 14]
# 打包 msgpack → WebSocket 发送
```

**客户端接收:**
```python
actions = policy.infer(payload)["actions"]  # [chunk_size, 14]
stream_buffer.integrate_new_chunk(actions, max_k=latency_k, min_m=8)

# 主循环:
act = stream_buffer.pop_next_action()  # [14], 经过时序平滑
left_action = act[:7].copy()
right_action = act[7:14].copy()
left_action[6] = max(0.0, left_action[6] - 0.003)   # gripper offset
right_action[6] = max(0.0, right_action[6] - 0.003)
ros_operator.puppet_arm_publish(left_action, right_action)
```

动作维度切分、gripper offset、发布格式均与 ROS1 temporal_smoothing 一致。

#### 12.8.5 `serve_policy.py` 启动方式

kai0 无专属 EnvMode (仅有 ALOHA/DROID/LIBERO), 需用 checkpoint 模式:

```bash
# 正确启动:
uv run scripts/serve_policy.py \
    --policy.config pi05_flatten_fold_normal \
    --policy.dir /path/to/checkpoint \
    --port 8000

# 客户端连接:
python agilex_inference_openpi_temporal_smoothing_ros2.py \
    --host <GPU_HOST_IP> --port 8000
```

### 12.9 总结: 客户端-服务端一致性风险矩阵

| 数据通道 | 客户端 (ROS2) | 服务端 (WebSocket) | 一致? | 风险 |
|---------|--------------|-------------------|-------|------|
| 图像格式 | uint8 CHW RGB 224x224 | CHW→HWC, resize no-op, uint8→float[-1,1] | ✅ | 无 |
| 图像压缩 | JPEG encode/decode | 无 (训练用 MP4) | ✅ | 无 (设计对齐) |
| 状态格式 | float64 [14] | pad+clamp+z-score | ✅ | 无 |
| Prompt | `"fold the sleeve"` (硬编码) | 被客户端值覆盖 | **❌** | **高: 与训练 prompt 不匹配** |
| 动作输出 | [chunk, 14] → StreamActionBuffer | Unnorm→[:14] | ✅ | 无 |
| 序列化 | msgpack-numpy | msgpack-numpy | ✅ | 无 |
| 超时 | 300s (ping/close/open) | 300s (ping/close) | ✅ | 无 |
| Normalize | 不做 (交给服务端) | z-score (norm_stats) | ✅ | 无 (无双重处理) |
| Resize | 客户端做 (224x224) | 服务端 no-op | ✅ | 无 (无双重处理) |

**唯一实质风险:** Prompt 不一致。需在推理部署前将 `lang_embeddings` 改为与训练配置匹配的值。

---

## 十三、ROS2 节点内推理 vs kai0 原版 WebSocket 模式差异审计 (2026-03-28)

> `policy_inference_node.py` 是本地新建的 ROS2 节点, 不在 kai0 原版 repo 中。
> 它将 JAX 推理直接嵌入 ROS2 节点进程, 绕过 WebSocket 通信层。
> 本节逐条审计其与 kai0 原版推理管线的差异。

### 13.1 架构对比

```
kai0 原版 (WebSocket 模式):                  本地新增 (ROS2 节点内模式):
┌──────────────────┐    WebSocket     ┌────────────────┐
│ inference_*.py   │──────────────►   │ serve_policy.py│
│ (ROS2 客户端)    │  msgpack-numpy   │ (GPU 服务端)   │
│                  │◄──────────────   │ Policy.infer() │
│ StreamActionBuf  │                  │ 全 transform 链│
│ 时序平滑 + 发布   │                  └────────────────┘
└──────────────────┘

┌─────────────────────────────────────────────┐
│ policy_inference_node.py (ROS2 节点)        │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │ ROS2 订阅    │→│ _get_observation()   │  │
│  │ cameras     │  │ → Policy.infer()     │  │
│  │ joints      │  │ → StreamActionBuffer │  │
│  └─────────────┘  │ → _publish_action()  │  │
│                   └──────────────────────┘  │
│  同一进程, 无 WebSocket 开销                  │
└─────────────────────────────────────────────┘
```

### 13.2 差异 #1: 图像色彩空间转换缺失 — 风险: **严重**

**kai0 原版 (`agilex_inference_openpi_temporal_smoothing_ros2.py`):**
```python
# 1. CvBridge: ROS Image (rgb8/bgr8) → OpenCV ndarray
# 2. JPEG encode/decode (训练对齐):
img = cv2.imencode(".jpg", img)[1].tobytes()
img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)  # → BGR
# 3. BGR → RGB:
imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
# 4. resize_with_pad (保持宽高比, 零填充):
imgs = image_tools.resize_with_pad(np.array(imgs), 224, 224)
# 5. HWC → CHW:
imgs[i].transpose(2, 0, 1)
```

**`policy_inference_node.py`:**
```python
# 1. CvBridge: passthrough (原始格式, D435 默认 rgb8)
self.img_front = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
# 2. 无 JPEG encode/decode ❌
# 3. 无 BGR→RGB 转换 ❌
# 4. cv2.resize (不保持宽高比, 无 padding):
imgs = [cv2.resize(im, (224, 224)) for im in imgs]   # ❌ 不同于 resize_with_pad
# 5. HWC → CHW:
imgs[0].transpose(2, 0, 1)
```

**问题清单:**

| # | 问题 | 影响 |
|---|------|------|
| A | **无 JPEG encode/decode** | 训练数据经 MP4 压缩, 推理缺少 JPEG 对齐, 图像比训练"更清晰", 分布偏移 |
| B | **无 BGR→RGB 转换** | RealSense D435 `image_raw` 默认 `rgb8`; `passthrough` 保持原样 → 是 RGB。但 kai0 原版假设 CvBridge 输出 BGR (经 `imencode` 隐式转换), 然后显式做 `cvtColor(BGR→RGB)`。如果相机输出实际是 `rgb8`, 则原版的 `cvtColor` 反而会把 RGB 错转为 BGR → **需要实测确认 D435 输出格式** |
| C | **`cv2.resize` vs `resize_with_pad`** | `cv2.resize` 直接拉伸到 224x224, 不保持宽高比; `resize_with_pad` 保持宽高比 + 零填充。640x480 → 拉伸会水平压缩 → **图像变形** |

**风险量化:**

- 差异 A (无 JPEG): 影响约 1-3% 像素值, 对训练/推理对齐有轻微影响
- 差异 B (色彩空间): 如果通道顺序错误 → **RGB↔BGR 全反, 模型看到的图像完全错乱**
- 差异 C (resize): 640:480=4:3 → 224x224 强制 1:1, 水平压缩 25% → **严重变形**

### 13.3 差异 #2: StreamActionBuffer 实现不同 — 风险: **中**

**kai0 原版 `StreamActionBuffer.integrate_new_chunk()`:**
```python
# 线性权重融合:
w_old = np.linspace(1.0, 0.0, overlap_len)   # 100% old → 0% old
w_new = 1.0 - w_old
smoothed = [w_old[i]*old + w_new[i]*new for i in range(overlap)]

# latency compensation:
drop_n = min(self.k, max_k)
new_chunk = actions[drop_n:]

# last_action 保存机制 (buffer 空时用上次最后一帧补到 min_m)
```

**`policy_inference_node.py` 的简化版本:**
```python
# 指数衰减融合 (不同的融合策略!):
alpha = self.decay_alpha ** (overlap - i)
merged.append(alpha * old + (1 - alpha) * new)

# latency compensation: 不同的 trim 逻辑
trim = min(max_k, max(0, len(new_chunk) - overlap))
for _ in range(trim):
    self.cur_chunk.popleft()

# 无 last_action 保存机制 (buffer 空时直接用新 chunk)
```

| 维度 | kai0 原版 | 节点内版本 | 差异 |
|------|----------|-----------|------|
| 融合方法 | 线性 linspace | 指数衰减 `α^(N-i)` | **不同** |
| latency trim | `new_chunk[drop_n:]` 裁剪新 chunk 前端 | 从 old chunk popleft | **不同** |
| buffer 空时行为 | `last_action` 补到 min_m | 直接用新 chunk | **不同** |

### 13.4 差异 #3: 推理节拍控制 — 风险: **低-中**

**kai0 原版:**
- 推理线程: `rate = create_rate(inference_rate)` + `rate.sleep()` — 补偿式
- 发布循环: `rate = create_rate(publish_rate)` + `rate.sleep()` — 补偿式

**节点内版本:**
- 推理线程: `time.sleep(max(0, period - elapsed))` — 手动补偿, 近似等价 ✅
- 发布循环: `self.create_timer(period, callback)` — ROS2 timer, 补偿式 ✅

节拍控制**基本等价**, 但推理线程的手动补偿在高负载时精度略低。

### 13.5 差异 #4: 帧同步机制缺失 — 风险: **高**

**kai0 原版:**
```python
# get_frame(): 基于 min(timestamps) 对齐 3 相机 + 2 关节
# 所有 deque 中找时间戳最接近的帧, 丢弃过期帧
# 确保一次推理中各传感器数据时间对齐
```

**节点内版本:**
```python
# _get_observation(): 直接取最新值, 无时间对齐
imgs = [self.img_front, self.img_right, self.img_left]
state = np.array(self.joint_left + self.joint_right)
# 各传感器回调独立更新, 无同步保证
```

3 个相机和 2 个关节的数据可能来自不同时刻 (最多差 1/30s ≈ 33ms), 在快速运动场景下可能导致观测不一致。

### 13.6 差异 #5: QoS 配置 — 风险: 低

| 参数 | kai0 原版 | 节点内版本 |
|------|----------|-----------|
| 图像 QoS | 默认 RELIABLE, depth=1000 | RELIABLE, depth=**1** |
| 关节 QoS | 默认 RELIABLE, depth=1000 | 默认 depth=**10** |

depth=1 意味着只保留最新一帧, 高负载时可能丢帧。但对推理场景 (只需最新观测) 影响不大。

### 13.7 差异 #6: Launch 配置中的 Topic 不一致 — 风险: **高**

**`inference_full_launch.py` (launch 文件):**
```python
'img_front_topic': '/camera_f/color/image_raw',
'img_left_topic': '/camera_l/color/image_rect_raw',    # ← rect_raw
'img_right_topic': '/camera_r/color/image_rect_raw',   # ← rect_raw
```

**`policy_inference_node.py` (默认参数):**
```python
self.declare_parameter('img_front_topic', '/camera_f/color/image_raw')
self.declare_parameter('img_left_topic', '/camera_l/color/image_raw')      # ← image_raw
self.declare_parameter('img_right_topic', '/camera_r/color/image_raw')     # ← image_raw
```

**kai0 原版默认:**
```python
--img_front_topic /camera_f/color/image_raw
--img_left_topic  /camera_l/color/image_raw   # ← image_raw
--img_right_topic /camera_r/color/image_raw   # ← image_raw
```

Launch 文件中左/右腕相机使用 `image_rect_raw` (去畸变后的图像), 而节点默认和 kai0 原版都用 `image_raw` (原始图像)。

`image_rect_raw` vs `image_raw`:
- `image_raw`: 原始畸变图像 (与训练数据一致)
- `image_rect_raw`: 去畸变后图像 (几何校正, 像素重采样)

**由于 launch 参数会覆盖节点默认值**, 实际运行时左右腕会收到去畸变图像, 与训练数据分布不一致。

### 13.8 差异 #7: Launch 默认 Checkpoint 仍指向 mixed_1 — 风险: **高**

```python
# inference_full_launch.py L41:
ckpt_arg = DeclareLaunchArgument('checkpoint_dir',
    default_value='/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/Task_A/mixed_1')
```

此问题与 §9.2 差异 #2 相同: `mixed_1` 是 Model Arithmetic 输出, 不是 π₀.₅ base。
正常微调后需要用自己训练出的 checkpoint, 不应默认用 mixed_1。

### 13.9 差异 #8: Launch 默认 Prompt — 风险: 中

```python
# inference_full_launch.py L44:
prompt_arg = DeclareLaunchArgument('prompt', default_value='fold the cloth')
```

| 来源 | Prompt |
|------|--------|
| Launch 默认 | `"fold the cloth"` |
| 节点默认 | `"fold the cloth"` |
| kai0 原版推理脚本 | `"fold the sleeve"` |
| 训练配置 default_prompt | `"Flatten and fold the cloth."` |
| 数据集 tasks.jsonl | `"flat the cloth"` |

现在是**四方不一致**。Launch 的 `"fold the cloth"` 是第四个不同的 prompt。

### 13.10 完整风险矩阵

| # | 差异 | 风险 | 影响 | 修复优先级 |
|---|------|------|------|-----------|
| 1C | `cv2.resize` vs `resize_with_pad` | **严重** | 图像 4:3→1:1 变形, 模型未见过 | P0 |
| 1B | 无 BGR↔RGB 转换链路 | **严重** | 可能通道反转, 需实测确认 | P0 |
| 1A | 无 JPEG encode/decode | **中** | 训练-推理图像分布偏移 | P1 |
| 2 | StreamActionBuffer 融合策略不同 | **中** | 时序平滑行为不同, 动作抖动 | P1 |
| 4 | 无多传感器帧同步 | **高** | 快速运动时观测不一致 | P1 |
| 6 | Launch 中 `image_rect_raw` vs `image_raw` | **高** | 去畸变图像 vs 原始图像 | P0 |
| 7 | Checkpoint 默认指向 mixed_1 | **高** | 起点权重错误 | P0 |
| 8 | Prompt 四方不一致 | **中** | 语言条件偏移 | P1 |
| 3 | 推理节拍手动补偿 | **低** | 高负载精度略低 | P2 |
| 5 | QoS depth=1 | **低** | 高负载可能丢帧 | P2 |

### 13.11 结论

**`policy_inference_node.py` 当前无法安全用于复现。** 存在 3 个严重问题 (图像 resize 变形、色彩空间链路不明、image_rect_raw topic 不匹配) 和多个中等问题 (StreamActionBuffer 实现不同、无帧同步、prompt 不一致)。

### 13.12 修复记录 (2026-03-28)

已完成全部 7 项修复, 修改文件:
- `ros2_ws/src/piper/scripts/policy_inference_node.py` (完整重写)
- `ros2_ws/src/piper/launch/inference_full_launch.py` (配置修复)

| # | 修复项 | 操作 |
|---|--------|------|
| 1 | 图像预处理 | 补全 `_jpeg_mapping()` + `cvtColor(BGR→RGB)` + `resize_with_pad(224,224)` |
| 2 | StreamActionBuffer | 替换为原版完整实现 (线性 linspace 融合 + last_action + latency trim) |
| 3 | 帧同步 | 回调改为 deque append (容量 2000), 新增 `_get_synced_frame()` 基于 min(timestamp) 对齐 |
| 4 | QoS depth | 1 → 1000 (匹配原版, 支持 deque 帧同步) |
| 5 | Prompt 默认值 | `'fold the cloth'` → `'Flatten and fold the cloth.'` |
| 6 | Launch topic | `image_rect_raw` → `image_raw` (左右腕相机) |
| 7 | Launch checkpoint | 参照 serve_policy.py 配置化设计: config_name 决定 transform 链, checkpoint_dir 决定权重; 推理默认 mixed_1 (kai0 best model), 含常见组合注释 |

**额外修复:**
- 图像 QoS: `RELIABLE` → `BEST_EFFORT` (匹配 RealSense ROS2 驱动默认发布的 QoS, 原版 kai0 用 RELIABLE 是因其环境不同)
- Launch 头顶相机: `name='camera_f', namespace=''` → `name='camera', namespace='camera_f'`, topic 从 `/camera/camera/...` 变为 `/camera_f/camera/color/image_raw`
- Launch policy node topic: 统一为 `/camera_f/camera/...`, `/camera_l/camera/...`, `/camera_r/camera/...`

### 13.13 环境修复: sim01 venv 升级 Python 3.11 → 3.12

sim01 的 kai0 venv 之前是 Python 3.11, 与 ROS2 Jazzy (3.12) 不兼容, 导致 rclpy 无法在 venv 中 import。

```bash
# 备份旧 venv
mv kai0/.venv kai0/.venv_311_bak

# 用系统 python3.12 重建 + 安装依赖
uv venv --python 3.12 kai0/.venv
uv sync --python 3.12
```

验证: venv python 3.12.3 → jax, rclpy, cv_bridge, openpi_client 全部 OK。

### 13.14 验证结果 (2026-03-29)

#### Test 1: 合成数据图像管线 (无 GPU)

```
脚本: scripts/test_inference_parity.py
输入: 3× 随机 640x480 RGB uint8 + 14D joint
```

| 对比 | images | state | prompt | 结论 |
|------|--------|-------|--------|------|
| 修复后 vs WS 原版 | 3 张 identical | equal | equal | **PASS ✅** |
| 修复前 vs WS 原版 | 99.5% 像素不同 | dtype 不同 | equal | FAIL ❌ (预期) |

resize 差异详解: `resize_with_pad` 保持 4:3 → 224x168 内容 + 上下各 28 行零填充; `cv2.resize` 拉伸到 224x224。

#### Test 2: 合成数据模型推理 (GPU, 固定 RNG)

```
模型: pi05_flatten_fold_normal, ckpt=mixed_1 (22GB)
GPU: RTX 5090 #0, 加载 15s
```

| 对比 | actions (50×14) | latency | 结论 |
|------|----------------|---------|------|
| 修复后 vs WS 原版 | **EXACT MATCH** | 64ms vs 64ms | **PASS ✅** |
| 修复前 vs WS 原版 | max_diff=0.55, mean=0.065 | — | FAIL ❌ (预期) |

#### Test 3: 真实相机端到端 (GPU + 3× RealSense + 2× Piper)

```
脚本: scripts/test_e2e_live_camera.py
相机: D435 头顶 (640x480 @15Hz) + D405 左/右腕 (848x480 @30Hz)
关节: 2× Piper 7DoF (当前零位)
```

| 对比 | images | actions (50×14) | 结论 |
|------|--------|----------------|------|
| 修复后 vs WS 原版 | 3 张 identical | **EXACT MATCH** | **PASS ✅** |
| 修复前 vs WS 原版 | 70-97% 像素不同 | max_diff=0.30 (~17°) | FAIL ❌ (预期) |

Actions 样本 (前 3 步, 左臂 7D):
```
step  WS管线                                          Node管线
   0  -0.0085 +0.1811 -0.4863 -0.0227 +0.3453 +0.0013 +0.0019  (完全相同)
   1  -0.0100 +0.1869 -0.4835 -0.0215 +0.3584 +0.0043 +0.0020  (完全相同)
   2  -0.0102 +0.1888 -0.4957 -0.0198 +0.3730 +0.0045 +0.0018  (完全相同)
```

**结论:** 修复后的 `policy_inference_node.py` 在真实相机 + 真实关节数据下, 与 kai0 原版 WebSocket 推理管线产生**完全相同**的 actions, 可安全用于复现。

#### Test 4: 实时连续推理 (mode=both, 10 秒)

```
脚本: scripts/test_e2e_realtime.py
模式: policy_inference_node (mode=both) — 节点内推理 + WS 服务同时运行
流程: 连续抓取真实相机帧 → 构建 obs → 送入 WS 推理 → 记录 actions
```

| 指标 | 值 |
|------|---|
| 采集步数 | 63 步 / 10 秒 |
| 推理频率 | **6.2 Hz** (目标 3 Hz, 余量充足) |
| 延迟 p50 | **101 ms** |
| 延迟 p95 | 161 ms |
| 延迟 max | 203 ms |
| 步间动作变化 | mean=0.003 rad, max=0.021 rad (平滑) |
| NaN/Inf | 0 ✅ |
| 超出 [-π,π] | 0/44100 ✅ |

Actions 统计 (14D, 63 步平均):
```
mean: [-0.009 +0.178 -0.486 -0.018 +0.349 +0.003 +0.002  +0.001 +0.196 -0.449 -0.006 +0.413 -0.015 +0.002]
std:  [ 0.001  0.005  0.003  0.002  0.004  0.002  0.000   0.002  0.005  0.005  0.002  0.007  0.001  0.000]
```

**结论:** mode=both 下实时推理稳定运行, 推理频率 6.2 Hz (超目标 2 倍), 延迟 p50=101ms, 无异常值, 步间变化小 (时序平滑有效)。

#### Test 5: 双模式独立运行对比 (mode=ros2 vs mode=websocket, 各 10s)

```
流程:
  Phase 1: policy_inference_node (mode=ros2) 独立运行 10s, 录制 /policy/actions
  Phase 2: serve_policy.py + policy_inference_node (mode=websocket) 独立运行 10s, 录制 /policy/actions
  Phase 3: 对比两组 actions 的统计分布
场景: 相机+机械臂静止 (输入不变)
```

| 指标 | mode=ros2 | mode=websocket |
|------|-----------|----------------|
| 录制步数 | 304 步 (30Hz × 10s) | 465 步 (46Hz × 10s) |
| warmup 延迟 | 2941 ms (JIT 编译) | 71 ms (WS 端已编译) |

均值对比 (14D, 场景静止):

| 维度 | ros2 mean | ws mean | diff (rad) |
|------|-----------|---------|------------|
| Lj1 | +1.01026 | +1.01551 | 0.00525 |
| Lj2 | -0.90703 | -0.91570 | **0.00867** (最大) |
| Rj1 | +0.73026 | +0.73429 | 0.00403 |
| Rj4 | +0.87868 | +0.87664 | 0.00204 |
| 其余 | — | — | < 0.002 |

```
均值差异: max=0.00867 rad (0.50°)  mean=0.00237 rad (0.14°)
阈值: 0.01 rad (0.57°)
结论: PASS ✅
```

差异来源: RNG 采样随机性 + 帧时间微差 (两次独立运行的帧不完全相同, 但静止场景下内容几乎一致)。0.50° 的最大偏差在 RNG 噪声范围内。

**额外修复 (测试中发现):**
- `policy_inference_node.py` 的 `resize_with_pad`: D435 (640x480) 和 D405 (848x480) 分辨率不同, `np.array(imgs)` 会报错。改为逐个 resize: `[image_tools.resize_with_pad(im[np.newaxis], 224, 224)[0] for im in imgs]`
- WebSocket localhost 连接需 `no_proxy=localhost` 避免代理干扰

---

## 十四、经验总结

### 国内机房 AI 训练环境搭建 checklist

1. **先摸清存储**: overlay 多大? 有没有共享 PFS? 容量够不够?
2. **代理不是万能的**: 区分哪些需要代理 (GitHub/HF) 哪些不需要 (PyPI)
3. **国内镜像优先**: 阿里云 PyPI + ModelScope 比代理快 10-50 倍
4. **uv 跨文件系统**: .venv 必须和 cache 在同一文件系统 (都在本地盘)
5. **共享存储策略**: 代码+数据放共享 PFS, .venv 放各自本地盘
6. **git 大 repo**: 通过代理不稳定, 在有外网的机器 clone 后 scp 过来
7. **下载脚本别信**: 先验证 HF repo 实际目录结构再下载
8. **aria2c 多段下载可能损坏数据**: Orbax/OCDBT 格式的 checkpoint 用单流 wget 下载, 并做 MD5/加载验证
9. **GCS 到国内限速 ~1-3MB/s**: pi0.5 base 12GB 需要 1-2 小时, 多机并行下载可加速
10. **TOS bucket 中转**: sim01↔gf0 公网仅 0.5MB/s, 用 TOS 对象存储中转大文件 (上传到 bucket → gf0 本地挂载读取)
11. **sim01↔gf0 带宽极低 (~0.5MB/s)**: 不要用 sim01 中转大文件, 让 gf0 自行从源头下载
12. **ModelScope 下载的视频可能截断**: moov atom 丢失, 需全量 ffprobe 扫描并重新下载损坏文件
13. **split_data.py 切分后时间戳错位**: 不要用切分数据, 用 `--data.episodes` 参数过滤 base 数据的 episode 子集
14. **训练前必须清理 GPU**: 脚本开头加 kill stale processes, 避免残留显存导致 NCCL 失败
15. **fsdp_devices=8, batch_size=128**: 论文实际参数 (不是 config.py 默认的 batch=256/fsdp=1)
16. **torchcodec 需禁用**: 与 CUDA 12 driver 不兼容, 在 override-dependencies 中排除
17. **缺少 ffmpeg 共享库**: `conda install -c conda-forge ffmpeg>=6`
18. **缺少 chex**: 手动加到 pyproject.toml dependencies
19. **wandb**: 离线环境用 `--no-wandb-enabled`
