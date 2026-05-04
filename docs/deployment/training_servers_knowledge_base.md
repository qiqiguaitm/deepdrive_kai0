# 训练服务器知识库 (gf0 / gf1 / gf2 / gf3)

> **作用**: 4 台 GPU 训练服务器的全方位参考 — 硬件、文件结构、环境、连接方式、训练命令、机器间差异、常见运维。
> **更新日期**: 2026-05-02
> **关联文档**:
> - [`gf2_gf3_deployment.md`](./gf2_gf3_deployment.md) — gf2/gf3 详细部署记录
> - [`sim01_deployment.md`](./sim01_deployment.md) — sim01 推理机部署
> - [`checkpoints_layout.md`](./checkpoints_layout.md) — ckpt 文件结构规范

---

## 1. 服务器全景

| 维度 | gf0 | gf1 | gf2 | gf3 |
|---|---|---|---|---|
| **GPU** | 8× A100-SXM4 80GB | 8× A100-SXM4 80GB | 8× A800-SXM4 80GB | 8× A800-SXM4 80GB |
| **GPU arch** | sm_80 | sm_80 | sm_80 (同 A100) | sm_80 |
| **驱动 / CUDA driver** | 535.129.03 / 12.2 | 535.129.03 / 12.2 | 550.144.03 / 12.4 | 550.144.03 / 12.4 |
| **CUDA toolkit** | 12.8 (用 `/usr/local/cuda-12.8`) | 12.8 | 12.4 | 12.4 |
| **CPU** | Xeon 8336C @ 2.30GHz, 112 cores | 同 gf0 | (多核 NUMA) | 同 gf2 |
| **RAM** | 1.8 TiB | 1.8 TiB | ~700+ GiB | ~700+ GiB |
| **OS** | Debian-velinux1u1 (5.4.250) | 同 gf0 | Ubuntu 22.04 | 同 gf2 |
| **Hostname** | `di-20260312174527-n5dw4` | `di-20260320201920-rzbrm` | `10-60-135-47` | `10-60-204-66` |
| **IP / 入口** | 跳板 `14.103.44.161:55555` (反向隧道) | 跳板 `14.103.44.161:11111` | `117.50.196.104` (直连) | `106.75.68.254` (直连) |
| **本地 SSH 别名** | `ssh -p 55555 tim@14.103.44.161` | `ssh -p 11111 tim@14.103.44.161` | `gf2` (alias in `~/.bashrc`) | `gf3` (alias in `~/.bashrc`) |
| **共享 FS** | /vePFS (gpfs, 跨 gf0/gf1) | 同 gf0 | **无** (每机独立 4TB) | **无** (每机独立 4TB) |
| **私网互通** | 同 gf1 (走 vePFS) | 同 gf0 | 与 gf3 SSH 双向密钥 + 内网 hostname 直连 | 与 gf2 同 |

---

## 2. 文件结构

### 2.1 工作目录路径速查

| 服务器 | 工作目录 | 实际存储 |
|---|---|---|
| gf0 | `/vePFS/tim/workspace/deepdive_kai0/` (= `/home/tim/workspace/deepdive_kai0` 软链) | gpfs 跨机共享 |
| gf1 | 同 gf0 (共享) | 同 |
| gf2 | `/home/tim/workspace/deepdive_kai0/` → `/data/shared/tim/workspace/deepdive_kai0/` | 本机 4TB ext4 |
| gf3 | 同 gf2 (各自独立, 不共享) | 同 gf2 |

### 2.2 Checkpoint 本地存储规范 ⭐ (2026-05-04 重要更新)

> **核心原则**: 每台服务器的 ckpt 写到独立的本地路径, 不跨机同步, 重启不丢失。

**统一路径**: 每台机器都使用 `/home/tim/local_ckpts/` 作为 ckpt 根目录 (其中是 symlink 还是 real dir 因机器而异)。

| Server | `/home/tim/local_ckpts/` 实现 | 物理后端 | 容量 | 持久性 |
|---|---|---|---|---|
| gf0 | symlink → `/vePFS/tim/gf0_local_ckpts/` | /vePFS (50T 共享 FS) | 看 /vePFS 余量 | ✓ 持久 |
| gf1 | symlink → `/vePFS/tim/gf1_local_ckpts/` | /vePFS | 看 /vePFS 余量 | ✓ 持久 |
| gf2 | 真实 dir | /dev/vda2 (492G ext4) | ~290G 可用 | ✓ 持久 |
| gf3 | 真实 dir | /dev/vda2 (492G ext4) | ~410G 可用 | ✓ 持久 |

**为何不放 `/dev/shm` (RAM)**:
- 重启数据丢失, 训练 ckpt 不能容忍
- /dev/shm 适合 dataset (可从源重建), 不适合 ckpt (训练成果)

**为何 gf0/gf1 没用 `/home/tim` 真实 dir**:
- gf0/gf1 上 `/home/tim` 在 overlay (~99G, 已 95% 用) — 没空间存 ckpt
- 唯一持久 + 大容量选项是 `/vePFS` (slow but persistent)
- 所以统一用 `/home/tim/local_ckpts` (symlink) → /vePFS 子目录, **每机独立子目录** 避免冲突

**怎么让训练写到 local_ckpts**:

openpi 默认把 ckpt 写到 `<KAI0_DATA_ROOT>/checkpoints/<config>/<exp>/`。我们用 **per-exp 软连接**, 在 launcher 启动训练前 pre-create 链接:

```bash
# 在 launcher 里:
CONFIG=pi05_flatten_fold_<your_config>
EXP=<your_exp_name>
LOCAL_DIR=/home/tim/local_ckpts/$CONFIG/$EXP
WORKSPACE_DIR=$KAI0_DATA_ROOT/checkpoints/$CONFIG/$EXP

mkdir -p "$LOCAL_DIR"
mkdir -p "$(dirname "$WORKSPACE_DIR")"
[ -e "$WORKSPACE_DIR" ] && [ ! -L "$WORKSPACE_DIR" ] && {
    echo "WARN: $WORKSPACE_DIR exists as real dir, please move first"
    exit 1
}
ln -sfn "$LOCAL_DIR" "$WORKSPACE_DIR"

# 然后正常启训练:
.venv/bin/python scripts/train.py $CONFIG --exp_name=$EXP --resume
```

`ln -sfn` (`-n` = no-deref existing symlink) 确保 idempotent, 重复 launcher 启动不出错。

**lsyncd 兼容性 (gf2/gf3)**:
- gf2/gf3 之间有 lsyncd 双向 mirror `/data/shared/` 目录
- `/home/tim/local_ckpts` 在 `/dev/vda2` 不在 lsyncd scope, 不会被同步 ✓
- 而 `/home/tim/workspace` 是 symlink → `/data/shared/...` 在 lsyncd 范围, 千万 **不要直接写 ckpt 到** `<kai0>/checkpoints/<config>/<exp>` 真实目录 (旧 bug 多次因此损坏)

**keep_period 设置**:
- 100k step 训练: `keep_period=10000` (保留 10 个) 比 `2_000` (保留 50 个) 减少 5× 占用
- 50k step: `keep_period=10000` (保留 5 个) 大约 165GB; 默认 `2_000` 时 825GB 可能撑爆 /dev/vda2

**已知 ckpt 路径**:

| 实验 | 当前 ckpt 真实路径 | 所有者 |
|---|---|---|
| gf2 实验1 | `/home/tim/local_ckpts/pi05_flatten_fold_mix_b6000_p1200_init_mixed_1/task_a_mix_base6000_pure1200_new_norm_base_mixed_1` | gf2 |
| gf3 实验2 | `/home/tim/local_ckpts/pi05_flatten_fold_mix_b6000_p1200_init_pi05_base/task_a_mix_base6000_pure1200_new_norm_base_pi0.5` | gf3 |
| gf0 实验3 | `/vePFS/tim/gf0_local_ckpts/pi05_flatten_fold_mix_b6000_p1200_init_pi05_base_100k/task_a_mix_base6000_pure1200_new_norm_base_pi0.5_100000` | gf0 |
| gf1 #25 (历史) | `/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_a_new_pure_1200/task_a_new_pure_1200_new_norm` | gf1 (训练完成后可移到 local_ckpts) |

### 2.3 数据集 / Checkpoint 目录约定 (传统 view)

```
deepdive_kai0/
├── kai0/                              # 主代码 (uv venv at .venv/)
│   ├── .venv/                         # Python 3.11/3.12 (uv 管理)
│   ├── src/openpi/                    # openpi 主代码
│   ├── scripts/                       # train.py / compute_norm_states_fast.py / ...
│   ├── checkpoints/                   # 训练 ckpt 落地
│   │   ├── Task_A/mixed_1/            # MA-merged init 模型 (paper-grade)
│   │   │   ├── _CHECKPOINT_METADATA
│   │   │   ├── norm_stats.json
│   │   │   └── params/                # ~12 GB JAX/Flax 权重
│   │   └── pi05_flatten_fold_*/<exp_name>/  # 各训练 exp 的 ckpts
│   └── data/                          # 数据集软链入口
│       └── Task_A/
│           ├── vis_base/              # → 真实/模拟采集数据集
│           ├── kai0_base/             # → HF 官方 kai0 base
│           ├── kai0_dagger/           # → HF 官方 kai0 dagger
│           ├── kai0_advantage/        # → HF 官方 advantage (gf2/gf3 only)
│           └── self_built/            # 用户构建的混合数据集
│               ├── A_pure_1200/{base,val}/
│               ├── A_new_pure_1200/{base,val}/
│               ├── mix_apr28_450/{base,val}/
│               └── ...
├── train_scripts/                     # 训练 launcher / 数据脚本
│   ├── data/
│   │   ├── build_task_a_*.py          # 数据集构建脚本
│   │   └── compute_delta_norm_stats_fast.py
│   └── launch/
│       ├── run_*_gf0.sh / run_*_gf1.sh
│       └── run_gf2.sh / run_gf2_adv_est.sh
├── docs/                              # 文档
├── setup_env.sh                       # KAI0_DATA_ROOT / OPENPI_DATA_HOME 自动配置
└── install.sh                         # 一键安装环境
```

### 2.3 数据集源 (按机器)

#### gf0 / gf1 (共享 vePFS)
```
/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/
  base/                # 自建 (来自 visrobot01)
  dagger/              # 自建
  vis_base/<date>/     # 按日期分子集 (~310-644 ep)
  kai0_base/, kai0_dagger/
  self_built/A_pure_1200, A_new_pure_1200, mix_apr28_450, ...

/vePFS/visrobot01/KAI0/Task_A/base/<date>/  # 原始采集 (跨用户共享)
```

#### gf2 / gf3 (独立 4TB ext4)
```
/data/shared/dataset/KAI0/Task_<X>/base/         # 自建 (rsync from /vePFS)
/data/shared/dataset/Kai0_official/Task_A/      # HF 官方 base/dagger/advantage
~/workspace/deepdive_kai0/kai0/data/Task_<X>/   # symlinks 指向上述路径
```

### 2.4 临时 / 加速存储 (按机器)

| 路径 | gf0/gf1 | gf2/gf3 |
|---|---|---|
| `/dev/shm` (tmpfs RAM) | **1.3 TB** ⭐ 训练数据可加速 | 大 (具体大小待测) |
| `/tmp` | overlay ~99GB | overlay ~99GB |
| `/dev/nvme0n1` | (如有, 通常无独立挂载) | `/nix` 3.5TB ext4 NVMe |
| `/transfer-shanghai` | TOS bucket FUSE 挂载 (跨机文件传输) | 同 |

> ⚠️ **gf1 vePFS I/O 约 gf0 慢 2× (实测 video decode)**, 训练时**强烈推荐**先把数据集 cp 到 `/dev/shm` 再训。

---

## 3. 环境 (Python 栈)

### 3.1 venv 路径

| 机器 | venv 路径 | Python |
|---|---|---|
| gf0 | `/vePFS/tim/workspace/deepdive_kai0/kai0/.venv` → `/home/tim/.kai0_venv` (本地 symlink) | 3.11 |
| gf1 | 同 (但本地 venv 物理独立) | 3.11 |
| gf2 | `/home/tim/workspace/deepdive_kai0/kai0/.venv` (uv 管理) | 3.12 |
| gf3 | 同 gf2 (本地独立) | 3.12 |

> **注意**: 虽然 gf0/gf1 venv 路径相同 (因 vePFS 共享), 但实际是 `/vePFS/.../kai0/.venv` → `/home/tim/.kai0_venv` 软链, **两机各自指向各自的本地** `/home/tim/.kai0_venv`。所以两机 venv 物理独立, lib 版本可能微差。

### 3.2 关键依赖 (各机基本一致)

- **JAX** 0.5.3 + cuda12 (含 GPU)
- **PyTorch** 2.7.1+cu126 (gf2/gf3) / 与之兼容版本 (gf0/gf1)
- **Flax** 0.10.2 / orbax-checkpoint 0.11.13
- **openpi** (editable, in `kai0/src/openpi/`)
- **lerobot** (HF 库) / transformers / sentencepiece
- **tos** 2.9.0 (Volcengine, 用于 TOS 文件传输)

### 3.3 环境变量 (`setup_env.sh` 自动设置)

| 变量 | gf0/gf1 (`profile=gf`) | gf2/gf3 (`profile=default`) |
|---|---|---|
| `KAI0_DATA_ROOT` | `/vePFS/tim/workspace/deepdive_kai0/kai0` | `$HOME/workspace/deepdive_kai0/kai0` |
| `OPENPI_DATA_HOME` | `/vePFS/tim/workspace/openpi_cache` | `$HOME/workspace/openpi_cache` |
| `PYTORCH_CKPT_BASE` | `/vePFS/tim/workspace/openpi_cache/modelscope_cache/lerobot` | `$HOME/workspace/openpi_cache/modelscope_cache/lerobot` |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | 0.9 (set per-launcher) | 同 |
| `WANDB_MODE` | `offline` (无外网) | `offline` |
| `LD_LIBRARY_PATH` | 含 `/usr/local/cuda-12.8/...` + `/home/tim/.cuda_compat` | 含 `/usr/local/cuda-12.4/...` |
| `TORCH_CUDA_ARCH_LIST` | (default) | `"8.0"` (设在 `~/.bashrc`) |

### 3.4 已知的机器特定 workaround

| 现象 | 解决 |
|---|---|
| **gf1** inline-eval 报 `StreamBeginCaptureToGraph is not implemented for CUDA below version 12.3` | launcher 加 `export XLA_FLAGS="--xla_gpu_enable_command_buffer="` |
| **gf1** vePFS 文件 I/O 比 gf0 慢 2× (page cache 不积极) | 训练前先 `cp -rL <data> /dev/shm/<data>`, config repo_id 指向 `/dev/shm/...` |
| gf0/gf1 共享 vePFS, ckpt 同时写不同 exp_name 不冲突 | 用不同 exp_name 即可并发 |
| gf2/gf3 HF 下载 429 限流 | 单机优先 + retry, 然后 rsync 到另一机 |

---

## 4. 连接方式 / 用户信息

### 4.1 SSH 速查

```bash
# gf0 / gf1 (从 sim01 / 任意公网机)
ssh -p 55555 tim@14.103.44.161   # gf0 (反向隧道经 14.103.44.161 跳板)
ssh -p 11111 tim@14.103.44.161   # gf1

# gf2 / gf3 (直连)
sshpass -p tim ssh tim@117.50.196.104   # gf2
sshpass -p tim ssh tim@106.75.68.254    # gf3

# 也可在 ~/.bashrc 设别名:
alias gf2='sshpass -p "tim" ssh -o StrictHostKeyChecking=no tim@117.50.196.104'
alias gf3='sshpass -p "tim" ssh -o StrictHostKeyChecking=no tim@106.75.68.254'
```

### 4.2 用户

- 用户名: `tim` (全部 4 台一致)
- 密码: `tim` (gf2/gf3 sudo 也是 `tim`, no NOPASSWD)
- gf0/gf1: 通过反向隧道, 无需密码 (key-based)
- gf2/gf3: 用 `sshpass -p tim` 或配置 SSH key

### 4.3 TOS 跨机传输 (gf 集群 ↔ sim01 ↔ gf2/gf3)

bucket: `transfer-shanghai` @ `tos-cn-shanghai.volces.com` (region `cn-shanghai`)

```bash
# 上传到 TOS (gf 任意机)
.venv/bin/python train_scripts/data/to_tos_file.py <local_file>

# 下载从 TOS (gf 任意机 / sim01 / gf2/gf3)
.venv/bin/python train_scripts/data/from_tos_file.py <bucket_path>
```

凭据已 hardcoded 在 `from_tos_file.py` (read-key, 公开). 写权限通过 `VOLC_TOS_AK / VOLC_TOS_SK` 环境变量。

---

## 5. 训练快速启动 (按机器)

### 5.1 通用启动模板 (适配任一 gf 机)

```bash
ssh tim@<host>
cd ~/workspace/deepdive_kai0/kai0   # gf2/gf3
# 或 /vePFS/tim/workspace/deepdive_kai0/kai0   # gf0/gf1

# Step 1: 计算 norm_stats (新建 dataset 时必做)
.venv/bin/python scripts/compute_norm_states_fast.py --config-name <config_name>

# Step 2: 启动训练 (JAX 全参微调)
nohup bash train_scripts/launch/run_<config>_gf<N>.sh > /tmp/train_<config>.log 2>&1 &
disown $!
```

### 5.2 通用 Launcher 模板

```bash
#!/bin/bash
set -euo pipefail

export PATH=/home/tim/miniconda3/bin:/home/tim/.local/bin:$PATH
export PYTHONUNBUFFERED=1
export KAI0_DATA_ROOT=<see table 3.3>
export OPENPI_DATA_HOME=<see table 3.3>
export PYTORCH_CKPT_BASE=<see table 3.3>
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_DATASETS_CACHE=/home/tim/.cache/huggingface/datasets
export WANDB_MODE=offline
export LD_LIBRARY_PATH=...   # 见 3.3 LD_LIBRARY_PATH

# gf1 specific (CUDA 12.2 workaround):
# export XLA_FLAGS="--xla_gpu_enable_command_buffer="

cd <KAI0_DATA_ROOT>
.venv/bin/python scripts/train.py <config_name> --exp_name=<exp_name> --resume
```

### 5.3 Resume 行为

- `--resume` (推荐): 从 `<KAI0_DATA_ROOT>/checkpoints/<config>/<exp_name>/` 找最大 step 的 ckpt resume; 若无 ckpt, fallback 到 `weight_loader` 指定的 init params 冷启
- ⚠️ **永远不要用 `--overwrite`**: 该 flag rmtree 整个 exp 目录, 导致**所有 ckpt 不可逆丢失** (历史教训: 2026-04-24 误用导致 5k ckpts 全失)

### 5.4 数据集放本地加速 (gf1 强烈推荐)

```bash
# Stop training first if running
# Copy to /dev/shm (tmpfs, ~3 GB/s read)
mkdir -p /dev/shm/<dataset>
cp -rL /vePFS/.../<dataset> /dev/shm/

# Edit config.py: change repo_id to /dev/shm/<dataset>/base
# Restart training
```

**实测 gf1 v3 用 /dev/shm 后**: 步速 5.5 → 3.16 s/step (1.74× 加速), GPU util 80% idle → 100% busy。

### 5.5 自动打包 best ckpt (训练 END 后)

`train_scripts/util/auto_pack_on_end.sh` (or `/tmp/auto_pack_on_end.sh`):
- 监控训练 log 中 `[train] === END` marker
- 解析 inline-eval, 选 best step (lowest MAE@1)
- tar 打包 `params + _CHECKPOINT_METADATA + assets/` (不含 `train_state/`)

```bash
nohup bash /tmp/auto_pack_on_end.sh \
  /tmp/train_<exp>.log \
  <ckpt_root>/<config>/<exp_name> \
  <out_tar_path> \
  > /tmp/auto_pack_<exp>.run.log 2>&1 &
disown $!
```

---

## 6. 机器间数据同步

### 6.1 gf0 ↔ gf1 (vePFS 共享, 无需同步)

直接读写 `/vePFS/...` 路径, 共享 GPFS。一边写另一边立即可见。

### 6.2 gf 集群 ↔ gf2/gf3

| 方法 | 适用 | 命令 |
|---|---|---|
| **TOS** | 大文件 (ckpt tar, 大 dataset) | `to_tos_file.py` 上传 + `from_tos_file.py` 下载, 走公网, ~85 MB/s |
| **rsync 直连** | 文档代码小文件 | gf2 ↔ gf3 内网直连 (gbps), gf0 → gf2 走公网 |
| **GitHub** | 代码 (`.gitignore` 排除大文件) | `git push origin main` + `git pull` |

### 6.3 sim01 ↔ gf 集群

历史路径: gf 集群通过 SSH 反向隧道 (端口 29290) 出公网. sim01 通过 TOS 拉 ckpt:

```bash
# gf 集群上传 ckpt
sudo cp <tar> /transfer-shanghai/KAI0/<name>.tar

# sim01 下载
cd /data1/DATA_IMP/KAI0/ckpt_downloads/<name>
.venv/bin/python ~/workspace/deepdive_kai0/web/data_manager/backend/tools/from_tos_file.py <name>.tar
tar -xf <name>.tar
```

---

## 7. 常见运维 / 故障排查

### 7.1 GPU 利用率低 (训练慢)

| 症状 | 可能原因 | 排查 / 解决 |
|---|---|---|
| GPU util 0% / 99% 周期性切换, 平均 20% | dataloader I/O 瓶颈 | 检查 `top` 看 `pt_data+` workers CPU; 数据放 `/dev/shm` |
| GPU util 99% 但步速慢 | 训练计算密集 (无瓶颈) | 正常, 不需修复 |
| 步速波动大 (3-15 s/step) | vePFS I/O 不稳 / NCCL 同步抖动 | 看 buff/cache 是否积累 (`free -h`) |
| inline-eval 报 CUDA 错 | gf1 CUDA 12.2 < 12.3 | launcher 加 `XLA_FLAGS="--xla_gpu_enable_command_buffer="` |

### 7.2 vePFS 满 (gf0/gf1)

vePFS 99% used (50T / ~533G 余量). 注意:
- 不要再多放训练 ckpt (每个 12-30 GB)
- 老 ckpt 主动清理 / 打包到 TOS
- 检查命令: `df -hT /vePFS`

### 7.3 训练崩溃 / GPU 占用未释放

```bash
# 找进程
pgrep -af 'pi05_flatten_fold' | head

# 优雅停止
kill -SIGTERM <pid>
sleep 10
ps -p <pid>   # 验证已停

# 强制停止 (慎用, 可能损坏 ckpt)
kill -SIGKILL <pid>

# 验证 GPU 释放
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

### 7.4 Locale warning (gf2/gf3)

每次 SSH 都会 `setlocale: LC_ALL: cannot change locale (zh_CN.UTF-8)`. 无功能影响, 可加 `export LC_ALL=C.UTF-8` 到 `~/.bashrc`。

---

## 8. 训练实验命名约定

```
<config_name>:    pi05_flatten_fold_<dataset_label>
<exp_name>:       <experiment_descriptor>_<version>
ckpt_path:        ${KAI0_DATA_ROOT}/checkpoints/<config>/<exp_name>/<step>/

例:
  config:   pi05_flatten_fold_mix_apr28_450
  exp_name: mix_apr28_450_v1
  ckpt:     /vePFS/.../checkpoints/pi05_flatten_fold_mix_apr28_450/mix_apr28_450_v1/28000/
```

---

## 9. 各机当前用途分工 (2026-05 状态)

| 机器 | 主用途 | 典型负载 |
|---|---|---|
| **gf0** | Task_A 全参 fine-tune (主战) | 50k step 长训, vePFS 数据 |
| **gf1** | Task_A 全参 fine-tune (副战) | 同 gf0, 但需 /dev/shm 加速 |
| **gf2** | (待用) Advantage Estimator / AWBC 训练 | 数据本地, 无 vePFS 拥挤问题 |
| **gf3** | (待用) gf2 副本 / 并行实验 | 同 gf2 |

---

## 10. 部署 Ckpt 工作流 (训练 → sim01 推理)

```
训练完成 (gf*)
  ↓
auto_pack_on_end.sh: 选 best step, 打包 params+assets+METADATA
  ↓
TOS upload (gf*) → /transfer-shanghai/KAI0/<name>.tar
  ↓
sim01 from_tos_file.py download
  ↓
sim01 解压 + symlink 到 kai0/checkpoints
  ↓
serve_policy.py 启动推理服务
```

详见 [`sim01_deployment.md`](./sim01_deployment.md) 与 [`checkpoints_layout.md`](./checkpoints_layout.md)。

---

## 11. 实测性能基线 (参考)

| 配置 | 机器 | 步速 (s/step) | 备注 |
|---|---|---:|---|
| pi05 全参 fine-tune, batch=128, fsdp=8, vePFS data | gf0 | **2.0** | 基准, 数据热 cache |
| 同上 | gf1 | 5.5 | vePFS 数据冷, dataloader bound |
| 同上, data on /dev/shm | gf1 | **3.16** | 修复后, GPU 100% util |
| 同上 | gf2/gf3 | (待测) | 期望 ~2-3 s/step |

inline-eval 时间 (200 frames 采样):
- 17 val ep: 660s
- 22 val ep: 850s
- 40 val ep: 1525s
- 57 val ep: 2300s
- 60 val ep: 1170s (gf0 mixed_173, val 集不同导致差异)

---

## 12. 修订历史

| 日期 | 内容 |
|---|---|
| 2026-05-02 | 初版: 整合 gf0/gf1/gf2/gf3, 含 v3 /dev/shm 加速实测 |

后续更新: 添加 gf2/gf3 实际训练性能基线 / sim01 ↔ gf 集群网络拓扑细节。
