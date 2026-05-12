# 训练服务器知识库 (gf0 / ~~gf1~~ / uc01 / uc02 / uc03 / js01 / js02 / js03 / js04)

> ⚠️ **2026-05-06 更新: gf1 已退役**
> ⚠️ **2026-05-08 更新: gf2/3/4 → uc01/02/03 重命名; uc03 (原 gf4) 加入训练**
> ⚠️ **2026-05-11 更新: 日期 leaf 命名统一为 `YYYY-MM-DD-v2` (见 §2.3 末)**
> ⭐ **2026-05-12 更新: js01–04 (A800×8 ×4 台) 加入集群** — 见 §14。共享 FS `/mnt/data` (JuiceFS:visincept 10T 跨 4 台), 本机 NVMe 14–28T, 200 Gb/s IB, CUDA toolkit 12.2。
>
> **当前 active 服务器: gf0, uc01, uc02, uc03, js01, js02, js03, js04** (8 台)。gf1 跳板 `14.103.44.161:11111` SSH 不通, 服务器已下线。
>
> 历史影响:
> - **gf1 #25** (`task_a_new_pure_1200_new_norm` 50k 训练) 在 step 40000 因 ENOSPC crash; best ckpt @ step 38000 (MAE@1=0.0104) 已通过 TOS 拉到 sim01 `/data1/DATA_IMP/checkpoints/task_a_new_pure_1200_new_norm_best_step38000/`
> - **gf1 上 /vePFS 仍存在** (gf0 共享访问), 历史 ckpt + dataset 仍可从 gf0 操作
> - **本地 .bashrc 已删 gf1 alias** (`alias gf1=...` 已注释)
> - 后续遇到 "需要 gf1" 的脚本 / 文档 → 切换到 gf0 (走 /vePFS 共享, 同访问权限)
>
> 下面表格中保留 gf1 列以保留历史信息, 但请视作 **"已下线 / 不可用"**。



> **作用**: 4 台 GPU 训练服务器的全方位参考 — 硬件、文件结构、环境、连接方式、训练命令、机器间差异、常见运维。
> **更新日期**: 2026-05-02
> **关联文档**:
> - [`gf2_gf3_deployment.md`](./gf2_gf3_deployment.md) — uc01/uc02 详细部署记录
> - [`sim01_deployment.md`](./sim01_deployment.md) — sim01 推理机部署
> - [`checkpoints_layout.md`](./checkpoints_layout.md) — ckpt 文件结构规范

---

## 1. 服务器全景

> **当前 active**: gf0 + uc01/02/03 + js01/02/03/04 = **8 台**。gf1 已下线 (2026-05-06)。
> 三大集群: **gf**(vePFS 共享) / **uc**(独立, lsyncd 部分镜像) / **js**(JuiceFS 半共享, 见 §14)。

| 维度 | **gf0** | ~~gf1~~ | **uc01** | **uc02** | **uc03** | **js01** | **js02** | **js03** | **js04** |
|---|---|---|---|---|---|---|---|---|---|
| **状态** | active | retired | active | active | active | active (新) | active (新) | active (新) | active (新) |
| **GPU** | 8× A100-80GB | 8× A100-80GB | 8× A800-80GB | 8× A800-80GB | 8× A800-80GB | 8× A800-80GB | 同 | 同 | 同 |
| **GPU arch** | sm_80 | sm_80 | sm_80 | sm_80 | sm_80 | sm_80 | sm_80 | sm_80 | sm_80 |
| **驱动 / CUDA driver** | 535.129.03 / 12.2 | 535.129.03 / 12.2 | 550.144.03 / 12.4 | 550.144.03 / 12.4 | 550.144.03 / 12.4 | 535.183.06 / 12.2 | 同 js01 | 同 | 同 |
| **CUDA toolkit** | 12.8 | 12.8 | 12.4 | 12.4 | 12.4 | 12.2 | 12.2 | 12.2 | 12.2 |
| **CPU** | Xeon 8336C, 112c | 同 gf0 | Xeon 8358P, 124c | 同 uc01 | 同 uc01 | Xeon 8380, 160c | 同 js01 | 同 | 同 |
| **RAM** | 1.8 TiB | 1.8 TiB | ~1.7 TiB | ~1.7 TiB | ~1.7 TiB | 1.0 TiB | 同 js01 | 同 | 同 |
| **/dev/shm** | 1.3 TB | 1.3 TB | (待测) | (待测) | (待测) | 504 GB | 504 GB | 504 GB | 504 GB |
| **OS** | Debian-velinux1u1 | 同 gf0 | Ubuntu 22.04 | 同 uc01 | 同 uc01 | Ubuntu 22.04.3 | 同 js01 | 同 | 同 |
| **Hostname** | `di-20260312174527-n5dw4` | `di-20260320201920-rzbrm` | `10-60-135-47` | `10-60-204-66` | (uc03) | `10-0-1-75` | `10-0-1-105` | `10-0-1-20` | `10-0-1-91` |
| **IP / 入口** | 跳板 `14.103.44.161:55555` (反向隧道) | 跳板 `14.103.44.161:11111` ❌ | `117.50.196.104` 直连 | `106.75.68.254` 直连 | `117.50.217.231` 直连 | **`120.92.149.234`** 公网直连 | 仅内网 `10.0.1.105` (经 js01) | 仅内网 `10.0.1.20` (经 js01) | 仅内网 `10.0.1.91` (经 js01) |
| **本地 SSH 别名** | `ssh -p 55555 tim@14.103.44.161` | `ssh -p 11111 tim@14.103.44.161` | `uc01` (bashrc) | `uc02` (bashrc) | `uc03` (bashrc) | `js01` (bashrc) | `js02` (bashrc + ProxyJump) | `js03` 同 | `js04` 同 |
| **共享 FS** | /vePFS (gpfs, 50T, 跨 gf0/gf1) | 同 gf0 | **无** (本机独立) | **无** | **无** | `/mnt/data` JuiceFS:visincept 10T (跨 js01-04) | 同 | 同 | 同 |
| **本机大盘** | (overlay 99G, 用 vePFS) | 同 gf0 | `/data/shared` 4TB ext4 | 同 uc01 | 同 uc01 + `/nix` 3.5T NVMe | `/DATA/disk0` 14T NVMe | `/DATA/disk0/1` 2×7T=14T | `/DATA/disk0..3` **28T** | `/DATA/disk0..3` **28T** |
| **InfiniBand** | (无) | (无) | 4× Mellanox CX-6 200 Gb/s RoCEv2 | 同 uc01 | 同 uc01 | mlx5 多块 200 Gb/s | 同 js01 | 同 | 同 |
| **多机训练** | 单机 | — | uc01+02+03 HSDP/FSDP (§13) | 同 | 同 | 4 台 NCCL+IB (规划, §14.10) | 同 | 同 | 同 |
| **Python / venv** | 3.11 | 3.11 | 3.12 | 同 | 同 | 3.12 / `/DATA/disk0/tim/.kai0_venv` | 同 js01 | 同 | 同 |

> **快速归类规则**:
> - **gf 集群** = vePFS 全共享 (代码+数据+ckpt 都共享, venv 各机独立)
> - **uc 集群** = 完全独立 + 200 Gb/s RoCEv2 多机训练 (§13)
> - **js 集群** = JuiceFS 半共享 (代码+数据共享, venv+ckpt 各机本地) + IB 多机训练潜力

---

## 2. 文件结构

### 2.1 工作目录路径速查

| 服务器 | 工作目录 | 实际存储 |
|---|---|---|
| gf0 | `/vePFS/tim/workspace/deepdive_kai0/` (= `/home/tim/workspace/deepdive_kai0` 软链) | gpfs 跨机共享 |
| gf1 | 同 gf0 (共享) | 同 |
| uc01 | `/home/tim/workspace/deepdive_kai0/` → `/data/shared/tim/workspace/deepdive_kai0/` | 本机 4TB ext4 |
| uc02 | 同 uc01 (各自独立, 不共享) | 同 uc01 |

### 2.2 Checkpoint 本地存储规范 ⭐ (2026-05-04 重要更新)

> **核心原则**: 每台服务器的 ckpt 写到独立的本地路径, 不跨机同步, 重启不丢失。

**统一路径**: 每台机器都使用 `/home/tim/local_ckpts/` 作为 ckpt 根目录 (其中是 symlink 还是 real dir 因机器而异)。

| Server | `/home/tim/local_ckpts/` 实现 | 物理后端 | 容量 | 持久性 |
|---|---|---|---|---|
| gf0 | symlink → `/vePFS/tim/gf0_local_ckpts/` | /vePFS (50T 共享 FS) | 看 /vePFS 余量 | ✓ 持久 |
| ~~gf1~~ (已退役) | symlink → `/vePFS/tim/gf1_local_ckpts/` | /vePFS | 已下线 | — |
| uc01 | 真实 dir | /dev/vda2 (492G ext4) | ~290G 可用 | ✓ 持久 |
| uc02 | 真实 dir | /dev/vda2 (492G ext4) | ~410G 可用 | ✓ 持久 |

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

**lsyncd 兼容性 (uc01/uc02)**:
- uc01/uc02 之间有 lsyncd 双向 mirror `/data/shared/` 目录
- `/home/tim/local_ckpts` 在 `/dev/vda2` 不在 lsyncd scope, 不会被同步 ✓
- 而 `/home/tim/workspace` 是 symlink → `/data/shared/...` 在 lsyncd 范围, 千万 **不要直接写 ckpt 到** `<kai0>/checkpoints/<config>/<exp>` 真实目录 (旧 bug 多次因此损坏)

**keep_period 设置**:
- 100k step 训练: `keep_period=10000` (保留 10 个) 比 `2_000` (保留 50 个) 减少 5× 占用
- 50k step: `keep_period=10000` (保留 5 个) 大约 165GB; 默认 `2_000` 时 825GB 可能撑爆 /dev/vda2

**已知 ckpt 路径**:

| 实验 | 当前 ckpt 真实路径 | 所有者 |
|---|---|---|
| uc01 实验1 | `/home/tim/local_ckpts/pi05_flatten_fold_mix_b6000_p1200_init_mixed_1/task_a_mix_base6000_pure1200_new_norm_base_mixed_1` | uc01 |
| uc02 实验2 | `/home/tim/local_ckpts/pi05_flatten_fold_mix_b6000_p1200_init_pi05_base/task_a_mix_base6000_pure1200_new_norm_base_pi0.5` | uc02 |
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
│           ├── kai0_advantage/        # → HF 官方 advantage (uc01/uc02 only)
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

#### uc01 / uc02 (独立 4TB ext4)
```
/data/shared/dataset/KAI0/Task_<X>/base/         # 自建 (rsync from /vePFS)
/data/shared/dataset/Kai0_official/Task_A/      # HF 官方 base/dagger/advantage
~/workspace/deepdive_kai0/kai0/data/Task_<X>/   # symlinks 指向上述路径
```

#### 日期 leaf 命名约定: `YYYY-MM-DD-v2` (2026-05-11 起)

历史上 `base/` 下日期 leaf 直接是 `YYYY-MM-DD`. 4-23 ~ 4-30 的数据被处理后另存为 `YYYY-MM-DD-v2`. **2026-05-11 起统一**: 所有新采集直接写 `YYYY-MM-DD-v2`, 不再区分"原始"与"处理后".

- **写入**: `web/data_manager/backend/app/layout.py:new_task_subset_root()` 给今日日期附加 `-v2`. 受影响调用方: `recorder.py::start_recording` (web UI 采集).
- **读取**: `_DATE_RE = r"^\d{4}-\d{2}-\d{2}(?:-v\d+)?$"` 同时匹配两种, `path_to_compound()` 把 `-v2` 保留在 task_id 中 (e.g. `Task_A_2026-05-11-v2`).
- **历史**: 2026-05-11 一次性把 5-06 ~ 5-09 (sim01 / TOS / uc01-uc03 共 4 端) 全部 `mv old → old-v2`. 期间 uc02/uc03 因 lsyncd `--update` 不删旧, 手动 rm 残留. Task_PP/5-09 当时仅 sim01 有, 下次 sync 直接以 -v2 上 TOS.

### 2.4 临时 / 加速存储 (按机器)

| 路径 | gf0/gf1 | uc01/uc02 |
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
| uc01 | `/home/tim/workspace/deepdive_kai0/kai0/.venv` (uv 管理) | 3.12 |
| uc02 | 同 uc01 (本地独立) | 3.12 |

> **注意**: 虽然 gf0/gf1 venv 路径相同 (因 vePFS 共享), 但实际是 `/vePFS/.../kai0/.venv` → `/home/tim/.kai0_venv` 软链, **两机各自指向各自的本地** `/home/tim/.kai0_venv`。所以两机 venv 物理独立, lib 版本可能微差。

### 3.2 关键依赖 (各机基本一致)

- **JAX** 0.5.3 + cuda12 (含 GPU)
- **PyTorch** 2.7.1+cu126 (uc01/uc02) / 与之兼容版本 (gf0/gf1)
- **Flax** 0.10.2 / orbax-checkpoint 0.11.13
- **openpi** (editable, in `kai0/src/openpi/`)
- **lerobot** (HF 库) / transformers / sentencepiece
- **tos** 2.9.0 (Volcengine, 用于 TOS 文件传输)

### 3.3 环境变量 (`setup_env.sh` 自动设置)

| 变量 | gf0/gf1 (`profile=gf`) | uc01/uc02 (`profile=default`) |
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
| uc01/uc02 HF 下载 429 限流 | 单机优先 + retry, 然后 rsync 到另一机 |

---

## 4. 连接方式 / 用户信息

### 4.1 SSH 速查

```bash
# gf0 / gf1 (从 sim01 / 任意公网机)
ssh -p 55555 tim@14.103.44.161   # gf0 (反向隧道经 14.103.44.161 跳板)
ssh -p 11111 tim@14.103.44.161   # gf1

# uc01 / uc02 (直连)
sshpass -p tim ssh tim@117.50.196.104   # uc01
sshpass -p tim ssh tim@106.75.68.254    # uc02

# 也可在 ~/.bashrc 设别名:
alias uc01='sshpass -p "tim" ssh -o StrictHostKeyChecking=no tim@117.50.196.104'
alias uc02='sshpass -p "tim" ssh -o StrictHostKeyChecking=no tim@106.75.68.254'
```

### 4.2 用户

- 用户名: `tim` (全部 4 台一致)
- 密码: `tim` (uc01/uc02 sudo 也是 `tim`, no NOPASSWD)
- gf0/gf1: 通过反向隧道, 无需密码 (key-based)
- uc01/uc02: 用 `sshpass -p tim` 或配置 SSH key

### 4.3 TOS 跨机传输 (gf 集群 ↔ sim01 ↔ uc01/uc02)

bucket: `transfer-shanghai` @ `tos-cn-shanghai.volces.com` (region `cn-shanghai`)

```bash
# 上传到 TOS (gf 任意机)
.venv/bin/python train_scripts/data/to_tos_file.py <local_file>

# 下载从 TOS (gf 任意机 / sim01 / uc01/uc02)
.venv/bin/python train_scripts/data/from_tos_file.py <bucket_path>
```

凭据已 hardcoded 在 `from_tos_file.py` (read-key, 公开). 写权限通过 `VOLC_TOS_AK / VOLC_TOS_SK` 环境变量。

---

## 5. 训练快速启动 (按机器)

### 5.1 通用启动模板 (适配任一 gf 机)

```bash
ssh tim@<host>
cd ~/workspace/deepdive_kai0/kai0   # uc01/uc02
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

### 6.2 gf 集群 ↔ uc01/uc02

| 方法 | 适用 | 命令 |
|---|---|---|
| **TOS** | 大文件 (ckpt tar, 大 dataset) | `to_tos_file.py` 上传 + `from_tos_file.py` 下载, 走公网, ~85 MB/s |
| **rsync 直连** | 文档代码小文件 | uc01 ↔ uc02 内网直连 (gbps), gf0 → uc01 走公网 |
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

### 7.4 Locale warning (uc01/uc02)

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
| **uc01** | (待用) Advantage Estimator / AWBC 训练 | 数据本地, 无 vePFS 拥挤问题 |
| **uc02** | (待用) uc01 副本 / 并行实验 | 同 uc01 |

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
| 同上 | uc01/uc02 | (待测) | 期望 ~2-3 s/step |

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
| 2026-05-12 | 添加 §14: js01-04 集群 (A800×8 ×4, JuiceFS 共享, IB, CUDA 12.2, 部署 + 数据同步全记录); §1 全景表扩到 8 台 |
| 2026-05-12 | 添加 section 13: 3-host HSDP/FSDP 集群训练 + RDMA + GDR + NCCL 配置 + 坑 |
| 2026-05-02 | 初版: 整合 gf0/gf1/uc01/uc02, 含 v3 /dev/shm 加速实测 |

后续更新: 添加 uc01/uc02 实际训练性能基线 / sim01 ↔ gf 集群网络拓扑细节 / js 集群多机 NCCL+IB 实测带宽 / 数据集落完后补 js 路径快查。

---

## 13. 3-Host HSDP/FSDP 集群训练 (uc01 + uc02 + uc03) ⭐ (2026-05-12)

**硬件**: 三台一致 — 8× A800-SXM4-80GB (NVLink 200 GB/s), 124 核 Xeon 8358P, 1.7 TB RAM, 4× Mellanox ConnectX-6 (200 Gb/s RoCEv2 each)

**关键能力**: 24 GPU 集群训练，RDMA + GPU Direct RDMA (GDR) 启用后跨主机带宽 ~800 Gb/s。

### 13.1 网络架构 (易误判)

| 网卡 | 用途 | 关键事实 |
|---|---|---|
| `eth0` (10.x.x.x) | 管理 / 公网 | virtio_net, MTU 1452, **慢，仅控制面** |
| `eth1-4` (192.168.{1-4}.x/24) | **训练通信** | **mlx5_core, 200 Gbps, RoCEv2, MTU 4200** |

**判断方法**:
```bash
# 网卡驱动 + 速率
for n in eth1 eth2 eth3 eth4; do
  echo -n "$n: "
  ethtool -i $n 2>/dev/null | grep "^driver:"
  cat /sys/class/net/$n/speed 2>/dev/null  # 应为 200000
done

# RoCE GID (NCCL 用 v2 / IPv4-mapped)
show_gids | head -20  # GID INDEX 3 = RoCEv2 IPv4 mapped

# nvidia-peermem 内核模块 (GDR 必需)
lsmod | grep nvidia_peermem
```

**主机间 IP 拓扑** (上 4 个 NIC 各独立 /24，平面无 router):
- uc01: 192.168.1.2, 192.168.2.2, 192.168.3.2, 192.168.4.2
- uc02: 192.168.1.3, 192.168.2.3, 192.168.3.3, 192.168.4.3
- uc03: 192.168.1.4, 192.168.2.4, 192.168.3.4, 192.168.4.4

### 13.2 NCCL 配置 — 必须启用 RDMA + GDR

❌ **错误（之前用过的，速度只有 ~26 Gbps × 4 NIC ≈ 100 Gbps）**:
```bash
export NCCL_IB_DISABLE=1           # ❌ 关 IB 走 TCP socket
export NCCL_NET_TYPE=Socket        # ❌
export NCCL_NET_GDR_LEVEL=0        # ❌ 关 GDR
```

✅ **正确（RDMA + GDR，~800 Gbps）**:
```bash
unset NCCL_IB_DISABLE NCCL_NET_TYPE NCCL_NET_GDR_LEVEL NCCL_NET_GDR_READ
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_GID_INDEX=3              # RoCEv2 IPv4 mapped
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_P2P_LEVEL=NVL               # 节点内 NVLink P2P
export NCCL_SOCKET_IFNAME=eth1          # 控制面 bootstrap
export NCCL_DEBUG=INFO                  # 第一次跑确认 transport
# 不要手动设 NCCL_MAX_NCHANNELS / NCCL_BUFFSIZE — 让 NCCL 自适应
```

**验证 NCCL 真用 IB**（log 应出现）:
```
NET/IB : Made virtual device [0..3] name=mlx5_0..3 speed=200000 ndevs=1
Using [0]mlx5_0:1/RoCE [1]mlx5_1:1/RoCE [2]mlx5_2:1/RoCE [3]mlx5_3:1/RoCE
NET/IB : GPU Direct RDMA Enabled for HCA 0..3
Channel XX/0 : N[i] -> M[i] [send] via NET/IB/X/GDRDMA
```

### 13.3 JAX/XLA 配置

```bash
# JAX distributed (3 host)
export JAX_COORDINATOR_ADDRESS=192.168.1.2:15830  # uc01 via Mellanox eth1
export JAX_NUM_PROCESSES=3
export JAX_PROCESS_INDEX=$PROC  # 0 (uc01) / 1 (uc02) / 2 (uc03)
export JAX_ENABLE_EMPTY_ARRAYS=true   # HSDP 必需

# 持久化编译缓存 (本地, 缓存按 HLO hash 索引)
# train.py:420 已设 jax_compilation_cache_dir = ~/.cache/jax
export JAX_COMPILATION_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_COMPILATION_CACHE_MIN_COMPILE_TIME_SECS=1

# 留余地给 NCCL RDMA buffer (默认 0.95 太激进会导致 NCCL alloc 失败)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

# 不要设 XLA_FLAGS='--xla_gpu_enable_command_buffer='（空值会禁用 CUDA Graph）
# 也不要传 COLLECTIVE token (不存在，会 flag parse failed → Aborted)
unset XLA_FLAGS  # 用 XLA 默认即可
```

### 13.4 Mesh / FSDP 选择 ⚠️ 关键

| 方案 | `fsdp_devices` | mesh | 编译时间 | rate (pi05) | 备注 |
|---|---|---|---|---|---|
| HSDP `[3,8]` | 8 | `[dp=3, fsdp=8]` | **30-45 分钟首次（命中后秒级）** | **~1.0 s/it** | 节点内 sharded, 节点间 replica。命中 cache 后最快 |
| 全 FSDP `[1,24]` | 24 | `[fsdp=24]` | **5-10 分钟首次** | ~1.2 s/it | 没有 mesh 转换 → SPMD partition 简单 |
| 单机 `[8]` | 8 | `[fsdp=8]` | 5-10 分钟 | 0.5-0.7 s/it | 备选，不需要多机 |

**HSDP 巨坑 ⚠️**: SPMD partitioner 在 mesh `[24]→[3,8]T(1,0)` 转换时如果命中"Involuntary full rematerialization"慢路径，会**死锁 50-100+ 分钟**。
- 触发条件: HLO 缓存未命中（weight_loader 路径变化 / batch / 模型架构改了）
- 症状: master CPU 满载 600-800%，但 ~/.cache/jax 不写新文件、日志 30 分钟+ 不动、NCCL clique init `for 10 seconds and may be stuck`
- 解决: 切到全 FSDP (`fsdp_devices=24`)，或者重用已编译过的相同 HLO

**HLO 缓存命中条件** (全部一致才命中):
- batch_size, mesh (fsdp_devices + num_processes)
- 模型架构 (pi05 / pi0)
- `weight_loader` 路径（**包括 ckpt path 字符串**，会进 HLO closure）
- dataset action_dim
- JAX / XLA 版本

### 13.5 共享存储 (NFS on uc01)

```
uc01 /etc/exports: /data/cluster_ckpt 192.168.1.0/24(rw,sync,no_subtree_check,no_root_squash)
uc02/uc03 /etc/fstab: 192.168.1.2:/data/cluster_ckpt /cluster_ckpt nfs vers=4,hard,intr,timeo=600,rsize=1048576,wsize=1048576
```

**用途**:
- Orbax CheckpointManager 跨主机一致性 (POSIX, 必须共享)
- 数据集 (~115 GB 训练数据集放 NFS 训练时实测 GPU 99% util, NFS 没成为瓶颈)

**带宽**: write ~219 MB/s, read ~2 GB/s (单 stream NFSv4 over TCP)，跨 host 直传走 RoCE NIC eth1。

### 13.6 集群训练启动脚本模板 (`/tmp/run_cluster_3host.sh`)

```bash
#!/bin/bash
set -euo pipefail

CONFIG="<your_config_name>"
EXP_NAME="<exp_name>"
COORD_ADDR="192.168.1.2:15830"
LOG_DIR=/home/tim/workspace/deepdive_kai0/logs
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
mkdir -p $LOG_DIR

NCCL_OPTS='
unset NCCL_IB_DISABLE NCCL_NET_TYPE NCCL_NET_GDR_LEVEL NCCL_NET_GDR_READ
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=eth1
unset NCCL_MAX_NCHANNELS NCCL_MIN_NCHANNELS NCCL_BUFFSIZE
export NCCL_DEBUG=INFO
'

TRAIN_CMD="cd /home/tim/workspace/deepdive_kai0/kai0 && .venv/bin/python -u scripts/train.py $CONFIG --exp_name=$EXP_NAME --seed=123 --overwrite --no-wandb-enabled"

launch_worker() {
  local TGT=$1 PROC=$2 TAG=$3
  ssh -o StrictHostKeyChecking=no $TGT "
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export PATH=/home/tim/miniconda3/bin:/home/tim/.local/bin:\$PATH
export PYTHONUNBUFFERED=1
export KAI0_DATA_ROOT=/home/tim/workspace/deepdive_kai0/kai0
export KAI0_LOCAL_ROOT=/home/tim/local_ckpts
export OPENPI_DATA_HOME=/home/tim/workspace/openpi_cache
export JAX_COORDINATOR_ADDRESS=$COORD_ADDR
export JAX_NUM_PROCESSES=3
export JAX_PROCESS_INDEX=$PROC
export JAX_ENABLE_EMPTY_ARRAYS=true
export JAX_COMPILATION_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_COMPILATION_CACHE_MIN_COMPILE_TIME_SECS=1
unset XLA_FLAGS
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
$NCCL_OPTS
export WANDB_MODE=offline
mkdir -p $LOG_DIR
nohup bash -c '$TRAIN_CMD' > $LOG_DIR/run_${TAG}_${TIMESTAMP}.log 2>&1 &
echo \"[${TAG} proc${PROC}] pid=\$!\"
disown
"
}

launch_worker "tim@192.168.1.3" 1 "uc02"
launch_worker "tim@192.168.1.4" 2 "uc03"
sleep 5

# uc01 master (local exec — 复制相同 env)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export PATH=/home/tim/miniconda3/bin:/home/tim/.local/bin:$PATH
export PYTHONUNBUFFERED=1
export KAI0_DATA_ROOT=/home/tim/workspace/deepdive_kai0/kai0
export KAI0_LOCAL_ROOT=/home/tim/local_ckpts
export OPENPI_DATA_HOME=/home/tim/workspace/openpi_cache
export JAX_COORDINATOR_ADDRESS=$COORD_ADDR
export JAX_NUM_PROCESSES=3 JAX_PROCESS_INDEX=0
export JAX_ENABLE_EMPTY_ARRAYS=true
export JAX_COMPILATION_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_COMPILATION_CACHE_MIN_COMPILE_TIME_SECS=1
unset XLA_FLAGS
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
unset NCCL_IB_DISABLE NCCL_NET_TYPE NCCL_NET_GDR_LEVEL NCCL_NET_GDR_READ
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23 NCCL_IB_RETRY_CNT=7 NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=eth1
unset NCCL_MAX_NCHANNELS NCCL_MIN_NCHANNELS NCCL_BUFFSIZE
export NCCL_DEBUG=INFO
export WANDB_MODE=offline
cd /home/tim/workspace/deepdive_kai0/kai0
nohup .venv/bin/python -u scripts/train.py $CONFIG --exp_name=$EXP_NAME --seed=123 --overwrite --no-wandb-enabled > $LOG_DIR/run_uc01_${TIMESTAMP}.log 2>&1 &
echo "[uc01 proc0] pid=$!"
disown
```

### 13.7 配置同步 (必做)

`config.py` 必须 3 host 一致 — worker 各自从本地读，不会自动从 master 拉:

```bash
# 改完 config.py 后:
CFG=/home/tim/workspace/deepdive_kai0/kai0/src/openpi/training/config.py
scp $CFG tim@192.168.1.3:$CFG
scp $CFG tim@192.168.1.4:$CFG
# verify
ssh tim@192.168.1.3 "grep -c '<new_config_name>' $CFG"
ssh tim@192.168.1.4 "grep -c '<new_config_name>' $CFG"
```

### 13.8 自建数据集时常见陷阱

**陷阱 A: parquet schema 不一致** (跨数据源合并)
- Task_A/base: 7 列标准 (`observation.state, action, timestamp, frame_index, episode_index, index, task_index`)
- Task_A/advantage: 12 列 (多 `progress_gt, stage_progress_gt, relative_advantage, absolute_value, absolute_advantage`)
- 合并后 `load_dataset("parquet", ...)` 会 `CastError: column names don't match`
- **修复**: 重写所有非标准 parquet 只 `select(KEEP_COLS)`

**陷阱 B: episode_index 重新编号必须改 parquet 列**
- 简单 rename 文件不够 — parquet 内 `episode_index` 和 `index` (running counter) 列必须同步重写
- mp4 可以直接 hardlink (同 `/dev/vdb` 分区秒级)

**陷阱 C: Orbax CheckpointManager metadata hash 不一致**
- 症状: `AssertionError: sync_global_devices name mismatch ('CheckpointManager:save_root_metadata') Expected: X; got: Y`
- 原因: NFS metadata stale + 上次启动残留 — 3 host `os.listdir()` 看到不同内容
- 修复: 先 `rm -rf $checkpoint_dir; sync; sleep 1` 再启动

**陷阱 D: train.py 不自动算 norm_stats**
- train.py:438 只 `shutil.copy(data.repo_id / 'norm_stats.json', ckpt_dir)`
- 必须先 `python scripts/compute_norm_states_fast.py --config-name <name>` 算好
- LeRobotDataset 在 init 时不验证 norm_stats，但 Normalize transform 会用

**陷阱 E: dataloader KeyError: 1 大量出现**
- 这是 lerobot 内部 retry 容错日志（`TransformedDataset.__getitem__` retry 50 次）
- 多数情况是 advantage 数据集 ~10% 视频缺失或 LeRobot timestamp lookup 失败
- 不是致命错误，会 skip + 重抽 — 但**会让日志极难看**，掩盖真正错误

**陷阱 F: 数据本地存储 / NFS 选择**
- 训练数据集放 NFS（uc01 export 到 uc02/uc03）实测 GPU 99% util — NFS 不是瓶颈
- 反之放 uc01 `/data/shared/...`（其他 host 没有）会导致 worker fail
- 大数据集（115GB+）合并时 hardlink mp4 + rewrite parquet → 几秒到几分钟

### 13.9 实测性能基线 (3-host 24 GPU)

| 配置 | mesh | 首次编译 | 步速 | ETA 50k |
|---|---|---:|---:|---:|
| pi05 HSDP, batch=120, fsdp=8 | `[3,8]` | 5-50 分钟* | **1.0 s/it** | 14 小时 |
| pi05 全 FSDP, batch=120, fsdp=24 | `[1,24]` | **8 分钟** | 1.2 s/it | 16.7 小时 |

\* HSDP 首次编译时长波动大: 命中缓存秒级；不命中可能 30-45 分钟，最坏死锁 50+ 分钟需要切 mesh

### 13.10 故障排查手册

| 症状 | 可能原因 | 修复 |
|---|---|---|
| 编译 30+ 分钟没出 Step, master 满载, cache 不写 | HSDP SPMD partitioner 死锁 | 切 `fsdp_devices=24` 全 FSDP |
| `Fatal: Check failed: tsl::Flags::Parse` | XLA_FLAGS 错误关键字 | `unset XLA_FLAGS` |
| `AssertionError: sync_global_devices ... CheckpointManager:save_root_metadata` | ckpt dir 残留 / NFS stale | `rm -rf $checkpoint_dir; sync` 再启 |
| `CastError: column names don't match` | parquet schema 不一致 | 重写非标准 parquet 只保 7 标准列 |
| NCCL `NET/Socket` 出现（不是 `NET/IB`） | `NCCL_IB_DISABLE=1` 错设 | unset, 改用 `NCCL_IB_HCA=mlx5_0..3` |
| `Shutdown barrier failed, 2/3 tasks reached` | 1 个 host 进程先死了 | 看那个 host 的 worker log 找根因 |
| GPU mem 满但 util 0% 长时间 | XLA 编译中 (正常) 或卡死 | check master CPU + ~/.cache/jax mtime |


---

## 14. js01-04 集群 ⭐ (2026-05-12 加入)

### 14.1 总览 (硬件 + 网络)

| 维度 | js01 | js02 | js03 | js04 |
|---|---|---|---|---|
| **公网 IP** | **120.92.149.234** | — | — | — |
| **内网 IP** | 10.0.1.75 | 10.0.1.105 | 10.0.1.20 | 10.0.1.91 |
| **入口** | 公网直连 | 经 js01 ProxyJump | 经 js01 ProxyJump | 经 js01 ProxyJump |
| **GPU** | 8× A800-SXM4-80GB (NV8 全互连) | 同 | 同 | 同 |
| **驱动 / CUDA** | 535.183.06 / 12.2 (toolkit `/usr/local/cuda-12.2`) | 同 | 同 | 同 |
| **CPU** | Xeon 8380 @ 2.30GHz, 160 cores | 同 | 同 | 同 |
| **RAM** | 1.0 TiB | 同 | 同 | 同 |
| **/dev/shm** | 504 GB tmpfs | 同 | 同 | 同 |
| **InfiniBand** | mlx5_0/1/4/5/6 200 Gb/s | mlx5_2/3/4 | mlx5_0..4 | mlx5_0..4 |
| **共享 FS** ⭐ | `/mnt/data` JuiceFS:visincept 10T (4 台共享) | 同 | 同 | 同 |
| **本机大盘** | `/DATA/disk0` 14T NVMe | `/DATA/disk0/1` 2×7T = 14T | `/DATA/disk0..3` 4×7T = **28T** | `/DATA/disk0..3` 4×7T = **28T** |
| **`/` 根盘** | 436G (29G used) | 同 | 同 | 同 |

> ⚠️ **`/` 仅 436G** — 不放数据集/ckpt。
> ⚠️ **CUDA 12.2 toolkit** (同 gf1): launcher 需加 `export XLA_FLAGS="--xla_gpu_enable_command_buffer="`，否则 inline-eval 报 `StreamBeginCaptureToGraph`。
> ⚡ **js03/js04 各 28T 本机 NVMe** — 重训练优先排到这两台。
> ⚠️ **TOS / 出公网仅 js01 通** — js02/03/04 内网 IP 无法访问 `tos-cn-shanghai.volces.com` (Connect timeout)，外部资源走 js01 中转 + 共享 FS 落地。

### 14.2 SSH 接入

**本机 `~/.bashrc`** (4 台同款风格):
```bash
alias js01='sshpass -p "tim" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null tim@120.92.149.234'
alias js02='sshpass -p "tim" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -J root@120.92.149.234 tim@10.0.1.105'
alias js03='sshpass -p "tim" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -J root@120.92.149.234 tim@10.0.1.20'
alias js04='sshpass -p "tim" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -J root@120.92.149.234 tim@10.0.1.91'
```

**本机 `~/.ssh/config`** (key 优先, sshpass fallback):
```
Host js01      HostName 120.92.149.234   User tim   StrictHostKeyChecking no
Host js01-root HostName 120.92.149.234   User root  StrictHostKeyChecking no
Host js02      HostName 10.0.1.105       User tim   ProxyJump js01-root
Host js03      HostName 10.0.1.20        User tim   ProxyJump js01-root
Host js04      HostName 10.0.1.91        User tim   ProxyJump js01-root
# (...类似 js02-root..js04-root)
```

**互信** (已配 ed25519 互写 `authorized_keys`): 4 台 tim 用户两两可免密 ssh，每台 `~/.ssh/config` 也含 js01-04 短名 (走 `10.0.1.x` 内网)。

**用户**: `tim` (uid=1000, 密码 `tim`, sudo 组有密码 sudo); `root` 也已植入本机 (gf0) 公钥, 应急用。

### 14.3 文件结构 (4 台一致)

| 路径 | 实现 | 用途 |
|---|---|---|
| `~/workspace` | symlink → `/mnt/data/tim/workspace` | 工作目录入口, 4 台共享 |
| `~/workspace/deepdive_kai0/` | 真目录 (在 JuiceFS) | 代码 (4 台同步可见) |
| `~/workspace/deepdive_kai0/kai0/.venv` | symlink → `/DATA/disk0/tim/.kai0_venv` | Python 3.12 venv (7.8G, 每机本地 NVMe) |
| `~/local_ckpts/` | symlink → `/DATA/disk0/tim/local_ckpts` | 训练 ckpt (每机独立, 符合 §2.2 规范) |
| `/mnt/data/tim/dataset/` | 真目录 (JuiceFS) | 数据集 (4 台共享, TOS 单次下载) |
| `/mnt/data/tim/bin/uv` | js01 拷贝的 uv 二进制 | 4 台 `~/.local/bin/uv` symlink 过来 |
| `/mnt/data/tim/uv_share/cache/` | js01 安装时生成的 uv wheel cache | js02-04 用 `UV_CACHE_DIR=` + `--offline` 离线安装 |
| `/mnt/data/tim/uv_share/python/` | js01 下载的 cpython-3.12.13 | js02-04 通过 symlink 共用 |

### 14.4 代码运行环境

**版本基线** (与 uc01-03 一致, 通过 `install.sh` 一键安装):
- Python **3.12.13** (uv 管理)
- JAX **0.5.3+cuda12** (含 `jax-cuda12-plugin`, `jax-cuda12-pjrt`, `nvidia-cudnn-cu12`, `nvidia-nccl-cu12`)
- PyTorch **2.7.1+cu126** + torchvision + torchcodec (CUDA 12.2 driver 跑 cu126 wheel 向前兼容)
- Flax **0.10.2**, orbax-checkpoint **0.11.13**, ml-dtypes 0.4.1, tensorstore 0.1.74
- Transformers **4.53.2**, sentencepiece, tokenizers
- openpi (editable, `kai0/src/openpi/`), openpi-client (editable, `kai0/packages/openpi-client/`)
- lerobot (HF git `0cf86487`), wandb, rerun-sdk, polars, dm-control, mujoco
- tos 2.9.0 (Volcengine TOS SDK, 用于数据下载)

**激活模板**:
```bash
ssh js0X
cd ~/workspace/deepdive_kai0/kai0
source ../setup_env.sh
# [setup_env] host=10-0-1-XX profile=default
# [setup_env] KAI0_DATA_ROOT=/home/tim/workspace/deepdive_kai0/kai0
# [setup_env] OPENPI_DATA_HOME=/home/tim/.cache/openpi
# [setup_env] PYTORCH_CKPT_BASE=/home/tim/.cache/openpi/modelscope_cache/lerobot
.venv/bin/python -c "import jax; print(jax.devices())"
```

**`~/.bashrc` 自动 export** (每台):
```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com           # HF 直连不通, 用国内镜像
export PATH="$HOME/.local/bin:$PATH"               # uv 在此
```

**Launcher 必加** (训练脚本里):
```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=offline
export XLA_FLAGS="--xla_gpu_enable_command_buffer="    # CUDA 12.2 workaround (同 gf1)
```

### 14.5 代码同步方式

**集群内 (4 台间)**: `/mnt/data/tim/workspace/deepdive_kai0/` 是 JuiceFS 上的**单一真目录**, 4 台 `~/workspace` symlink 到同一处。任一 js 上 `git pull` / 编辑, **其他 3 台立即可见**, 无需手工同步。

> ⚠️ `.venv/` 不在共享 FS — 每台本机 NVMe (`/DATA/disk0/tim/.kai0_venv`), uv 在不同机器 sync 会冲突, 故独立。

**gf0 → js 集群 (初次部署)**:
```bash
# 在 gf0 跑 (排除大件)
rsync -a --delete \
  --exclude='kai0/checkpoints/' \
  --exclude='kai0/data/Task_*/' \
  --exclude='kai0/wandb/' --exclude='kai0/cache/' \
  --exclude='kai0/.venv' --exclude='logs/' \
  /home/tim/workspace/deepdive_kai0/ \
  js01:/mnt/data/tim/workspace/deepdive_kai0/
```

**日常更新** (gf0 是主源):
```bash
# 选项 A: 在 gf0 改, 走 rsync (同步范围可控)
# 选项 B: 在任一 js 改 (4 台立即可见), 再回 rsync 到 gf0
# 当前 js 上无 GitHub 私钥 (只 pull, 不能 push)
```

### 14.6 数据集构建 / 同步方式 (TOS → /mnt/data 共享 FS)

**核心约束** (2026-05-12 实测):
- **TOS endpoint 只 js01 通**: 公网 `120.92.149.234` 的 js01 可达 `tos-cn-shanghai.volces.com`, js02/03/04 内网 IP 完全 Connect timeout
- 后果: 即便 4 台想分片并行下载, **必须全部走 js01**
- 缓解: 利用 `/mnt/data` 共享 — js01 单机下到 `/mnt/data/tim/dataset/KAI0/`, **4 台立即可见**, 无需 4× 下载

**典型 TOS → 集群 工作流**:
```bash
# 1. 在 js01 跑递归下载器 (代码在 train_scripts/data/from_tos_recursive.py)
ssh js01
cd ~/workspace/deepdive_kai0/kai0
nohup .venv/bin/python ../train_scripts/data/from_tos_recursive.py \
    --prefix KAI0/Task_A/ \
    --dest /mnt/data/tim/dataset/KAI0 \
    --concurrency 32 \
    --skip-existing \
    --manifest /mnt/data/tim/dataset/kai0_manifest.tsv \
    > /tmp/tos_pull_taskA.log 2>&1 &
disown $!

# 2. 排队多个 prefix (顺序, 避免 TOS bucket 多 list 限速)
for PFX in KAI0/Task_E/ KAI0/Task_P/ KAI0/Task_PS/ KAI0/Task_H/ KAI0/Task_HP/; do
  .venv/bin/python ../train_scripts/data/from_tos_recursive.py \
    --prefix "$PFX" --dest /mnt/data/tim/dataset/KAI0 \
    --concurrency 32 --skip-existing
done
```

**`from_tos_recursive.py`** 关键特性:
- AK/SK 硬编码 (与 `from_tos_file.py` 同, 来自 TOS 上的 `KAI0/from_tos_file.py`)
- `--skip-existing`: 按 size 校验, 已存在 + 大小相同则跳过, 支持断点续传
- `--concurrency`: 单进程线程数, 推荐 32 (再大 TOS 端反而触发限速)
- `--manifest`: 写所有 object 列表到 TSV, 后续重跑只读 manifest 不重新 list

**实测速度** (单进程 32 并发):
- ~100-130 files/sec, ~13 MB/s (受限于 TOS API 每请求 ~200ms 延迟, 不是带宽)
- 多进程并行不同 prefix 触发 TOS list_objects 限速 → 卡死, 不可行
- KAI0/ 全量 (361 GB, 2.37M 文件) 单机预计 **6-7 小时** 拉完

**数据集落地后** — 4 台共用一份, 各机训练时:
```bash
# 把 hot subset 拷到本机 /dev/shm (504G tmpfs) 加速 dataloader
mkdir -p /dev/shm/A_new_pure_1200
cp -rL /mnt/data/tim/dataset/KAI0/Task_A/self_built/A_new_pure_1200/ /dev/shm/
# config.py 里 repo_id 指 /dev/shm/A_new_pure_1200/base
```

### 14.7 Ckpt 落地 (本机, 不共享)

每台 `~/local_ckpts/` → `/DATA/disk0/tim/local_ckpts/` (本机 NVMe, 持久, 不跨机同步), 符合 §2.2 规范。

每个训练 exp 启动前在 launcher 中 pre-create 软链:
```bash
LOCAL_DIR=/home/tim/local_ckpts/$CONFIG/$EXP
WORKSPACE_DIR=$KAI0_DATA_ROOT/checkpoints/$CONFIG/$EXP
mkdir -p "$LOCAL_DIR"
mkdir -p "$(dirname "$WORKSPACE_DIR")"
ln -sfn "$LOCAL_DIR" "$WORKSPACE_DIR"   # idempotent
```

> ⚠️ **不要直接写 ckpt 到 `/mnt/data/.../kai0/checkpoints/`** 真目录 — JuiceFS 元数据同步会显著拖慢且可能损坏 ckpt 历史教训 (类似 uc01/02 的 lsyncd 坑)。

### 14.8 训练快速启动 (单机 8 GPU)

```bash
ssh js0X   # 任意一台
cd ~/workspace/deepdive_kai0/kai0
source ../setup_env.sh

# 计算 norm_stats (新建 dataset 时必做)
.venv/bin/python scripts/compute_norm_states_fast.py --config-name <config>

# 启动训练
nohup bash train_scripts/launch/run_<config>_js0X.sh > /tmp/train_<config>.log 2>&1 &
disown $!
```

### 14.9 离线安装最佳实践 (新机入网避坑)

js02/03/04 因内网 IP 受限, 初次装环境踩坑总结:
1. **astral.sh 不通** → uv 二进制走 `/mnt/data/tim/bin/uv` (从 js01 拷贝, 4 台 symlink)
2. **github releases (python-build-standalone) 不通** → uv-managed cpython 走 `/mnt/data/tim/uv_share/python/` symlink
3. **pypi.tuna 偶尔超时** → uv cache 走 `/mnt/data/tim/uv_share/cache/` (从 js01 rsync), 配合 `uv sync --offline`
4. **.venv 不能是 symlink 时 uv 拒建** → 用 `UV_PROJECT_ENVIRONMENT=/DATA/disk0/tim/.kai0_venv` 显式指定 venv 真实路径
5. **快速法**: 装好一台 (js01) 后, `rsync -a /DATA/disk0/tim/.kai0_venv/ tim@js0X:/DATA/disk0/tim/.kai0_venv/` 直接整盘复制 (venv 路径绝对一致即可移植)

### 14.10 多机训练 (规划, 待实测)

4 台之间:
- 200 Gb/s InfiniBand × 多 HCA per 节点 (mlx5_0..6)
- 4 台 SSH 互信 + 短名解析 OK
- 与 §13 uc 集群同样的 NCCL+IB+GDR 思路应可复用, 但 IP 拓扑不同 (uc 走 192.168.X.0/24 ×4 平面, js 走 10.0.1.0/24 单平面)
- 跨机 NCCL `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` 待实测

详细调研另见 [`js_servers_probe.md`](./js_servers_probe.md)。
