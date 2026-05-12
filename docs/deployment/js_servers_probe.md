# js01-04 服务器调研报告 (2026-05-12)

> **状态**: 仅完成 Phase 1 (SSH 互信 + tim 账号) + Phase 2 (硬件调研); Phase 3+ 部署待启动。
> **关联**: 见 [`training_servers_knowledge_base.md`](./training_servers_knowledge_base.md) 主表; 部署落地后会把该文件并入主表。

---

## 1. 总览

| 维度 | js01 | js02 | js03 | js04 |
|---|---|---|---|---|
| 公网 IP | **120.92.149.234** | — | — | — |
| 内网 IP | 10.0.1.75 | 10.0.1.105 | 10.0.1.20 | 10.0.1.91 |
| 入口 | 公网直连 | 经 js01 ProxyJump | 经 js01 ProxyJump | 经 js01 ProxyJump |
| GPU | 8× A800-SXM4-80GB | 同 | 同 | 同 |
| 驱动 / CUDA 驱动 | 535.183.06 / 12.2 | 同 | 同 | 同 |
| CUDA toolkit | `/usr/local/cuda-12.2` | 同 | 同 | 同 |
| CPU | Xeon 8380 @ 2.30GHz, 160 cores | 同 | 同 | 同 |
| RAM | 1.0 TiB | 同 | 同 | 同 |
| OS | Ubuntu 22.04.3 LTS, kernel 5.15 | 同 | 同 | 同 |
| hostname | `10-0-1-75` | `10-0-1-105` | `10-0-1-20` | `10-0-1-91` |
| /dev/shm | **504 GB** tmpfs ⭐ | 同 | 同 | 同 |
| InfiniBand | mlx5_0/1/4/5/6 (200 Gb/s active) | mlx5_2/3/4 (200 Gb/s) | mlx5_0..4 | mlx5_0..4 |
| GPU 拓扑 | 8 GPU NV8 全互连 | 同 | 同 | 同 |
| Python (system) | 3.10.12 | 同 | 同 | 同 |
| uv / docker / conda | **全部缺失** (需装) | 同 | 同 | 同 |
| git / rsync / tmux | ✓ | ✓ | ✓ | ✓ |

## 2. 磁盘布局 (差异最大的维度)

| 挂载 | js01 | js02 | js03 | js04 |
|---|---|---|---|---|
| `/` (`/dev/sda3` ext4) | 436G (392G avail) | 同 | 同 | 同 |
| `/boot` (`/dev/sda2` ext4) | 3.7G | 同 | 同 | 同 |
| `/mnt/data` ⭐ **共享** (`JuiceFS:visincept`) | 10T, 用 1M | 同 (4 台共享同一卷) | 同 | 同 |
| `/DATA/disk0` (本机 NVMe) | **14T** | **7T** | 7T | 7T |
| `/DATA/disk1` | 440G HDD | 7T NVMe | 7T NVMe | 7T NVMe |
| `/DATA/disk2` | — | 440G HDD | 7T NVMe | 7T NVMe |
| `/DATA/disk3` | — | — | 7T NVMe | 7T NVMe |
| `/DATA/disk4` | — | — | 440G HDD | 440G HDD |
| **本机 NVMe 总和** | **14T** | **14T** | **28T** ⭐ | **28T** ⭐ |

**关键观察 (写入 js01 立刻在其他 3 台可见)**:
- `/mnt/data` (`JuiceFS:visincept`, 10T) **4 台共享**，等价于 gf0/gf1 的 vePFS — 适合放代码主线、共享数据集、跨机 ckpt 检视。
- 本机 NVMe (`/DATA/disk0/...`) 是每台独占，js03/04 比 js01/02 多两块 7T NVMe，**重训练应优先排到 js03/js04**。
- `/dev/shm` 504 GB tmpfs 比 gf0 (1.3T) 小但远比 uc01/uc02 强，适合训练加速 dataset cache。
- `/` 只有 436G，**不要在 / 上塞数据 / ckpt**。

## 3. 网络

| 项 | 结果 |
|---|---|
| 4 台内网互通 (10.0.1.0/24) | ✓ TCP 22 / SSH 全通 |
| InfiniBand | 多块 mlx5_X HCA，单口 200 Gb/s，全部 LinkUp Active — 多机训练 NCCL+IB 应能跑满 |
| 出公网 GitHub | HTTP 200 |
| 出公网 PyPI tuna | HTTP 200 (清华源可用) |
| 出公网 huggingface.co | HTTP 000 (不通; 用 hf-mirror 或 modelscope) |
| 出公网 TOS shanghai | HTTP 403 (域名通; 需 AK/SK) |
| 私网代理 | 暂无 (gf0 上的 sim01:29290 反向隧道在 4 台**不可达**, 因不同 VPC) |

## 4. 用户 / SSH

- 4 台都已建好 `tim` 用户 (uid=1000, sudo, 密码 `tim`)。
- 4 台 `/root/.ssh/authorized_keys` 和 `/home/tim/.ssh/authorized_keys` 都有本机 (gf0) `~/.ssh/id_rsa.pub`。
- 4 台之间 **tim 用户互信** 已建立: 每台 ed25519 私钥 + 其他 3 台的公钥追加进 `authorized_keys`，12 条路径 SSH 测试全通。
- 4 台 `~tim/.ssh/config` 配置好 js01/02/03/04 短名 (走 10.0.1.x 内网)。
- 本机 (gf0) `~/.ssh/config` 已加 8 个 Host: `js01..js04` (tim) + `js01-root..js04-root` (root)，js02-04 用 `ProxyJump js01-root`。

## 5. CUDA 兼容性提示

- 4 台 CUDA toolkit 是 12.2，**与 gf1 同**。按知识库 §3.4，inline-eval 时可能遇到 `StreamBeginCaptureToGraph` 报错 → launcher 加 `export XLA_FLAGS="--xla_gpu_enable_command_buffer="`。
- JAX 0.5.3+cuda12 应该可用 (需要 driver >= 12.x，当前 driver 12.2 ✓)。如遇兼容问题，可参考 gf0 的 cuda-12.8 安装思路 (`/home/tim/.cuda_compat` 目录补 .so)。

## 6. 待办 (Phase 3+)

1. **每台装 uv** (走清华 PyPI 源)
2. **代码放置**: 推荐 `/mnt/data/tim/workspace/deepdive_kai0` (4 台共享真目录) + 每台 `~/workspace/deepdive_kai0` symlink。
3. **venv 放置**: 每台 `/DATA/disk0/tim/.kai0_venv` (本机 NVMe, 快) ← `kai0/.venv` 软链。**venv 不能放共享 FS**，会有 lib path / 文件锁问题。
4. **数据集**: 从 TOS `tos://transfer-shanghai/KAI0/...` 拉到 `/mnt/data/tim/dataset/KAI0/` (一次下载 4 台见)。
5. **ckpt**: 每台 `/home/tim/local_ckpts/` → `/DATA/disk0/tim/local_ckpts/` (本机 NVMe，与知识库 §2.2 规范一致)。
6. **MNT_DATA root**: 共享文件根目录建议 `/mnt/data/tim/`，每台先 `sudo mkdir -p /mnt/data/tim && sudo chown tim:tim /mnt/data/tim`。

## 7. 与现有 (gf0/gf1/uc01/02/03) 风格对比

| 风格 | gf0/gf1 (vePFS) | uc01/02/03 (独立) | **js01-04 (推荐)** |
|---|---|---|---|
| 共享 FS | ✓ /vePFS | ✗ | ✓ `/mnt/data` JuiceFS |
| 工作目录 | 共享 | 独立 | 共享 (走 /mnt/data) |
| venv | 独立 (软链) | 独立 | 独立 (软链到 /DATA/disk0) |
| ckpt | 共享 vePFS 子目录 | 独立 ext4 | 独立 /DATA/disk0 (规范同 uc) |
| 数据集 | 共享 | 独立 (rsync/TOS) | 共享 (单次 TOS 拉取) |

---

修订历史:
- 2026-05-12: 初版 (Phase 1+2 完成)
