# 双节点分布式训练部署方案与跟进计划

**创建时间**: 2026-04-18  
**状态**: 进行中（XLA 编译阶段）  
**负责人**: Tim

---

## 一、架构方案

### 硬件拓扑

```
Node0 (192.168.0.144 / gf0)    Node1 (192.168.0.161 / gf1)
┌──────────────────────────┐    ┌──────────────────────────┐
│  8x A100 80GB GPU        │    │  8x A100 80GB GPU        │
│  mlx5_1~4 (4x 200GbE)   │◄──►│  mlx5_1~4 (4x 200GbE)   │
│  JAX process_id=0        │    │  JAX process_id=1        │
│  JAX gRPC coordinator    │    │                          │
└──────────────────────────┘    └──────────────────────────┘
         ↑
    port 15701 (gRPC)
```

### 软件配置

| 组件 | 配置 |
|------|------|
| 框架 | JAX multi-controller，FSDP 16 devices |
| 模型 | pi05（~3B params，PaliGemma VLM + 扩散策略） |
| 训练任务 | `pi05_flatten_fold_awbc`（AWBC, Task_A advantage 数据集） |
| 全局 batch size | 256（每节点 128） |
| 训练步数 | 100,000 steps |
| 通讯协议 | NCCL over TCP（4x 200GbE per node，1103 连接） |
| 共享存储 | vePFS（训练数据 + checkpoint） |

### 启动脚本

启动脚本位于 `/tmp/launch_multinode_fast.sh`，关键环境变量：

```bash
export JAX_COORDINATOR_ADDRESS=192.168.0.144:15701
export JAX_NUM_PROCESSES=2
export NCCL_IB_DISABLE=1                    # 当前：TCP 模式（待切换 RoCE）
export NCCL_SOCKET_IFNAME=eth2,eth3,eth4,eth5
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_CUMEM_ENABLE=0
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

Node1 通过跳板访问：
```bash
ssh -p 2222 root@192.168.0.144 \
  "ssh -p 2222 -i /root/.ssh/ssh_worker_rsa_key root@192.168.0.161 '<cmd>'"
```

---

## 二、问题记录与解决方案

### 问题 1：XLA "编译" 时间极长（300+ 分钟）✅ 已解决（根因不是 XLA 编译）

**现象**：早期 run DataLoader 完成后"进入 XLA jit(init) 编译"，3+ 小时未完成。

**真实根因（2026-04-18 定位）**：
**不是** XLA HLO pass pipeline 慢，而是 `scripts/train.py:247-252` 的 wandb image 生成：
```python
if jax.process_index() == 0:  # 原代码，未 check wandb_enabled
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(...)
    ]
```
`img[i]` 对多主机 FSDP-sharded 数组做 Python int 索引，走 `apply_primitive(gather)` 路径（**非** jit），每次触发完整 NCCL clique init + HLO 编译，多主机下每次 5+ 分钟，5 次循环累计 25+ 分钟，而且可能永远卡住。

**诊断过程**：
1. 最小复现：一个 FSDP 2D-mesh 的 jit 函数（matmul + grad）在 16 设备跨主机编译**只要 6.6 秒**
2. py-spy 抓栈发现主线程持续在 `pxla.py:1297 __call__`，调用来自 `scripts/train.py:249 的 <listcomp> img[i]`
3. 8 个 R-state 线程 100% CPU 持续 5+ 分钟在同一个 gather primitive 上

**修复**（已应用 `scripts/train.py:247`）：
```python
# 1. 加 wandb_enabled 门禁（wandb 禁用时直接跳过）
if config.wandb_enabled and jax.process_index() == 0:
    # 2. 先整体 device_get 到 host，再用 numpy 索引（绕过 apply_primitive）
    host_images = {k: np.asarray(v) for k, v in batch[0].images.items()}
    images_to_log = [
        wandb.Image(np.concatenate([host_images[k][i] for k in host_images], axis=1))
        for i in range(min(5, len(next(iter(host_images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)
```

**验证结果**（2026-04-18 04:27 UTC 修复后实测）：
| 阶段 | 耗时 |
|------|------|
| DataLoader 初始化 | 20:14 (vePFS Arrow 表扫描，见问题 2) |
| init_train_state jit 编译 | **37 秒** ✅ |
| ptrain_step jit 编译 | ~90 秒 |
| 达到 Step 0 | 共计 ~22:20（首次） |
| GPU 利用率 | 16/16 全部 100% |
| 初始 loss | 0.7964（正常） |

**状态**: ✅ 已解决

---

### 问题 2：LeRobotDataset 初始化慢（~20 分钟）⚠️ 已知，可缓存

**真实根因（2026-04-18 py-spy 定位）**：
不是 12,220 次 stat() 调用（那只是辅助），主瓶颈在 `lerobot_dataset.py:508`：
```python
timestamps = torch.stack(self.hf_dataset["timestamp"]).numpy()
episode_indices = torch.stack(self.hf_dataset["episode_index"]).numpy()
check_timestamps_sync(timestamps, episode_indices, ...)
```
遍历整个 HuggingFace Arrow Table（百万行）走 Python GIL 做 Arrow→Torch→numpy 格式转换，CPU 100% 单线程。

**py-spy 栈证据**：`datasets/formatting/formatting.py:extract_batch/format_batch/format_row`。

**缓解方案**：
| 方案 | 改动 | 启动节省 |
|------|------|----------|
| A. 跳过 check_timestamps_sync（DataLoader 已有 skipping 兜底） | 1 行代码 | 20min → 0s |
| B. 首次算完存 `.npy`，后续 load | lerobot patch | 20min → <1s |
| C. 用 `pyarrow.Table.column("x").to_numpy(zero_copy=True)` 替代 torch.stack | lerobot patch | 20min → ~30s |

**状态**: [ ] 低优先级，暂不处理（单次训练可接受）

---

### 问题 3：节点间带宽未达最优（NCCL TCP vs RoCE）⚠️ 待优化

**现状**：
- 硬件：4 × 200GbE（mlx5 RoCE）per node → 理论 800 Gbps
- 实际：`NCCL_IB_DISABLE=1` → TCP/IP → 约 480-560 Gbps（损失 ~30%）

**优化方案**：
```bash
# 移除 NCCL_IB_DISABLE=1，改为 RoCE 模式
unset NCCL_IB_DISABLE
export NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4
export NCCL_IB_GID_INDEX=3     # RoCEv2
export NCCL_NET_GDR_LEVEL=2    # GPU Direct RDMA
```

**验证方法**：
```bash
# 在双节点上运行 nccl-tests 确认 RoCE 通路
./all_reduce_perf -b 1G -e 4G -f 2 -g 8
```

**状态**: [ ] 训练稳定后执行 RoCE 验证

---

### 问题 4：训练进程以 root 运行，tim 无法直接管理 ✅ 已解决

**解决方案**：
```bash
# Kill Node0 进程
ssh -p 2222 root@192.168.0.144 "kill -9 <PID>"

# Kill Node1 进程
ssh -p 2222 root@192.168.0.144 \
  "ssh -p 2222 -i /root/.ssh/ssh_worker_rsa_key root@192.168.0.161 'kill -9 <PID>'"
```

---

### 问题 5：Node0/Node1 编译期间 GPU 不对称 ✅ 已知设计行为

**现象**：编译期间 Node0 GPU 0%（CPU-bound），Node1 GPU 100%（NCCL barrier 自旋）。

**根因**：JAX multi-controller 架构固有行为，Node0 主进程编译时 Node1 GPU 在 dummy op 上自旋等待。

**影响**：编译期间 Node1 浪费约 8 × 80GB GPU 计算资源（3+ 小时）。

**状态**: 无解，接受此限制

---

## 三、跟进计划

### 阶段 1：完成首次训练启动（当前）

- [ ] 等待 XLA jit(init) 编译完成，出现 `Initialized train state`
- [ ] 等待 XLA jit(ptrain_step) 编译完成（第二次 JIT，时间较短）
- [ ] 确认第一个 `Step 0: loss=X.XXX` 出现
- [ ] 记录初始 loss 值（预期 1-3 之间）
- [ ] 确认双节点 GPU 利用率（应达 80-100%）
- [ ] 测量训练速度（steps/sec）

### 阶段 2：编译完成后立即执行

- [ ] 同步 XLA 编译缓存到 Node1
  ```bash
  ssh -p 2222 root@192.168.0.144 \
    "scp -P 2222 -i /root/.ssh/ssh_worker_rsa_key \
     /root/.cache/jax/jit__* root@192.168.0.161:/root/.cache/jax/"
  ```
- [ ] 记录 checkpoint 保存路径
- [ ] 确认 WandB loss 曲线正常（如果启用）

### 阶段 3：下次重启优化

- [ ] 更新 launch 脚本，加入 `--xla_gpu_enable_triton_gemm=false`
- [ ] 加入 `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`
- [ ] 验证缓存命中后编译时间（目标 < 10 分钟）

### 阶段 4：网络优化（训练稳定后）

- [ ] 运行 `nccl-tests` 验证 RoCE 通路
- [ ] 对比 TCP vs RoCE 带宽数据
- [ ] 如验证通过，更新 launch 脚本启用 RoCE
- [ ] 测量启用 RoCE 后训练速度提升

---

## 四、常用运维命令

```bash
# 查看训练日志
tail -f /tmp/train_node0.log
tail -f /tmp/train_node1.log  # 在 Node0 上

# 查看 GPU 状态
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# 查看 Node1 GPU（从本机）
ssh -p 2222 root@192.168.0.144 \
  "ssh -p 2222 -i /root/.ssh/ssh_worker_rsa_key root@192.168.0.161 \
   'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'"

# 确认进程
ps aux | grep train.py | grep -v grep

# Kill 所有训练进程
ssh -p 2222 root@192.168.0.144 "kill -9 $(ps aux | grep train.py | grep -v grep | awk '{print $2}')"
ssh -p 2222 root@192.168.0.144 \
  "ssh -p 2222 -i /root/.ssh/ssh_worker_rsa_key root@192.168.0.161 \
   'kill -9 \$(ps aux | grep train.py | grep -v grep | awk \"{print \\\$2}\")'"
```

---

## 五、训练进度记录

| 时间 (UTC) | 事件 | 备注 |
|-----------|------|------|
| 2026-04-17 21:11 | 第二次 run 启动 | 加 `XLA_FLAGS=--xla_gpu_autotune_level=0` |
| 2026-04-17 21:32 | DataLoader 初始化完成 | batch shape 正常，全局 batch=256 |
| 2026-04-17 21:32 | XLA jit(init) 编译开始 | Node0 CPU-bound，Node1 GPU 100% |
| 2026-04-18 00:48+ | XLA 编译仍在进行 | 188+ 分钟，进程健康（8线程活跃） |
| TBD | Initialized train state | — |
| TBD | Step 0: loss=? | — |
| TBD | 训练正常运行 | — |
