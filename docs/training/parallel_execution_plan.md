# 双机并行执行计划

本文档是 [`training_plans.md`](./training_plans.md) 的子计划，详细说明如何在两台机器上并行执行 `kai0_mixed_1` 和 `kai0_full` 训练。

## 硬件环境

| 机器 | 地址 | 角色 | GPU |
|------|------|------|-----|
| **gf0** | `ssh -p 2222 root@192.168.0.144` | Worker A | 8 × A100 80GB |
| **gf1** | `ssh -p 2222 root@192.168.0.161` | Worker B / Master (当前机器) | 8 × A100 80GB |

### 共享基础设施

- **NFS 共享目录**: `~/workspace/deepdive_kai0/` — 两机可见同一份数据和 checkpoints
- **SSH 互信**: 两机已配置免密 SSH
- **总 GPU**: 16 × A100 80GB

### 机器标识与命名约定

- `gf1` 视为 **Master**（当前用户所在机器），负责协调、监控、MA 合并等单点任务
- `gf0` 视为 **Worker**，主要执行可并行的训练任务
- 所有命令标注执行机器：`[gf1]`, `[gf0]`, `[both]`, `[either]`

---

## 全局并行策略

```
时间轴 →                                                  
gf1 ████████████████████████████████████████████████████
gf0 ████████████████████████████████████████████████████

         ↑ 可并行区段              ↑ 串行区段 (MA merge 等)
```

**核心原则**：
1. **NFS 保证数据一致性**：两机读写同一份文件系统，无需 rsync
2. **Checkpoint 隔离**：每个训练任务写入不同子目录，避免冲突
3. **时间线协调**：依赖关系明确 (e.g., Step A 完成后才能开始 Step B)
4. **故障隔离**：单机失败不影响另一机正在跑的任务

---

## 方案 1: `kai0_mixed_1` 并行执行

### 时间对比

| 执行方式 | 总时间 |
|---------|--------|
| 单机串行 (原方案) | ~5 天 |
| **双机并行** | **~3 天** (节省 40%) |

### 任务分配

```
准备阶段 (gf1 单机):
  Step 1: 合并数据                    ~2h
  Step 2: 计算 norm_stats             ~20min
  Step 3: 切分 episodes (生成 JSON)   ~1min
  Step 4: 修改 config.py              ~15min
  
训练阶段 (并行):
  gf1:  split_0 (~28h) → split_2 (~28h) = 56h
  gf0:  split_1 (~28h) → split_3 (~28h) = 56h
  
合并阶段 (gf1 单机):
  Step 5: Dump validation             ~30min  
  Step 6: MA 合并                     ~1h
```

### 执行步骤

#### Phase 0: 准备阶段 (gf1)

```bash
# [gf1] 进入工作目录
cd ~/workspace/deepdive_kai0/kai0

# Step 1: 合并 base + dagger
python train_deploy_alignment/data_augment/merge_lerobot.py \
  --src_paths data/Task_A/base data/Task_A/dagger \
  --tgt_path data/Task_A/kai0_mixed_1_data \
  --repo_id kai0_mixed_1_data

# 验证合并结果
python3 -c "
import json
info = json.load(open('data/Task_A/kai0_mixed_1_data/meta/info.json'))
print(f'episodes: {info[\"total_episodes\"]}, frames: {info[\"total_frames\"]}')
assert info['total_episodes'] == 6512, 'Episode count mismatch'
"

# Step 2: 在 config.py 新增 kai0_mixed_1_split_0~3 配置 (见 training_plans.md)
# 然后计算 norm_stats
uv run python scripts/compute_norm_states_fast.py --config-name kai0_mixed_1_split_0

# Step 3: 生成 4-split episode 列表
python3 << 'EOF'
import json, random
random.seed(42)
episodes = list(range(6512))
random.shuffle(episodes)
for i in range(4):
    split_eps = sorted(episodes[i::4])
    json.dump(split_eps, open(f"data/Task_A/kai0_mixed_1_split_{i}.json", "w"))
    print(f"split_{i}: {len(split_eps)} episodes")
EOF

# Step 4: 验证 gf0 可访问相同数据
ssh -p 2222 root@192.168.0.144 "ls /home/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_mixed_1_data/meta/info.json"
```

#### Phase 1: 并行训练启动

**在 gf1 上创建启动脚本 `run_kai0_mixed_1_gf1.sh`**:

```bash
#!/bin/bash
# [gf1] 运行 split_0 和 split_2
export PATH=/home/tim/miniconda3/bin:/home/tim/.local/bin:$PATH
export PYTHONUNBUFFERED=1
export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_DATASETS_CACHE=/home/tim/.cache/huggingface/datasets
export WANDB_MODE=offline
export LD_LIBRARY_PATH=/home/tim/miniconda3/lib:/home/tim/.cuda_compat:/usr/local/cuda-12.8/targets/x86_64-linux/lib
for d in /home/tim/.kai0_venv/lib/python3.11/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH=$d:$LD_LIBRARY_PATH
done

cd /home/tim/workspace/deepdive_kai0/kai0

for i in 0 2; do
    echo "[gf1] === START split_$i $(date) ==="
    rm -rf checkpoints/kai0_mixed_1_split_$i/split_${i}_v1
    EPISODES=$(python3 -c "import json; print(' '.join(map(str, json.load(open('data/Task_A/kai0_mixed_1_split_$i.json')))))")
    uv run scripts/train.py kai0_mixed_1_split_$i \
      --exp_name=split_${i}_v1 \
      --fsdp-devices 8 \
      --batch-size 256 \
      --data.episodes $EPISODES \
      || echo "[gf1] split_$i FAILED"
    echo "[gf1] === END split_$i $(date) ==="
done
```

**在 gf0 上创建启动脚本 `run_kai0_mixed_1_gf0.sh`**（通过 NFS 直接创建）:

```bash
#!/bin/bash
# [gf0] 运行 split_1 和 split_3 (同 gf1 环境变量)
export PATH=/home/tim/miniconda3/bin:/home/tim/.local/bin:$PATH
export PYTHONUNBUFFERED=1
export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_DATASETS_CACHE=/home/tim/.cache/huggingface/datasets
export WANDB_MODE=offline
export LD_LIBRARY_PATH=/home/tim/miniconda3/lib:/home/tim/.cuda_compat:/usr/local/cuda-12.8/targets/x86_64-linux/lib
for d in /home/tim/.kai0_venv/lib/python3.11/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH=$d:$LD_LIBRARY_PATH
done

cd /home/tim/workspace/deepdive_kai0/kai0

for i in 1 3; do
    echo "[gf0] === START split_$i $(date) ==="
    rm -rf checkpoints/kai0_mixed_1_split_$i/split_${i}_v1
    EPISODES=$(python3 -c "import json; print(' '.join(map(str, json.load(open('data/Task_A/kai0_mixed_1_split_$i.json')))))")
    uv run scripts/train.py kai0_mixed_1_split_$i \
      --exp_name=split_${i}_v1 \
      --fsdp-devices 8 \
      --batch-size 256 \
      --data.episodes $EPISODES \
      || echo "[gf0] split_$i FAILED"
    echo "[gf0] === END split_$i $(date) ==="
done
```

**启动训练**:

```bash
# [gf1] 启动本机训练 (后台)
cd ~/workspace/deepdive_kai0
nohup bash run_kai0_mixed_1_gf1.sh > kai0_mixed_1_gf1.log 2>&1 &
echo "gf1 PID: $!"

# [gf1] 启动 gf0 远程训练 (通过 SSH)
ssh -p 2222 root@192.168.0.144 \
  "cd /home/tim/workspace/deepdive_kai0 && nohup bash run_kai0_mixed_1_gf0.sh > kai0_mixed_1_gf0.log 2>&1 & echo gf0 PID: \$!"
```

#### Phase 2: 监控与协调

**监控脚本 `monitor_kai0_mixed_1.sh`** ([gf1] 运行):

```bash
#!/bin/bash
# 监控两机训练进度
cd ~/workspace/deepdive_kai0

while true; do
    clear
    echo "=== $(date) ==="
    
    echo ""
    echo "=== gf1 (local) status ==="
    grep "Step " kai0_mixed_1_gf1.log 2>/dev/null | tail -3
    grep "START\|END\|FAILED" kai0_mixed_1_gf1.log 2>/dev/null | tail -5
    
    echo ""
    echo "=== gf0 (remote) status ==="
    ssh -p 2222 root@192.168.0.144 \
      "cd /home/tim/workspace/deepdive_kai0 && grep 'Step ' kai0_mixed_1_gf0.log 2>/dev/null | tail -3 && grep 'START\|END\|FAILED' kai0_mixed_1_gf0.log 2>/dev/null | tail -5"
    
    echo ""
    echo "=== GPU Usage ==="
    echo "-- gf1 --"
    nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | head -8
    echo "-- gf0 --"
    ssh -p 2222 root@192.168.0.144 \
      "nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | head -8"
    
    echo ""
    echo "=== Checkpoints on NFS ==="
    for i in 0 1 2 3; do
        steps=$(ls checkpoints/kai0_mixed_1_split_$i/split_${i}_v1/ 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')
        echo "  split_$i: $steps"
    done
    
    sleep 300  # 5 分钟刷新
done
```

#### Phase 3: 合并阶段 (gf1 单机)

**等待所有 4 个 split 都完成**:

```bash
# [gf1] 检查所有 split 完成
python3 << 'EOF'
import os
all_done = True
for i in range(4):
    ckpt = f"checkpoints/kai0_mixed_1_split_{i}/split_{i}_v1/24999"
    if os.path.exists(ckpt):
        print(f"✓ split_{i}: {ckpt}")
    else:
        print(f"✗ split_{i}: MISSING")
        all_done = False
print("\nAll splits ready" if all_done else "\nNot ready yet")
EOF
```

**Step 5: Dump validation** (gf1 单机，I/O 密集):

```bash
# [gf1]
cd ~/workspace/deepdive_kai0/kai0
export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache

uv run python model_arithmetic/dump_data.py \
  --dataset kai0_mixed_1_split_0 \
  --output model_arithmetic/kai0_mixed_1_val.pkl \
  --batch-size 16
```

**Step 6: MA 合并** (gf1 单机，单 GPU):

```bash
# [gf1]
uv run python model_arithmetic/arithmetic.py \
  --config kai0_mixed_1_split_0 \
  --data-path model_arithmetic/kai0_mixed_1_val.pkl \
  --checkpoints \
    $(pwd)/checkpoints/kai0_mixed_1_split_0/split_0_v1/24999 \
    $(pwd)/checkpoints/kai0_mixed_1_split_1/split_1_v1/24999 \
    $(pwd)/checkpoints/kai0_mixed_1_split_2/split_2_v1/24999 \
    $(pwd)/checkpoints/kai0_mixed_1_split_3/split_3_v1/24999 \
  --output $(pwd)/checkpoints/kai0_mixed_1 \
  --optimize_method greedy \
  --gpu_ids "0"
```

#### Phase 4: 验证 (gf1)

```bash
# [gf1] 使用 8 GPU 并行评测
cd ~/workspace/deepdive_kai0/kai0
uv run python model_arithmetic/evaluate_heldout.py
# 将 kai0_mixed_1 加入 MODELS dict 后重跑
```

### 预期总时间

```
Phase 0 准备:     ~2.5h   (gf1)
Phase 1 训练:     ~56h    (gf0/gf1 并行)
Phase 2 监控:     (与训练并行)
Phase 3 合并:     ~1.5h   (gf1)
Phase 4 验证:     ~0.5h   (gf1)
───────────────────────
总计:             ~60h ≈ 2.5 天
```

**节省时间**: 5 天 → 2.5 天（50% 节省）

---

## 方案 2: `kai0_full` 并行执行

### 时间对比

| 执行方式 | 总时间 |
|---------|--------|
| 单机串行 | ~10 天 |
| **双机并行** | **~6 天** (节省 40%) |

### 并行策略

完整方案有更多可并行的机会：

1. **Step A 训练时** (1.5d)，gf0 可做 Step C-D 的准备工作（但主要是 CPU 任务）
2. **Step B (dagger advantage 预测)** 两机可分工：每机预测一半数据
3. **Step D (数据增强)** 两机并行：gf0 做 space_mirroring，gf1 做 time_scaling
4. **Step E (4 splits 训练)** 和 kai0_mixed_1 一样 2+2 分配
5. **Step G (AWBC fine-tune)** 单机（单个训练任务）

### 任务分配时间线

```
Day 0-1.5:  [gf1] Step A: Advantage Estimator     (1.5d)
            [gf0] Step 0: Merge base+dagger (prep)  (idle 部分可用)

Day 1.5-1.7: [gf1] Step B (第一半 data)    | [gf0] Step B (第二半 data)
             (2小时并行)

Day 1.7-1.8: [gf1] Step C: 合并+离散化    (0.1d)

Day 1.8-2.8: [gf1] space_mirroring (0.5d) | [gf0] time_scaling (0.5d)
             → 两机结果 merge 到一起
             (1d 并行)

Day 2.8-5.8: [gf1] split_0 + split_2      | [gf0] split_1 + split_3
             (4 splits × 1.5d = 3d 并行)

Day 5.8-6:  [gf1] Step F: MA merge         (0.2d)

Day 6-7.2:  [gf1] Step G: AWBC fine-tune  (可选, 1.2d)
```

### 详细步骤

#### Phase 0: Advantage Estimator (gf1 单机)

```bash
# [gf1] 训练 Advantage Estimator
cd ~/workspace/deepdive_kai0/kai0
nohup uv run torchrun --standalone --nproc_per_node=8 \
  scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD \
  --exp_name=adv_est_v2 \
  --save_interval 10000 \
  > kai0_full_adv_est.log 2>&1 &
echo "adv_est PID: $!"
```

**同时 gf0 可做的准备工作**:

```bash
# [gf0] 准备合并数据目录结构 (不占 GPU)
ssh -p 2222 root@192.168.0.144 << 'EOF'
cd /home/tim/workspace/deepdive_kai0/kai0
# 预先合并 base + dagger (同 kai0_mixed_1 流程)
python train_deploy_alignment/data_augment/merge_lerobot.py \
  --src_paths data/Task_A/base data/Task_A/dagger \
  --tgt_path data/Task_A/kai0_full_base_merged \
  --repo_id kai0_full_base_merged &
echo "merge PID: $!"
EOF
```

#### Phase 1: Advantage 预测并行 (Step B)

**等 Step A 完成后**:

```bash
# [gf1] 对 base+dagger 的前半预测 advantage
cd ~/workspace/deepdive_kai0/kai0
uv run python stage_advantage/annotation/eval.py \
  Task-A KAI0 \
  data/Task_A/kai0_full_base_merged \
  --ckpt-dir checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v2/100000 \
  --episode-range 0-3255 \
  --output data/Task_A/kai0_full_adv_part1 &

# [gf0] 对后半预测
ssh -p 2222 root@192.168.0.144 \
  "cd /home/tim/workspace/deepdive_kai0/kai0 && \
   uv run python stage_advantage/annotation/eval.py \
     Task-A KAI0 \
     data/Task_A/kai0_full_base_merged \
     --ckpt-dir checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v2/100000 \
     --episode-range 3256-6511 \
     --output data/Task_A/kai0_full_adv_part2 &"

# 等待两机都完成后，合并两个 advantage 标注数据集
wait
python train_deploy_alignment/data_augment/merge_lerobot.py \
  --src_paths data/Task_A/kai0_full_adv_part1 data/Task_A/kai0_full_adv_part2 \
  --tgt_path data/Task_A/kai0_full_base_data
```

**注意**: `eval.py` 当前实现可能不支持 `--episode-range` 参数，需要先确认或改写。

#### Phase 2: 离散化 (Step C, gf1 单机)

```bash
# [gf1]
python stage_advantage/annotation/discretize_advantage.py \
  data/Task_A/kai0_full_base_data \
  --threshold 30 \
  --discretion-type binary \
  --advantage-source absolute_advantage
```

#### Phase 3: 数据增强并行 (Step D)

```bash
# [gf1] space_mirroring
nohup python train_deploy_alignment/data_augment/space_mirroring.py \
  --src_path data/Task_A/kai0_full_base_data \
  --tgt_path data/Task_A/kai0_full_mirrored \
  --repo_id kai0_full_mirrored \
  --merge_with_src \
  > kai0_full_mirror.log 2>&1 &

# [gf0] time_scaling (同时跑在另一个目录)
ssh -p 2222 root@192.168.0.144 << 'EOF'
cd /home/tim/workspace/deepdive_kai0/kai0
nohup python train_deploy_alignment/data_augment/time_scaling.py \
  --src_path data/Task_A/kai0_full_base_data \
  --tgt_path data/Task_A/kai0_full_time_scaled \
  --repo_id kai0_full_time_scaled \
  --extraction_factor 2 \
  > kai0_full_time.log 2>&1 &
EOF

# 等待两机都完成
wait

# [gf1] 合并两种增强数据为最终训练数据
python train_deploy_alignment/data_augment/merge_lerobot.py \
  --src_paths data/Task_A/kai0_full_mirrored data/Task_A/kai0_full_time_scaled \
  --tgt_path data/Task_A/kai0_full_data

# 重新计算 norm_stats
uv run python scripts/compute_norm_states_fast.py --config-name kai0_full_split_0
```

#### Phase 4: 4 Splits 训练并行 (Step E)

结构与 `kai0_mixed_1` Phase 1 完全相同：gf1 跑 split_0+2，gf0 跑 split_1+3。

```bash
# 生成 splits
python3 << 'EOF'
import json, random
random.seed(42)
# 假设 kai0_full_data 有 ~24000 episodes
import os, json
info = json.load(open("data/Task_A/kai0_full_data/meta/info.json"))
n_episodes = info["total_episodes"]
episodes = list(range(n_episodes))
random.shuffle(episodes)
for i in range(4):
    split_eps = sorted(episodes[i::4])
    json.dump(split_eps, open(f"data/Task_A/kai0_full_split_{i}.json", "w"))
    print(f"split_{i}: {len(split_eps)} eps")
EOF

# [gf1] 启动 split_0, split_2
nohup bash run_kai0_full_gf1.sh > kai0_full_gf1.log 2>&1 &

# [gf0] 启动 split_1, split_3
ssh -p 2222 root@192.168.0.144 \
  "cd ~/workspace/deepdive_kai0 && nohup bash run_kai0_full_gf0.sh > kai0_full_gf0.log 2>&1 &"
```

#### Phase 5: MA 合并 + AWBC Fine-tune (Step F/G, gf1 单机)

```bash
# [gf1] MA 合并
uv run python model_arithmetic/dump_data.py \
  --dataset kai0_full_split_0 \
  --output model_arithmetic/kai0_full_val.pkl --batch-size 16

uv run python model_arithmetic/arithmetic.py \
  --config kai0_full_split_0 \
  --data-path model_arithmetic/kai0_full_val.pkl \
  --checkpoints \
    $(pwd)/checkpoints/kai0_full_split_0/split_0_v1/24999 \
    $(pwd)/checkpoints/kai0_full_split_1/split_1_v1/24999 \
    $(pwd)/checkpoints/kai0_full_split_2/split_2_v1/24999 \
    $(pwd)/checkpoints/kai0_full_split_3/split_3_v1/24999 \
  --output $(pwd)/checkpoints/kai0_full_ma \
  --optimize_method greedy

# [gf1] (可选) AWBC fine-tune
uv run scripts/train.py kai0_full_awbc_finetune \
  --exp_name=awbc_ft_v1 \
  --fsdp-devices 8 \
  --batch-size 256
```

---

## 协调机制

### 1. 命名空间隔离

两机写入不同 checkpoint 子目录，避免覆盖：

```
checkpoints/
├── kai0_mixed_1_split_0/     # gf1 负责
├── kai0_mixed_1_split_1/     # gf0 负责
├── kai0_mixed_1_split_2/     # gf1 负责
└── kai0_mixed_1_split_3/     # gf0 负责
```

### 2. NFS 读写注意事项

- **Dataset 读取**: 两机同时读 NFS 数据无冲突（只读）
- **Checkpoint 写入**: 每机写入独立目录，无锁竞争
- **HuggingFace 缓存**: 可能冲突，两机应使用**本地缓存**:
  ```bash
  # gf0 设置本地缓存路径 (避免读/写 NFS 缓存)
  export HF_DATASETS_CACHE=/home/tim/.cache/huggingface/datasets  # 已在脚本中
  ```
- **Wandb offline**: 两机独立 run，不互相干扰

### 3. 训练日志

两机的日志文件写在 NFS 上，统一查看：

```
~/workspace/deepdive_kai0/
├── kai0_mixed_1_gf0.log
├── kai0_mixed_1_gf1.log
├── kai0_full_gf0.log
└── kai0_full_gf1.log
```

### 4. 故障恢复

**单机失败**：
- 失败的 split 可在任一机器重试（数据 NFS 共享）
- 示例: gf0 的 split_1 失败 → gf1 空闲时启动 split_1

**检测失败**:
```bash
# [gf1] 检查两机状态
for host in gf0 gf1; do
  if [[ "$host" == "gf0" ]]; then
    log=$(ssh -p 2222 root@192.168.0.144 "cat ~/workspace/deepdive_kai0/kai0_mixed_1_gf0.log" 2>/dev/null)
  else
    log=$(cat ~/workspace/deepdive_kai0/kai0_mixed_1_gf1.log 2>/dev/null)
  fi
  if echo "$log" | grep -q "FAILED"; then
    echo "❌ $host has failures:"
    echo "$log" | grep "FAILED"
  else
    echo "✅ $host running"
  fi
done
```

### 5. SSH 长连接

为避免 SSH 意外断开导致任务中断，训练任务一律用 `nohup ... &` 后台执行，然后断开 SSH 也不影响。

**启动模板**:
```bash
ssh -p 2222 root@192.168.0.144 \
  "cd /home/tim/workspace/deepdive_kai0 && \
   nohup bash run_task.sh > task.log 2>&1 < /dev/null & \
   disown && echo 'Started PID: '\$!"
```

---

## 验证 checklist (开始前必做)

### 环境一致性

```bash
# 两机 Python 环境一致
for host in gf1 gf0; do
  if [[ "$host" == "gf1" ]]; then
    cmd="python3 --version && python3 -c 'import jax; print(jax.__version__)'"
  else
    cmd="ssh -p 2222 root@192.168.0.144 'python3 --version && python3 -c \"import jax; print(jax.__version__)\"'"
  fi
  echo "=== $host ==="
  eval "$cmd"
done
```

### NFS 挂载验证

```bash
# [gf1] 写入测试文件
echo "test from gf1 $(date)" > ~/workspace/deepdive_kai0/nfs_test.txt

# [gf0] 读取确认
ssh -p 2222 root@192.168.0.144 "cat ~/workspace/deepdive_kai0/nfs_test.txt"

# 清理
rm ~/workspace/deepdive_kai0/nfs_test.txt
```

### GPU 可用性

```bash
# 两机 GPU 都空闲
for host in gf1 gf0; do
  echo "=== $host ==="
  if [[ "$host" == "gf1" ]]; then
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
  else
    ssh -p 2222 root@192.168.0.144 "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader"
  fi
done
```

### pi05_base 权重

```bash
# 两机都能访问 pi05_base
for host in gf1 gf0; do
  echo "=== $host ==="
  if [[ "$host" == "gf1" ]]; then
    ls /vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params/ | head -3
  else
    ssh -p 2222 root@192.168.0.144 "ls /vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params/ | head -3"
  fi
done
```

---

## 时间预算总结

| 方案 | 单机串行 | 双机并行 | 节省 |
|------|---------|---------|-----|
| `kai0_mixed_1` | 5 天 | **2.5 天** | 50% |
| `kai0_full` | 10 天 | **6 天** | 40% |
| **总计** | **15 天** | **8.5 天** | **43%** |

### 关键路径

```
Day 0:   准备 kai0_mixed_1 合并数据 + config
Day 0-2.5: 并行训 4 splits (kai0_mixed_1) + 合并
Day 2.5: 验证 kai0_mixed_1 与官方 mixed_1 对齐
Day 2.5-4: Advantage estimator 训练 (kai0_full Phase 0)
Day 4-4.2: Dagger advantage 预测并行
Day 4.2-5.2: 数据增强并行
Day 5.2-8.2: kai0_full 4 splits 并行训练
Day 8.2-8.5: MA 合并
Day 8.5: 完成 (含 AWBC fine-tune 则 9.7 天)
```

---

## 运行顺序建议

**第一阶段** (Day 0-2.5): 完成 `kai0_mixed_1`

```bash
# [gf1] 启动整个 pipeline
cd ~/workspace/deepdive_kai0
./run_kai0_mixed_1_pipeline.sh  # (待写, 封装 Phase 0-3)
```

**第二阶段** (Day 2.5-8.5): 完成 `kai0_full`

```bash
# [gf1] 确认 mixed_1 成功后，启动 full pipeline
./run_kai0_full_pipeline.sh  # (待写, 封装 Phase 0-5)
```

**第三阶段**: 评测对比

```bash
# [gf1] 在 evaluate_heldout.py 中添加 kai0_mixed_1 和 kai0_full
uv run python model_arithmetic/evaluate_heldout.py
```

---

## 已知风险与缓解

### 1. NFS I/O 瓶颈

**风险**: 两机同时读大量视频 → NFS 带宽饱和 → 训练数据加载慢

**缓解**:
- 监控 NFS 带宽: `iftop` 或 `nload`
- 如果瓶颈严重: 预先把 videos 复制到每机本地 SSD
- 降低 `num_workers` 从 8 到 4

### 2. SSH 持久连接

**风险**: 网络抖动导致 ssh 断开

**缓解**:
- 用 `nohup` + `disown` 启动
- 远程日志写 NFS，随时可在 gf1 查看
- 备用方案: 用 `tmux` 持久 session

### 3. gf0 环境差异

**风险**: gf0 可能缺失某些 Python 包 / CUDA 库

**缓解**:
- 开始前先跑 `验证 checklist`
- 发现差异立即修复
- 如果 `uv` 版本不同，考虑用 `pip` 直接装依赖

### 4. Checkpoint 写入冲突

**风险**: 两机同时写同一目录

**缓解**:
- 命名空间隔离（已设计）
- 用 `rm -rf` 清理旧 checkpoint 前加 confirmation

### 5. 时间同步

**风险**: 两机时间不同步导致日志时间混乱

**缓解**:
```bash
# 两机都启用 NTP
ssh -p 2222 root@192.168.0.144 "date"
date
# 如果差异 > 1s, 手动同步或启用 chronyd
```

---

## 快速启动命令卡片

### Pre-flight checks
```bash
# SSH 连通性
ssh -p 2222 root@192.168.0.144 "echo OK"

# NFS 共享
ls ~/workspace/deepdive_kai0/kai0/data/Task_A/base/meta/info.json

# GPU 空闲
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
ssh -p 2222 root@192.168.0.144 "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader"
```

### Start kai0_mixed_1
```bash
cd ~/workspace/deepdive_kai0
# Phase 0 准备 (见 Phase 0 section)
# Phase 1 启动 (gf1)
nohup bash run_kai0_mixed_1_gf1.sh > kai0_mixed_1_gf1.log 2>&1 &
# Phase 1 启动 (gf0)
ssh -p 2222 root@192.168.0.144 \
  "cd ~/workspace/deepdive_kai0 && nohup bash run_kai0_mixed_1_gf0.sh > kai0_mixed_1_gf0.log 2>&1 &"
```

### Monitor
```bash
bash monitor_kai0_mixed_1.sh
```

### Stop (emergency)
```bash
# gf1
pkill -9 -f "train.py kai0_mixed_1"

# gf0
ssh -p 2222 root@192.168.0.144 "pkill -9 -f 'train.py kai0_mixed_1'"
```

---

## 参考

- 主训练方案: [`training_plans.md`](./training_plans.md)
- 官方 kai0 GitHub: [OpenDriveLab/KAI0](https://github.com/OpenDriveLab/KAI0)
- kai0 HF: [OpenDriveLab-org/Kai0](https://huggingface.co/OpenDriveLab-org/Kai0)
