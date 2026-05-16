# Backup Manifest — uc 集群重装前备份

**备份日期**: 2026-05-16
**备份原因**: uc 集群重装 (因挖矿木马入侵, 详见 `docs/security/2026-05-16_rvn_miner_incident.md`)
**TOS 路径根**: `tos://transfer-shanghai/backup_uc_reinstall_20260516/`
**总大小**: ~45.2 GB
**备份执行人**: tim + Claude Code (自动化 tosutil push)
**备份完成**: 2026-05-16 19:18 CST ✅
**实测速度**: 135 MB/s (cron `pull_tos_to_shared.sh` 已禁用避免抢带宽)
**TOS 验证**:
- `datasets/A_new_pure_200/`: **815 objects, 3.14 GB** ✓
- `datasets/A_new_pure_200_val/`: **94 objects, 262.74 MB** ✓ (补传 2026-05-16 19:23 CST)
- `ckpts/pi05init_step_4000/`: **104 objects, 41.67 GB** ✓
- 总计: **1013 objects, 45.07 GB**

---

## 1. 备份内容清单

| # | 对象 | 大小 | TOS 路径 | 原 uc 路径 | 用途 |
|---|---|---|---|---|---|
| 1 | **uc01 pi05init ckpt step 4000** | 42 GB | `tos://transfer-shanghai/backup_uc_reinstall_20260516/ckpts/pi05init_step_4000/` | `uc01:/home/tim/local_ckpts/checkpoints/pi05_flatten_fold_a_new_pure_200_js/task_a_pure200_new_norm_base_pi0.5/4000/` | **resume 训练 (paused at step 4330)** |
| 2 | **A_new_pure_200 dataset** | 3.2 GB | `tos://transfer-shanghai/backup_uc_reinstall_20260516/datasets/A_new_pure_200/` | `uc01:/home/tim/local_ckpts/data/Task_A/self_built/A_new_pure_200/` | resume 训练必需的数据 |
| 3 | **A_new_pure_200_val** (inline_eval) | 263 MB | `tos://transfer-shanghai/backup_uc_reinstall_20260516/datasets/A_new_pure_200_val/` | `uc01:/home/tim/local_ckpts/data/Task_A/self_built/A_new_pure_200_val/` | inline_eval 必需 |

---

## 2. 实验上下文

### 2.1 训练详情 (paused state)

| 参数 | 值 |
|---|---|
| Config name | `pi05_flatten_fold_a_new_pure_200_js` |
| Exp name | `task_a_pure200_new_norm_base_pi0.5` |
| Init | pi05_base (raw pretrained) |
| Init 路径 (原) | `uc01:/home/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params` |
| Data | A_new_pure_200 (200 ep `-new` 精选 + hflip mirror) |
| Steps target | 50,000 |
| **Steps reached** | **4330** (latest ckpt 在 step 4000) |
| Batch | 120, FSDP=8 |
| num_workers | 64 (uc 单机 + 本地 SSD 推荐值) |
| LR | 1.5e-5 → 1.5e-6 cosine, warmup 1k |
| EMA | 0.9999 |
| WandB | offline (`--no-wandb-enabled`) |
| Status | **paused 2026-05-16 18:36 CST** (用户主动停止, ckpt 4000 完整保存) |

### 2.2 已有 step 4000 eval (前期数据)

| step | MAE@1 | @10 | @25 | @50 |
|---|---|---|---|---|
| 4000 | 0.0507 | 0.0663 | 0.0927 | 0.1274 |

(pi05_base init 起点, 与 pure2_1800_6000 在 step 4k = 0.0534 接近, 收敛轨迹符合 pi05_base 起点预期)

---

## 3. uc 重装后 resume 流程

### Step 1: 从 TOS 拉回数据 + ckpt

假设重装后服务器 IP 仍为 uc01 (or 改名), 准备好新机器后:

```bash
# 假设 tosutil 已安装 + 配置好 ~/.tosutilconfig (endpoint = tos-cn-shanghai.volces.com)
# 假设有 /home/tim/local_ckpts/ 这种目录结构

# 1. 拉回 dataset
mkdir -p /home/tim/local_ckpts/data/Task_A/self_built/
tosutil cp -r tos://transfer-shanghai/backup_uc_reinstall_20260516/datasets/A_new_pure_200/ \
  /home/tim/local_ckpts/data/Task_A/self_built/A_new_pure_200/ \
  -j 32 -p 8

# 2. 拉回 ckpt step 4000
mkdir -p /home/tim/local_ckpts/checkpoints/pi05_flatten_fold_a_new_pure_200_js/task_a_pure200_new_norm_base_pi0.5/
tosutil cp -r tos://transfer-shanghai/backup_uc_reinstall_20260516/ckpts/pi05init_step_4000/ \
  /home/tim/local_ckpts/checkpoints/pi05_flatten_fold_a_new_pure_200_js/task_a_pure200_new_norm_base_pi0.5/4000/ \
  -j 32 -p 8

# 3. 验证大小 (应该是 3.2G + 42G)
du -sh /home/tim/local_ckpts/data/Task_A/self_built/A_new_pure_200/
du -sh /home/tim/local_ckpts/checkpoints/pi05_flatten_fold_a_new_pure_200_js/task_a_pure200_new_norm_base_pi0.5/4000/
```

### Step 2: 验证 ckpt 完整性

```bash
ls /home/tim/local_ckpts/checkpoints/pi05_flatten_fold_a_new_pure_200_js/task_a_pure200_new_norm_base_pi0.5/4000/
# 应该看到:
#   _CHECKPOINT_METADATA
#   assets/
#   params/
#   train_state/
```

如果缺失 `train_state/`, **resume 无法工作** (需要优化器状态)。检查 manifest.ocdbt 是否完整。

### Step 3: 准备 pi05_base init (新机器上不在?)

`pi05_base` 在备份中未包含 (12G, 可从 HF 重下)。重装后需要:

```bash
# 选 1: 从 HuggingFace 拉 (推荐)
mkdir -p /home/tim/workspace/openpi_cache/openpi-assets/checkpoints/
huggingface-cli download openpi-assets/pi05_base \
  --local-dir /home/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base

# 选 2: 从其他 uc / js 节点 scp 拉 (如果还有)

# 选 3: 让 openpi 训练时自动从 gs:// 下载 (默认行为, 但要 GCS 访问)
```

### Step 4: 预防 cron 拖慢 (重要!)

⚠️ **resume 前先禁掉 `pull_tos_to_shared.sh` cron**, 否则该 cron 会持续拉数据 → 训练 rate 退化到 5.5 s/it (上次原始训练就是被这个拖到 5h 才到 step 4k):

```bash
crontab -l > /tmp/tim_cron_backup_$(date +%Y%m%d).txt   # 备份原 cron
crontab -l | sed 's|^\(\*/5 \* \* \* \* /home/tim/scripts/pull_tos_to_shared.sh\)|#\1|' | crontab -
crontab -l   # 验证 pull_tos 那行已注释
```

训练完成后 resume cron:
```bash
crontab /tmp/tim_cron_backup_*.txt
```

### Step 5: 启动 resume 训练

```bash
cd /data/shared/tim/workspace/deepdive_kai0/kai0
nohup env \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  OPENPI_DATA_HOME=/home/tim/workspace/openpi_cache \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  JAX_COMPILATION_CACHE_DIR=/home/tim/workspace/xla_cache_uc01 \
  XLA_FLAGS=--xla_gpu_autotune_level=0 \
  .venv/bin/python -u scripts/train.py pi05_flatten_fold_a_new_pure_200_js \
    --exp-name task_a_pure200_new_norm_base_pi0.5 \
    --batch-size 120 \
    --fsdp-devices 8 \
    --num-workers 64 \
    --data.repo-id /home/tim/local_ckpts/data/Task_A/self_built/A_new_pure_200 \
    --inline-eval-val-root /home/tim/local_ckpts/data/Task_A/self_built/A_new_pure_200_val \
    --weight-loader.params-path /home/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params \
    --checkpoint-base-dir /home/tim/local_ckpts/checkpoints \
    --no-wandb-enabled \
    --resume \
  > /data/shared/tim/logs/train_task_a_pure200_new_norm_base_pi0.5_uc01.log 2>&1 < /dev/null & disown -a
```

**关键 flag**: `--resume` (不要用 `--overwrite`, 否则会从 step 0 重启)

### Step 6: 验证恢复

```bash
# 看 log 应当显示 "Restoring checkpoint from .../4000"
tail -f /data/shared/tim/logs/train_task_a_pure200_new_norm_base_pi0.5_uc01.log
```

预期: ~3 min 编译 + 重启 → 第一个 Progress 行 rate ≈ 1.9 s/it (本地盘 + nw=64 优化)。

如果 rate ≥ 5 s/it, 检查 `pull_tos_to_shared.sh` 是否还在跑:
```bash
ps -ef | grep tosutil | grep -v grep   # 应该为空
```

---

## 4. ⚠️ 备份缺失项 + 注意事项

### 4.1 未备份 (重要!)

以下未备份, 重装后可能需要重新获取:

| 项 | 大小 | 重新获取方式 |
|---|---|---|
| ~~`A_new_pure_200_val` (验证集)~~ | ~~263MB~~ | ✅ 已补传 (见 §1 #3) |
| pi05_base init | 12 GB | HuggingFace 下载 (`openpi-assets/pi05_base`) |
| mixed_1 init | 22 GB | 已废弃 (本次实验用 pi05_base init) |
| deepdive_kai0 代码 | (变化) | git clone from GitHub |
| `.venv` Python 环境 | (变化) | `pip install -e .` 重建 |
| **uc02 / uc03 ckpt** | 各 42 GB | 已在 uc02/uc03 ckpt 49999 (final SOTA), 但 uc 重装后会丢失. **如果你需要这两个 ckpt 用于 deploy, 应该现在补备份!** |

### 4.2 关于 `A_new_pure_200_val`

inline_eval 需要这个。如果没有, 训练能跑但 inline_eval 会失败 (或跳过)。

建议补备份: 
```bash
ssh uc01 "tosutil cp -r /home/tim/local_ckpts/data/Task_A/self_built/A_new_pure_200_val \
  tos://transfer-shanghai/backup_uc_reinstall_20260516/datasets/A_new_pure_200_val/ -j 32"
```

### 4.3 uc02 / uc03 best ckpt

如果你需要保留 best SOTA (MAE@1=0.0088 / 0.0089) 作为后续 deploy 候选:

```bash
ssh uc02 "tosutil cp -r /cluster_ckpt/checkpoints/pi05_flatten_fold_a_new_pure2_1800/task_a_new_pure_1800_new_norm_base_mixed1/49999 \
  tos://transfer-shanghai/backup_uc_reinstall_20260516/ckpts/pure_1800_mixed1_step_49999/ -j 32"
ssh uc03 "tosutil cp -r /data/shared/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_a_new_smooth_800_new_norm/task_a_new_smooth_800_new_norm/49999 \
  tos://transfer-shanghai/backup_uc_reinstall_20260516/ckpts/smooth_800_step_49999/ -j 32"
```

(各 42G, 共 84G 额外)

---

## 5. 参考

- Mining 事件: `deepdive_kai0/docs/security/2026-05-16_rvn_miner_incident.md`
- 训练范式对比: `deepdive_kai0/docs/training/training_paradigm_comparison.md`
- Task A SOTA 排行: `deepdive_kai0/docs/training/00_action_only_finetune_history.md`

## 6. Backup 完成验证 (push 完后自动跑)

push 完成后, 可以用 tosutil ls 确认:
```bash
tosutil ls tos://transfer-shanghai/backup_uc_reinstall_20260516/ -d -r 2>&1 | head -30
# 应该看到:
# datasets/A_new_pure_200/    Size: ~3.2G
# ckpts/pi05init_step_4000/   Size: ~42G
```

记录 backup 完成时间: 待 push 完成后填入。

---

**备份执行命令** (历史记录, 已跑):
```bash
ssh uc01 "tosutil cp -r /home/tim/local_ckpts/data/Task_A/self_built/A_new_pure_200 \
  tos://transfer-shanghai/backup_uc_reinstall_20260516/datasets/A_new_pure_200/ -j 32 -p 8"
ssh uc01 "tosutil cp -r /home/tim/local_ckpts/checkpoints/pi05_flatten_fold_a_new_pure_200_js/task_a_pure200_new_norm_base_pi0.5/4000 \
  tos://transfer-shanghai/backup_uc_reinstall_20260516/ckpts/pi05init_step_4000/ -j 32 -p 8"
```
