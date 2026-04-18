# run_gf1.sh 实验计划

## 实验背景

本项目 **kai0** 基于 openpi (Physical Intelligence 的 pi0/pi0.5 模型) 框架，旨在通过三大模块提升双臂机器人操作的鲁棒性：
1. **Model Arithmetic** (权重空间合并) — `run_gf1.sh`
2. **Stage Advantage** (阶段性优势估计) — `run_gf2.sh`
3. **Train-Deploy Alignment** (训练部署对齐)

`run_gf1.sh` 对应 **Model Arithmetic** 模块：将 3055 条演示数据分为 4 个子集，分别训练独立模型，最后通过权重合并(inverse_loss/gradient_descent等方法)生成更鲁棒的混合策略。

## 实验目标

1. 在 Task_A (Flatten and Fold cloth) 的 4 个数据子集上，各训练 25K 步的 pi0.5 模型
2. 生成 4 个子集模型 checkpoint，为后续 model arithmetic 权重合并做准备
3. 验证分割训练策略的可行性，对比全量训练(run_gf0.sh, 100K步)的效果

### 训练配置
- **模型**: pi0.5 (pi05=True)
- **数据**: Task_A/base, 共 3055 episodes, 分为 4 份 (764, 764, 764, 763)
- **步数**: 每份 25,000 步
- **batch_size**: 命令行覆盖为 128 (config 默认 256)
- **FSDP**: 8 GPU (A100 80GB)
- **预训练权重**: pi05_base checkpoint
- **Prompt**: "Flatten and fold the cloth."

## 遇到的问题

### 问题 1: 进程被 SIGPIPE 意外终止
- **现象**: 首次运行时用 `bash run_gf1.sh 2>&1 | head -100` 管道方式执行，`head -100` 读完 100 行后关闭管道，导致训练进程收到 SIGPIPE 信号被 kill
- **表现**: 训练进程(PID 3300)在数据加载阶段被静默终止，日志无错误信息，GPU 内存释放

### 问题 2: 初始化耗时较长
- **现象**: 从启动到首个训练步需要 ~15 分钟
- **原因**:
  - 数据加载: 8 个 worker 加载 764 个 episode 的视频数据，耗时 ~12 分钟
  - 模型初始化: JAX/XLA 编译 + FSDP 权重分片，耗时 ~2 分钟
  - 首步 JIT 编译

### 问题 3 (非阻塞): batch_size 不一致
- **config.py** 默认 `batch_size=256`
- **run_gf1.sh** 命令行传 `--batch-size 128`
- 实际 `local_batch_size=128` (每设备 16)

## 分析与解决

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| SIGPIPE 终止 | 管道 `\| head -100` 导致 | 改用 `nohup ... > log 2>&1 &` 后台运行 |
| 初始化慢 | 数据加载+XLA编译 | 正常现象，无需修复 |
| 无错误日志 | TPU/ROCm 后端不可用 | 正常警告，实际使用 CUDA 后端 |

## 验证方案

### 1. 训练过程验证
- [x] 所有 8 GPU 利用率 100%，各 ~77GB 显存
- [x] 训练速度 ~2s/step
- [x] Loss 正常下降: step 0 → 0.6831, step 200 → 0.0419
- [x] grad_norm 稳定: step 200 → 0.2477
- [ ] 4 个 split 依次完成训练 (split_0 预计 ~14h)

### 2. 输出检查
- [ ] 每个 split 生成 checkpoint: `checkpoints/pi05_flatten_fold_split_$i/split_${i}_v1/`
- [ ] keep_period=5000 保留关键 checkpoint (step 5000, 10000, 15000, 20000, 25000)
- [ ] norm_stats.json 已复制到 checkpoint 目录

### 3. 后续步骤 (Model Arithmetic)
- [ ] 使用 `model_arithmetic/dump_data.py` 导出验证集
- [ ] 使用 `model_arithmetic/arithmetic.py` 或 `arithmetic_torch.py` 合并 4 个 checkpoint
- [ ] 可选方法: inverse_loss, gradient_descent, average, greedy
- [ ] 评估合并模型 vs 全量训练(run_gf0.sh)的性能对比

### 4. 监控命令
```bash
# 查看训练进度
grep "Step\|Progress" /vePFS/tim/workspace/deepdive_kai0/gf1_train.log | tail -10

# 查看 GPU 状态
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# 检查进程
ps aux | grep train.py | grep -v grep

# 查看 loss 趋势
grep "^Step" /vePFS/tim/workspace/deepdive_kai0/gf1_train.log
```

## 当前状态

- **时间**: 2026-03-29 12:00 启动
- **进度**: split_0 训练中，step ~380/25,000
- **日志**: `/vePFS/tim/workspace/deepdive_kai0/gf1_train.log`
- **PID**: 651960 (bash), 651975 (train.py)
- **预计完成**: split_0 ~14h, 全部 4 splits ~56h
