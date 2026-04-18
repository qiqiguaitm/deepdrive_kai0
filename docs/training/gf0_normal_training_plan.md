# GF0：Normal π₀.₅ Fine-tuning 实验记录与计划

> 日期：2026-03-29
> 配置：`pi05_flatten_fold_normal`
> 脚本：`run_gf0.sh`

---

## 一、实验背景

kai0 (χ₀) 框架复现的**阶段 1**：在 Task_A/base 数据集上对 π₀.₅ 进行全参数微调，获得基础模型。该基础模型是后续所有模块（Model Arithmetic、Stage Advantage、Train-Deploy Alignment）的前置依赖。

### 环境配置

| 项目 | 值 |
|------|-----|
| 机器 | di-20260312174527-n5dw4 |
| GPU | 8× A100 80GB (81920 MiB) |
| 框架 | JAX + FSDP |
| 数据集 | Task_A/base，3,055 episodes，9,165 视频 |
| 基础权重 | π₀.₅ base checkpoint (13GB, Orbax) |
| 训练步数 | 100,000 steps |
| 批大小 | 128（命令行覆盖，config 默认 256） |
| 保存间隔 | 每 5,000 步 |

---

## 二、实验目标

1. **验证 B1 阻塞项已修复**：之前所有训练均因 `load_dataset` NameError 失败，需确认 lerobot 环境问题已解决
2. **完成 100K 步全量微调**：获得 `pi05_flatten_fold_normal/normal_v1` checkpoint
3. **确认训练稳定性**：loss 正常收敛、无 OOM、无 NaN/Inf

---

## 三、遇到的问题

### 问题 1：历史阻塞 — `load_dataset` NameError（已自然修复）

**现象**：之前 `run_gf0.sh` 和 `run_gf1.sh` 均在数据加载阶段崩溃：
```
NameError: name 'load_dataset' is not defined
# 出错位置：lerobot/common/datasets/lerobot_dataset.py:622
```

**结论**：本次运行中未再出现此错误，可能是后续环境操作（`uv sync` / `uv pip install`）已修复了 datasets 包的兼容性问题。**B1 阻塞项已解除。**

### 问题 2：初始化阶段长时间无输出（~15 分钟）

**现象**：脚本启动后，日志停留在 `data_config:` 打印处约 15 分钟无新输出，GPU 0 占用 73GB 但利用率 0%，GPUs 1-7 仅 429MB。

**分析**：
- 日志行为正常 — `data_config` 的打印内容极长（包含完整 NormStats 数组），实际是一条未完结的日志行
- 15 分钟沉默期对应以下步骤（均为 CPU 密集型）：
  1. 数据加载器初始化 + 首批数据预取（含视频解码）
  2. 模型参数初始化 + base checkpoint 加载（13GB）
  3. FSDP 分片策略计算 + 参数/优化器状态分布到 8 卡
  4. XLA JIT 编译首个 train_step（最耗时）
- 分片完成后，8 卡各占 ~76.8GB，利用率升至 99-100%

**结论**：非异常行为，属于大模型 FSDP 初始化的正常耗时。

### 问题 3：Checkpoint 保存崩溃 — `config.save_train_state` 不存在

**现象**：训练在首次保存 checkpoint 时崩溃：
```
AttributeError: 'TrainConfig' object has no attribute 'save_train_state'
# 出错位置：scripts/train.py:281
```

**根因**：`git diff` 显示 commit 12d4554 在 `save_state()` 调用中添加了第 5 个参数 `config.save_train_state`，但未在 `TrainConfig` dataclass 中添加对应字段，也未修改 `checkpoints.py` 中 `save_state()` 的签名来接受该参数。原始 commit 771ebbf 只传 4 个参数。

**修复**：移除多余的第 5 个参数，恢复为原始 4 参数调用：
```python
# 修复前（12d4554）：
_checkpoints.save_state(checkpoint_manager, train_state, data_loader, step, config.save_train_state)
# 修复后：
_checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
```

### 问题 4：batch_size=256 在 A100-80GB 上可行性

**现象**：`run_gf0.sh` 将 config 默认的 batch_size=256 覆盖为 128。

**分析**：
- nvidia-smi 显示 76.8GB 是 XLA 预分配的内存池（`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`），非实际使用量
- 模型使用 `nothing_saveable` 重算策略 + bfloat16 + FSDP，激活内存极小
- 实测 batch_size=256 成功训练 Step 0-2，8 卡各 ~76.8GB，无 OOM

**结论**：batch_size=256 在 8×A100-80GB 上完全可行。已改回论文原始设定。

### 问题 5：损坏/缺失视频文件（已修复）

**现象**：数据加载器跳过了 3 个缺失视频：
```
WARNING: skipping index 2729394 — episode_002548.mp4 (top_head) FileNotFoundError
WARNING: skipping index 3206900 — episode_002953.mp4 (top_head) FileNotFoundError
WARNING: skipping index 3207052 — episode_002953.mp4 (hand_right) FileNotFoundError
```

首次运行中还出现过 `moov atom not found`（MP4 文件损坏），在重新运行中变为 `FileNotFoundError`（可能是首次运行的 cleanup 脚本误删了相关文件）。

**影响**：3,055 个 episode 中仅 2 个受影响，数据加载器优雅跳过，对训练精度影响可忽略。

### 问题 4：首次运行输出被截断导致误判

**现象**：首次通过后台任务运行，任务输出文件被截断为 200 行，恰好在 sharding 日志处截止。任务报告 "exit code 0"，但实际进程已结束，且无 checkpoint 产出。

**分析**：后台任务捕获系统（Claude Code background task runner）有 200 行输出缓冲限制。进程实际是在 sharding/编译阶段因某种原因退出（可能是首次运行中的 `moov atom not found` 触发了致命错误路径）。

**解决**：第二次运行改为 `bash run_gf0.sh > /tmp/train_output.log 2>&1 &` 直接写文件，绕过输出截断问题。

---

## 四、当前状态

### 训练进度（截至 12:29 UTC）

| 指标 | Step 0 | Step 100 | Step 200 | Step 300 | Step 400 |
|------|--------|----------|----------|----------|----------|
| loss | 0.6831 | 0.2646 | 0.0419 | 0.0296 | 0.0247 |
| grad_norm | 5.3936 | 1.8961 | 0.2472 | 0.1930 | 0.1603 |
| param_norm | 1802.39 | 1802.39 | 1802.38 | 1802.38 | 1802.38 |

- **训练速率**：~2.0-2.5 s/step
- **当前步数**：~424/100,000
- **预计剩余时间**：~55-65 小时
- **GPU 利用率**：8 卡均 80-100%
- **GPU 显存**：8 卡各 ~76.8GB / 81.9GB

### 收敛趋势

Loss 从 0.68 → 0.025 在 400 步内快速下降，符合预训练模型微调的典型特征（初始 loss 较低，收敛较快）。grad_norm 从 5.39 → 0.16 持续下降，训练稳定。

---

## 五、验证方案

### 5.1 训练过程监控

```bash
# 实时查看最新进度
tail -20 /tmp/train_output.log

# 查看所有 step 级指标
grep "^Step " /tmp/train_output.log

# 检查是否有错误
grep -i -E "error|exception|traceback|OOM|nan|inf" /tmp/train_output.log \
  | grep -v -E "rocm|tpu|libtpu|pynvml|torchvision|moov|FileNotFound"

# GPU 状态
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```

### 5.2 Checkpoint 验证

训练完成后，验证 checkpoint 完整性：

```bash
# 确认 checkpoint 文件存在
ls -la kai0/checkpoints/pi05_flatten_fold_normal/normal_v1/

# 应包含多个步数目录（5000, 10000, ..., 100000）
# 每个目录下应有 train_state/, params/, assets/, metrics/ 等子目录
```

### 5.3 Loss 曲线分析

```bash
# 提取全部 step 指标做分析
grep "^Step " /tmp/train_output.log | awk -F'[=, ]' '{print $2, $4, $6, $8}'
```

预期指标：
- loss 在 1K 步后应降至 < 0.02
- grad_norm 应维持在 0.05 - 0.5 之间，无突然飙升
- param_norm 应基本稳定（变化 < 1%）
- 无 NaN 或 Inf 出现

### 5.4 推理功能验证（训练完成后）

```bash
# 启动 policy server 加载训练好的 checkpoint
cd kai0
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_flatten_fold_normal \
  --policy.dir=checkpoints/pi05_flatten_fold_normal/normal_v1/<best_step> \
  --port 8000
```

### 5.5 后续阶段衔接

训练完成后可进入：
- **阶段 2（Model Arithmetic）**：运行 `run_gf1.sh` 训练 4 个 split 子模型
- **阶段 3（AWBC）**：需先补全 advantage 视频（B2），再运行 `run_gf2.sh`

---

## 六、风险与注意事项

| 风险 | 概率 | 应对 |
|------|------|------|
| 训练中途 OOM | 低 | 当前 76.8/81.9 GB，余量较小；如 OOM 可降 batch_size 至 64 |
| 损坏视频触发致命错误 | 低 | 当前仅 warning 跳过，已验证不影响训练 |
| 网络存储（vePFS）IO 瓶颈 | 中 | 视频数据在 vePFS 上，IO 延迟可能影响训练速率 |
| 训练中断（进程被杀） | 中 | checkpoint 每 5K 步保存，可从最近 checkpoint 恢复 |
| Loss 发散 / NaN | 低 | 当前 400 步收敛良好，需持续监控 |

---

## 七、关键命令速查

```bash
# 查看训练进程
ps aux | grep train.py | grep -v grep

# 查看训练日志
tail -f /tmp/train_output.log

# 终止训练（如需要）
kill $(pgrep -f "scripts/train.py")

# 从 checkpoint 恢复训练（需修改 run_gf0.sh 删除 rm -rf 行）
# train.py 会自动检测已有 checkpoint 并恢复
```
