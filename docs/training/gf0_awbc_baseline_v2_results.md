# gf0 AWBC baseline v2 — 训练结果记录

**状态**: ⏸ 提前停止（2026-04-19 15:10 UTC，达 21,000 / 100,000 steps，21.0%）
**停止原因**: 验证本 baseline 的 loss/eval 已进入高边际成本区，释放 8×A100 以并行推进 stage classifier 工作

---

## 背景原理与核心设置

### 实验目标
复现 χ₀ 论文 **Stage-Advantage (SA) 模块**：在 Task_A 专家示教数据上，用 **Advantage-Weighted Behavior Cloning (AWBC)** 把"优势"信号注入到 π₀.₅ 策略中，让推理时可通过 prompt 选择高优势行为。本 run 作为 **binary-AWBC baseline**（3055 ep base 数据），供后续 awbc_v2（+dagger + space_mirroring 扩到 13K ep）做 A/B 对比。

### AWBC 的工作机理
和普通 BC 唯一区别：**训练 prompt 根据该帧的 advantage 标签切换**。
- `task_index=1` → `"Flatten and fold the cloth. Advantage: positive"`
- `task_index=0` → `"Flatten and fold the cloth. Advantage: negative"`

prompt 通过 `DataConfig(prompt_from_task=True)` 读 `meta/tasks.jsonl`，作为语言条件送入 PaliGemma。策略学到 "看到 positive prompt → 执行像高优势样本的动作" 的映射，其他架构/loss 全与 full-FT BC 一致（flow matching on normalized 14-dim actions）。

### 数据与标签生成流水线
数据集 `Task_A/advantage` (3055 ep) 由前置步骤产出：

1. **人工标注**：对 base 数据集每帧标 `stage_progress_gt ∈ [0, 1]`（手工划分 stage）。
2. **训练 Advantage Estimator (AE)**：以 `(obs, action, language)` 为输入，回归 `stage_progress_gt` 作为 progress 预测器。
3. **预测连续 advantage**：AE 对每帧预测 `stage_progress_pred`，定义
   `absolute_advantage[t] = stage_progress_pred[t+50] − stage_progress_pred[t]`
   （未来 50 帧的 progress 增量，衡量该时刻行为"推进任务"的程度）。
4. **Discretize（分桶）**：`discretize_advantage.py --stage-nums 2 --threshold 30` 按 stage 分组算 30 百分位，把每帧 `absolute_advantage` 分成 **binary** → 写入 parquet 的 `task_index` 列。
5. **tasks.jsonl**：`task_index` → prompt 字符串 的映射，如上。

### 模型架构
- π₀.₅ 3B（PaliGemma + flow matching action head），配置 `pi05=True`
- 输入：3 摄像头 (top_head/hand_left/hand_right 224×224) + 14-dim 机器人状态 + language token
- 输出：14-dim × 50-frame action chunk（flow matching 去噪，inference 用 10 步 Euler）

### 推理使用方式
部署时发送固定 positive prompt：
```python
policy.infer(obs, task="Flatten and fold the cloth. Advantage: positive")
```
这触发模型执行"像高优势数据那样"的动作分布。实验和论文均显示该方式带来显著成功率提升（χ₀ paper Fig 7）。**必须使用训练时完全相同的 prompt 格式**（句号分隔、大小写、"positive/negative" 关键词）；格式偏差会显著掉分。

### 关键超参
| 项 | 值 | 说明 |
|----|----|------|
| AWBC prompt 模式 | binary | `Advantage: positive/negative`（非 n_slices 连续） |
| `use_delta_joint_actions` | False | 绝对关节角，非 Δ |
| Action chunk 长度 | 50 frames | 30fps → 1.67 s 预测窗 |
| val_ratio | 0.1 | 10% 作 val（eval_every 1000 步） |
| eval_batches | 4 | 每次 eval 采样 batch 数 |
| eval_interval_early/late | 100 / 1000 | 前 3 次密 eval，之后每 1000 步 |

## 运行参数

| 项目 | 值 |
|------|------|
| Config | `pi05_flatten_fold_awbc` |
| exp_name | `gf0_awbc_baseline_v2` |
| 数据集 | `Task_A/advantage`（3055 ep，binary AWBC prompt） |
| 基础权重 | `gs://openpi-assets/checkpoints/pi05_base/params` |
| Batch size | 256 (FSDP 8×A100) |
| 目标步数 | 100,000（config 默认） |
| 启动 | 2026-04-18 17:26:07 UTC |
| 停止 | 2026-04-19 15:10 UTC |
| 训练时长 | 21h 44m |
| 单步耗时 | 3.73 s/step |
| Log | `logs/gf0_awbc_baseline_20260418_172607.log`（542 MB） |
| Checkpoint | `kai0/checkpoints/pi05_flatten_fold_awbc/gf0_awbc_baseline_v2/{5000,10000,15000,20000,21000}` |
| 最新 ckpt 大小 | 42 GB/ckpt (step 21000) |

## Loss curve (training loss)

| Step | loss | grad_norm | param_norm |
|------|------|-----------|------------|
| 100 | 0.3013 | 2.058 | 1802.39 |
| 500 | 0.0144 | 0.134 | 1802.39 |
| 1000 | 0.0114 | 0.126 | 1802.45 |
| 5000 | 0.0046 | 0.057 | 1803.64 |
| 10000 | 0.0027 | 0.048 | 1804.91 |
| 15000 | 0.0018 | 0.046 | 1805.68 |
| 20000 | 0.0013 | 0.037 | 1806.01 |
| 20900 | 0.0013 | 0.040 | 1806.04 |

## Eval MAE curve (val set)

| Step | joint_1 | joint_10 | joint_50 | grip_1 | grip_10 | grip_50 |
|------|---------|----------|----------|--------|---------|---------|
| 100 | 0.2079 | 0.2119 | 0.2309 | 0.0184 | 0.0189 | 0.0208 |
| 1000 | 0.0551 | 0.0638 | 0.0892 | 0.0049 | 0.0056 | 0.0073 |
| 5000 | 0.0183 | 0.0248 | 0.0409 | 0.0014 | 0.0019 | 0.0028 |
| 10000 | 0.0103 | 0.0155 | 0.0246 | 0.0007 | 0.0013 | 0.0017 |
| 15000 | 0.0069 | 0.0116 | 0.0174 | 0.0004 | 0.0008 | 0.0013 |
| 18000 | 0.0059 | 0.0096 | 0.0142 | 0.0003 | 0.0006 | 0.0009 |
| **20000** | **0.0050** | **0.0085** | **0.0117** | **0.0003** | **0.0006** | **0.0009** |

## 分析

- **收敛健康**：loss 从 0.30 平滑降至 0.0013（230×），无抖动/爆炸。
- **Eval 曲线与训练 loss 一致下降**：joint_50 MAE 从 0.23 rad 降到 0.012 rad (~0.7°)，grip_50 降到 1e-3 数值精度。
- **边际收益骤减**：15K→20K 仅 ~30% 进一步改善；预计剩 79K 步（3.4 天）只能带来 ~15% 额外下降，性价比低。
- **用途**：`step_20000` 作为 binary-AWBC baseline checkpoint，可用于后续 A/B vs awbc_v2（dagger + mirror 扩容）和 χ₀ SA 消融。

## 产物

- 最佳 checkpoint: `checkpoints/pi05_flatten_fold_awbc/gf0_awbc_baseline_v2/20000/`（eval MAE 最优）
- 兼容 `serve_policy.py` 直接推理（JAX Orbax 格式）
- 后续 awbc_v2 实验的 head-to-head baseline
