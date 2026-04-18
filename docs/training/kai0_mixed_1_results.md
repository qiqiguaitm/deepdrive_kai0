# kai0_mixed_1 复现实验总结

本文档记录 `kai0_mixed_1`（论文 Model Arithmetic 模块）的复现实验、发现与结论。

## 实验目标

复现 OpenDriveLab 论文中 `Task_A/mixed_1` 的训练 recipe：在 base + dagger 合并数据上训练 4 个 split，然后用 Model Arithmetic 合并。

**参考文档**：
- [training_plans.md](./training_plans.md) - 方案总体设计
- [parallel_execution_plan.md](./parallel_execution_plan.md) - 双机并行执行计划

## 实验流程

### 1. 数据准备（~30 min）

```
base (3055 ep, 3.36M frames) + dagger (3457 ep, 2.42M frames)
   ↓ merge_lerobot.py
kai0_mixed_1_data (6512 ep, 5.78M frames, 84 GB)
   ↓ compute_norm_states_fast.py
norm_stats.json
```

**发现**：合并后的 norm_stats 与官方 `Task_A/mixed_1/norm_stats.json` **不完全一致**（state mean[0]: ours=-0.096 vs official=-0.120，差异 10%）。

**分析**：官方训练数据可能包含额外的增强或不同版本的 DAgger 数据。我们采用可用数据继续实验，接受"配方近似"而非"精确复现"。

### 2. 4-Split 切分

```python
# seed=42, 随机均分 6512 episodes
split_0: 1628 episodes (767 base + 861 dagger)
split_1: 1628 episodes (807 base + 821 dagger)
split_2: 1628 episodes (721 base + 907 dagger)
split_3: 1628 episodes (760 base + 868 dagger)

# 4 splits 合起来覆盖全部 6512 episodes (train set leakage!)
```

### 3. 双机并行训练（~71 hours）

| 机器 | Splits | 开始时间 | 结束时间 | 耗时 |
|------|--------|---------|---------|------|
| **gf1** | split_0, split_2 | 2026-04-05 10:32 | 2026-04-08 08:22 | ~70h |
| **gf0** | split_1, split_3 | 2026-04-05 10:32 | 2026-04-08 09:12 | ~71h |

**共 4 × 25,000 steps, batch_size=256, 8 GPU FSDP per machine**

训练时间分布：
```
split_0 (gf1):  4/05 10:32 → 4/06 21:23  (35h)
split_2 (gf1):  4/06 21:23 → 4/08 08:22  (35h)
split_1 (gf0):  4/05 10:32 → 4/06 22:03  (36h)
split_3 (gf0):  4/06 22:04 → 4/08 09:12  (35h)
```

**节省时间**: 双机并行相比单机串行（~142h）节省了 50%，实际 71h。

### 4. 最终 Loss 对齐

所有 4 个 split 收敛到几乎相同的 loss（same seed + 相同初始化 + 相似数据）：

```
split_0: final loss = 0.0047 (step 24999)
split_1: final loss = 0.0048
split_2: final loss = 0.0048
split_3: final loss = 0.0048
```

**差异 < 0.0001**，4 个模型基本重合。

### 5. Model Arithmetic 合并（3 种方法）

尝试了全部 3 种合并方法：

| 方法 | 权重 | Mixed Loss (on train val, 50 batches) |
|------|------|----------------------------------------|
| **greedy** | `[0, 0, 1.0, 0]` (退化到 split_2) | **0.009040** |
| **inverse_loss** | `[0.249, 0.248, 0.252, 0.252]` (几乎均匀) | 0.012228 |
| **gradient_descent** | `[0.115, 0.127, 0.628, 0.130]` | 0.010771 |

**最佳单 checkpoint 性能** (split_2 alone): 0.009042

### 6. Held-out 评测 (DAgger dataset)

5 个候选模型在 DAgger 数据上的评测：

```
┌────────────────────────────┬───────┬────────┬────────┬────────┐
│ Model                      │  Loss │  MSE   │   L1   │ CosDist│
├────────────────────────────┼───────┼────────┼────────┼────────┤
│ normal_99999 (base only)   │ 0.195 │ 0.0711 │ 0.0894 │ 0.0547 │
│ official_mixed (base+?)    │ 0.237 │ 0.0655 │ 0.0868 │ 0.0443 │
│ kai0_mixed_1_inverse_loss  │ 0.015 │ 0.0077 │ 0.0327 │ 0.0058 │
│ kai0_mixed_1_greedy/split_2│ 0.011 │ 0.0056 │ 0.0271 │ 0.0041 │
└────────────────────────────┴───────┴────────┴────────┴────────┘
```

**表面看** `kai0_mixed_1` 完胜 `official_mixed`（21 倍 loss 优势），但这是数据泄漏造成的。

---

## 核心发现

### Finding 1: Same-seed MA 在我们的设置下无效

**现象**：
- 4 个 split 的 final loss 几乎完全相同 (0.0047-0.0048)
- 3 种 MA 方法都无法超过最佳单 checkpoint
- Greedy 直接退化为"单选最佳"

**原因**：
- 相同 seed=42 → 相同参数初始化
- 相同 pi05_base 预训练权重
- 随机 episode 切分后，每份数据分布几乎相同
- → 4 个模型在参数空间收敛到**几乎同一个点**
- → 平均它们 = 加噪声，而不是聚合多样性

**启示**：论文 MA 有效必然依赖 **model diversity**，可能来自：
- 不同随机种子
- 不同初始化
- 不同数据增强 per split
- 按语义而非随机切分数据

我们的复现未能引入这些多样性，因此 MA 退化为"无效平均"。

### Finding 2: DAgger held-out eval 对 kai0_mixed_1 有数据泄漏

**问题**：
- `kai0_mixed_1` 的训练数据 = base (3055) + **dagger (3457)**
- 评测用的 held-out 数据 = 从 **同一份 dagger 数据** dump 出的 50 batches
- 4 个 split 加起来覆盖了**全部 6512 个 episodes**

**后果**：
- kai0_mixed_1 在 dagger eval 上的 loss (0.011) 实际上是**训练集性能**，不是 held-out 性能
- 与 `normal_99999` (loss=0.195) 的 21x 差距大部分来自"见过 vs 没见过"的差异
- **这不是公平对比**

**真正的 held-out 评测方案**（未执行，成本太高）：
- 从 dagger 中预留 10% 作为 held-out，不参与训练
- 重新训练 4 个 split 和 MA 合并
- 成本：额外 ~3 天

### Finding 3: 我们的合并数据 ≠ 官方训练数据

**证据**：
```
state mean[0] 三方对比:
  base:                   -0.0782
  our (base+dagger):      -0.0962
  official mixed_1:       -0.1199
  
反推 dagger:
  从 our:      [-0.121, 1.620, ...]
  从 official: [-0.178, 1.387, ...]  ← 差异 0.26
```

**推测**：官方训练数据可能包含：
- 额外的 DAgger 版本（HF 未发布）
- Space mirroring 增强
- Time scaling 增强
- 其他我们没有的数据

**验证方案**（未执行）：用 space_mirroring 处理 base+dagger，重算 norm_stats，看是否与官方对齐。

### Finding 4: MA 的"有效性"是一个相对概念

论文声称 MA 能带来 monotonic 性能提升，但：

1. **我们的 offline MSE 显示 MA 无帮助**（有数据泄漏+同 seed 问题）
2. **论文的 success rate 可能有帮助**（不同评测指标、不同评测数据、可能有 diversity）
3. **offline MSE vs success rate 的相关性弱**（真实部署场景与训练损失不对齐）

**结论**：要判断 MA 的真实效果，必须：
- 使用真机 success rate 指标
- 确保 model diversity
- 使用公平 held-out 评测

这些都超出了 offline 评测的能力范围。

---

## 推荐的 `kai0_mixed_1` 最终产物

基于上述发现，我们将 **`split_2` 直接作为 `kai0_mixed_1`** 的正式版本：

```
kai0_mixed_1 = checkpoints/kai0_mixed_1_split_2/split_2_v1/24999
norm_stats   = data/Task_A/kai0_mixed_1_data/norm_stats.json
```

### 理由

1. **它是 4 个 split 中 loss 最低的** (0.009042)
2. **避免引入 MA averaging noise** (inverse_loss/gradient_descent 都更差)
3. **等价于 greedy MA 结果** (greedy 选了它)
4. **公平反映 "MA on base+dagger" 在 same-seed 下的真实性能**

### 保留的候选 checkpoint

```
checkpoints/
├── kai0_mixed_1_split_0/split_0_v1/24999    # 源 split 0
├── kai0_mixed_1_split_1/split_1_v1/24999    # 源 split 1
├── kai0_mixed_1_split_2/split_2_v1/24999    # 源 split 2 ← 推荐的 kai0_mixed_1
├── kai0_mixed_1_split_3/split_3_v1/24999    # 源 split 3
├── kai0_mixed_1_greedy_only_split2/         # greedy 合并结果 (= split_2)
├── kai0_mixed_1_inverse_loss/               # inverse_loss 合并
└── kai0_mixed_1_grad/                       # gradient_descent 合并
```

这些 checkpoint 都可以用于未来的 ablation study 或进一步实验。

---

## 时间与成本统计

| 步骤 | 耗时 | GPU 小时 |
|------|-----|---------|
| 数据合并 | 30 min | 0 (CPU) |
| norm_stats 计算 | 10 min | 0 (CPU) |
| 双机并行训练（4 splits） | ~71h | 71 × 16 = 1136 GPU-hours |
| Validation dump | 15 min | 1 GPU-hour |
| MA 合并 (3 方法 + eval) | ~3h | 3 GPU-hours |
| Held-out 评测 | ~1h | 8 GPU-hours |
| **总计** | **~75 hours** | **~1150 GPU-hours** |

**硬件**: 2 × 8 × A100 80GB (16 GPU 总计)
**存储**: ~100 GB (合并数据 84GB + 7 个 checkpoint ~2GB 每个)

---

## 下一步

### 不做的事

- ❌ 重训 4 splits with different seeds (3 天，ROI 低)
- ❌ 重新预留 held-out 数据（3 天，需重训）
- ❌ 精确复现官方训练数据（需要 space_mirror + time_scale 实验）

### 做的事

- ✅ 用 `split_2` 作为 `kai0_mixed_1` 最终版
- ✅ 保留所有候选 checkpoint 供未来分析
- ➡️ 进入 `kai0_full` 实验（参见 [training_plans.md](./training_plans.md) 第二部分）
  - Advantage estimator 训练
  - 数据增强（mirror + time_scale）
  - AWBC fine-tune
  - 完整 chi0 pipeline 复现

### 评测局限性必须记住

即使 `kai0_full` 跑完，**offline MSE/MSE 指标无法可靠判断 SA/TDA 的贡献**，除非：
- 有真机测试能力
- 或者仿真环境有足够保真度
- 或者有公平的 held-out 测试集

当前条件下，`kai0_full` 的最主要价值是**产出一个可供真机测试的完整 checkpoint**，而不是离线评测证明"SA/TDA 有多少提升"。

---

## 附录：关键数据表

### A.1 训练 loss 曲线（split_0 为代表）

```
Step      Loss      Grad Norm    Param Norm
────────────────────────────────────────────
0         0.6950    5.258        1802.39
100       0.2720    1.861        1802.39
500       0.0263    0.134        1802.39
1000      0.0208    0.103        1802.45
5000      0.0127    0.060        1803.65
10000     0.0096    0.051        1805.04
15000     0.0066    0.045        1805.92
20000     0.0055    0.040        1806.30
24999     0.0047    0.039        1806.54
```

### A.2 Greedy Search 详细日志

```
Phase 1 - Evaluate individual checkpoints:
  Checkpoint 1 (split_0): loss=0.009081
  Checkpoint 2 (split_1): loss=0.009101
  Checkpoint 3 (split_2): loss=0.009019  ← selected
  Checkpoint 4 (split_3): loss=0.009028

Phase 2 - Try adding to [split_2]:
  + Checkpoint 1: loss=0.011142  (worse)
  + Checkpoint 2: loss=0.011031  (worse)
  + Checkpoint 4: loss=0.011086  (worse)

-> No improvement found. Stopping.
Final greedy weights: [0, 0, 1, 0]
Final mixed loss: 0.009040  (= split_2 alone)
```

### A.3 Gradient Descent 最终权重

```
50 iterations of Adam optimization on simplex (softmax of log_weights):

Best iteration loss (single batch): 0.006474
Best weights: [0.115, 0.127, 0.628, 0.130]

50-batch average Mixed loss: 0.010771  ← 真实 mixed 性能
```

### A.4 Inverse Loss 权重

```
Computed individual losses:
  split_0: 0.00910
  split_1: 0.00912
  split_2: 0.00904
  split_3: 0.00904

Inverse loss weights (w_i ∝ 1/loss_i²):
  [0.2487, 0.2475, 0.2519, 0.2518]  ← 几乎均匀

Mixed loss: 0.012228  (worse than any single)
```

### A.5 Episode 分布

```
kai0_mixed_1_data (6512 episodes total):
  - Episodes 0-3054: from base dataset
  - Episodes 3055-6511: from dagger dataset

Random split (seed=42):
  split_0: 767 base + 861 dagger = 1628 total
  split_1: 807 base + 821 dagger = 1628 total
  split_2: 721 base + 907 dagger = 1628 total
  split_3: 760 base + 868 dagger = 1628 total
  
Total coverage: 3055 base + 3457 dagger = 6512 ✓ (full)
```

---

## 参考

- [训练方案: training_plans.md](./training_plans.md)
- [并行执行: parallel_execution_plan.md](./parallel_execution_plan.md)
- [kai0 论文](https://arxiv.org/abs/2602.09021)
- [官方 HF Kai0](https://huggingface.co/OpenDriveLab-org/Kai0)
