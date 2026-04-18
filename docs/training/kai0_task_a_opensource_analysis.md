# kai0 Task_A (叠衣服) 开源情况梳理

## 一、开源了什么

### 模型 (HuggingFace Model Repo: `OpenDriveLab-org/Kai0`)

| Checkpoint | 说明 | 类型 |
|-----------|------|------|
| `Task_A/mixed_1` | Task A 最佳模型 | Model Arithmetic 混合后的最终模型 (JAX Orbax, 22GB) |
| `Task_B/mixed_9` | Task B 最佳模型 | 同上 |
| `Task_C/mixed_5` | Task C 最佳模型 | 同上 |

**注意**：
- 这三个是 **Model Arithmetic (MA) 的产出物**（多个子模型加权混合），不是 AWBC 模型，也不是全流程最终模型
- `mixed_N` 中的数字是实验编号（非混合方法名称），checkpoint 元数据中未记录使用的混合方法
- 论文称 greedy search 为最优混合方法（Section IV-E），但未确认 mixed_1 是否使用该方法

### 数据 (HuggingFace Dataset Repo: `OpenDriveLab-org/Kai0`)

| 数据集 | Episodes | 大小 | 说明 |
|--------|----------|------|------|
| `Task_A/base` | 3,055 ep (~42h) | 46GB | 专家示范数据 |
| `Task_A/dagger` | 3,457 ep (~13h) | 39GB | DAgger 在线纠正数据 |
| `Task_A/advantage` | 3,055 ep | 31GB | base 的副本 + 官方 Advantage Estimator 标注列 + 离散化 `tasks.jsonl` |

advantage 数据包含的标注列：
- `stage_progress_gt` — 人工标注的分阶段进度 (0→1)
- `absolute_value` — Estimator 预测的累积进度
- `absolute_advantage` — 前后帧 absolute_value 差值，clip [-1,1]
- `relative_advantage` — 双时间步直接预测的相对进度差
- `task_index` — 二值化后的正/负 advantage 标签（threshold=30%）

### 代码 (GitHub: `OpenDriveLab/KAI0`)

三个模块代码全部开源（Apache 2.0）：
- Model Arithmetic：混合脚本 + 6 种混合方法
- Stage Advantage：Estimator 训练、eval 推理、离散化、AWBC 训练
- Train-Deploy Alignment：数据增强、DAgger 采集、多种推理模式

---

## 二、没有开源什么

| 未开源项 | 说明 |
|----------|------|
| **Normal fine-tune checkpoint** | 用户需自行训练 `pi05_flatten_fold_normal` (100K steps) |
| **Split 子模型 checkpoint (×4)** | Model Arithmetic 的输入，用户需自行训练 4 个 split (各 25K steps) |
| **Advantage Estimator checkpoint** | 用户需自行训练 `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD` (100K steps) |
| **AWBC checkpoint** | 用户需自行训练 `pi05_flatten_fold_awbc` (100K steps) |
| **全流程最终模型 (MA+SA+TDA)** | mixed_1 仅含 MA，不含 SA/TDA 效果的最终部署模型 |

---

## 三、Task A 各方法成功率

> **数据来源**：3.1 来自论文 Figure 6 柱状图（近似读数）；3.2-3.5 来自博客页面 JS 源码中的精确图表数据。不同实验条件下同一 baseline 成功率存在差异（20%~50%），论文称之为 "masking effect"。

### 3.1 模块消融实验（论文 Figure 6，近似读数）

以 π₀.₅ fine-tune 为 baseline，逐步叠加模块：

| 方法 | 使用模块 | 成功率 (%) |
|------|----------|-----------|
| π₀.₅ baseline | 无 | ~50 |
| +TDA | Train-Deploy Alignment | ~55 |
| +MA | Model Arithmetic | ~60 |
| +SA | Stage Advantage (AWBC) | ~55 |
| +MA +TDA | 两模块 | ~80 |
| +SA +TDA | 两模块 | ~85 |
| +MA +SA | 两模块 | ~80 |
| **χ₀ (MA+SA+TDA)** | **三模块全开** | **~93-95** |

### 3.2 Model Arithmetic 独立消融（博客精确数据）

**数据来源**：博客页面 JS 源码，变量名 `s`，图表标题 "Model Arithmetic"

```javascript
// 原始数据
{trick:"Task A", single:60, single_std:[9.4,9.4], full:73.3, full_std:[4.7,4.7], model:90, model_std:[4.7,4.7]}
```

| 方法 | Task A 成功率 (%) | 说明 |
|------|-------------------|------|
| Best Candidate（最好的单个子模型） | 60.0 ± 9.4 | 4 个 split 模型中表现最好的 |
| Full Data（全量数据训练单模型） | 73.3 ± 4.7 | `pi05_flatten_fold_normal` 的效果 |
| **Model Arithmetic（混合模型）** | **90.0 ± 4.7** | **mixed_1 对应此数据** |

**确认**：`mixed_1` 是 MA 模块的产物，**90.0 ± 4.7% 是 MA 单模块在 Task A 上的成功率**，不含 SA 和 TDA。这是博客 Model Arithmetic 消融图表中 Task A 的精确数值。

### 3.3 Stage Advantage 独立消融（博客精确数据）

```javascript
// 原始数据
{trick:"chrome", SuccessRate:66.7, SuccessRate_std:[.001,.001]}  // Value-diff
{trick:"safari", SuccessRate:76.7, SuccessRate_std:[4.7,4.7]}    // Direct
{trick:"firefox", SuccessRate:80, SuccessRate_std:[18,18]}        // Direct+Stage
```

| SA 方法 | 成功率 (%) | 说明 |
|---------|-----------|------|
| Value-diff (π₀.₆ 风格) | 66.7 | 非分阶段的 advantage |
| Direct | 76.7 ± 4.7 | 直接预测 advantage |
| **Direct + Stage（kai0 方法）** | **80.0 ± 18.0** | 分阶段 advantage（方差较大） |

### 3.4 DAgger 消融（博客精确数据）

```javascript
// 原始数据
{trick:"baseline", pi05:20, pi05_std:[2.5,2.5], pi0:0, pi0_std:[0,.001]}
{trick:"+ heuristic DAgger", pi05:83.3, pi05_std:[2.8,2.8], pi0:73.3, pi0_std:[3.2,3.2]}
{trick:"+ DAgger", pi05:93.3, pi05_std:[2.2,2.2], pi0:80, pi0_std:[2.8,2.8]}
```

| 方法 | π₀.₅ 成功率 (%) | π₀ 成功率 (%) |
|------|----------------|---------------|
| Baseline | 20.0 ± 2.5 | 0.0 |
| + Heuristic DAgger | 83.3 ± 2.8 | 73.3 ± 3.2 |
| + DAgger | 93.3 ± 2.2 | 80.0 ± 2.8 |

### 3.5 推理优化消融（博客精确数据）

```javascript
// 原始数据
{trick:"baseline", absolute:36.7, absolute_std:[4,4], delta:36.7, delta_std:[4.7,4.7]}
{trick:"+ inchunk smooth", absolute:30, absolute_std:[3,3], delta:66.7, delta_std:[9.4,9.4]}
{trick:"+ temp. smooth", absolute:76.7, absolute_std:[6,6], delta:83.3, delta_std:[4.7,4.7]}
{trick:"+ RTC", absolute:90, absolute_std:[4.7,4.7], delta:83.3, delta_std:[4.7,4.7]}
```

| 推理方式 | Absolute Joint (%) | Delta Joint (%) |
|----------|-------------------|-----------------|
| Sync (baseline) | 36.7 ± 4.0 | 36.7 ± 4.7 |
| + Inchunk smooth | 30.0 ± 3.0 | 66.7 ± 9.4 |
| + Temporal smooth | 76.7 ± 6.0 | 83.3 ± 4.7 |
| + RTC | **90.0 ± 4.7** | 83.3 ± 4.7 |

---

## 四、开源模型 vs 未开源模型成功率对照

| 模型 | 是否开源 | 成功率 (%) | 数据来源 |
|------|---------|-----------|----------|
| π₀.₅ base | ✓ (openpi 官方) | 起始点，未直接评测 | — |
| π₀.₅ fine-tune (normal, full data) | ✗ checkpoint 未开源 | 73.3 ± 4.7 | 博客 MA 图 "full data" |
| **mixed_1 (MA 混合模型)** | **✓ 已开源** | **90.0 ± 4.7** | **博客 MA 图 "model arithmetic"** |
| AWBC 模型 (SA 产物) | ✗ 未开源 | 80.0 ± 18.0 (SA 独立) | 博客 SA 图 "Direct+Stage" |
| Advantage Estimator | ✗ 未开源 | N/A（中间工具） | — |
| **χ₀ 全流程 (MA+SA+TDA)** | **✗ 未开源** | **~93-95** | 论文 Figure 6（近似读数） |

### 关键结论

1. **已开源的 `mixed_1` 成功率 90.0 ± 4.7%** — 来自博客 Model Arithmetic 消融图表的精确数据，是 MA 单模块在 Task A 上的最佳结果
2. **论文最高成功率 ~95% 的全流程模型未开源** — MA + SA + TDA 三模块组合
3. 从 mixed_1 (90%) 到全流程 (~95%) 的约 5% 提升来自 SA 和 TDA 模块
4. Normal fine-tune (full data) 成功率 73.3% — gf0/gf1 正在复现的阶段一模型的预期效果
5. SA 模块独立贡献 80% 成功率，但方差大 (±18%)
6. 论文提到相同设置下成功率有 20%~60% 波动（"masking effect"），实验可复现性受此影响

---

## 五、gf2 复现的定位

gf2 复现的两个训练（AWBC + Advantage Estimator）属于 **Stage Advantage 模块**：

```
                    已开源 ✓           gf2 正在复现            未开源 ✗
                 ┌──────────┐     ┌──────────────────┐    ┌────────────┐
π₀.₅ base ──→   │ mixed_1  │     │ Advantage Est.   │    │ χ₀ 全流程    │
  (openpi)       │ (MA 产物)│     │ AWBC model       │    │ MA+SA+TDA  │
                 │ SR: 90%  │     │ SR: ~80%         │    │ SR: ~95%   │
                 └──────────┘     └──────────────────┘    └────────────┘
```

各阶段预期效果：
- **gf0/gf1 阶段一**（normal + split + MA）：预期达到 mixed_1 水平 (~90%)
- **gf2 阶段二**（AWBC）：SA 模块独立贡献 ~80%，叠加 MA 后预期 ~80-90%
- **全流程**：需要实体机器人的 TDA 模块才能达到 ~95%
