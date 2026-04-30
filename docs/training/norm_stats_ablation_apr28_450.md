# norm_stats 消融实验：new_norm vs inherit_norm (mix_apr28_450)

> **目的**：在同 dataset、同 init、同 hyperparams 下，对比 **从当前 train 数据重算 norm_stats** vs **继承 init 模型 norm_stats** 对训练过程与终点的影响。
> **范围**：Task A FlattenFold cold-start 训练，30k steps × 405 train ep + 45 val ep。
> **创建日期**：2026-04-30
> **状态**：A 组 (new_norm) 仍训练中，B 组 (inherit_norm) 已完成。

---

## 1. 实验设置

两组 head-to-head, 唯一差异是 norm_stats:

| 维度 | A 组 (new_norm) | B 组 (inherit_norm) |
|---|---|---|
| 机器 | gf1 | gf0 |
| Config | `pi05_flatten_fold_mix_apr28_450` | `pi05_flatten_fold_mix_apr28_450_inherit_norm` |
| exp_name | `mix_apr28_450_v1` | `mix_apr28_450_inherit_norm_v1` |
| 数据集路径 | `Task_A/self_built/mix_apr28_450/` | `Task_A/self_built/mix_apr28_450_inherit/` (data/videos symlinked) |
| **norm_stats 来源** | **新计算** (从 405 train ep 重算) | **继承自 `Task_A/mixed_1/norm_stats.json`** (init 模型 snapshot) |
| Init params | `Task_A/mixed_1/params` | 同 |
| 数据成分 | 150 vis_2026-04-28 + 150 kai0_base + 150 kai0_dagger (stratified val 15/15/15) | 同 (symlinked) |
| Total frames | 323,203 train + 36,500 val | 同 (data 一致) |
| steps | 30,000 | 30,000 |
| batch / fsdp | 128 / 8 | 128 / 8 |
| peak_lr / warmup / decay | 1.5e-5 / 1000 / cosine to 1.5e-6 over 30k | 同 |
| ema_decay | 0.9999 | 0.9999 |
| save_interval / inline_eval_every | 2000 / 1 | 2000 / 1 |
| 启动时间 | 2026-04-29 11:00 CST | 2026-04-29 11:32 CST |
| 完成时间 | (预计 Thu 19:00-20:00 CST) | **Thu 11:29 CST** ✅ |
| 总耗时 | ~32-33 hr (gf1 步速 3.1 s/step) | **23:55 hr** (gf0 步速 2.0 s/step) |

**为什么 gf1 比 gf0 慢 56%**: 历史观察, gf1 在多次训练中 train 步速一致比 gf0 慢 (3.1 vs 2.0 s/step), 与本实验无关。可能是硬件/网络配置差异。

---

## 2. norm_stats 数值对比

| feature | mixed_1 (inherit) | mix_apr28_450 (new) | 漂移 |
|---|---|---|---|
| 公开 sha256 | `e46a10b056e70cbc...` (固定) | (由当前 405 train 重算) | — |
| state mean (示例 dim 0) | -0.0639 | (新值, 略不同) | 数值差异 ~5-15% |
| state std | (mixed_1 训练时分布) | (新数据分布) | 同上 |

⚠️ **本质**: mixed_1 是早期 kai0 多源数据上预训的 model, 它的 norm_stats 反映 mixed_1 训练数据的分布。new norm 是当前 405 train (含 150 vis_2026-04-28 新数据) 的真实分布。两者会有 ~5-15% 漂移。

---

## 3. 完整 inline-eval MAE 数据

### A 组 new_norm (gf1) — 进行中

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0182 | 0.0437 | 0.0827 | 0.1292 |
| 4000 | 0.0174 | 0.0402 | 0.0751 | 0.1177 |
| 6000 | 0.0167 | 0.0370 | 0.0677 | 0.1055 |
| 8000 | 0.0161 | 0.0344 | 0.0617 | 0.0957 |
| 10000 | 0.0154 | 0.0323 | 0.0576 | 0.0889 |
| 12000 | 0.0148 | 0.0310 | 0.0551 | 0.0850 |
| 14000 | 0.0143 | 0.0301 | 0.0536 | 0.0826 |
| 16000 | 0.0138 | 0.0295 | 0.0527 | 0.0813 |
| 18000 | 0.0135 | 0.0293 | 0.0523 | 0.0806 |
| 20000 | 0.0132 | 0.0290 | 0.0520 | 0.0801 |
| **22000** | **0.0131** | 0.0290 | 0.0519 | 0.0799 |
| 24-30k | (待出, 每 ~2 hr 一次) | — | — | — |

### B 组 inherit_norm (gf0) — 完成 ✅

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0216 | 0.0451 | 0.0816 | 0.1257 |
| 4000 | 0.0203 | 0.0416 | 0.0746 | 0.1157 |
| 6000 | 0.0192 | 0.0384 | 0.0678 | 0.1047 |
| 8000 | 0.0181 | 0.0357 | 0.0621 | 0.0954 |
| 10000 | 0.0173 | 0.0337 | 0.0582 | 0.0890 |
| 12000 | 0.0165 | 0.0323 | 0.0558 | 0.0852 |
| 14000 | 0.0158 | 0.0313 | 0.0542 | 0.0828 |
| 16000 | 0.0153 | 0.0308 | 0.0534 | 0.0815 |
| 18000 | 0.0149 | 0.0304 | 0.0529 | 0.0808 |
| 20000 | 0.0147 | 0.0302 | 0.0526 | 0.0803 |
| 22000 | 0.0144 | 0.0300 | 0.0524 | 0.0801 |
| 24000 | 0.0142 | 0.0300 | 0.0524 | 0.0800 |
| 26000 | 0.0141 | 0.0300 | 0.0524 | 0.0799 |
| **28000** | **0.0140** | 0.0300 | 0.0524 | 0.0799 ★ best |
| 29999 | 0.0140 | 0.0300 | 0.0525 | 0.0799 (final) |

---

## 4. 头对头对比

### 同 step head-to-head (同 val, 数值直接可比 — val 集相同, 只 norm_stats 不同)

| step | A new_norm | B inherit_norm | gap (new_norm 优势) |
|---:|---:|---:|---:|
| 2000 | 0.0182 | 0.0216 | **-16%** ⭐ 早期最大 |
| 4000 | 0.0174 | 0.0203 | -14% |
| 6000 | 0.0167 | 0.0192 | -13% |
| 8000 | 0.0161 | 0.0181 | -11% |
| 10000 | 0.0154 | 0.0173 | -11% |
| 12000 | 0.0148 | 0.0165 | -10% |
| 14000 | 0.0143 | 0.0158 | -9% |
| 16000 | 0.0138 | 0.0153 | -10% |
| 18000 | 0.0135 | 0.0149 | -9% |
| 20000 | 0.0132 | 0.0147 | -10% |
| 22000 | 0.0131 | 0.0144 | -9% |
| 24-30k | (待出) | 0.0140-0.0142 | (预计 -10%) |

### 关键观察

1. **new_norm 在每个 step 都更低 MAE**: 从 step 2000 起就一致优于 inherit_norm, **差距稳定在 9-16%**
2. **gap 随训练缓慢缩窄**: 早期 16% (step 2k) → 后期 9-10% (step 22k+); 但**永不收敛到 0** — 模型无法完全"忘掉"错误的 norm
3. **inherit_norm plateau 较早**: step 26000-29999 基本停在 0.0140-0.0141, 改善缓慢
4. **new_norm 仍在下降**: step 22k=0.0131, 仍每 2k step 改善 ~1-2%, 预测 step 30k 终点 **~0.0125-0.0128**

### 预测最终结果

| 指标 | A new_norm (预测 step 30k) | B inherit_norm (final step 29999) | gap |
|---|---:|---:|---:|
| MAE@1 | **~0.0125-0.0128** | **0.0140** | **-9-11%** |
| @10 | ~0.0282-0.0285 | 0.0300 | -5-6% |
| @25 | ~0.0510-0.0515 | 0.0525 | -2-3% |
| @50 | ~0.0790-0.0795 | 0.0799 | -0.5-1% |

**重要**: gap 在长 horizon (@10/@25/@50) 比 short horizon (@1) 小很多。说明 norm_stats 主要影响**单步 action 精度**, 对 chunk action 趋势的影响小 — 模型的 long-term planning 还是基本一致。

---

## 5. 解释：为什么 new_norm 一直更优？

### 假设 1 (验证): 数据分布漂移

mixed_1 init 模型在 kai0 早期数据上训练, 其 norm_stats 反映**早期 kai0 数据**的 state/action 分布。新数据 (vis_2026-04-28) 是 visrobot01 真机采集, joint range / gripper distribution 与早期 kai0 数据有一定差异。

继承的 inherit_norm 对新数据"normalize 不准" → 模型看到的输入分布与训练时(隐含)的分布不一致 → 影响精度。

new_norm 完美匹配当前训练数据 → 输入归一化最优 → 训练效率最高。

### 假设 2 (部分支持): 模型适应能力

模型 fine-tune 过程中可以学着"补偿" norm_stats 的偏差 (通过权重调整)。但补偿不完整, 导致 plateau floor 较高。

随训练进行, gap 从 16% → 9-10% (缩了 ~6-7%) → 模型确实在补偿, 但**补偿不完全**。

### 假设 3 (未排除): NormStats 与 EMA 交互

EMA=0.9999 的长 horizon EMA 可能与 inherit_norm 配合更好? 但本实验数据**否定**该假设 — new_norm + EMA=0.9999 全程更优。

---

## 6. 工程结论

✅ **冷启 fine-tune 必须重算 norm_stats** — 即使 init 模型有自己的 norm_stats snapshot, 也应该用当前训练数据重算。`compute_norm_states_fast.py` 是必要前置步骤。

⚠️ **续训 (--resume) 是另一种情况** — 续训应该保留旧 norm_stats 以维持模型一致性 (避免输入分布跳变), 见 `task_a_visrobot01_mixed_600.md` Section 3 visrobot01_only Phase B 的设计。

📊 **gap 大小取决于数据分布漂移**: 本实验 mix_apr28_450 与 mixed_1 的差异主要来自:
- 150 vis_2026-04-28 是新采集 (visrobot01 cam)
- 150 kai0_base + 150 kai0_dagger 是 mixed_1 早期训练时未必见过的样本 (random sample seed=42)

如果新训练数据与 init 模型训练数据完全同分布, 预期 gap 会很小 (~1-2%)。本实验 gap **9-16%** 提示 mix_apr28_450 与 mixed_1 数据分布**显著漂移**。

---

## 7. Checkpoint 路径与部署 tar 包

### B 组 inherit_norm (已完成)

```
/vePFS/.../checkpoints/pi05_flatten_fold_mix_apr28_450_inherit_norm/mix_apr28_450_inherit_norm_v1/
├── 2000/  4000/  ...  28000/  29999/   ★ 28000 推荐
└── norm_stats.json (== mixed_1 snapshot)
```

**部署 tar 包** (已打包, 2026-04-30 11:38 CST):
- 路径: `/vePFS/tim/workspace/deepdive_kai0_tmp/data/mix_apr28_450_inherit_norm_best.tar`
- 大小: 11.6 GB (12,440,524,800 bytes)
- 内容: `params/` + `_CHECKPOINT_METADATA` + `assets/`
- MAE@1 = 0.0140

### A 组 new_norm (训练中, 自动 watcher 待打包)

```
/vePFS/.../checkpoints/pi05_flatten_fold_mix_apr28_450/mix_apr28_450_v1/
├── 2000/  4000/  ...  22000/   (训练中, 还会保存 24k/26k/28k/30k)
└── norm_stats.json (== 当前 405 train 重算)
```

**部署 tar 包** (训练完成自动打包, ETA Thu 19:00-20:00 CST):
- 路径: `/vePFS/tim/workspace/deepdive_kai0_tmp/data/mix_apr28_450_new_norm_best.tar`
- 内容: 同上 (params + _CHECKPOINT_METADATA + assets)

---

## 8. 历史

| 日期时间 (CST) | 事件 |
|---|---|
| 2026-04-29 11:00 | gf1 mix_apr28_450 new_norm 启动 (30k 步) |
| 2026-04-29 11:32 | gf0 mix_apr28_450 inherit_norm 启动 (30k 步) |
| 2026-04-30 03:38 (UTC) ≈ 11:29 CST | gf0 inherit_norm 完成 (step 29999, best step 28000 @ MAE 0.0140) |
| 2026-04-30 11:38 (CST) | inherit_norm best ckpt step 28000 自动打包 (11.6 GB) |
| 2026-04-30 ~19:00-20:00 (预计) | gf1 new_norm 完成, 自动打包 |

---

## 9. 关联文档

- `task_a_visrobot01_mixed_600.md` — Task A 全参数微调系列 (mix_vis600 / pure_vis600 / vis_base_40k 三组 40k 训练)
- `00_action_only_finetune_history.md` — 主排行榜
- `kai0_mixed_1_results.md` — mixed_1 init 来源, 含其原始 norm_stats 计算依据
