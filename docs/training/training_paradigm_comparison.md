# Task A 训练范式对比 (single-stage vs two-stage with mixed_1 intermediate)

> **范围**: 比较三种使用 ≈7000-8000 ep 数据训练 Task_A 的范式, 重点对比 **single-stage (pi05_base 一次性)** vs **two-stage (经 mixed_1 中转)**。
>
> **背景**: `mixed_1` = pi05_base 在 6000 ep 官方数据上 finetune 的产物 (Task_A 通用表示)。后续可作为下游小数据 finetune 的高质量 init。
>
> **3 个实验**:
> - **A. `task_a_new_pure2_1800_6000_new_norm`** (老 SOTA) — pi05_base + 7900 ep **一次性** (一阶段)
> - **B. `task_a_new_pure2_1800_new_norm_base_pi0.5`** ⏳ TBD — pi05_base + **1800 ep 直接** (一阶段, 但数据少)
> - **C. `task_a_new_pure_1800_new_norm_base_mixed1`** (uc02 新, 已完成) — mixed_1 + 1800 ep (**两阶段** 的 Stage 2; Stage 1 = pi05_base + 6000 已隐含在 mixed_1)

---

## 1. 总览对比表

| 维度 | A. SOTA 一阶段 | B. ⏳ 1800ep 一阶段 | C. uc02 mixed1 两阶段 |
|---|---|---|---|
| Config | `pi05_flatten_fold_a_new_pure2_1800_6000` | (待定; 可复用 `pi05_flatten_fold_a_new_pure2_1800` + 改 init = pi05_base) | `pi05_flatten_fold_a_new_pure2_1800` |
| Init | pi05_base | pi05_base | **mixed_1** (= pi05_base + 6000 ep) |
| Stage 1 data | n/a | n/a | **6000 ep 官方** (Task_A mixed_1 训练用) |
| Stage 2 data | 7900 ep one-shot (1790 v2-mirror + 3055 base + 3055 advantage) | 1800 ep (`A_new_pure2_1800`) | 1800 ep (`A_new_pure2_1800`) |
| Stage 1 cost | n/a | n/a | (历史成本, 已支付) |
| Stage 2 cost | 50k step × 24 GPU × ~1 s/it ≈ **560 GPU·h** | 50k step × 8 GPU × ~1.9 s/it ≈ **210 GPU·h** | 50k step × 8 GPU × ~1.9 s/it ≈ **210 GPU·h** |
| Final MAE@1 | **0.0085** | TBD | **0.0088** (+3.5%) |
| Final @10 | 0.0168 | TBD | **0.0153** (-8.9%) |
| Final @25 | 0.0254 | TBD | **0.0203** (-20.1%) |
| Final @50 | 0.0337 | TBD | **0.0258** (-23.4%) |
| 完成日期 | 2026-05-13 | (未跑) | 2026-05-16 |
| 详细 doc | `task_a_new_mixed_pure2_1800_6000_results.md` | (待) | `task_a_new_pure_1800_new_norm_base_mixed1_results.md` |

---

## 2. 范式 A: pi05_base + 7900 ep 一阶段

### 配置
- Init: pi05_base (raw pretrained, 无 Task_A 适配)
- Data: 7900 ep 大杂烩 = 1790 `-new` mirror + 3055 `kai0_base` + 3055 `kai0_advantage`
- Cluster: uc01+02+03 (24 GPU HSDP)
- 50k step, batch=120

### 训练动力学
- 起点 (step 4k) MAE@1 = 0.0534 (pi05_base 未适配 Task_A 状态)
- step 24k MAE@1 = 0.0106 (-80%)
- step 49999 MAE@1 = **0.0085** (SOTA at the time, beaten only by pure_200 #7 后续 NEW SOTA)
- 收敛特征: 前 24k 步快速 -80%, 后 24k 步缓慢 -20%

### 优势
1. **单步 (@1) 最优**: 0.0085, 比两阶段范式 (0.0088) 低 3.5%
2. 数据多样性最高 (3 个来源混合), 模型见的样本最丰富

### 劣势
1. **Long-horizon (@50) 较差**: 0.0337 vs 两阶段 0.0258 (差 30.6%)
2. **训练成本最高**: 24 GPU × 24h = ~560 GPU·h (Stage 2 只)
3. mixed_1/2 等中间产物未保留, 不能复用

---

## 3. 范式 B: pi05_base + 1800 ep 直接 ⏳ TBD

### 假设配置
- Init: pi05_base (raw)
- Data: 1800 ep `A_new_pure2_1800` (与 C 相同)
- Cluster: 8 GPU 单机 (或 16 GPU 双机)
- 50k step, batch=120

### 预期结果 (根据 A 和 C 推断)

由于数据少 (1800 vs 7900), 但 init 是 cold pi05_base, 预期:
- step 4k MAE@1 ≈ 0.0534 (起点应同 A)
- **收敛 ceiling 假设比 C (0.0088) 差** 因为缺 Stage 1 的 6000 ep 通用表示积累
- **预测 final MAE@1 ≈ 0.0095~0.0105** (类似 1800 v5 0.0090 但 init 更冷 → 应略差)
- @50 预测 ≈ 0.0400~0.0500 (类似 1800 v5 0.0328 但更弱)

### 为什么要做这个对照?

填充 2x2 因子设计:
| Init \ Data | 1800 ep | 7900 ep |
|---|---|---|
| **pi05_base** | **B (空缺)** | A = 0.0085 |
| **mixed_1** | C = 0.0088 | (无对应) |

若做完 B, 能干净 disentangle:
- 数据规模影响 (B vs A, init 同 = pi05_base)
- Init 选择影响 (B vs C, data 同 = 1800 ep)

### 推荐方案
- 与 C 同 server (uc02 或 uc03), 同 hyperparams, 仅换 `--weight-loader.params-path` → pi05_base
- 预算: 26h 训练 + 2h eval = 28h
- Exp name 建议: `task_a_new_pure2_1800_base_pi0.5` (与 B 系列命名对齐)

---

## 4. 范式 C: mixed_1 + 1800 ep 两阶段 (uc02 已完成)

### 配置
- Init: mixed_1 (= pi05_base + 6000 ep, 已预训练)
- Data: 1800 ep `A_new_pure2_1800` (Stage 2)
- Cluster: uc02 单机 8 GPU
- 50k step, batch=128

### 训练动力学
- 起点 (step 4k) MAE@1 = **0.0128** (比 A 的 0.0534 低 4.2x — warm init 优势明显)
- step 24k MAE@1 = 0.0095
- step 49999 = **0.0088** (@10=0.0153 @25=0.0203 @50=0.0258)
- 后段 plateau 明显 (step 40k 后只小幅改善)

### 优势
1. **Long-horizon (@10/@25/@50) 显著最优**: 比 A 全线领先 9-23%
2. **训练成本节省**: Stage 2 仅 ~210 GPU·h (含 Stage 1 摊薄成本依然总比 A 省, 因为 mixed_1 可复用)
3. **mixed_1 可复用**: 后续 multiple Task_A 微调实验都用同一 mixed_1, 摊薄 Stage 1 成本
4. **warm start** = 起点 MAE 低 4x, 早期 step 即可达到 A 在中期才达到的水平

### 劣势
1. **单步 (@1) 略弱**: 0.0088 vs A 的 0.0085, 差 3.5%
2. mixed_1 是 fixed checkpoint, 无法在 Stage 2 中重新调整 6000 ep 数据见过的内容
3. Stage 2 数据 (1800 ep) 必须不与 Stage 1 数据严重重复, 否则浪费

---

## 5. 范式对比 — 关键洞察

### 5.1 单步精度 vs 长程精度的权衡

| 指标 | A. 一阶段 SOTA | C. 两阶段 mixed1 | 谁好 |
|---|---|---|---|
| @1 (单步精度) | 0.0085 | 0.0088 | A (+3.5%) |
| @10 | 0.0168 | 0.0153 | **C (-8.9%)** |
| @25 | 0.0254 | 0.0203 | **C (-20.1%)** |
| @50 (50步规划) | 0.0337 | 0.0258 | **C (-23.4%)** |

**假设**: chunk planner 的 long-horizon 表示需要"成熟"的低层 representations。
- A: 一阶段训练, low-level 和 chunk planner 同步学, 后者可能"过拟合"早期不稳定的 low-level
- C: 两阶段, low-level 在 Stage 1 已稳定, Stage 2 专注 chunk planner refinement

**实际意义**: 部署 (real-robot action 选择) 看哪个 horizon 主导。
- 短时反应任务 (单步控制): A 更好
- 多步规划任务 (chunk-level decision): C 更好

### 5.2 训练成本对比

| 范式 | Stage 1 (GPU·h) | Stage 2 (GPU·h) | 总 (本次实验) | 摊薄 (后续 N=5 个 Task_A 实验) |
|---|---|---|---|---|
| A 一阶段 | 0 | 560 | 560 | 5×560 = **2800** |
| C 两阶段 | 1000 (假设 6000 ep ×8GPU×42h ≈ 1000) | 210 | **1210** (首次) | 1000 + 5×210 = **2050** |

**N≥3 时 C 范式更省**, 复用 mixed_1 摊薄 Stage 1 成本。

### 5.3 范式选择决策树

```
是否第一次训 Task_A?
├─ 是 → 用 A 范式 (一次性 7900 ep, 简单直接, SOTA @1)
│        - 同时**保存中间 step 6000-ish ckpt 作为未来 mixed_1 候选**
│
└─ 否 (已有 mixed_1)?
    ├─ 部署侧重单步精度 → A 范式 (@1 略优)
    └─ 部署侧重长程规划 → **C 范式** (@10/@25/@50 全胜, 还省 GPU·h)
```

---

## 6. 与 pure_200 NEW SOTA 的进一步对比

`task_a_new_pure_200_new_norm` (pure_200 #7) 是 **NEW SOTA**: 200 ep `-new` 精选 + mirror + mixed_1_clean init, final MAE@1=**0.0065**.

| 实验 | 范式 | 数据 | Init | MAE@1 | @50 |
|---|---|---|---|---|---|
| pure_200 #7 (NEW SOTA) | 两阶段 (mixed_1_clean = mixed_1 cleaned 版本) | **200 ep cleaned + mirror** | mixed_1_clean | **0.0065** | **0.0079** |
| A 老 SOTA | 一阶段 | 7900 ep | pi05_base | 0.0085 | 0.0337 |
| C uc02 mixed1 | 两阶段 | 1800 ep | mixed_1 | 0.0088 | 0.0258 |

**核心发现**: 
- **数据 quality + 精选 >> 数据 quantity**: 200 ep 精选 (with mirror) 完胜 7900 ep 大杂烩
- **mixed_1 类 init 在小数据上极强**: 200 ep 时优势最大 (MAE 仅 0.0065)
- **Mirror augmentation 关键**: 对 long-horizon (@50) 帮助巨大

更新决策树:

```
有多少 Task_A 精选数据?
├─ ≥200 ep -new + mirror → 用 mixed_1/mixed_1_clean + small data **优先尝试 200 ep ⭐**
├─ ≥1800 ep                → C 范式 (mixed_1 + 1800 ep, 长程优化)
└─ 完全新场景               → A 范式 (pi05_base + 7900 ep 一次性)
```

---

## 7. 后续行动

- [ ] **跑 B 范式** (`task_a_new_pure2_1800_base_pi0.5`): 与 C 同 server 同 hyperparam, 只换 init = pi05_base, 填补 2x2 因子设计
- [ ] 用 mixed_1 + 200 ep mirror (复制 pure_200 #7 setup, 但用 mixed_1 而非 mixed_1_clean) 测试 mixed_1 vs mixed_1_clean 在小数据下的差异
- [ ] 评估 uc02 ckpt 49999 在 sim01 真机 long-horizon 表现 (@25/@50 优势是否转化为真机增益)
- [ ] 探索 Stage 1 用更大数据 (e.g., 10000 ep mixed_2) 是否能进一步提升 Stage 2 后的 @1

## 8. 参考

- A 详细: `task_a_new_mixed_pure2_1800_6000_results.md`
- C 详细: `task_a_new_pure_1800_new_norm_base_mixed1_results.md`
- NEW SOTA 详细: `task_a_new_pure_200_new_norm_results.md`
- uc03 smooth_800 (另一 1800 ep 类似实验, mixed_1_clean init): `task_a_new_smooth_800_new_norm_results.md`
- Mining 安全事件 (uc02/03 训练受影响): `docs/security/2026-05-16_rvn_miner_incident.md`
