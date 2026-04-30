# 实验历史汇总（pi05 / kai0）

> **作用**：集成本机所有训练实验的历史记录与结果，**含每步 inline-eval MAE@{1,10,25,50} 完整曲线**。涵盖训练类型: action-only freeze、全参数解冻 (full-finetune)、LoRA (r=16/32)、AWBC、cold-start 混合数据训练 (mix_vis600 / pure_vis600 / mixed_visrobot01) 等。
> **范围**：Task E（扶起倒箱）+ Task P（抓放盒子）+ Task A（FlattenFold） — 三个任务下的所有 train run, 每条 run 的 best step / best MAE / 数据规模 / freeze 策略 / LoRA r 都在此聚合; 详细超参 / 数据配方移到下方 "关联详细文档" 列表的对应专题文件。
> **最近更新**：2026-04-30 21:40 CST (norm_stats 消融全部完成: new_norm best 0.0127 vs inherit_norm best 0.0140, gap -9.3%)
> **数据来源**：`logs/train_*.log` 中 `[inline-eval] step=N MAE@1=… @10=… @25=… @50=…` 行（9 val ep × 20 frames，~30s/eval），与 `logs/eval_history_v2/v2_step_*.json` 离线归档（9 val ep × 50 queries）。
> **命名前缀 `00_` 用于按文件名排序时置顶。**
>
> **关联详细文档**：
> - `task_e_master_plan.md` — Task E 完整规划与所有 Phase 1/2 实验细节
> - `task_p_unfreeze_8k_20k_analysis.md` — Task P 全参数解冻对照
> - **`task_a_visrobot01_mixed_600.md`** — Task A 全参数微调系列 (mixed_gf0_173 / visrobot01_only / mix_vis600 / pure_vis600)
> - **`norm_stats_ablation_apr28_450.md`** — norm_stats 消融实验 (new_norm vs inherit_norm, head-to-head 同 dataset 同 hparams)
> - `kai0_mixed_1_results.md` — Task A 迁移 init 来源
> - `training_plans.md` — kai0_mixed_1 / kai0_full 训练 recipe
> - `project_complete_guide.md` — freeze_filter / inline eval 总览

---

## 1. TL;DR 全实验 best MAE 排行榜

> 每行的 best 取整个训练过程中 inline-eval MAE@1 最低的 step（**不是 final step**）。

| 排名 | 实验 | 任务 | 数据 | 步数 | best step | **best MAE@1** | @10 | @25 | @50 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 🥇 | **t10_allgood_25k** | Task E | base_allgood | 25k | **24999** | **0.0233** | 0.0415 | 0.0683 | 0.1066 |
| 🥇 | **t16_allgood_wd** | Task E | base_allgood | 25k | 24999 | **0.0233** | 0.0415 | 0.0683 | 0.1065 |
| 🥈 | **t15_allgood_40k** | Task E | base_allgood | 40k | 39999 | **0.0234** | 0.0412 | 0.0667 | 0.1021 |
| 🥉 | **t6_kai0_allgood** | Task E | base_allgood | 15k | 14999 | 0.0245 | 0.0423 | 0.0694 | 0.1087 |
| 🥉 | **t14_allgood_lora** | Task E | base_allgood + LoRA | 15k | 14999 | 0.0245 | 0.0415 | 0.0675 | 0.1072 |
| 5 | **t7_lora16_v2** | Task E | base_merged + LoRA r=16 (fixed init) | 25k | 24000 | 0.0257 | 0.0418 | 0.0646 | 0.0986 |
| 6 | **t1_lora16** | Task E | base + LoRA r=16 (E2 init) | 15k | 14999 | 0.0260 | 0.0442 | 0.0709 | 0.1081 |
| 6 | t1_lora32 | Task E | base + LoRA r=32 (E2 init) | 15k | 14000 | 0.0261 | 0.0442 | 0.0706 | 0.1068 |
| 7 | **v3e_lowlr (E2)** | Task E | base 64ep | 15k | 14999 | **0.0262** | 0.0446 | 0.0713 | 0.1080 |
| 8 | v3e_combo (E3)¹ | Task E | base 64ep | (8k crash) | 12000 | 0.0277 | 0.0460 | 0.0716 | 0.1086 |
| 9 | v3e_ema (E1) | Task E | base 64ep | 15k | 14999 | 0.0297 | 0.0482 | 0.0734 | 0.1105 |
| 10 | v3e_long (E4) | Task E | base 64ep + 默认 LR | 15k | 14999 | 0.0327 | 0.0508 | 0.0785 | 0.1175 |
| 11 | **v3_kai0_base** | Task E | base 64ep | 15k | 12000 | 0.0333 | 0.0514 | 0.0775 | 0.1164 |
| 12 | v5_kai0_aug | Task E | full_aug 256ep | 15k | 14999 | 0.0351 | 0.0552 | 0.0881 | 0.1329 |
| 13 | v2 baseline (16k)² | Task E | base 64ep | 16k | 16000 | 0.0382 | 0.0543 | 0.0790 | 0.1119 |
| 14 | v2 baseline (canon)² | Task E | base 64ep | 15k | 14000 | 0.0411 | 0.0574 | 0.0820 | 0.1153 |
| 15 | v8_pi05_mirror | Task E | mirror 128ep | 15k | 14000 | 0.0421 | 0.0598 | 0.0876 | 0.1283 |
| 16 | t21_bs64³ | Task E | base_allgood + bs=64 | 25k | 14000 | 0.0418 | 0.0676 | 0.1163 | 0.2121 |
| 16 | t22_bs128³ | Task E | base_allgood + bs=128 | 25k | 6000 | 0.0422 | 0.0702 | 0.1246 | 0.2307 |
| 17 | v4_pi05_aug | Task E | full_aug 256ep | 15k | 10000 | 0.0452 | 0.0677 | 0.0978 | 0.1350 |
| 对照 | **unfreeze_20k** ⚡ | Task P | base 24k frames | 20k | 4000 | **0.0195** | 0.0367 | 0.0600 | 0.0797 |
| 对照 | unfreeze_8k ⚡ | Task P | base 24k frames | 8k | 3000 | 0.0206 | 0.0380 | 0.0610 | 0.0806 |
| 对照 | p_t10_allgood_25k | Task P | base_allgood | 25k | 14000 | 0.0626 | 0.0860 | 0.1265 | 0.1842 |
| 对照 | p_v3_kai0init (P-T10) | Task P | base 24k frames + freeze | 15k | 6000 | 0.0703 | 0.0965 | 0.1380 | 0.1950 |
| Task A | **mixed_gf0_173** ⚡ | Task A | mix 173+173+173 ep | 13k | 9000 | **0.0129** | 0.0296 | 0.0521 | 0.0786 |
| Task A | **mix_vis600_v1** ⚡ | Task A | mix 310+145+145 (540 train) | 40k | **38000** | **0.0146** | 0.0320 | 0.0554 | 0.0834 |
| Task A | **pure_vis600_v1** ⚡ | Task A | 309 orig + 291 hflip mirror (560 train) | 40k | **39999** | **0.0151** | 0.0268 | 0.0404 | 0.0558 |
| Task A | **mix_apr28_450 (new_norm)** 🏆 | Task A | 150 vis_apr28 + 150 kai0_base + 150 dagger | 30k | **28000** | **0.0127** | 0.0289 | 0.0518 | 0.0796 |
| Task A | mix_apr28_450 (inherit_norm) ⚡ | Task A | 同上 (norm_stats inherited from mixed_1) | 30k | **28000** | **0.0140** | 0.0300 | 0.0524 | 0.0799 |
| Task A | **vis_base_40k_v1** ⚡ | Task A | vis_base 288+22 ONLY (no mix, no mirror) | 40k | **36000** | **0.0168** | 0.0365 | 0.0606 | 0.0907 |
| Task A | visrobot01_only_v1 (B end) ⚡ | Task A | vis_base 288+22 (Phase A→B) | 12k | 11999 | 0.0171 | 0.0373 | 0.0625 | 0.0943 |
| Task A | visrobot01_only_v1 (Phase A end) ⚡ | Task A | visrobot01-only 193+17 | 9k | 9000 | 0.0179 | 0.0389 | 0.0648 | 0.0974 |
| Task A | visrobot01_only_2k_gf0 ⚡ | Task A | visrobot01-only 193+17 | 2k | 1999 | 0.0202 | 0.0411 | 0.0680 | 0.1017 |

¹ E3 (combo) 在 step ~8k 因 GPU 1 NUMA SIGSEGV 中断，best 在 step 12000 之前；step 10000=0.0284, step 12000=0.0277。
² v2 训练超过 nominal 15k 步，step 16000 实测最佳 (0.0382)；master plan 中 0.0411@14000 是 canonical 数。
³ bs=64/128 series 没做 LR 缩放 → 大 bs 下 effective LR 过低 → 早期反而更差，且 long-horizon (@50) 严重退化（0.21+）。
⚡ Task P / Task A 用 8×A100 全解冻，与 Task E 不同基线，**仅供对照**，不在同一榜单。
⏳ 训练中，详见 `task_a_visrobot01_mixed_600.md`。
**4-way ablation 全部完成 (2026-04-29 01:21 CST 最后一个 vis_base_40k 落地)**: 数据 hierarchy 已确认 — kai0 跨域 (mix 0.0146) > hflip mirror (pure 0.0151) > 单源 (vis_only 0.0168), 三者 final gap 各 3.4% / 10.1% / 13.1%。详见 `task_a_visrobot01_mixed_600.md` Section 6.3 + Section 7。

**norm_stats 消融 (mix_apr28_450 同 dataset 头对头, ✅ 全部完成 2026-04-30)**: 两组 final plateau 后 — **new_norm 0.0127 vs inherit_norm 0.0140 (-9.3%)**。head-to-head gap 从早期 16% 缩到 final-plateau 9.3%, 不会完全闭合。@1/@10/@25/@50 horizon-dependent gap: 9.3%/3.7%/1.1%/0.4% (norm 主要影响**单步精度**, 对 chunk planning 影响小)。**冷启必须重算 norm_stats**。详见 `norm_stats_ablation_apr28_450.md`。

**核心修正**（vs 上一版）：
1. **真正的 Task E 最佳是 t10_allgood_25k = 0.0233**，不是 E2 的 0.0262。允许更长训练 + "allgood" 增广数据后，性能再下一台阶（-12%）。
2. **t6/t10/t14/t15/t16 系列**（kai0_allgood + 长训）此前未在 master plan 中归档，本汇总首次系统化。
3. v3 / v5 / v3e_lowlr / v3e_ema / t7 等 best step 已用完整曲线核实。
4. **Task A 系列首次归档** (2026-04-25): mixed_gf0_173 = 0.0129, visrobot01_only B end = 0.0171, 详见 `task_a_visrobot01_mixed_600.md`。

**两条核心结论**：
1. Task E 小数据（64 ep base）：action-only + 短训打到 ~0.033；用 base_allgood（增广扩到 ~256 ep 等价规模）+ 25k 步可压到 **0.0233**，比 v2 基线 -43%。
2. Task P 中等数据（24k frames）：**全解冻显著优于 action-only**（0.0195 vs 0.0703，-72%），但**真机抓取仍偏 2-3cm**，说明 offline MAE 不是唯一指标。

---

## 2. 标准 freeze 策略（全 action-only 实验通用）

```python
from openpi.shared import nnx_utils
import flax.nnx as nnx

freeze_filter = nnx.All(
    nnx_utils.PathRegex(".*PaliGemma.*"),         # 冻整个 PaliGemma（视觉塔 + LLM 主干）
    nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),  # 但放开 Action Expert（llm 内并行塔，带 _1 后缀）
)
# 顶层 action_in_proj / action_out_proj / time_mlp_* 不在 PaliGemma 下，天然可训
ema_decay = None  # 冻 backbone 下 EMA 收益小，省 ~6.5 GB/卡
```

**冻结后规模**：trainable ≈ 800 M（占总 3.3 B 的 24%）；train_state 从 ~65 GB → ~22 GB；单卡 5090 batch=4 稳在 ~21 GB。

**Phase 2 LoRA 变体**：
```python
freeze_filter = nnx.All(
    nnx_utils.PathRegex(".*PaliGemma.*"),
    nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),
    nnx.Not(nnx_utils.PathRegex(".*lora_[ab].*")),  # 同时放开 LoRA 分支
)
# LoRA init 必须 w_b = zeros（identity start），normal(0.01) 会扰动 E2 已收敛权重 ~5%
```

---

## 3. 完整实验矩阵（含 per-step 曲线）

> 所有数据来自训练时 inline-eval（9 val ep × 20 frames）。**粗体 = 该实验最佳 step**。
> v2 baseline 来自 `logs/eval_history_v2/v2_step_*.json` 离线归档（9 val ep × 50 queries），样本数不同因此与 inline-eval 数值不严格可比，但趋势一致。

### 3.1 Task E — Phase 0：v2 baseline（2026-04-17，pi05_base + base 64ep + 全冻 PaliGemma）

| step | MAE@1 | @10 | @25 | @50 | 备注 |
|---:|---:|---:|---:|---:|---|
| 2000 | 0.0541 | 0.0723 | 0.0975 | 0.1327 | |
| 4000 | 0.0537 | 0.0721 | 0.0995 | 0.1366 | |
| 6000 | 0.0466 | 0.0648 | 0.0916 | 0.1293 | |
| 8000 | 0.0453 | 0.0620 | 0.0867 | 0.1191 | |
| 10000 | 0.0423 | 0.0591 | 0.0853 | 0.1216 | |
| 12000 | 0.0418 | 0.0589 | 0.0850 | 0.1249 | |
| 14000 | 0.0411 | 0.0574 | 0.0820 | 0.1153 | canonical 引用值 |
| **16000** | **0.0382** | **0.0543** | **0.0790** | **0.1119** | **真正最佳**（超过 nominal 15k 续跑） |

### 3.2 Task E — Phase 1 差异化 2×2（2026-04-18）

**因子**：init (kai0_mixed_1 vs pi05_base) × 数据 (base 64ep vs full_aug 256ep / mirror 128ep)

#### v3 (`pi05_stand_box_kai0init` / kai0_mixed_1 + base 64ep) → GPU 1

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0427 | 0.0611 | 0.0854 | 0.1214 |
| 4000 | 0.0409 | 0.0604 | 0.0880 | 0.1310 |
| 6000 | 0.0380 | 0.0565 | 0.0854 | 0.1275 |
| 8000 | 0.0374 | 0.0561 | 0.0828 | 0.1191 |
| 10000 | 0.0349 | 0.0530 | 0.0789 | 0.1157 |
| **12000** | **0.0333** | **0.0514** | **0.0775** | **0.1164** |

#### v4 (`pi05_stand_box_aug` / pi05_base + full_aug 256ep) → GPU 2

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0636 | 0.0830 | 0.1112 | 0.1549 |
| 4000 | 0.0533 | 0.0764 | 0.1083 | 0.1479 |
| 6000 | 0.0532 | 0.0750 | 0.1038 | 0.1431 |
| 8000 | 0.0454 | 0.0677 | 0.0991 | 0.1396 |
| **10000** | **0.0452** | **0.0677** | **0.0978** | **0.1350** |
| 12000 | 0.0469 | 0.0708 | 0.1068 | 0.1537 |
| 14000 | 0.0469 | 0.0652 | 0.0948 | 0.1364 |

#### v5 (`pi05_stand_box_kai0_aug` / kai0_mixed_1 + full_aug 256ep) → GPU 3

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0467 | 0.0701 | 0.1021 | 0.1507 |
| 4000 | 0.0404 | 0.0638 | 0.0974 | 0.1432 |
| 6000 | 0.0421 | 0.0617 | 0.0905 | 0.1345 |
| 8000 | 0.0372 | 0.0573 | 0.0873 | 0.1295 |
| 10000 | 0.0360 | 0.0563 | 0.0872 | 0.1299 |
| 12000 | 0.0377 | 0.0584 | 0.0910 | 0.1398 |
| 14000 | 0.0407 | 0.0565 | 0.0803 | 0.1180 |
| **14999** | **0.0351** | **0.0552** | **0.0881** | **0.1329** |

#### v8 (`pi05_stand_box_mirror` / pi05_base + mirror 128ep) → GPU 0

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0688 | 0.0853 | 0.1143 | 0.1553 |
| 4000 | 0.0585 | 0.0787 | 0.1077 | 0.1472 |
| 6000 | 0.0491 | 0.0673 | 0.0983 | 0.1414 |
| 8000 | 0.0468 | 0.0646 | 0.0895 | 0.1256 |
| 10000 | 0.0462 | 0.0636 | 0.0914 | 0.1323 |
| 12000 | 0.0462 | 0.0629 | 0.0888 | 0.1320 |
| **14000** | **0.0421** | **0.0598** | **0.0876** | **0.1283** |
| 14999 | 0.0440 | 0.0608 | 0.0899 | 0.1312 |

**Phase 1 结论**：
1. **init 主导**：v3 (kai0+base) vs v4 (pi05+aug) — kai0_mixed_1 init 提升 **26%** @1。
2. **time_scaling 负贡献**：v3 (base) > v5 (full_aug)；v8 (mirror only) > v4 (full_aug)。time_scaling 在 64 ep 小数据上稀释信号。
3. **kai0 init 5× 加速**：v3 step 2k 已 0.0427 ≈ v2 step 14k (0.0411)。

### 3.3 Task E — Phase 1-FT v3 续训（E1/E2/E3/E4，2026-04-18）

从 `v3/12000` init 续训 15k 步。

#### E2 `v3e_lowlr` （peak_lr 2.5e-5 → **1.25e-5**）→ 🥇 best 续训方案

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0312 | 0.0492 | 0.0745 | 0.1107 |
| 4000 | 0.0319 | 0.0501 | 0.0784 | 0.1213 |
| 6000 | 0.0297 | 0.0470 | 0.0715 | 0.1076 |
| 8000 | 0.0285 | 0.0466 | 0.0732 | 0.1096 |
| 10000 | 0.0276 | 0.0458 | 0.0716 | 0.1062 |
| 12000 | 0.0268 | 0.0453 | 0.0717 | 0.1074 |
| 14000 | 0.0263 | 0.0448 | 0.0709 | 0.1059 |
| **14999** | **0.0262** | **0.0446** | **0.0713** | **0.1080** |

#### E1 `v3e_ema`（ema_decay=0.9999）

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0322 | 0.0505 | 0.0764 | 0.1148 |
| 4000 | 0.0314 | 0.0496 | 0.0756 | 0.1131 |
| 6000 | 0.0307 | 0.0491 | 0.0749 | 0.1117 |
| 8000 | 0.0304 | 0.0489 | 0.0746 | 0.1115 |
| 10000 | 0.0302 | 0.0487 | 0.0744 | 0.1114 |
| 12000 | 0.0299 | 0.0486 | 0.0740 | 0.1111 |
| 14000 | 0.0298 | 0.0484 | 0.0735 | 0.1107 |
| **14999** | **0.0297** | **0.0482** | **0.0734** | **0.1105** |

#### E3 `v3e_combo` （EMA + lowLR）— GPU 1 NUMA SIGSEGV，仅 step 12k 数据

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 10000 | 0.0284 | 0.0467 | 0.0723 | 0.1094 |
| **12000** | **0.0277** | **0.0460** | **0.0716** | **0.1086** |
| (>12k) | 崩 | | | |

#### E4 `v3e_long`（默认 LR 续训）

| step | MAE@1 | @10 | @25 | @50 | 备注 |
|---:|---:|---:|---:|---:|---|
| 14000 | 0.0334 | 0.0496 | 0.0739 | 0.1102 | 比 v3/12k (0.0333) 还差 |
| **14999** | **0.0327** | **0.0508** | **0.0785** | **0.1175** | 默认 LR 在已收敛权重上震荡 |

**Phase 1-FT 结论**：lowLR (1.25e-5) 是最强续训手段（-21% vs v3 起点）；EMA 仅 10% 增益；combo 边际；默认 LR 续训 = 破坏。

### 3.4 Task E — Phase 2 vision LoRA & 长训实验（2026-04-19 ~ 04-21）

> 此阶段重点：Phase 1-FT 的 E2 (0.0262) 不是天花板。**长训 + base_allgood 数据可压到 0.0233**。

#### T1-1 `pi05_stand_box_vision_lora16` / t1_lora16 (E2 init + LoRA r=16 on vision MLP)

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0296 | 0.0472 | 0.0734 | 0.1098 |
| 4000 | 0.0317 | 0.0497 | 0.0764 | 0.1108 |
| 6000 | 0.0303 | 0.0480 | 0.0729 | 0.1075 |
| 8000 | 0.0283 | 0.0464 | 0.0730 | 0.1111 |
| 10000 | 0.0277 | 0.0456 | 0.0716 | 0.1070 |
| 12000 | 0.0267 | 0.0445 | 0.0712 | 0.1084 |
| 14000 | 0.0261 | 0.0442 | 0.0706 | 0.1067 |
| **14999** | **0.0260** | **0.0442** | **0.0709** | **0.1081** |

#### T1-2 `pi05_stand_box_vision_lora32` / t1_lora32 (E2 init + LoRA r=32)

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0296 | 0.0471 | 0.0733 | 0.1099 |
| 4000 | 0.0316 | 0.0497 | 0.0764 | 0.1108 |
| 6000 | 0.0303 | 0.0480 | 0.0729 | 0.1084 |
| 8000 | 0.0282 | 0.0463 | 0.0729 | 0.1114 |
| 10000 | 0.0276 | 0.0455 | 0.0714 | 0.1066 |
| 12000 | 0.0267 | 0.0444 | 0.0711 | 0.1078 |
| **14000** | **0.0261** | **0.0442** | **0.0706** | **0.1068** |
| 14999 | 0.0261 | 0.0443 | 0.0709 | 0.1081 |

> r=16 vs r=32 在每个 step 差距 ≤ 0.0001 → **rank 不是瓶颈**。

#### T7 `pi05_stand_box_vision_lora16_v2` / t7_lora16_v2 (**w_b=0 fix** + 25k + base_merged 128ep)

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0337 | 0.0519 | 0.0789 | 0.1202 |
| 4000 | 0.0328 | 0.0513 | 0.0785 | 0.1188 |
| 6000 | 0.0301 | 0.0467 | 0.0709 | 0.1076 |
| 8000 | 0.0299 | 0.0474 | 0.0713 | 0.1055 |
| 10000 | 0.0307 | 0.0481 | 0.0741 | 0.1095 |
| 12000 | 0.0297 | 0.0468 | 0.0711 | 0.1042 |
| 14000 | 0.0287 | 0.0446 | 0.0682 | 0.1037 |
| 16000 | 0.0270 | 0.0437 | 0.0673 | 0.1013 |
| 18000 | 0.0263 | 0.0432 | 0.0672 | 0.1005 |
| 20000 | 0.0261 | 0.0432 | 0.0668 | 0.0995 |
| 22000 | 0.0257 | 0.0424 | 0.0664 | 0.1013 |
| **24000** | **0.0257** | **0.0418** | **0.0646** | **0.0986** |
| 24999 | 0.0257 | 0.0423 | 0.0661 | 0.1001 |

> 25k 步把 @50 进一步压到 0.0986（T1-1 的 0.1081 → -8.8%），说明长训对 long-horizon 收益最大。

#### T6 `pi05_stand_box_kai0_allgood` / t6_kai0_allgood (kai0 init + base_allgood + 15k)

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0384 | 0.0634 | 0.1117 | 0.2060 |
| 4000 | 0.0348 | 0.0567 | 0.0971 | 0.1715 |
| 6000 | 0.0321 | 0.0521 | 0.0867 | 0.1441 |
| 8000 | 0.0298 | 0.0488 | 0.0791 | 0.1260 |
| 10000 | 0.0276 | 0.0457 | 0.0739 | 0.1166 |
| 12000 | 0.0262 | 0.0439 | 0.0714 | 0.1126 |
| 14000 | 0.0252 | 0.0428 | 0.0698 | 0.1097 |
| **14999** | **0.0245** | **0.0423** | **0.0694** | **0.1087** |

#### T10 `pi05_stand_box_kai0_allgood_25k` / t10_allgood_25k 🥇 **最佳记录**

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0384 | 0.0634 | 0.1117 | 0.2060 |
| 6000 | 0.0322 | 0.0522 | 0.0866 | 0.1437 |
| 10000 | 0.0278 | 0.0459 | 0.0739 | 0.1166 |
| 14000 | 0.0255 | 0.0431 | 0.0699 | 0.1099 |
| 18000 | 0.0242 | 0.0419 | 0.0688 | 0.1077 |
| 20000 | 0.0239 | 0.0418 | 0.0688 | 0.1076 |
| 22000 | 0.0237 | 0.0418 | 0.0688 | 0.1073 |
| 24000 | 0.0234 | 0.0416 | 0.0684 | 0.1067 |
| **24999** | **0.0233** | **0.0415** | **0.0683** | **0.1066** |

#### T14 `pi05_stand_box_kai0_allgood_lora16` / t14_allgood_lora (allgood + LoRA r=16, 15k)

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0384 | 0.0633 | 0.1117 | 0.2063 |
| 6000 | 0.0321 | 0.0521 | 0.0868 | 0.1454 |
| 10000 | 0.0277 | 0.0456 | 0.0737 | 0.1168 |
| 12000 | 0.0261 | 0.0434 | 0.0707 | 0.1120 |
| 14000 | 0.0252 | 0.0421 | 0.0685 | 0.1091 |
| **14999** | **0.0245** | **0.0415** | **0.0675** | **0.1072** |

#### T15 `pi05_stand_box_kai0_allgood_40k` / t15_allgood_40k (allgood + 40k 长训)

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0384 | 0.0634 | 0.1117 | 0.2060 |
| 8000 | 0.0298 | 0.0488 | 0.0788 | 0.1254 |
| 14000 | 0.0257 | 0.0433 | 0.0700 | 0.1100 |
| 20000 | 0.0244 | 0.0423 | 0.0691 | 0.1077 |
| 24000 | 0.0241 | 0.0423 | 0.0686 | 0.1062 |
| 28000 | 0.0238 | 0.0420 | 0.0681 | 0.1055 |
| 32000 | 0.0236 | 0.0418 | 0.0676 | 0.1042 |
| 36000 | 0.0235 | 0.0416 | 0.0672 | 0.1030 |
| 38000 | 0.0234 | 0.0414 | 0.0669 | 0.1025 |
| **39999** | **0.0234** | **0.0412** | **0.0667** | **0.1021** |

> 25k → 40k 增加 60% 训练量，@1 仅从 0.0233 → 0.0234（持平）；但 @50 从 0.1066 → 0.1021（-4%），即**长 horizon 仍受益于长训**，@1 已饱和。

#### T16 `pi05_stand_box_kai0_allgood_wd2` / t16_allgood_wd (allgood + 25k + wd=0.01)

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 12000 | 0.0264 | 0.0441 | 0.0715 | 0.1126 |
| 18000 | 0.0242 | 0.0419 | 0.0688 | 0.1077 |
| 22000 | 0.0236 | 0.0418 | 0.0688 | 0.1073 |
| 24000 | 0.0234 | 0.0416 | 0.0684 | 0.1067 |
| **24999** | **0.0233** | **0.0415** | **0.0683** | **0.1065** |

> wd=0.01 vs T10 wd=1e-4：差异 ≤ 0.0001，weight decay 在此规模下**不是瓶颈**。

#### T21 `pi05_stand_box_kai0_allgood_bs64` & T22 `pi05_stand_box_kai0_allgood_bs128` ❌ 失败

| exp | bs | best step | best @1 | @50 | 备注 |
|---|---:|---:|---:|---:|---|
| t21_bs64 | 64 | 14000 | 0.0418 | 0.2121 | 无 LR linear-scale → effective LR 过低 |
| t22_bs128 | 128 | 6000 | 0.0422 | 0.2307 | 同上，更严重 |

> bs 增大但 LR 没按 sqrt/linear 缩放 → 整体收敛慢且 @50 灾难（0.21+ vs 正常 0.10）。**未来用大 bs 必须缩放 LR**。

**Phase 2 阶段总结**：
1. **真正最佳 = T10 allgood 25k = 0.0233**，比 E2 的 0.0262 再提升 11%。
2. **base_allgood 数据 + 长训 = 主力组合**：T6 (15k) 0.0245 → T10 (25k) 0.0233 → T15 (40k) 0.0234（@1 饱和，@50 仍降）。
3. **LoRA 不是必需**：T14 (allgood+LoRA r=16 15k) = T6 (allgood 15k) = 0.0245，LoRA 没显著帮助 small dataset 上的 vision 适配。
4. **rank / wd / bs scaling 都不是 Task E 瓶颈**：变化 ≤ 0.0001。

### 3.5 Task P — 全解冻 vs action-only 对照

> Task P (抓放盒子) 数据规模约 24k frames，比 Task E 大 ~30%。用于回答 "action-only 何时是局限"。

#### P-T10 frozen `pi05_pick_place_box_kai0init` / p_v3_kai0init (action-only, 15k 步, sim01)

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0756 | 0.0991 | 0.1421 | 0.2034 |
| 4000 | 0.0757 | 0.1029 | 0.1483 | 0.2036 |
| **6000** | **0.0703** | **0.0965** | **0.1380** | **0.1950** |
| 8000 | 0.0743 | 0.1028 | 0.1469 | 0.2013 |
| 10000 | 0.0734 | 0.1043 | 0.1489 | 0.2013 |
| 12000 | 0.0753 | 0.1026 | 0.1424 | 0.1920 |
| 14000 | 0.0728 | 0.1050 | 0.1522 | 0.2094 |
| 14999 | 0.0740 | 0.1036 | 0.1477 | 0.2010 |

> action-only 在 Task P 上**完全卡死**在 0.07，trajectory 不再下降。这是 action-only 的能力天花板。

#### p_t10_allgood_25k (Task P + base_allgood + 25k action-only)

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0741 | 0.0927 | 0.1282 | 0.2026 |
| 6000 | 0.0668 | 0.0846 | 0.1203 | 0.1877 |
| 10000 | 0.0633 | 0.0837 | 0.1227 | 0.1838 |
| 12000 | 0.0629 | 0.0849 | 0.1248 | 0.1837 |
| **14000** | **0.0626** | **0.0860** | **0.1265** | **0.1842** |
| 16000 | 0.0626 | 0.0872 | 0.1281 | 0.1848 |
| 24999 | 0.0639 | 0.0918 | 0.1338 | 0.1874 |

> allgood 数据让 action-only 的 Task P 从 0.0703 → 0.0626（-11%），但仍远高于全解冻的 0.0195。

#### Task P 全解冻（gf1 8×A100，bs=128）— 来自 `task_p_unfreeze_8k_20k_analysis.md`

| 配置 | peak_lr | best step | MAE@1 | @10 | @25 | @50 |
|---|---:|---:|---:|---:|---:|---:|
| unfreeze_8k | 2.5e-5 | 3000 | **0.0206** | 0.0380 | 0.0610 | 0.0806 |
| unfreeze_20k | 1.5e-5 | **4000** | **0.0195** | 0.0367 | 0.0600 | 0.0797 |
| unfreeze_2k (smoke) | — | early | 0.0362 | — | — | — |

**Task P 结论**：
1. **action-only 在 Task P 上是天花板**：base 0.0703，allgood 0.0626，再加数据/步数也压不下去。
2. **全解冻 = -69% MAE@1**：0.0703 → 0.0195。中等数据下 vision 适配收益巨大。
3. **真机现实**：全解冻 0.0206 ckpt 部署到 sim01，抓取瞬间偏 2-3 cm；offline MAE 不捕获关键瞬间误差。
4. **过拟合识别 3 标志**（来自 unfreeze_20k）：val MAE 反弹 +3-5%；train_loss/val_MAE gap > 10×；gradient norm 继续降但 val 不动 → 过拟合震荡。
5. **早停信号**：必须用 val MAE@1，不能用 train_loss（unfreeze_8k step 7999 train=0.0009 但 val MAE=0.0219，gap 24×）。

### 3.6 Task A — 全解冻全参数微调系列（gf0/gf1，2026-04-24 ~ 04-26）

> **完整 per-step 曲线**: `task_a_visrobot01_mixed_600.md`
> 全部 init from `Task_A/mixed_1/params`，全部 8×A100 FSDP=8 bs=128，全部 cosine schedule。

#### mixed_gf0_173 (gf0, 13k 步) ✅
| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 1000 | 0.0153 | 0.0352 | 0.0647 | 0.1020 |
| 5000 | 0.0133 | 0.0303 | 0.0532 | 0.0804 |
| 7000 | 0.0130 | 0.0298 | 0.0523 | 0.0789 |
| **9000** | **0.0129** | 0.0296 | 0.0521 | 0.0786 |
| 12999 | 0.0129 | 0.0296 | 0.0520 | 0.0785 |

step 7-12999 完全 plateau。519 ep mix (173 vis + 173 base + 173 dagger) → MAE@1=0.0129。

#### visrobot01_only_v1 (gf1, Phase A 9k → Phase B --resume 12k) ✅
| step | MAE@1 | val | 阶段 |
|---:|---:|---|---|
| 1000 | 0.0241 | 17 ep | A |
| 5000 | 0.0183 | 17 ep | A |
| 8000 | 0.0179 | 17 ep | A |
| 9000 | **0.0179** | 17 ep | A end (plateau) |
| 10000 | 0.0175 | 22 ep | B (vis_base 288 ep 重建) |
| 11000 | 0.0172 | 22 ep | B |
| **11999** | **0.0171** | 22 ep | B end |

⚠️ Phase A vs B val 集不同, 不可直接比 (17 ep 单日期 vs 22 ep 跨 3 日期)。Phase B 内部 step-step 同 val 可比。

**关键发现**: 在 Phase A 完全 plateau (step 8-9k) 后, 加 95 个新 vis_base ep + 极低 LR (3.66e-6 → 1.5e-6) 续训, 仍能压低 4.5%。续训突破 plateau 是真信号 (不是 LR 退火噪声)。

#### visrobot01_only_2k_gf0 (gf0, 2k sanity) ✅
step 1999 MAE@1=0.0202。同样数据 2k 比 9k (Phase A) 差 11%。

#### mix_vis600_v1 (gf0, 40k 长训) ✅ 已完成 (2026-04-27 03:48 CST, 33:21 hr)
540 train (310 vis + 145 base + 145 dagger) + 59 val (60 → 59 修复 corrupt val ep 35)。peak_lr=1.5e-5, ema=0.9999, save_interval=2k。

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 8000 | 0.0189 | 0.0385 | 0.0672 | 0.1033 |
| 12000 | 0.0173 | 0.0348 | 0.0601 | 0.0914 |
| 16000 | 0.0161 | 0.0331 | 0.0573 | 0.0871 |
| 20000 | 0.0154 | 0.0323 | 0.0561 | 0.0852 |
| 24000 | 0.0150 | 0.0320 | 0.0556 | 0.0842 |
| 28000 | 0.0148 | 0.0319 | 0.0554 | 0.0837 |
| 30000 | 0.0147 | 0.0319 | 0.0553 | 0.0835 |
| **36000** | **0.0146** | 0.0320 | 0.0554 | 0.0834 |
| **38000** | **0.0146** | 0.0320 | 0.0554 | 0.0834 |
| **39999** | **0.0146** | 0.0321 | 0.0555 | 0.0835 |

step 30000 起 plateau, step 36k/38k/39999 三连 tied @ 0.0146 (best, **推荐部署 step 38000**)。完整 17 个数据点见 `task_a_visrobot01_mixed_600.md`。

**部署 tar 包**: `/vePFS/.../deepdive_kai0_tmp/data/mix_vis600_best_step38000.tar` (11.6 GB, params + assets + _CHECKPOINT_METADATA, MAE@1=0.0146)。

#### pure_vis600_v1 (gf1, 40k 长训) ✅ 已完成 (2026-04-29 00:14 CST, 36:05 hr)
560 train (309 orig + 291 hflip mirror) + 40 val (paired by source ep 防 hflip leakage)。**0 kai0 source**, 全部 visrobot01 域 + 镜像增强。

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0268 | 0.0589 | 0.1074 | 0.1698 |
| 8000 | 0.0222 | 0.0422 | 0.0684 | 0.1017 |
| 14000 | 0.0190 | 0.0346 | 0.0542 | 0.0773 |
| 20000 | 0.0171 | 0.0308 | 0.0475 | 0.0667 |
| 26000 | 0.0160 | 0.0287 | 0.0438 | 0.0610 |
| 32000 | 0.0154 | 0.0275 | 0.0417 | 0.0579 |
| **38000** | **0.0151** | 0.0268 | 0.0406 | 0.0562 |
| **39999** | **0.0151** | 0.0268 | 0.0404 | 0.0558 ★ best |

step 38000/39999 tied @ 0.0151, **推荐部署 step 39999** (@25/@50 微优)。**部署 tar 包**: `/vePFS/.../deepdive_kai0_tmp/data/pure_vis600_best_step39999.tar` (11.6 GB)。

**codec 修复 (commit 18e3942)**: Mon 12:00 CST step 14000 时, 重编码 873 mirror mp4 (preset=ultrafast, bf=0, keyint=15) → random-seek 3.16ms → 0.85ms, 训练 8.4 → 2.5 s/step。

#### vis_base_40k_v1 (gf0, 40k 长训) ✅ 已完成 (2026-04-29 01:21 CST, 35:52 hr)
288 train + 22 val from vis_base 310 ep ONLY。**4-way ablation 中最纯净的单源 baseline** (无 kai0, 无 mirror)。Same hyperparams as pure_vis600 / mix_vis600, init from Task_A/mixed_1。

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0270 | 0.0568 | 0.1031 | 0.1625 |
| 8000 | 0.0224 | 0.0425 | 0.0699 | 0.1064 |
| 14000 | 0.0194 | 0.0379 | 0.0626 | 0.0939 |
| 22000 | 0.0176 | 0.0365 | 0.0608 | 0.0913 |
| 30000 | 0.0170 | 0.0364 | 0.0606 | 0.0907 |
| **36000** | **0.0168** | 0.0365 | 0.0606 | 0.0907 ★ best |
| 39999 | 0.0169 | 0.0366 | 0.0607 | 0.0907 |

step 36000 单点 best @ MAE@1=0.0168, **推荐部署**。**部署 tar 包**: `/vePFS/.../deepdive_kai0_tmp/data/vis_base_40k_best_step36000.tar` (11.6 GB)。

**40k vs 12k 单源对比**: vis_base_40k 40k = 0.0168 vs visrobot01_only_v1 12k = 0.0171 (-1.8%) → **单源数据 12k 步即近 plateau, 加步数收益极低**。

#### 4-way ablation final hierarchy (val 不同, 数值不严格可比, 但同 init+同 schedule 可比)
| step | mix_vis600 (vis+kai0) | pure_vis600 (vis+mirror) | vis_base_40k (vis only) | gap (vs mix) |
|---:|---:|---:|---:|---|
| 2000 | (failed) | 0.0268 | 0.0270 | — |
| 8000 | 0.0189 | 0.0222 | 0.0224 | pure +17%, vis +18% |
| 16000 | 0.0161 | 0.0182 | 0.0187 | pure +13%, vis +16% |
| 24000 | 0.0150 | 0.0163 | 0.0173 | pure +9%, vis +15% |
| 32000 | 0.0147 | 0.0154 | 0.0169 | pure +5%, vis +15% |
| **final (38k+)** | **0.0146** | **0.0151** | **0.0168** | **pure +3.4%, vis +15%** |

**hierarchy 终极结论**: 
- **kai0 跨域 > hflip mirror > 单源** (在每个 step 都成立)
- kai0 vs mirror gap **从 17% (early) 缩到 3.4% (final)** — long-horizon 训练显著缩小 gap
- mirror vs vis_only gap **从 1% (early) 扩到 10.1% (final)** — mirror 是 late-game amplifier

### 3.7 Task A — norm_stats 消融实验 (gf0+gf1，2026-04-29 ~ 04-30)

> **完整 per-step 曲线**: `norm_stats_ablation_apr28_450.md`
> 同 dataset (mix_apr28_450, 150 vis_apr28 + 150 kai0_base + 150 kai0_dagger), 同 hparams (30k, peak_lr=1.5e-5, ema=0.9999), 唯一差异 = norm_stats 来源。

#### A 组 new_norm (gf1) ✅ 已完成 (Thu 19:56 CST, 32:53 hr)
norm_stats 从当前 405 train 重算 (default 行为)。

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0182 | 0.0437 | 0.0827 | 0.1292 |
| 8000 | 0.0161 | 0.0344 | 0.0617 | 0.0957 |
| 14000 | 0.0143 | 0.0301 | 0.0536 | 0.0826 |
| 20000 | 0.0132 | 0.0290 | 0.0520 | 0.0801 |
| 24000 | 0.0129 | 0.0289 | 0.0518 | 0.0797 |
| 26000 | 0.0128 | 0.0289 | 0.0517 | 0.0796 |
| **28000** | **0.0127** | 0.0289 | 0.0518 | 0.0796 ★ best |
| 29999 | 0.0127 | 0.0290 | 0.0518 | 0.0796 (final, tied) |

step 26-29999 plateau @ 0.0127-0.0128。**部署 ckpt: step 28000**, deploy-ready folder + tar at `/vePFS/.../deepdive_kai0_tmp/data/mix_apr28_450_best_step28000{,.tar}` (含 README.md 描述本 ckpt + 顶层 norm_stats.json)。

#### B 组 inherit_norm (gf0) ✅ 已完成 (Thu 11:29 CST, 23:55 hr)
norm_stats 直接复制自 `Task_A/mixed_1/norm_stats.json` (init 模型 snapshot), 不重算。

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0216 | 0.0451 | 0.0816 | 0.1257 |
| 8000 | 0.0181 | 0.0357 | 0.0621 | 0.0954 |
| 14000 | 0.0158 | 0.0313 | 0.0542 | 0.0828 |
| 20000 | 0.0147 | 0.0302 | 0.0526 | 0.0803 |
| 26000 | 0.0141 | 0.0300 | 0.0524 | 0.0799 |
| **28000** | **0.0140** | 0.0300 | 0.0524 | 0.0799 ★ best |
| 29999 | 0.0140 | 0.0300 | 0.0525 | 0.0799 (final) |

step 26-29999 plateau @ 0.0140-0.0141。**部署 tar**: `/vePFS/.../deepdive_kai0_tmp/data/mix_apr28_450_inherit_norm_best.tar` (11.6 GB)。

#### head-to-head gap 总览 (同 val, 数值直接可比, ✅ final 数据)

| step | A new_norm | B inherit_norm | gap |
|---:|---:|---:|---:|
| 2000 | 0.0182 | 0.0216 | -16% (early max) |
| 8000 | 0.0161 | 0.0181 | -11% |
| 14000 | 0.0143 | 0.0158 | -9% |
| 20000 | 0.0132 | 0.0147 | -10% |
| 24000 | 0.0129 | 0.0142 | -9% |
| **28000** | **0.0127** | **0.0140** | **-9.3%** (final-plateau) |

#### Horizon-dependent gap (final step 28000)

| horizon | A new_norm | B inherit_norm | gap |
|---|---:|---:|---:|
| @1 (单步) | 0.0127 | 0.0140 | **-9.3%** |
| @10 | 0.0289 | 0.0300 | -3.7% |
| @25 | 0.0518 | 0.0524 | -1.1% |
| @50 | 0.0796 | 0.0799 | -0.4% |

**核心结论**: **冷启 fine-tune 应重算 norm_stats** — inherit init 模型的 norm 会一致地差 9-16% MAE@1, 即使 30k 长训也无法完全补偿 (final 仍 -9.3%)。gap 在 short-horizon (@1) 大, long-horizon (@50) 几乎消失 → norm 主要影响**单步 action 精度**, 对 chunk planning 影响小。

---

## 4. 关键工程经验（按通用性排序）

### 4.1 LR 调度（最强信号）

| 场景 | 推荐 peak_lr | 备注 |
|---|---|---|
| Action-only 续训（已收敛权重）| **1.25e-5** | E2 验证最优；默认 2.5e-5 会 overshoot |
| Action-only 从 pi05_base 训新数据 | 2.5e-5 | v3 / v4 / v5 / v8 共用 |
| 全解冻 + bs=128 + 短训（≤5k）| 1.5e-5 | 8k run 用 2.5e-5 过激进 |
| 全解冻 + bs=128 + 长训（≥10k）| 1.5e-5 ~ 2e-5 | 20k run 验证 |

### 4.2 EMA

- Action-only freeze 下 EMA 收益小，建议 `ema_decay=None`（省 6.5 GB/卡）
- 全解冻下 `ema_decay=0.999` 是金标准（半衰期 ~700 步）
- `ema_decay=0.9999`（半衰期 ~10k）在小数据 / 短训会稀释信号，stage 1 已弃用

### 4.3 Init 选择优先级

1. **kai0_mixed_1**（Task A warmed-up）→ 同硬件平台迁移**首选**，预期 +20–26%
2. **pi05_base**（GCS 官方）→ 跨平台 / 任务差异大时使用
3. 上一个最佳 ckpt（如 E2/14999）→ Phase 2 续训 / LoRA 起点

#### 4.3.1 norm_stats 是否重算 (✅ 严格验证完成, 见 `norm_stats_ablation_apr28_450.md`)
- **冷启 fine-tune (--overwrite OR weight_loader 而非 --resume): 必须重算 norm_stats**。继承 init 模型的 norm 会一致地差 **9-16% MAE@1 早期, 9.3% final**, 即使 30k 长训也无法完全补偿。
- **续训 (--resume 同 exp_name)**: 应保留旧 norm_stats (已写在 ckpt assets/ 里), 维持模型一致性, 避免输入分布跳变 (见 visrobot01_only Phase B 的设计)。
- 数据漂移越大 (新数据 vs init 训练数据), gap 越大。完全同分布数据 gap 接近 0。
- **horizon-dependent**: gap @1=9.3%, @10=3.7%, @25=1.1%, @50=0.4% — 主要影响**单步精度**, 不破坏 chunk planning。
- 实验对照 (mix_apr28_450, 30k step, head-to-head 完美控制):
  - new_norm best MAE@1 = **0.0127** (step 28000)
  - inherit_norm best MAE@1 = **0.0140** (step 28000)

### 4.4 Save 策略

- `save_interval=2000` + `keep_period=5000` 适合 action-only（15k 步 → 保留 5k/10k/15k 三个）
- `save_interval=1000` + `keep_period=1000` 适合全解冻短训（避免 keep_period=5000 丢失 best_step 拐点）

### 4.5 数据增强（小数据场景）

| 增强 | base 64ep | base_merged 128ep | full_aug 256ep |
|---|---|---|---|
| mirror（左右臂 swap + 视频翻转）| — | ✅ +10% 边际 | ✅ |
| time_scaling（2× 降采样）| — | — | ❌ **负贡献** |

**判据**：mirror 总是开；time_scaling 仅在 ep > 256 时再考虑。

### 4.6 5090 单卡 32 GB 的硬墙

| 配置 | 显存 | 能否运行 |
|---|---|---|
| Action-only freeze + bs=4 + inline-eval | ~21 GB | ✅ 稳 |
| LoRA r=16/32 vision MLP + AE + bs=4 | ~24 GB | ✅ |
| 全 vision 解冻 + FSDP=2 + bs=2 | > 32 GB | ❌ OOM |

**绕法**：远程 gf1 (8×A100 80GB) 用于全解冻实验。

### 4.7 sim01 NUMA 0/2 故障（已知绕过）

- GPU 1/2 在 inline-eval 后 SIGSEGV（NUMA 1/2 节点 0-byte DIMM）
- 用户态修复（numactl / cgroup / faulthandler）**全部失效** —— driver 显式 NUMA hint 不可绕
- 路径：仅用 GPU 0 + GPU 3；坏卡只能跑无 inline-eval 的 smoke
- 长期解：socket 1+2 各补 ≥8 GB DIMM

---

## 5. Checkpoint 路径索引（已物理验证存在）

### 5.1 Init 来源

| 用途 | 路径 |
|---|---|
| Task A 迁移 init | `kai0/checkpoints/Task_A/mixed_1/params/` |
| pi05_base GCS init | `openpi_cache/openpi-assets/checkpoints/pi05_base/params/` |

### 5.2 Task E action-only checkpoint（Phase 0/1/1-FT）

| exp | 路径 | best step (best @1) |
|---|---|---|
| v2 baseline | `kai0/checkpoints/pi05_stand_box_normal/stand_box_v2_pi05base/` | 16000 (0.0382) |
| v3 kai0+base | `kai0/checkpoints/pi05_stand_box_kai0init/v3_kai0_base/` | 12000 (0.0333) |
| v4 pi05+aug | `kai0/checkpoints/pi05_stand_box_aug/v4_pi05_aug/` | 10000 (0.0452) |
| v5 kai0+aug | `kai0/checkpoints/pi05_stand_box_kai0_aug/v5_kai0_aug/` | 14999 (0.0351) |
| v8 pi05+mirror | `kai0/checkpoints/pi05_stand_box_mirror/v8_pi05_mirror/` | 14000 (0.0421) |
| **E2 v3+lowLR** | `kai0/checkpoints/pi05_stand_box_kai0init_lowlr/v3e_lowlr/` | **14999 (0.0262)** |
| E1 v3+EMA | `kai0/checkpoints/pi05_stand_box_kai0init_ema/v3e_ema/` | 14999 (0.0297) |
| E3 v3+combo | `kai0/checkpoints/pi05_stand_box_kai0init_combo/` | 12000 (0.0277, NUMA 崩) |
| E4 v3+long | `kai0/checkpoints/pi05_stand_box_kai0init_long/v3e_long/` | 14999 (0.0327) |

### 5.3 Task E Phase 2 — LoRA & 长训 allgood 系列 checkpoint

| exp | 路径 | best step (best @1) |
|---|---|---|
| T1-1 LoRA r=16 | `kai0/checkpoints/pi05_stand_box_vision_lora16/t1_lora16/` | 14999 (0.0260) |
| T1-2 LoRA r=32 | `kai0/checkpoints/pi05_stand_box_vision_lora32/t1_lora32/` | 14000 (0.0261) |
| T7 LoRA r=16 v2 (fixed init + 25k) | `kai0/checkpoints/pi05_stand_box_vision_lora16_v2/t7_lora16_v2/` | 24000 (0.0257) |
| T6 allgood 15k | `kai0/checkpoints/pi05_stand_box_kai0_allgood/t6_kai0_allgood/` | 14999 (0.0245) |
| **T10 allgood 25k** 🥇 | `kai0/checkpoints/pi05_stand_box_kai0_allgood_25k/` | **24999 (0.0233)** |
| T14 allgood + LoRA 16 | `kai0/checkpoints/pi05_stand_box_kai0_allgood_lora16/` | 14999 (0.0245) |
| T15 allgood 40k | `kai0/checkpoints/pi05_stand_box_kai0_allgood_40k/` | 39999 (0.0234) |
| T16 allgood 25k + wd=0.01 | `kai0/checkpoints/pi05_stand_box_kai0_allgood_wd2/` | 24999 (0.0233) |
| T21 allgood bs=64 ❌ | `kai0/checkpoints/pi05_stand_box_kai0_allgood_bs64/` | 14000 (0.0418) |
| T22 allgood bs=128 ❌ | `kai0/checkpoints/pi05_stand_box_kai0_allgood_bs128/` | 6000 (0.0422) |

### 5.4 Task P checkpoint

| exp | 路径 | best step (best @1) |
|---|---|---|
| P-T10 frozen baseline | `kai0/checkpoints/pi05_pick_place_box_kai0init/p_v3_kai0init/` | 6000 (0.0703) |
| P allgood 25k action-only | `kai0/checkpoints/pi05_pick_place_box_kai0_allgood_25k/p_t10_allgood_25k/` | 14000 (0.0626) |
| unfreeze_2k smoke | `kai0/checkpoints/pi05_pick_place_box_kai0_unfreeze_2k/` | early (0.0362) |
| **unfreeze_8k** | `kai0/checkpoints/pi05_pick_place_box_kai0_unfreeze_8k/p_unfreeze_8k_v1/` | **3000 (0.0206)** |
| **unfreeze_20k** 🥇 | `kai0/checkpoints/pi05_pick_place_box_kai0_unfreeze_20k/` | **4000 (0.0195)** |

### 5.5 Task A 全参数微调 checkpoint (`task_a_visrobot01_mixed_600.md`)

| exp | 路径 | best step (best @1) |
|---|---|---|
| **mixed_gf0_173_v1** | `/vePFS/.../checkpoints/pi05_flatten_fold_mixed_gf0/mixed_gf0_173_v1/` | 9000-12999 tied @ 0.0129 (推荐 12999) |
| **visrobot01_only_v1** (Phase A end + Phase B end) | `/vePFS/.../checkpoints/pi05_flatten_fold_visrobot01_only/visrobot01_only_v1/` | A: 9000 (0.0179); B: **11999 (0.0171)** |
| visrobot01_only_2k_gf0_v1 | `/vePFS/.../checkpoints/pi05_flatten_fold_visrobot01_only_2k/visrobot01_only_2k_gf0_v1/` | 1999 (0.0202) |
| **mix_vis600_v1** ✅ | `/vePFS/.../checkpoints/pi05_flatten_fold_mix_vis600/mix_vis600_v1/` | **38000 (0.0146)** ★ 推荐部署; tar 包 `/vePFS/.../deepdive_kai0_tmp/data/mix_vis600_best_step38000.tar` 11.6 GB |
| **pure_vis600_v1** ✅ | `/vePFS/.../checkpoints/pi05_flatten_fold_pure_vis600/pure_vis600_v1/` | **39999 (0.0151)** ★ 推荐部署; tar 包 `/vePFS/.../deepdive_kai0_tmp/data/pure_vis600_best_step39999.tar` 11.6 GB |
| **vis_base_40k_v1** ✅ | `/vePFS/.../checkpoints/pi05_flatten_fold_vis_base_40k/vis_base_40k_v1/` | **36000 (0.0168)** ★ 推荐部署; tar 包 `/vePFS/.../deepdive_kai0_tmp/data/vis_base_40k_best_step36000.tar` 11.6 GB |
| **mix_apr28_450 (new_norm)** ✅🏆 | `/vePFS/.../checkpoints/pi05_flatten_fold_mix_apr28_450/mix_apr28_450_v1/` | **28000 (0.0127)** ★★★ Task A 最佳; deploy-ready folder + tar `/vePFS/.../deepdive_kai0_tmp/data/mix_apr28_450_best_step28000{,.tar}` 11.6 GB (含顶层 norm_stats.json + README.md, 按 mixed_1 init 部署格式) |
| **mix_apr28_450 (inherit_norm)** ✅ | `/vePFS/.../checkpoints/pi05_flatten_fold_mix_apr28_450_inherit_norm/mix_apr28_450_inherit_norm_v1/` | **28000 (0.0140)** 消融对照; tar 包 `/vePFS/.../deepdive_kai0_tmp/data/mix_apr28_450_inherit_norm_best.tar` 11.6 GB |

### 5.6 其他相关（探索性 / 已废弃）

| exp | 路径 | 状态 |
|---|---|---|
| pi05_stand_box_delta | `kai0/checkpoints/pi05_stand_box_delta/v5_delta_freeze/` | abs vs delta 对比，delta 输 |
| pi05_stand_box_kai0_allgood (bs64/bs128/wd2/lora16/25k/40k) | `kai0/checkpoints/pi05_stand_box_kai0_allgood*/` | 多变体探索 |
| pi05_stand_box_e2_ft | `kai0/checkpoints/pi05_stand_box_e2_ft/` | E2 续训探索 |
| pi05_stand_box_normal_v15 / v15_delta | `kai0/checkpoints/pi05_stand_box_normal_v15/`, `pi05_stand_box_v15_delta/` | v15 系列探索 |
| pi05_stand_box_vision_fsdp | `kai0/checkpoints/pi05_stand_box_vision_fsdp/` | 全解冻 FSDP 尝试，OOM |
| pi05_stand_box_kai0_unfreeze_2k | `kai0/checkpoints/pi05_stand_box_kai0_unfreeze_2k/` | Task E 全解冻 smoke |
| pi05_pick_place_box_kai0_allgood_25k | `kai0/checkpoints/pi05_pick_place_box_kai0_allgood_25k/` | Task P allgood 长训 |
| pi05_flatten_fold_awbc(_v2 / _v2_robust / from_official_mixed) | `kai0/checkpoints/pi05_flatten_fold_awbc*/` | Task A AWBC 变体 |
| pi05_flatten_fold_normal_v1 | `kai0/checkpoints/pi05_flatten_fold_normal_v1/` | flatten/fold 任务 |
| mixed_gf0_best_at_4k / mixed_gf0_step12999_final | `kai0/checkpoints/mixed_gf0*/` | gf0 上 Task A mixed 训练快照 |
| ma_E1_E2_equal | `kai0/checkpoints/ma_E1_E2_equal/` | E1+E2 等权 Model Arithmetic 合并 |
| visrobot01_only_best_step6000 | `kai0/checkpoints/visrobot01_only_best_step6000/` | visrobot 单平台 |
| kai0_mixed_1_grad | `kai0/checkpoints/kai0_mixed_1_grad/` | Task A mixed gradient 实验 |

---

## 6. 关键代码与配置位置

| 用途 | 文件 |
|---|---|
| 训练配置注册中心（30+ TrainConfig） | `kai0/src/openpi/training/config.py` |
| 训练主循环 + inline eval (`_run_inline_eval`, `_build_eval_policy`) | `kai0/scripts/train.py` |
| pi05 模型 + Action Expert 定义 | `kai0/src/openpi/models/pi0.py` |
| `get_freeze_filter()` 工具 | `kai0/src/openpi/models/pi0_config.py` |
| Agilex policy 输入/输出变换 | `kai0/src/openpi/policies/agilex_policy.py` |
| 训练启动入口 | `scripts/start_train.sh` / `train_scripts/launch/start_train.sh` |
| 独立评测脚本 | `scripts/eval_val_action_mse.py` |
| Ckpt 评测自动归档 | `scripts/auto_eval_v2.sh` |
| 空间镜像 | `kai0/train_deploy_alignment/space_mirroring.py` |
| 时间缩放 | `kai0/train_deploy_alignment/time_scaling.py` |
| 数据切分 / 修复（Task E） | `scripts/prepare_task_e_splits.py`, `scripts/generate_episodes_stats.py` |

---

## 7. 教训与决策原则（汇总自所有实验）

1. **Init > Aug**（小数据）：找一个同平台 warmed-up ckpt 比堆数据增强收益更大。
2. **Action-only vs 全解冻分水岭**：
   - 数据 ≤ 64 ep（~20k frames）→ action-only 是优解，避免过拟合。
   - 数据 ≥ 24k frames → 全解冻显著优于 action-only（Task P -69%），但需 80GB 卡。
3. **lowLR 是续训第一原则**：从已收敛 ckpt 接力时，peak_lr 砍半。
4. **MAE 不是真机指标**：offline MAE@1=0.0206 仍可在抓取瞬间偏 2-3cm。当 MAE < 0.025 后，应停止追 MAE，转 DAgger / RTC / Stage Advantage。
5. **Train loss 不能做 early stop**：必须用 val MAE@1，gap > 10× 即过拟合警示。
6. **rank 不是 LoRA 瓶颈**：r=16 已饱和小任务视觉 delta；升 rank 不如加 MHA 覆盖或换 DoRA。
7. **time_scaling 在小数据下负贡献**：mirror 是安全 aug，time_scaling 仅在 ep > 256 考虑。
8. **EMA 看场景**：冻结下 None；全解冻下 0.999；不要用 0.9999（会稀释信号）。

---

## 8. 后续方向（基于完整数据更新 2026-04-25）

> 当前最佳 = T10 allgood 25k = **0.0233**。所有 LoRA / DoRA / wd / rank / bs 调参均**已饱和**（≤ 0.0001 差异）。
> Phase 2 长训 + base_allgood 已榨干 action-only 极限。

### 已验证不再有显著收益的方向（实验结论）

| 方向 | 实证结果 | 结论 |
|---|---|---|
| LoRA rank 16 vs 32 | 差 0.0001（T1-1 vs T1-2）| **饱和** |
| weight decay 1e-4 vs 0.01 | 差 0.0001（T10 vs T16）| **饱和** |
| Batch size 缩放（无 LR scale）| 0.042 stuck（T21/T22）| 必须配 LR scaling |
| 25k → 40k | @1 持平 0.0234 / 0.0233（T15 vs T10）| @1 饱和；@50 仍降 -4% |
| LoRA + allgood | T14 = T6 = 0.0245 | LoRA 不是瓶颈 |
| time_scaling 增广 | v5 < v3，v4 < v8 | **负贡献** |

### 仍有空间的方向（按 EV/成本排序）

| 优先级 | 手段 | 理由 / 预期 |
|---|---|---|
| **P0** | **DAgger 补强 grasp phase** | Task P 真机失败模式是 grasp 瞬间偏 2-3cm；MAE@1=0.0195 已不再敏感于此。+30~50% 真机成功率 |
| **P0** | **RTC（real-time chunk）推理** | rtc3 / rtc5 推理时混合多 chunk，+15~30% 真机成功率 |
| **P1** | **Stage Advantage 加权 grasp loss** | kai0 module 3，给关键瞬间 loss 高权重；+20~40% 真机 |
| **P1** | Horizon-weighted loss（@1-@10 加权 1.5×）| 让训练直接优化 short-horizon 而不是 chunk 平均 loss；预期 -1~3% MAE@1 |
| **P2** | 远程 gf1 8×A100 全 vision 解冻 + 25k | Task P 已验证 -69%；Task E 上需测但训练成本 5h + 数据传输 1h |
| **P2** | DoRA / MHA LoRA 替换 vanilla LoRA | 上限可能突破 0.022~0.023，但 LoRA 已不在瓶颈上 |
| P3 | rsLoRA / LoRA dropout / curriculum / 多任务 co-train | 边际，仅在 P0/P1 全部跑过后再考虑 |

**核心判断**：**MAE@1 已不再是 Task E/P 的主要瓶颈**。继续压 MAE 收益递减；真机成功率取决于关键瞬间的精度（grasp / pre-grasp）和闭环能力（RTC + DAgger），转方向到这些手段。

---

_本文件由 history sweep 生成于 2026-04-25，覆盖 deepdive_kai0 已完成的所有 action-only 与对照实验。新实验完成后请追加到对应章节并更新第 1 节排行榜。_
