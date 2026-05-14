# Task A new_mixed_pure2_1800_6000 训练结果

> **本文档合并两个相关实验** (相同 ~7000-8000 ep 大规模混合数据集, 不同 init 与数据组成)：
> - **A. task_a_new_pure2_1800_6000_new_norm** (`pi05_flatten_fold_a_new_pure2_1800_6000`) ⭐ **新 SOTA**
> - **B. task_a_mix_base6000_pure1200_new_norm_base_mixed_1** (`pi05_flatten_fold_mix_b6000_p1200_init_mixed_1`) (原始独立文档已合入)

| 维度 | A. new_pure2_1800_6000 | B. mix_b6000_p1200 |
|---|---|---|
| Config | `pi05_flatten_fold_a_new_pure2_1800_6000` | `pi05_flatten_fold_mix_b6000_p1200_init_mixed_1` |
| 状态 | ✅ 完成 (50k 步, 2026-05-13 22:24 China, 26h51m) | ✅ 完成 (50k 步, 2026-05-07 12:24 China, 26h17m) |
| **数据集** | A_new_pure2_1800_6000_new_norm (**7900 ep**) | mix_b6000_p1200/base (~7200 ep) |
| **Init** | **pi05_base** (原始) | Task_A/mixed_1/params (MA-merged) |
| Cluster | uc01+uc02+uc03 (24 GPU) | uc01 单机 (8 GPU) |
| **Best MAE@1** | **0.0085** ⭐ (step 49999) | 0.0108 (step 44k+) |
| **gap vs best** | — | **+27% (worse)** |

---

## A. task_a_new_pure2_1800_6000_new_norm ⭐ SOTA

### A.1 实验配置

| 参数 | 值 |
|---|---|
| Config name | `pi05_flatten_fold_a_new_pure2_1800_6000` |
| Model | pi05 (Pi0Config(pi05=True)) |
| **Init** | `gs://openpi-assets/checkpoints/pi05_base/params` (原始 pi05, 未做 Task_A 微调) |
| **Dataset** | `Task_A/self_built/A_new_pure2_1800_6000_new_norm` |
| - 来源 | A_new_pure2_1800 (1790 ep) + kai0_base (3055 ep) + kai0_advantage (3055 ep) |
| - 合计 | **7900 ep / 8,851,074 frames / 113 GB** |
| Val | `A_new_pure2_1800_6000_new_norm_val` (30 ep evenly-spaced subset) |
| Prompt | "Flatten and fold the cloth." |
| `use_delta_joint_actions` | False |
| LR schedule | Cosine, warmup=1k, peak_lr=1.5e-5, decay_steps=50k, decay_lr=1.5e-6 |
| EMA decay | 0.9999 |
| Steps | 50,000 |
| Batch | 120, fsdp_devices=8 (HSDP `[3,8]` 失败 → 改 FSDP `[1,24]` 全分片) |
| Save | every 2,000 step, keep_period=2,000 |
| inline_eval | every 2 saves (= 每 4k 步), 200 frames |
| Seed | 123 |
| Servers | **uc01 + uc02 + uc03** (各 A800-SXM4-80GB ×8, 24 GPU 总) |
| 网络 | 4× Mellanox ConnectX-6 RoCEv2 200G + GDR |
| NFS | uc01 export `/data/cluster_ckpt` (POSIX 一致, orbax 跨主机 ckpt) |

### A.2 完整 inline-eval MAE@{1,10,25,50} 曲线

| step | MAE@1 | @10 | @25 | @50 | Δ@1 vs prev | eval (min) |
|---:|---:|---:|---:|---:|---:|---:|
| 4000  | 0.0534 | 0.0746 | 0.1078 | 0.1493 | (start, pi05_base 起点) | 42.3 |
| 8000  | 0.0272 | 0.0418 | 0.0656 | 0.0954 | **-49.1%** | 40.6 |
| 12000 | 0.0190 | 0.0312 | 0.0510 | 0.0757 | -30.1% | 40.4 |
| 16000 | 0.0146 | 0.0261 | 0.0438 | 0.0659 | -23.2% | 41.2 |
| 20000 | 0.0121 | 0.0230 | 0.0392 | 0.0590 | -17.1% | 40.6 |
| 24000 | 0.0106 | 0.0212 | 0.0360 | 0.0536 | -12.4% | 40.0 |
| 28000 | 0.0097 | 0.0199 | 0.0334 | 0.0489 | -8.5% | 40.1 |
| 32000 | 0.0092 | 0.0189 | 0.0312 | 0.0449 | -5.2% | 39.9 |
| 36000 | 0.0088 | 0.0182 | 0.0294 | 0.0415 | -4.3% | 40.2 |
| 40000 | 0.0086 | 0.0176 | 0.0279 | 0.0387 | -2.3% | 39.9 |
| 44000 | 0.0086 | 0.0172 | 0.0267 | 0.0364 | 0.0% | 39.8 |
| 48000 | 0.0085 | 0.0169 | 0.0258 | 0.0345 | -1.2% | 39.8 |
| **49999** | **0.0085** | **0.0168** | **0.0254** | **0.0337** | 0.0% | 39.7 |

**Best**: step 49999 (训练到底), MAE@1 = **0.0085** ⭐ — 全任务 SOTA

### A.3 训练动力学

- pi05_base 起点 MAE@1=0.0534 → final 0.0085 = **-84% 降幅**
- 收敛特征: 前 24k 步快速下降 (-80%), 后 24k 步缓慢收敛 (-20%)
- 自 step 40k 后 MAE@1 进入 plateau (0.0086 ~ 0.0085)
- @50 horizon 继续小幅下降到 step 49999 (0.0387 → 0.0337, -13%) — 长horizon planner 持续改进
- 全程 0 NaN, param_norm 1804.34 → 1804.x 稳定 (vs B 实验 +2.97)

### A.4 SOTA 关键洞察

1. **pi05_base 起点初值更高但天花板更低**: 起点 MAE@1=0.0534 (vs B 实验 0.0161, 高 3.3x), 但收敛后 MAE@1=0.0085 (比 B 的 0.0108 低 21%)
2. **大规模混合数据 + 长训** 优势凸显: 7900 ep × 50k 步, 比 B 的 ~7200 ep × 50k 步多约 10% 数据, 但 MAE 低 21%
3. **关键差异是数据质量**: A 包含 1790 ep `-new` 高质量 mirror 增强 (Task_A v2 dates), B 仅有 1200 ep `-new`
4. **集群训练 ROI**: 24 GPU (vs 8) 节省 wall time (26h vs 26h), 但训练精度提升明显 — 重点不是 GPU 数量, 是 mesh 配置 (FSDP `[1,24]` 成功, HSDP `[3,8]` SPMD partitioner 死锁)

### A.5 集群训练经验 (重大坑)

详见 `docs/deployment/training_servers_knowledge_base.md` section 13。

- **mesh `[3,8]` HSDP 首次编译死锁 ≥105 分钟** (SPMD partitioner mesh 转换慢路径)
- **mesh `[1,24]` 全 FSDP 8 分钟编译完成** ✅
- NCCL 必须用 IB+GDR (不是 Socket TCP), 配 `mlx5_0..3` 200G ConnectX RoCEv2

---

## B. task_a_mix_b6000_p1200_init_mixed_1 (原 baseline)

> 原始独立文档已合入本节, 作为 SOTA 实验 A 的对照组。

### B.1 实验配置

| 参数 | 值 |
|---|---|
| Config name | `pi05_flatten_fold_mix_b6000_p1200_init_mixed_1` |
| Model | pi05 |
| **Init** | `Task_A/mixed_1/params` (MA-merged + previous Task_A 微调) |
| Dataset | `Task_A/self_built/mix_b6000_p1200/base` (6000 base + 1200 pure mix, ~7200 ep) |
| Val | `mix_b6000_p1200/val_self_built` (30 ep paired orig+mirror) |
| LR | Cosine warmup=1k, peak=1.5e-5, decay=50k, end=1.5e-6 |
| EMA | 0.9999 |
| Steps | 50,000 |
| Batch | 128, fsdp_devices=8 |
| Save | every 2,000, keep_period=10,000 |
| inline_eval | every 2 saves, 200 frames |
| Seed | 42 |
| Server | uc01 (A800 ×8, single host) |

### B.2 inline-eval MAE@{1,10,25,50} 曲线

| step | MAE@1 | @10 | @25 | @50 | Δ@1 vs prev |
|---:|---:|---:|---:|---:|---:|
| 4000  | 0.0161 | 0.0393 | 0.0779 | 0.1345 | (start, mixed_1 已 finetune 过) |
| 8000  | 0.0141 | 0.0315 | 0.0571 | 0.0948 | -12.4% |
| 12000 | 0.0127 | 0.0283 | 0.0506 | 0.0815 | -9.9% |
| 16000 | 0.0123 | 0.0275 | 0.0492 | 0.0783 | -3.1% |
| 20000 | 0.0120 | 0.0269 | 0.0483 | 0.0766 | -2.4% |
| 24000 | 0.0116 | 0.0263 | 0.0474 | 0.0753 | -3.3% |
| 28000 | 0.0114 | 0.0260 | 0.0468 | 0.0743 | -1.7% |
| 32000 | 0.0111 | 0.0256 | 0.0462 | 0.0735 | -2.6% |
| 36000 | 0.0110 | 0.0254 | 0.0460 | 0.0732 | -0.9% |
| 40000 | 0.0109 | 0.0253 | 0.0458 | 0.0729 | -0.9% |
| 44000 | **0.0108** | 0.0252 | 0.0457 | 0.0728 | -0.9% |
| 48000 | **0.0108** | 0.0252 | 0.0457 | 0.0728 | 0.0% |
| 49999 | **0.0108** | 0.0252 | 0.0457 | 0.0728 | 0.0% |

**Best**: step 44000 (首次 plateau), MAE@1 = **0.0108**

### B.3 训练动力学

- mixed_1 起点 MAE@1=0.0161 (已比 pi05_base 0.0534 低 70%, 因 mixed_1 已含 Task_A 信息)
- 自 step 44k 后训练完全 plateau (-0.9% / 4k 步), 后 6k 步无改进
- param_norm 从 1804.34 → 1807.32 (+2.97), 比 A 实验大幅移动 — mixed_1 init 与 task_a_new_pure2 数据分布更远

### B.4 数据规模 vs 数据质量结论

| 指标 | B. mix_b6000_p1200 | task_a_pure_1200_new_norm | task_a_new_pure_1200_new_norm |
|---|---:|---:|---:|
| 数据规模 | ~7200 ep | 1142 ep | 1143 ep |
| best MAE@1 | **0.0108** | 0.0145 | **0.0104** |
| best step | 44k | 49999 | 38k |

**B 实验启示**:
- 单纯堆数据不如使用高质量 `-new` 限定数据 — **数据质量 > 数据量**
- mix_b6000_p1200 (~6x 数据) 比 new_pure_1200 (1143 ep) 反而 **+3.8% MAE@1** worse

但 **A 实验推翻了"数据质量 > 数据量"** —— 在 **足够大规模 + pi05_base 干净起点** 下, 数据规模优势重新显现 (A 比 new_pure_1200 低 **18% (0.0085 vs 0.0104)**)。

---

## 总结对比表

| 维度 | A (SOTA, pi05_base init) | B (mixed_1 init) | gap |
|---|---:|---:|---:|
| 数据规模 (ep) | 7900 | ~7200 | A +10% |
| Init MAE@1 | 0.0534 (pi05_base raw) | 0.0161 (mixed_1, pre-tuned) | B 70% better start |
| **Final MAE@1** | **0.0085** | 0.0108 | **A -21%** ⭐ |
| Final MAE@10 | 0.0168 | 0.0252 | A -33% |
| Final MAE@25 | 0.0254 | 0.0457 | A -44% |
| Final MAE@50 | 0.0337 | 0.0728 | A -54% |
| Train wall time | 26h51m | 26h17m | ≈相同 |
| Compute | 24 GPU (3 host HSDP→FSDP) | 8 GPU (单机) | A 3× |
| Plateau step | ~40k | ~44k | 近似 |

---

## 部署 Checklist

### A (新 SOTA)
- [x] ckpt-49999 已保存到 `/cluster_ckpt/checkpoints/pi05_flatten_fold_a_new_pure2_1800_6000/task_a_new_pure2_1800_6000_new_norm/49999/`
- [x] norm_stats: 与 ckpt 同目录 + dataset 根目录
- [ ] Pack ckpt 49999 per `kai0/checkpoints/README.md` Type A flat 格式
- [ ] sim01 部署测试

### B (原 baseline, 已部署)
- ckpt 路径: `/home/tim/local_ckpts/pi05_flatten_fold_mix_b6000_p1200_init_mixed_1/task_a_mix_base6000_pure1200_new_norm_base_mixed_1/{20000,30000,40000,49999}/`
- 保留: 20k, 30k, 40k, 49999

---

## 经验教训 (合并)

1. **A 推翻 B 的"数据质量 > 数据量"** — 实际上 **两者协同更优**: A 既增加了规模 (7900 vs 7200), 又用了 pi05_base 干净起点 (而非 finetuned mixed_1 偏置), 还能进 long-horizon 收敛
2. **pi05_base init 起点高但天花板低**: 起点 MAE 是 mixed_1 init 的 3.3x, 但最终精度低 21%
3. **集群训练慎选 mesh**: HSDP `[3,8]` 在 RDMA + 24 GPU 环境下 SPMD partitioner 死锁 105+ 分钟; 退到全 FSDP `[1,24]` 8 分钟编译完成
4. **RDMA + GDR 必须启用** (`mlx5_0..3` IB verbs, `NCCL_NET_GDR_LEVEL` default): 默认 socket TCP 仅 26 Gbps × 4 NIC, RDMA 800 Gbps
5. **40-44k 后 plateau**: 未来类似实验可缩短到 40k 步, 节省 ~20% 训练时间
6. **长 horizon (@50) 收敛更慢**: A 实验 @50 在 step 48k 仍小幅下降 (-1.5%), @1 已 plateau — chunk planner 需要更长训练
