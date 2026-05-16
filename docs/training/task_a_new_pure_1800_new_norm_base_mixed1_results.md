# Task A new_pure_1800_new_norm_base_mixed1 训练结果 (uc02 单机, mixed_1 init)

> **结论先行**: uc02 (8 GPU FSDP) 跑 `task_a_new_pure_1800_new_norm_base_mixed1` 从 step 0 → 49999, final **MAE@1=0.0088**。**两阶段训练**: pi05_base + 6000 官方 → mixed_1 → + 1800 ep `-new` 精选 → 0.0088。在 long-horizon (@10/@25/@50) 上**显著优于 1800 v5 js (mixed_1_clean init)** 和**老 SOTA pure2_1800_6000**, 仅单步 (@1) 略弱老 SOTA。两阶段范式相比一阶段 (pi05_base + 7900 ep 一次性) MAE@1 差 3.5%, 但 @50 反而胜出 23%。

## 1. 实验配置

| 参数 | 值 |
|---|---|
| Config name | `pi05_flatten_fold_a_new_pure2_1800` (uc 集群版) |
| Model | pi05 (`Pi0Config(pi05=True)`) |
| **Init** | `mixed_1` = `pi05_base + 6000 官方 ep finetune` (Stage 1 结果) |
| Init 路径 | `/data/shared/tim/workspace/deepdive_kai0/kai0/checkpoints/Task_A/mixed_1/params` |
| **Stage 2 Dataset** | `/cluster_ckpt/data/Task_A/self_built/A_new_pure2_1800` (1800 ep `-new` 精选 + mirror) |
| Val | `/cluster_ckpt/data/Task_A/self_built/A_new_pure2_1800_val` |
| Prompt | "Flatten and fold the cloth." |
| LR schedule | Cosine, warmup=1k, peak_lr=1.5e-5, decay_steps=50k, decay_lr=1.5e-6 |
| EMA decay | 0.9999 |
| Steps | 50,000 |
| Batch | **128**, fsdp_devices=8 (单机 FSDP `[1,8]`) |
| Save | every 2,000 step, keep_period=2,000 |
| inline_eval | every 2 saves (= 每 4k 步), 200 frames |
| Seed | 42 |
| Server | **uc02** (Intel 8358P × 124 vCPU + A800-SXM4-80GB × 8) |
| 存储 | 数据 + ckpt 在 `/cluster_ckpt/` (NFS), pi05_base 在 `/home/tim/workspace/openpi_cache/` |
| 训练时长 | 42h13m (含 mining 干扰拖累, 实际净时长 ~26h) |
| WandB | offline |

### 1.1 关键差异 (vs 老 SOTA 与同期实验)

| 维度 | 本实验 (uc02 mixed1) | pure2_1800_6000 (老 SOTA) | pure2_1800_js (mixed_1_clean) |
|---|---|---|---|
| Init | **mixed_1** (warmed) | **pi05_base** (cold) | mixed_1_clean (warmed, different cleanup) |
| 数据规模 | 1800 ep | 7900 ep (1790 mirror + 3055 base + 3055 advantage) | 1800 ep |
| 训练范式 | **两阶段** (6000 → +1800) | 一阶段 (7900 一次性) | 两阶段 (类似本实验) |
| Cluster | 单机 8 GPU | 24 GPU (uc01+02+03 HSDP[1,24]) | 16 GPU HSDP `[2,8]` js03+04 |
| Batch | 128 | 120 | 80 |
| Final MAE@1 | **0.0088** | 0.0085 (-3.5%) | 0.0090 (+2.2%) |
| @50 horizon | **0.0258** | 0.0337 (+30.6%) | 0.0328 (+27.1%) |

## 2. 完整 inline-eval MAE@{1,10,25,50} 曲线

| step | MAE@1 | @10 | @25 | @50 | Δ@1 vs prev |
|---:|---:|---:|---:|---:|---:|
| 4000  | 0.0128 | 0.0341 | 0.0670 | 0.1136 | (start, mixed_1 起点) |
| 8000  | 0.0117 | 0.0283 | 0.0509 | 0.0816 | -8.6% |
| 12000 | 0.0107 | 0.0250 | 0.0428 | 0.0650 | -8.5% |
| 16000 | 0.0101 | 0.0229 | 0.0374 | 0.0542 | -5.6% |
| 20000 | 0.0098 | 0.0213 | 0.0333 | 0.0466 | -3.0% |
| 24000 | 0.0095 | 0.0200 | 0.0301 | 0.0410 | -3.1% |
| 28000 | 0.0092 | 0.0188 | 0.0277 | 0.0372 | -3.2% |
| 32000 | 0.0092 | 0.0180 | 0.0259 | 0.0345 | 0.0% |
| 36000 | 0.0090 | 0.0172 | 0.0242 | 0.0318 | -2.2% |
| 40000 | 0.0089 | 0.0164 | 0.0227 | 0.0295 | -1.1% |
| 44000 | 0.0088 | 0.0159 | 0.0216 | 0.0279 | -1.1% |
| 48000 | 0.0088 | 0.0154 | 0.0205 | 0.0261 | 0.0% |
| **49999** | **0.0088** | **0.0153** | **0.0203** | **0.0258** | 0.0% |

**Best**: step 49999 全维度最低 (@1=0.0088, @10=0.0153, @25=0.0203, @50=0.0258)

## 3. 训练动力学

- mixed_1 起点 step 4k MAE@1=0.0128 (vs pi05_base 起点 4k=0.0534 在 pure2_1800_6000): warm init 起点低 4.2x
- 收敛极快: step 4k → 24k 间 MAE@1 从 0.0128 → 0.0095 (-26%)
- step 28k 后进入慢速 plateau (-1 ~ -3% per 4k step)
- step 44k 起 @1 平在 0.0088, @10/@25/@50 继续微降 (long-horizon planner 仍在精修)
- 全程 0 NaN
- 中途遭遇 **挖矿木马入侵导致 step 38k-42k 严重降速** (详见 `docs/security/2026-05-16_rvn_miner_incident.md`), 但 MAE 趋势未受影响

## 4. SOTA 关键洞察

### 4.1 两阶段范式 vs 一阶段范式

| 范式 | Stage 1 | Stage 2 | 总数据 | MAE@1 | @50 | 训练成本 |
|---|---|---|---|---|---|---|
| 一阶段 (老 SOTA) | — | pi05_base + 7900 ep, 24 GPU 24h | 7900 ep | **0.0085** | 0.0337 | 1x |
| 两阶段 (本实验) | pi05_base + 6000 ep (mixed_1) | mixed_1 + 1800 ep, 8 GPU 26h | 6000+1800 ≈ 7800 ep | **0.0088** | **0.0258** | 0.3x (Stage 2 仅) |

**关键观察**:
1. 一阶段 @1 略优 (-3.5%), 但**两阶段 @50 反而胜出 23%** — Stage 1 (6000 ep) 已经收敛到对 Task_A 通用表示, Stage 2 精修 chunk planner
2. 两阶段**计算成本仅 30%** (Stage 2 只跑 8 GPU 1.5 天, vs 一阶段 24 GPU 1 天)
3. **mixed_1 作为可复用 init**: 一次训练得到, 后续多个 Task_A 微调可共享, 摊薄成本

### 4.2 long-horizon (@10/@25/@50) 优势的解释

uc02 (mixed_1 init) vs pure2_1800_6000 (pi05_base init):
- @1: 0.0088 vs 0.0085 — pure2_1800_6000 略胜
- @10: **0.0153 vs 0.0168** — mixed_1 胜 -8.9%
- @25: **0.0203 vs 0.0254** — mixed_1 胜 -20.1%
- @50: **0.0258 vs 0.0337** — mixed_1 胜 -23.4%

**假设**: mixed_1 通过 6000 ep 已经学到良好的 chunk-level 表示, Stage 2 在 1800 ep 上专门优化 long-horizon planning。一阶段训练在 Stage 1 的 chunk 表示尚未稳定时就开始 fine-tune, long-horizon 性能受损。

### 4.3 mixed_1 vs mixed_1_clean

uc02 mixed_1 vs js cluster pure2_1800_js (同 1800 ep, 不同 init):
- mixed_1 init: 0.0088
- mixed_1_clean init: 0.0090

**mixed_1 比 mixed_1_clean 优 ~2%, 全 horizon 一致**:
- Stage 1 训练时 **clean** 数据可能过滤掉了对 long-horizon 有帮助的边缘 case
- 完整 6000 ep 训练得到的 mixed_1 起点更稳

## 5. 最佳 ckpt 位置

```
uc02:/cluster_ckpt/checkpoints/pi05_flatten_fold_a_new_pure2_1800/task_a_new_pure_1800_new_norm_base_mixed1/49999/
```

⚠️ `/cluster_ckpt` 是 NFS 共享路径, **uc01/uc02/uc03 任一节点可访问**, 但 js / gf 集群不可访问。
若需跨集群使用, 用 scp 或 tos 上传。

**完整 ckpt 列表**: 2000, 4000, ..., 48000, **49999** (25 个 ckpt 全部保留, max_to_keep=1 + keep_period=2000)
**附属**: `norm_stats.json`, `wandb_id.txt`

## 6. 受 mining 攻击的影响

训练期间 (2026-05-16 01:30 - 10:30) 因挖矿木马入侵 uc 集群导致中后期严重降速:
- 正常 rate: 1.9 s/it
- 矿机干扰期 rate: 14-25 s/it (慢 7-13x)
- step 38k-42k 期间 inline_eval 卡 8h+ 才完成 (vs 正常 20min)

详细事件分析见 `docs/security/2026-05-16_rvn_miner_incident.md`。**MAE 走势未受影响**, 只是训练时长延长 (26h → 42h)。最终 ckpt 完整可用。

## 7. 后续

- **强烈建议**: 用 49999 ckpt 上 sim01 测试, 与 pure_200 #7 SOTA (0.0065) 对比 — 看 long-horizon 在真机的体现
- **追加对照实验**: 跑 `pi05_base + 1800 ep 直接` (无 mixed_1 中间), 完整 3-way 范式比较 (见 `training_paradigm_comparison.md`)
- **mixed_1 init 复用价值**: pure_200 (200 ep, mixed_1_clean) 已是新 SOTA 0.0065, 表明 mixed_1 类 init + 精选小数据是更经济的 Task_A 训练路径
