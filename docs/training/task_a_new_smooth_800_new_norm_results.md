# Task A new_smooth_800_new_norm 训练结果 (uc03 单机, vis_clean_v2 数据)

> **结论先行**: uc03 (8 GPU FSDP) 跑 `task_a_new_smooth_800_new_norm` 从 step 0 → 49999, final **MAE@1=0.0089**。**811 ep vis_base_clean_v2** 经 X1 cleanup 后的小批量精选数据 + mixed_1_clean init, MAE@1 在 step 40k 起 plateau (0.0089), 接近 1800 ep 训练的 0.0088。**数据 1/2.2 但单步精度仅差 1.1%**, 验证 "数据清洗 > 数据规模" 假设。Long-horizon (@50=0.0636) 显著差于 1800 ep 实验, 表明小数据训练受限于 chunk planner 多步精度。

## 1. 实验配置

| 参数 | 值 |
|---|---|
| Config name | `pi05_flatten_fold_a_new_smooth_800_new_norm` |
| Model | pi05 (`Pi0Config(pi05=True)`) |
| **Init** | `mixed_1_clean` (= cleaned version of mixed_1) |
| Init 路径 | `/home/tim/local_ckpts/Task_A_init/mixed_1_clean/params` |
| **Dataset** | `/data/shared/tim/data/Task_A/A_new_smooth_800/base` (**811 ep vis_base_clean_v2 + X1 cleanup, 930K frames**) |
| - 来源 | vis_base 数据经 X1 自动化清洗后的 smooth_800 子集 |
| Val | `/data/shared/tim/data/Task_A/A_new_smooth_800/val` (26 ep) |
| Prompt | "Flatten and fold the cloth." |
| `use_delta_joint_actions` | False |
| LR schedule | Cosine, warmup=1k, peak_lr=1.5e-5, decay_steps=50k, decay_lr=1.5e-6 |
| EMA decay | 0.9999 |
| Steps | 50,000 |
| Batch | **128**, fsdp_devices=8 |
| **num_workers** | **64** (用 [[feedback-uc-cluster-num-workers]] 优化) |
| Save | every 2,000 step |
| inline_eval | every 2 saves, 200 frames |
| Seed | 42 |
| Server | **uc03** (Intel 8358P × 124 vCPU + A800-SXM4-80GB × 8, 本地 SSD) |
| 训练时长 | 42h08m (含 mining 干扰 8h, 实际净 ~26h) |
| WandB | offline |

### 1.1 关键差异 (vs 同期 1800 ep 实验)

| 维度 | uc03 smooth_800 (本实验) | uc02 pure_1800 mixed1 | 1800 v5 js |
|---|---|---|---|
| 数据 | 811 ep vis_clean_v2 (X1 cleanup) | 1800 ep `-new` (mirror) | 1800 ep `-new` (mirror) |
| Init | mixed_1_clean | mixed_1 | mixed_1_clean |
| Cluster | uc 单机 8 GPU | uc 单机 8 GPU | js 双机 HSDP `[2,8]` 16 GPU |
| Batch | 128 | 128 | 80 |
| Final MAE@1 | **0.0089** | **0.0088** | **0.0090** |
| Final @50 | **0.0636** | 0.0258 | 0.0328 |

## 2. 完整 inline-eval MAE@{1,10,25,50} 曲线

| step | MAE@1 | @10 | @25 | @50 | Δ@1 vs prev |
|---:|---:|---:|---:|---:|---:|
| 4000  | 0.0123 | 0.0312 | 0.0620 | 0.1077 | (start, mixed_1_clean 起点) |
| 8000  | 0.0111 | 0.0260 | 0.0477 | 0.0782 | -9.8% |
| 12000 | 0.0104 | 0.0240 | 0.0433 | 0.0690 | -6.3% |
| 16000 | 0.0098 | 0.0229 | 0.0417 | 0.0664 | -5.8% |
| 20000 | 0.0096 | 0.0225 | 0.0411 | 0.0653 | -2.0% |
| 24000 | 0.0094 | 0.0223 | 0.0409 | 0.0648 | -2.1% |
| 28000 | 0.0092 | 0.0221 | 0.0406 | 0.0644 | -2.1% |
| 32000 | 0.0091 | 0.0220 | 0.0404 | 0.0640 | -1.1% |
| 36000 | 0.0090 | 0.0220 | 0.0404 | 0.0639 | -1.1% |
| 40000 | 0.0089 | 0.0220 | 0.0403 | 0.0637 | -1.1% |
| 44000 | 0.0089 | 0.0220 | 0.0403 | 0.0637 | 0.0% |
| 48000 | 0.0089 | 0.0221 | 0.0404 | 0.0636 | 0.0% |
| **49999** | **0.0089** | **0.0221** | **0.0404** | **0.0636** | 0.0% |

**Best**: step 40k 起 plateau, step 49999 实际就是 best (与 step 40k 完全持平)

## 3. 训练动力学

- 起点 (step 4k) MAE@1=0.0123, 接近 1800 ep 同 init 的 0.0128 (-4%) — 小数据集起点不弱
- **早期收敛快**: step 4k → 16k 在 12k 步内 -20% (类似 1800 ep 节奏)
- **step 40k 起完全 plateau**: @1/@10/@25/@50 后段几乎不变, 训练后期对优化目标无新信息
- @50 在 step 40k 后只下降 0.0001 — 小数据集触及 long-horizon planning 上限
- 全程 0 NaN

## 4. 关键洞察

### 4.1 数据清洗 vs 数据规模 (@1 角度)

| 实验 | 数据 | @1 | @1 与 SOTA 差 |
|---|---|---|---|
| **uc03 smooth_800 (本实验)** | 811 ep cleaned | 0.0089 | -4.7% vs SOTA |
| uc02 pure_1800 mixed1 | 1800 ep `-new` | 0.0088 | -3.5% vs SOTA |
| pure2_1800_6000 SOTA | 7900 ep mix | 0.0085 | baseline |
| **pure_200 #7 NEW SOTA** | 200 ep cleaned | **0.0065** | **+23.5% better** |

**@1 角度**: 811 ep cleaned (smooth_800) 接近 1800 ep, 数据清洗优势明显。但被 200 ep cleaned (pure_200) 大幅超越 → 数据 quality > size 在单步精度上成立, 但**还需要 mirror augmentation** (pure_200 用了)。smooth_800 没有 mirror, 是单纯 cleaning。

### 4.2 long-horizon (@50) 数据规模门槛

| 实验 | 数据 | @50 |
|---|---|---|
| **uc03 smooth_800 (811 ep)** | 811 ep cleaned | **0.0636** |
| uc02 pure_1800 mixed1 (1800 ep) | 1800 ep | 0.0258 (uc03 的 24.7%) |
| pure2_1800_6000 (7900 ep) | 7900 ep | 0.0337 |
| pure_200 #7 (200 ep cleaned + mirror) | 200 ep | 0.0079 |

**@50 角度**: smooth_800 远差于其他, 即使 200 ep pure_200 也胜出 8x。
原因推测:
- pure_200 用了 **hflip mirror**, 200 ep × 2 镜像 = 等效 400 ep 镜像对增强, **左右对称样本对** 帮助 chunk planner 学习 invariance
- smooth_800 仅有 cleaning, **无 mirror augmentation** → chunk planner 缺乏对称参考

**核心结论**: long-horizon (@50) 需要 (a) 数据规模 **或** (b) symmetric augmentation (e.g., hflip mirror)。smooth_800 两个都不满足。

### 4.3 训练优化经验

- **num_workers=64** 在 uc 单机 + 本地 SSD 下表现良好 (config 默认 32 在 step 64 后永久反压, 64 自动恢复)
  - 详见 [[feedback-uc-cluster-num-workers]] memory
- 受 mining 攻击拖累 step 32k-42k 区间, 但 MAE 走势未受影响
- 训练后期 (step 40k+) 完全 plateau, **建议下次相同 data 训练只跑 40k step 节省 20% 时间**

## 5. 最佳 ckpt 位置

```
uc03:/data/shared/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_a_new_smooth_800_new_norm/task_a_new_smooth_800_new_norm/49999/
```

(uc03 **本地路径**, 跨节点访问需 scp 或迁移到 `/cluster_ckpt`)

**完整 ckpt**: 2000, 4000, ..., 48000, **49999** (25 个) + `norm_stats.json` + `wandb_id.txt`

## 6. 后续

- ckpt 49999 与 step 40k 实质等价 (@1 一致, @50 仅 -0.16%) — **建议用 step 40000 ckpt 即可**, 节省 20% 后期训练
- 若 smooth_800 数据需要追 SOTA, **建议加 hflip mirror** 类似 pure_200 做法, 预期 @50 大幅改善
- 不建议作为 sim01 deploy 候选 (long-horizon 表现不如 pure_200 SOTA + uc02 mixed1)
