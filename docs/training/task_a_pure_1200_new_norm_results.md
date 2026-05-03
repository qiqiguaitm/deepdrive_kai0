# Task A pure_1200 系列 (new_norm) 训练结果

> **范围**: 两个 Task A action-only 微调实验, 共用相似配方 (pi05 + mixed_1 init + 50k step + new norm_stats), 数据集差异:
>   - **`task_a_pure_1200_new_norm`** (gf0 #24, ✅ 完成): A_pure_1200 数据集 (1142 train + 58 val), 包含所有日期 (8 个 base + 4 个 -new)
>   - **`task_a_new_pure_1200_new_norm`** (gf1 #25, ⏳ 训练中, 67%): A_new_pure_1200 数据集 (1143 train + 57 val), 只用 6 个 -new 日期
>
> **目的**: 对比 -new 限定 vs 全日期 + mirror 增强对 single-source flatten/fold 的影响。两次都 init 自 Task_A/mixed_1 (MA-merged checkpoint), 用 cold-start 重算 norm_stats。

---

## 1. 实验配置 (两次相同)

| 参数 | 值 |
|---|---|
| Model | pi05 (Pi0Config(pi05=True)) |
| Init | `Task_A/mixed_1/params` (MA-merged + previous Task A finetune) |
| LR schedule | Cosine, warmup=1k, peak_lr=1.5e-5, decay_steps=50000, decay_lr=1.5e-6 |
| EMA | 0.9999 |
| Batch | 128, fsdp_devices=8 |
| Steps | 50,000 |
| Save | every 2,000 step (keep_period=2000) |
| inline_eval_every | 2 (i.e., 每 4,000 步 eval 一次, 12 evals total) |

## 2. 数据集对比

| 维度 | A_pure_1200 (#24) | A_new_pure_1200 (#25) |
|---|---|---|
| 来源 | `KAI0/Task_A/base/` 全 8 日期 | `KAI0/Task_A/base/` 仅 6 个 `-new` 日期 |
| 日期 | 04-16, 04-23, 04-23-new, 04-24, 04-24-new, 04-25, 04-25-new, 04-27, 04-28, 04-28-new, 04-29-new, 04-30-new | 04-23-new, 04-24-new, 04-25-new, 04-28-new, 04-29-new, 04-30-new |
| Train | 1,142 ep (620 orig + 580 hflip mirror) | 1,143 ep (613 orig + 530 mirror) |
| Val | 58 ep | 57 ep (paired orig+mirror, 防止 hflip leakage) |
| seed | 42 | 42 |

---

## 3. 完整 inline-eval MAE@{1,10,25,50} 曲线

### 3.1 task_a_pure_1200_new_norm (gf0 #24, 完成)

| step | MAE@1 | @10 | @25 | @50 | Δ@1 vs prev |
|---:|---:|---:|---:|---:|---:|
| 4000 | 0.0265 | 0.0510 | 0.0875 | 0.1386 | (start) |
| 8000 | 0.0233 | 0.0416 | 0.0663 | 0.0991 | -12.1% |
| 12000 | 0.0208 | 0.0366 | 0.0570 | 0.0828 | -10.7% |
| 16000 | 0.0188 | 0.0335 | 0.0519 | 0.0745 | -9.6% |
| 20000 | 0.0175 | 0.0314 | 0.0483 | 0.0688 | -6.9% |
| 24000 | 0.0165 | 0.0299 | 0.0457 | 0.0648 | -5.7% |
| 28000 | 0.0158 | 0.0286 | 0.0435 | 0.0615 | -4.2% |
| 32000 | 0.0154 | 0.0277 | 0.0420 | 0.0592 | -2.5% |
| 36000 | 0.0150 | 0.0269 | 0.0407 | 0.0573 | -2.6% |
| 40000 | 0.0148 | 0.0264 | 0.0399 | 0.0562 | -1.3% |
| 44000 | 0.0147 | 0.0261 | 0.0393 | 0.0552 | -0.7% |
| 48000 | **0.0145** | 0.0257 | 0.0387 | 0.0544 | -1.4% |
| 49999 | **0.0145** | 0.0255 | 0.0384 | 0.0539 | 0.0% |

**Best**: step 49999 (or 48000, 同分) — **MAE@1=0.0145**, @10=0.0255, @25=0.0384, @50=0.0539

### 3.2 task_a_new_pure_1200_new_norm (gf1 #25, 训练中, ~67%)

| step | MAE@1 | @10 | @25 | @50 | Δ@1 vs prev |
|---:|---:|---:|---:|---:|---:|
| 4000 | (失败) | — | — | — | XLA bug, 已修 |
| 8000 | (失败) | — | — | — | 同上 |
| 12000 | (失败) | — | — | — | 同上 |
| 16000 | 0.0118 | 0.0268 | 0.0462 | 0.0709 | (recovered) |
| 20000 | 0.0113 | 0.0254 | 0.0433 | 0.0658 | -4.2% |
| 24000 | 0.0109 | 0.0244 | 0.0413 | 0.0624 | -3.5% |
| 28000 | 0.0108 | 0.0237 | 0.0398 | 0.0600 | -0.9% |
| 32000 | **0.0105** | 0.0231 | 0.0386 | 0.0582 | -2.8% |
| 36000 | (待出, ETA 4h) | — | — | — | — |
| 40000 | (待出) | — | — | — | — |
| 44000 | (待出) | — | — | — | — |
| 48000 | (待出) | — | — | — | — |

**Best so far (step 32000)**: MAE@1=**0.0105**, @10=0.0231, @25=0.0386, @50=0.0582
**ETA 完成**: ~May 5 中午 (rate 6.8 s/it, 剩余 16k step ~30h)
**预测最终 MAE@1**: ~0.0098-0.0103 (按 step 32k 趋势 -2.8%/4k step 推断)

> ⚠️ **inline-eval CUDA bug**: 早期 step 4k/8k/12k 的 inline-eval 失败, 错误为 `StreamBeginCaptureToGraph is not implemented for CUDA below version 12.3`。修复方法是 launcher 加 `XLA_FLAGS=--xla_gpu_enable_command_buffer=`, 这部分在 v3 log 之后生效 (step 16000 起恢复)。

---

## 4. 头对头对比 (同 step)

| step | A_pure_1200 MAE@1 | A_new_pure_1200 MAE@1 | gap (-new lower) |
|---:|---:|---:|---:|
| 16000 | 0.0188 | **0.0118** | **-37.2%** |
| 20000 | 0.0175 | **0.0113** | **-35.4%** |
| 24000 | 0.0165 | **0.0109** | **-33.9%** |
| 28000 | 0.0158 | **0.0108** | **-31.6%** |
| 32000 | 0.0154 | **0.0105** | **-31.8%** |

**关键观察:**
- **`-new` 限定数据 (gf1 #25) 大幅领先 (-32% to -37%)**
- 两组同样 init / 同 hparams / 同步数, 唯一差别在数据 source: 全日期 vs 仅 -new
- `-new` 日期 = 最新且更高质量的采集; "non-new" 早期日期数据可能含更多采集噪声 / 不一致 prompt
- gap 随 step 缓慢收窄 (37% → 32%), 但远未闭合 — 数据质量主导

---

## 5. ckpt 路径

| 实验 | 路径 |
|---|---|
| gf0 #24 完成 | `/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_a_pure_1200/task_a_pure_1200_new_norm/49999/params/` |
| gf0 #24 best tar | `/vePFS/tim/workspace/deepdive_kai0_tmp/data/task_a_pure_1200_new_norm_best.tar` (auto-pack) |
| gf1 #25 best (current) | `/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_a_new_pure_1200/task_a_new_pure_1200_new_norm/32000/params/` |

---

## 6. 后续

- gf1 #25 训练中, ~30h 后完成。完成后:
  1. 拉取 final ckpt MAE
  2. 跑 offline eval 在更大 val 集 (50 queries) 验证
  3. 部署到 sim01 / 真机 测试 flatten/fold
- 若 final MAE@1 < 0.010, 这将是 Task A 当前 best

---

**最近更新**: 2026-05-03 23:00 CST (gf1 step 33,900, latest MAE@1=0.0105 @ step 32k)
