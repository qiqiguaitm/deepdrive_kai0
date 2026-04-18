# gf2 复现方案：Advantage Estimator + AWBC

## 背景

- gf0、gf1 正在进行阶段一（Normal fine-tune + Split 训练），尚未完成
- gf2（本机，8× GPU）基于 **kai0 官方发布的模型和数据**，独立复现阶段二：AWBC 和 Advantage Estimator
- Task_A 数据已完整（base 3,055 ep / advantage 3,055 ep / dagger 3,457 ep，视频全部可用）

## 现有资源

| 资源 | 路径 | 状态 |
|------|------|------|
| π₀.₅ base checkpoint（**两者共用起始点**） | `openpi_cache/.../pi05_base/params` (13GB, Orbax) | ✓ |
| 官方 mixed_1 checkpoint | `checkpoints/Task_A/mixed_1/` (22GB, Orbax) | ✓ |
| Task_A/advantage 数据 | `data/Task_A/advantage/` (3,055 ep, 含全部标注列) | ✓ |
| 官方预标注列（Estimator 验证基准） | `absolute_advantage`, `relative_advantage`, `absolute_value` | ✓ 在 parquet 中 |
| advantage tasks.jsonl | `0` → "fold the cloth, Advantage: negative" / `1` → "...positive" | ✓ 已离散化 |

**注意**：官方未发布 Advantage Estimator 和 AWBC 的结果 checkpoint，只有 Model Arithmetic 产物 (`mixed_1`)。

---

## 执行计划

### Step 1：AWBC 训练（JAX FSDP）

**Config**: `pi05_flatten_fold_awbc` — 从 π₀.₅ base 初始化，在官方预标注 advantage 数据上训练。

```bash
cd /home/tim/workspace/deepdive_kai0/kai0

# 前置：计算 norm stats
uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_awbc

# 训练（100K steps, 8 GPU FSDP）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_awbc \
  --exp_name=awbc_v1 --no-wandb-enabled --fsdp-devices 8 --batch-size 128
```

产出：`checkpoints/pi05_flatten_fold_awbc/awbc_v1/{5000,...,100000}/`

### Step 2：Advantage Estimator 训练（PyTorch DDP）

**Config**: `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD` — 从 π₀.₅ base 初始化，学习预测 stage progress advantage。

```bash
# 训练（100K steps, 8 GPU DDP, skip_norm_stats=True 无需预计算）
uv run torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py \
  ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD \
  --exp_name=adv_est_v1 --save_interval 10000 --no-wandb-enabled --batch-size 144
```

产出：`experiment/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1/{10000,...,100000}/model.safetensors`

### 执行顺序

两者都需 8 GPU，必须串行。先 AWBC（直接用官方数据出结果），后 Advantage Estimator。

---

## 验证方案

### 验证 1：Advantage Estimator — 自训预测 vs 官方预标注

**原理**：官方 `data/Task_A/advantage/` 中的 `absolute_advantage` / `relative_advantage` / `absolute_value` 列是官方 Estimator 标注的。用自训 Estimator 对同一数据重新推理，逐帧对比。

**步骤**：

```bash
# 1. 修改 eval.py 指向自训 checkpoint
#    stage_advantage/annotation/eval.py 中 MODELS_CONFIG_MAP:
#    'ckpt_dir': 'experiment/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1'
#    'ckpt_steps': 100000

# 2. 推理标注（产出 data_KAI0_100000/ 目录）
uv run python stage_advantage/annotation/eval.py Flatten-Fold KAI0 data/Task_A/advantage

# 3. 对比验证
python scripts/validate_advantage_estimator.py \
  --dataset data/Task_A/advantage \
  --pred-suffix KAI0_100000 \
  --output logs/adv_est_validation.json
```

**对比指标**：

| 指标 | 含义 | 通过标准 |
|------|------|----------|
| Pearson r (absolute_value) | 累积进度预测的线性相关 | > 0.85 |
| Spearman r (absolute_advantage) | advantage 排序一致性 | > 0.80 |
| MAE / RMSE | 预测绝对误差 | 参考值，越小越好 |
| 离散化一致率 (top 30%) | 二值化后正负标签匹配率 | > 85% |
| Per-episode Pearson 均值 | 每个 episode 内部一致性 | mean > 0.85 |

**脚本**：`scripts/validate_advantage_estimator.py`

---

### 验证 2：AWBC — 训练 loss 收敛

**原理**：检查 loss 曲线是否单调下降并收敛到稳定值。

```bash
# 从训练日志提取 loss 曲线
python scripts/validate_awbc.py loss --log logs/gf2_awbc_v1.log
```

**通过标准**：
- 后 10% 的平均 loss < 前 10% 的 95%（显著下降）
- 最终 loss 稳定（最后 1000 步标准差小）

---

### 验证 3：AWBC — 验证集 loss 对比

**原理**：用 `model_arithmetic/` 的 loss 计算功能，在相同验证数据上对比不同模型。

```bash
# dump 验证数据
python model_arithmetic/dump_data.py --dataset pi05_flatten_fold_normal --output val_base.pkl

# 自训 AWBC loss
python model_arithmetic/arithmetic.py \
  --config pi05_flatten_fold_awbc --data-path val_base.pkl \
  --checkpoints checkpoints/pi05_flatten_fold_awbc/awbc_v1/100000 \
  --output /tmp/eval_awbc --optimize_method average --use_gpu --gpu_ids "0"

# 官方 mixed_1 loss（参考，训练路径不同）
python model_arithmetic/arithmetic.py \
  --config pi05_flatten_fold_normal --data-path val_base.pkl \
  --checkpoints checkpoints/Task_A/mixed_1 \
  --output /tmp/eval_mixed --optimize_method average --use_gpu --gpu_ids "0"

# gf0 normal baseline loss（gf0 完成后）
python model_arithmetic/arithmetic.py \
  --config pi05_flatten_fold_normal --data-path val_base.pkl \
  --checkpoints checkpoints/pi05_flatten_fold_normal/normal_v1/100000 \
  --output /tmp/eval_normal --optimize_method average --use_gpu --gpu_ids "0"
```

每次运行会打印：`Mixed: X.XXXXXX`（验证 loss 值），三者直接比较。

**预期排序**：`AWBC ≤ mixed_1 < normal`（AWBC 在 advantage 加权下应有更好或接近的 loss）

---

### 验证 4：AWBC — 推理 action 合理性

**原理**：启动 policy server，送入真实观测，检查输出 action 的维度和数值。

```bash
# 终端 1：启动 server
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_flatten_fold_awbc \
  --policy.dir=checkpoints/pi05_flatten_fold_awbc/awbc_v1/100000 \
  --default-prompt "fold the cloth, Advantage: positive" \
  --port 8000

# 终端 2：推理测试
python scripts/validate_awbc.py infer \
  --host localhost --port 8000 \
  --dataset data/Task_A/advantage \
  --num-episodes 5
```

**通过标准**：
- action shape = (50, 14)：50 步 chunk × 14 维（双臂各 6 关节 + 1 gripper）
- 数值在 [-3, 3] 范围内
- 不同 episode 的 action std > 0.01（非退化常量）

---

### 验证 5：AWBC — Prompt 条件化差异

**原理**：AWBC 的核心是 prompt 条件化。同一观测分别用 "Advantage: positive" 和 "Advantage: negative" 推理，action 应有显著差异。

```bash
# server 同验证 4，终端 2 运行：
python scripts/validate_awbc.py prompt-diff \
  --host localhost --port 8000 \
  --dataset data/Task_A/advantage \
  --num-episodes 10
```

**通过标准**：
- 正/负 prompt action 的 L2 差异 mean > 0.01
- 若差异接近 0 → 模型未学到 advantage 条件化，训练有问题

---

## 验证汇总

| # | 验证对象 | 方法 | 对比基准 | 脚本 |
|---|----------|------|----------|------|
| 1 | Advantage Estimator | 重新标注 → 逐帧对比 | 官方预标注列 | `scripts/validate_advantage_estimator.py` |
| 2 | AWBC 收敛 | loss 曲线 | 自身趋势 | `scripts/validate_awbc.py loss` |
| 3 | AWBC loss | 验证集 loss | mixed_1 / normal baseline | `model_arithmetic/arithmetic.py` |
| 4 | AWBC 推理 | action shape + range | 期望值 (50,14) | `scripts/validate_awbc.py infer` |
| 5 | AWBC 条件化 | pos vs neg prompt diff | L2 > 0.01 | `scripts/validate_awbc.py prompt-diff` |

---

## 时间线

```
Day 1:
  ├─ Step 1: AWBC norm_stats + 训练 (100K steps)
  └─ 训练结束后立即运行验证 2 (loss 曲线)

Day 2:
  ├─ Step 2: Advantage Estimator 训练 (100K steps)
  ├─ 验证 3: AWBC 验证集 loss 对比
  └─ 验证 4+5: AWBC 推理 + prompt 条件化测试

Day 3:
  ├─ 验证 1: Estimator eval.py 推理 + 对比（3055 episodes，耗时较长）
  └─ 汇总所有验证结果
```
