# kai0 复现训练方案

本文档定义两个 kai0 复现方案：`kai0_mixed_1`（基础版）和 `kai0_full`（完整版）。

## 背景

本项目旨在复现 OpenDriveLab 的 kai0 (χ₀) 论文结果。官方发布了三个任务的最终合并模型 (`Task_A/mixed_1`, `Task_B/mixed_9`, `Task_C/mixed_5`)，但**未发布**：
- 各个 split 的源 checkpoint
- Advantage Estimator 模型
- AWBC 训练的中间产物
- 完整训练 recipe 的超参数细节

通过对 `Task_A/mixed_1` 的分析（norm_stats 反推、prompt 响应测试），我们确定：
- 官方 `mixed_1` 的 norm_stats **数学上等于** base + dagger 合并数据的统计量
- 对 `"Advantage: positive/negative"` prompt **无响应**（adv+/- 输出 MSE < 0.00003）
- 左右臂对称性与 base 几乎一致 → **未做 space mirroring**

**结论**：`mixed_1` 实际上是 **MA + TDA(data) 的产物**，不包含 SA，也不含数据增强。

---

## 方案对比

| 维度 | `kai0_mixed_1` (基础版) | `kai0_full` (完整版) |
|------|----------------------|---------------------|
| **对应论文模块** | MA + TDA(data) | MA + SA + TDA(data + augment) |
| **等效官方模型** | `Task_A/mixed_1` (复现) | 论文 Figure 6 的 full chi0 |
| **训练数据** | base + dagger (6512 ep) | base + dagger + 镜像 + 时间缩放 (~24000 ep) |
| **Advantage 标签** | 否 | 是 (需训 advantage estimator) |
| **AWBC** | 否 | 是 (MA 基础上 fine-tune) |
| **数据增强** | 否 | space_mirroring + time_scaling |
| **新训 checkpoint 数** | 5 (4 splits + 1 merge) | 11 (1 estimator + 1 augmented base + 4 splits + 1 merge + 4 AWBC splits + merge) |
| **总训练时间** | ~5 天 | ~13 天 |
| **GPU 资源** | 8 × A100 80GB | 8 × A100 80GB |
| **风险等级** | 低 | 中 |
| **验证难度** | 低（可对比官方 mixed_1）| 高（需真机 success rate）|

---

## 评测局限性声明（重要）

**在纯离线评测下，无法可靠判断 SA 的贡献**：

1. **离线 MSE 与 success rate 脱钩**：我们之前的测试显示 `mixed_1` vs `our_mixed` 的 MSE 差异 < 10%，但论文声称 `chi0` 相对 baseline 有 250% 成功率提升。MSE 不能反映 long-horizon 任务的完成能力。

2. **论文的 monotonic increase 是 success rate**，不是 MSE。我们无法用 offline MSE 验证 "SA 加上去会不会提升"。

3. **可靠的 SA 贡献判断需要**：
   - 完整方案的 A/B 对比（`kai0_mixed_1` vs `kai0_full`）
   - 真机部署或高质量仿真
   - 衡量 task success rate 而非 action MSE

**因此两个方案的目标不同**：
- `kai0_mixed_1`: **可验证** —— 复现官方 `mixed_1`，用 offline MSE 对比检查 pipeline 正确性
- `kai0_full`: **研究性** —— 构建完整 chi0 候选，但离线指标可能不显示优势，最终价值依赖真机测试

---

## 方案 1: `kai0_mixed_1` (基础版)

### 目标

复现 `Task_A/mixed_1` 的训练配方：在 base + dagger 合并数据上做 4-split Model Arithmetic。

### 输入

| 资源 | 来源 | 大小 |
|------|------|-----|
| base 数据集 | `data/Task_A/base` | 3055 ep, 3.36M frames |
| dagger 数据集 | `data/Task_A/dagger` | 3457 ep, 2.42M frames |
| pi05_base 权重 | `openpi_cache/openpi-assets/checkpoints/pi05_base/params` | ~8GB |

### 流程

```
Step 1: Merge datasets
  base (3055 ep) + dagger (3457 ep) → combined (6512 ep)
  
Step 2: Compute norm_stats
  combined → norm_stats.json (quantile: q01, q99, mean, std)
  
Step 3: Split into 4 subsets
  6512 ep → split_0 (1628) + split_1 (1628) + split_2 (1628) + split_3 (1628)
  random shuffle with seed=42
  
Step 4: Train 4 MA members
  split_i (i=0..3):
    pi05_base → 25,000 steps → pi05_kai0_mixed_1_split_i/ckpt
  batch_size=256, fsdp_devices=8, log_interval=100
  
Step 5: Dump validation data
  combined 数据集抽 50 batches → kai0_mixed_1_val.pkl
  
Step 6: Model Arithmetic merge
  4 × split ckpts + val pkl → inverse_loss weighted merge → kai0_mixed_1/0/
```

### 详细步骤

#### Step 1: 合并数据集

```bash
cd kai0/
python train_deploy_alignment/data_augment/merge_lerobot.py \
  --src_paths data/Task_A/base data/Task_A/dagger \
  --tgt_path data/Task_A/kai0_mixed_1_data \
  --repo_id kai0_mixed_1_data
```

预期输出：
```
data/Task_A/kai0_mixed_1_data/
├── data/chunk-000/ ... chunk-006/  (6512 episodes)
├── videos/chunk-000/ ... chunk-006/
└── meta/
    ├── info.json (total_episodes=6512, total_frames=5777710)
    ├── episodes.jsonl
    └── tasks.jsonl
```

#### Step 2: 计算 norm_stats

```bash
# 先在 config.py 添加 kai0_mixed_1_normal config (repo_id 指向合并数据)
uv run python scripts/compute_norm_states_fast.py --config-name kai0_mixed_1_normal
```

验证 norm_stats 与官方 `mixed_1/norm_stats.json` 在 <1% 误差内（因为都是 base+dagger）。

#### Step 3: 切分 episodes

```python
# scripts/generate_kai0_splits.py
import json, random
random.seed(42)
episodes = list(range(6512))
random.shuffle(episodes)
splits = [episodes[i::4] for i in range(4)]
for i, s in enumerate(splits):
    json.dump(sorted(s), open(f"data/Task_A/kai0_mixed_1_split_{i}.json", "w"))
```

#### Step 4: 在 `config.py` 添加 4 个 split configs

```python
# kai0/src/openpi/training/config.py
TrainConfig(
    name="kai0_mixed_1_split_0",
    model=pi0_config.Pi0Config(pi05=True),
    data=LerobotAgilexDataConfig(
        repo_id="data/Task_A/kai0_mixed_1_data",
        default_prompt="Flatten and fold the cloth.",
        use_delta_joint_actions=False,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=25_000,
    keep_period=5000,
    num_workers=8,
    batch_size=256,
),
# ...重复 split_1, split_2, split_3
```

#### Step 5: 启动训练

```bash
# run_kai0_mixed_1.sh
export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=offline

cd kai0/
for i in 0 1 2 3; do
  EPISODES=$(python scripts/get_kai0_episodes.py $i)
  uv run scripts/train.py kai0_mixed_1_split_$i \
    --exp_name=split_${i}_v1 \
    --fsdp-devices 8 \
    --batch-size 256 \
    --data.episodes $EPISODES \
    || echo "[train] split_$i FAILED"
done
```

#### Step 6: Model Arithmetic 合并

```bash
# Dump validation data (从合并数据集)
uv run python model_arithmetic/dump_data.py \
  --dataset kai0_mixed_1_split_0 \
  --output kai0_mixed_1_val.pkl \
  --batch-size 16

# 合并 (优先 greedy, 退化到 inverse_loss)
uv run python model_arithmetic/arithmetic.py \
  --config kai0_mixed_1_split_0 \
  --data-path kai0_mixed_1_val.pkl \
  --checkpoints \
    checkpoints/kai0_mixed_1_split_0/split_0_v1/24999 \
    checkpoints/kai0_mixed_1_split_1/split_1_v1/24999 \
    checkpoints/kai0_mixed_1_split_2/split_2_v1/24999 \
    checkpoints/kai0_mixed_1_split_3/split_3_v1/24999 \
  --output $(pwd)/checkpoints/kai0_mixed_1 \
  --optimize_method greedy \
  --gpu_ids "0"
```

### 时间估算

| 步骤 | 耗时 | 备注 |
|------|------|-----|
| Step 1 合并数据 | ~2h | I/O 密集，视频 symlink 即可 |
| Step 2 norm_stats | ~20min | compute_norm_states_fast.py |
| Step 3 切分 | ~1min | JSON 文件生成 |
| Step 4 写 config | 手动 | ~15min |
| Step 5 训练 4 splits | **~114h (4.75 天)** | 每 split ~28h |
| Step 6 MA 合并 | ~1h | 评估 + 合并 + 保存 |
| **总计** | **~5 天** | |

### 输出产物

```
checkpoints/
├── kai0_mixed_1_split_0/split_0_v1/24999/     # Split 0 checkpoint
├── kai0_mixed_1_split_1/split_1_v1/24999/     # Split 1 checkpoint
├── kai0_mixed_1_split_2/split_2_v1/24999/     # Split 2 checkpoint
├── kai0_mixed_1_split_3/split_3_v1/24999/     # Split 3 checkpoint
└── kai0_mixed_1/                              # 最终合并模型
    ├── 0/params/                              # Orbax checkpoint
    └── norm_stats.json
```

### 验证标准

与官方 `Task_A/mixed_1` 对比：

| 指标 | 预期 |
|------|------|
| norm_stats.mean[0] | ≈ -0.1199 (官方) ± 0.01 |
| 参数 L2 距离 | < 150 (两个合并模型间) |
| held-out DAgger MSE | ≈ 0.066 (官方) ± 0.015 |
| offline loss | ≈ 0.237 (官方) ± 0.05 |

---

## 方案 2: `kai0_full` (完整版)

### 目标

构建完整 chi0 候选模型：MA + SA + TDA 三模块全部启用。

### 输入

`kai0_mixed_1` 的所有输入 +

| 资源 | 来源 | 备注 |
|------|------|------|
| stage_progress_gt 标注 | `data/Task_A/advantage` 已有 | 仅 base 数据，需扩展到 dagger |
| data_augment 工具 | `train_deploy_alignment/data_augment/` | space_mirroring.py, time_scaling.py |

### 流程

```
Step A: Advantage Estimator (SA Step 1)
  pi05_base + base(stage_progress_gt) → 100K steps → advantage_estimator

Step B: Advantage 预测 (SA Step 2)
  advantage_estimator × (base + dagger) → 预测 absolute/relative advantage
  为 dagger 生成之前缺失的 advantage 标签

Step C: Advantage 离散化 (SA Step 3)
  阈值=30% → task_index ∈ {0, 1}
  更新 tasks.jsonl: "fold the cloth, Advantage: positive/negative"

Step D: 数据增强 (TDA)
  (base + dagger with labels) → space_mirroring → 翻倍
                               → time_scaling → 追加"加速"版本
  合并所有增强变体 → augmented_dataset (~24000 ep)

Step E: MA 训练 (4 splits)
  augmented_dataset → 切 4 splits
  4 × pi05_base → 25K steps each
  (prompts 从 tasks.jsonl 读取，含 advantage 标签)

Step F: MA 合并
  4 × checkpoints → greedy merge → kai0_full_ma

Step G: (可选) AWBC fine-tune
  kai0_full_ma → 继续训 25K steps on augmented_dataset → kai0_full
```

### 详细步骤

#### Step A: 训练 Advantage Estimator

```bash
# Config: ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD (已有)
# 数据: data/Task_A/advantage (含 stage_progress_gt)
uv run torchrun --standalone --nproc_per_node=8 \
  scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD \
  --exp_name=adv_est_v2 \
  --save_interval 10000
```

输出: `checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v2/100000/`

**耗时**: ~1.5 天

#### Step B: 为 base + dagger 预测 advantage

```bash
# 现有 advantage 数据只针对 base
# 需要对 dagger 重新预测
uv run python stage_advantage/annotation/eval.py \
  Task-A KAI0 \
  data/Task_A/dagger \
  --ckpt-dir checkpoints/.../adv_est_v2/100000 \
  --output data/Task_A/dagger_with_advantage
```

为 dagger 每帧生成 `absolute_advantage`, `relative_advantage`, `absolute_value` 列。

**耗时**: ~4 小时 (2.4M frames × 3 cameras 视频解码 + GPU 推理)

#### Step C: 离散化 advantage

```bash
# 对 base + dagger 合并后的数据做离散化
# 先合并:
python train_deploy_alignment/data_augment/merge_lerobot.py \
  --src_paths data/Task_A/advantage data/Task_A/dagger_with_advantage \
  --tgt_path data/Task_A/kai0_full_base_data \
  --repo_id kai0_full_base_data

# 然后离散化:
python stage_advantage/annotation/discretize_advantage.py \
  data/Task_A/kai0_full_base_data \
  --threshold 30 \
  --discretion-type binary \
  --advantage-source absolute_advantage
```

输出：更新 `tasks.jsonl`、每帧 `task_index ∈ {0, 1}`。

**耗时**: ~30 分钟

#### Step D: 数据增强

```bash
# Space mirroring (数据翻倍)
python train_deploy_alignment/data_augment/space_mirroring.py \
  --src_path data/Task_A/kai0_full_base_data \
  --tgt_path data/Task_A/kai0_full_mirrored \
  --repo_id kai0_full_mirrored \
  --merge_with_src

# Time scaling (factor=2, 追加加速版本)
python train_deploy_alignment/data_augment/time_scaling.py \
  --src_path data/Task_A/kai0_full_mirrored \
  --tgt_path data/Task_A/kai0_full_data \
  --repo_id kai0_full_data \
  --extraction_factor 2 \
  --split_ratio 0.5
```

最终数据集 `data/Task_A/kai0_full_data` 预计 ~24000 episodes。

**耗时**: ~1 天 (视频 flip + frame extraction, CPU bound)

#### Step E: MA 训练

```python
# config.py: 新增 kai0_full_split_0~3
TrainConfig(
    name="kai0_full_split_0",
    model=pi0_config.Pi0Config(pi05=True),
    data=LerobotAgilexDataConfig(
        repo_id="data/Task_A/kai0_full_data",
        default_prompt="Flatten and fold the cloth.",
        use_delta_joint_actions=False,
        base_config=DataConfig(prompt_from_task=True),  # ← 启用 advantage prompts
    ),
    weight_loader=CheckpointWeightLoader("pi05_base/params"),
    num_train_steps=25_000,
    batch_size=256,
),
```

```bash
# run_kai0_full.sh
for i in 0 1 2 3; do
  EPISODES=$(python scripts/get_kai0_full_episodes.py $i)
  uv run scripts/train.py kai0_full_split_$i \
    --exp_name=split_${i}_v1 \
    --fsdp-devices 8 \
    --batch-size 256 \
    --data.episodes $EPISODES
done
```

**耗时**: ~4.75 天 (4 × 28h)

#### Step F: MA 合并

```bash
uv run python model_arithmetic/dump_data.py \
  --dataset kai0_full_split_0 \
  --output kai0_full_val.pkl --batch-size 16

uv run python model_arithmetic/arithmetic.py \
  --config kai0_full_split_0 \
  --data-path kai0_full_val.pkl \
  --checkpoints \
    checkpoints/kai0_full_split_0/split_0_v1/24999 \
    checkpoints/kai0_full_split_1/split_1_v1/24999 \
    checkpoints/kai0_full_split_2/split_2_v1/24999 \
    checkpoints/kai0_full_split_3/split_3_v1/24999 \
  --output $(pwd)/checkpoints/kai0_full_ma \
  --optimize_method greedy \
  --gpu_ids "0"
```

**耗时**: ~1 天

#### Step G: (可选) AWBC fine-tune

```python
# config.py: kai0_full_awbc_finetune
TrainConfig(
    name="kai0_full_awbc_finetune",
    model=pi0_config.Pi0Config(pi05=True),
    data=LerobotAgilexDataConfig(
        repo_id="data/Task_A/kai0_full_data",
        default_prompt="Flatten and fold the cloth.",
        use_delta_joint_actions=False,
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=CheckpointWeightLoader(
        "checkpoints/kai0_full_ma/0/params"  # ← 从 MA 输出继续
    ),
    num_train_steps=25_000,
    batch_size=256,
),
```

```bash
uv run scripts/train.py kai0_full_awbc_finetune \
  --exp_name=awbc_ft_v1 \
  --fsdp-devices 8 \
  --batch-size 256
```

**耗时**: ~1.2 天

### 时间估算

| 步骤 | 耗时 | 累计 |
|------|------|-----|
| A: Advantage estimator 训练 | 1.5 天 | 1.5 天 |
| B: Dagger advantage 预测 | 0.2 天 | 1.7 天 |
| C: 合并 + 离散化 | 0.1 天 | 1.8 天 |
| D: 数据增强 (mirror + time) | 1 天 | 2.8 天 |
| E: 4 splits 训练 | 4.75 天 | 7.55 天 |
| F: MA 合并 | 1 天 | 8.55 天 |
| G: AWBC fine-tune (可选) | 1.2 天 | 9.75 天 |
| **总计 (含 G)** | **~10 天** | |
| **总计 (不含 G)** | **~8.5 天** | |

### 输出产物

```
checkpoints/
├── ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v2/100000/    # Advantage estimator
├── kai0_full_split_0~3/split_*_v1/24999/                   # 4 MA splits
├── kai0_full_ma/                                            # MA merge result
│   ├── 0/params/
│   └── norm_stats.json
├── kai0_full_awbc_finetune/awbc_ft_v1/24999/               # (可选) AWBC fine-tune
└── kai0_full/                                               # 最终模型 (= ma 或 awbc_finetune)
    └── ...

data/Task_A/
├── kai0_full_base_data/        # base + dagger + advantage labels
├── kai0_full_mirrored/         # + space_mirroring
└── kai0_full_data/             # + time_scaling (最终训练数据)
```

### 验证标准

1. **离线 MSE/loss** (与 `kai0_mixed_1` 对比):
   - 预期: 差异 < 10% (MSE 不敏感指标)
   - 如果显著变差: pipeline 有 bug

2. **Advantage prompt 响应性测试**:
   - 喂 `"Advantage: positive"` vs `"Advantage: negative"` 到同观测
   - 预期: MSE > 0.01 (baked-in SA)
   - 如果 < 0.001: SA 没有生效，可能 fine-tune 步数不够

3. **真机 success rate** (理想但非必需):
   - 如果能做: 对比 `kai0_mixed_1` 应有显著提升
   - 如果不能: 记录为限制，不作为验证标准

---

## 决策流程

```
是否需要真机测试能力？
├── 是 → 两个方案都做
│   ├── 先做 kai0_mixed_1 验证 pipeline
│   ├── 再做 kai0_full 验证论文 monotonic increase
│   └── 真机对比 success rate
│
└── 否 → 只做 kai0_mixed_1
    ├── 目标: 复现 Task_A/mixed_1 的性能
    ├── 验证: offline MSE + norm_stats 对齐
    └── 跳过 kai0_full (离线指标无法验证 SA 贡献)
```

---

## 运行前 checklist

### 通用
- [ ] 8 × A100 80GB GPU 可用
- [ ] 至少 500GB 磁盘空间 (数据 + checkpoints)
- [ ] pi05_base 预训练权重已下载到 `openpi_cache/`
- [ ] `run_gf1.sh` 已调通 (验证训练环境)
- [ ] wandb offline 配置 (避免训练中断)

### kai0_mixed_1 特定
- [ ] `data/Task_A/base` 和 `data/Task_A/dagger` 完整
- [ ] `merge_lerobot.py` 脚本可用 (在 `data_augment/`)
- [ ] `compute_norm_states_fast.py` 可跑

### kai0_full 特定
- [ ] `data/Task_A/advantage` 存在 (含 `stage_progress_gt` 列)
- [ ] Advantage estimator config (`ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD`) 可用
- [ ] `space_mirroring.py`, `time_scaling.py` 脚本可用
- [ ] `stage_advantage/annotation/eval.py` 和 `discretize_advantage.py` 可用

---

## 已知风险

### kai0_mixed_1
1. **MA 合并方法选择**: inverse_loss 在 4 个 loss 相近的 splits 上退化为简单平均；推荐 `greedy` 或 `gradient_descent`
2. **数据合并脚本兼容性**: `merge_lerobot.py` 需要测试 base + dagger 的 schema 是否一致
3. **norm_stats 计算精度**: 6512 ep × 3.2M+ frames 需要 quantile (q01, q99) 的流式计算

### kai0_full
1. **Advantage estimator 训练质量**: 100K 步是否充分？过拟合风险？
2. **DAgger 数据的 advantage 预测**: DAgger 是失败恢复场景，advantage estimator 在 OOD 上可能不可靠
3. **数据增强 I/O**: space_mirroring 需要重新编码 ~40000 个视频文件 (base+dagger × 3 cameras)
4. **Augmented 数据的训练成本**: 24000 ep 是 6512 的 3.7 倍，但 25K steps 不变，可能欠拟合

---

## 参考

- [kai0 论文 (arXiv:2602.09021)](https://arxiv.org/abs/2602.09021)
- [官方 HF 仓库](https://huggingface.co/OpenDriveLab-org/Kai0)
- [本地 stage_advantage/README.md](../stage_advantage/README.md)
- [本地 model_arithmetic/README.md](../model_arithmetic/README.md)
- [本地 train_deploy_alignment/README.md](../train_deploy_alignment/README.md)
