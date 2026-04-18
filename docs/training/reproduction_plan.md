# kai0 完整复现训练计划

> 相关文档：
> - [Task_A 开源情况梳理](kai0_task_a_opensource_analysis.md) — 开源模型/数据/代码、各方法成功率、未开源内容
> - [gf2 Advantage + AWBC 方案](gf2_advantage_awbc_plan.md) — gf2 具体执行计划与验证步骤

## 当前数据集状态

### Task_A 数据（唯一已下载的 Task）

| 数据集 | 路径 | Episodes | Frames | Videos (mp4) | 预期 Videos | 状态 |
|--------|------|----------|--------|-------------|-------------|------|
| **base** | `data/Task_A/base/` | 3,055 | 3,362,369 | 9,165 | 9,165 (3055×3) | **完整** ✓ |
| **advantage** | `data/Task_A/advantage/` | 3,055 | 3,362,369 | 6,669 | 9,165 (3055×3) | **视频不完整** ✗ |
| **dagger** | `data/Task_A/dagger/` | 3,457 | 2,415,341 | 10,371 | 10,371 (3457×3) | **完整** ✓ |
| **splits (×4)** | `data/Task_A/splits/split_[0-3]/` | 764+764+764+763 | — | 9,165 | 9,165 | **完整** ✓ |

### advantage 视频缺失详情

advantage 数据集的 parquet 完整（3,055 个 episode 均有），但 **视频下载中断**：

| chunk | top_head | hand_left | hand_right |
|-------|----------|-----------|------------|
| chunk-000 | 1,000 ✓ | 1,000 ✓ | 1,000 ✓ |
| chunk-001 | 1,000 ✓ | 1,000 ✓ | 1,000 ✓ |
| chunk-002 | **0** ✗ | 669 ✗ | **0** ✗ |
| chunk-003 | **0** ✗ | **0** ✗ | **0** ✗ |

缺失 2,496 个视频文件，占总数 27%。advantage 的视频与 base 相同 episode（同为 3,055 个 episode），可考虑从 base 建立符号链接而非重新下载。

### Task_B / Task_C 数据

**未下载**。`config.py` 中 `pi05_tee_shirt_sort_normal` 和 `pi05_hang_cloth_normal` 的 `repo_id` 仍是 `<path_to_repo_root>/data/...` 占位符。

### 损坏视频文件

`logs/bad_videos.txt` 记录了 8 个损坏的 mp4 文件，`logs/bad_videos_gf1.txt` 记录了 12 个（含重叠），分布在 base (3个)、advantage (1个)、dagger (8个+)。

---

## 当前 Checkpoint 状态

| Checkpoint | 路径 | 状态 |
|-----------|------|------|
| **π₀.₅ base** | `openpi_cache/.../pi05_base/params` | **完整** ✓ (13GB, Orbax格式) |
| **pi05_flatten_fold_normal/normal_v1** | `checkpoints/pi05_flatten_fold_normal/normal_v1/` | **仅 norm_stats.json**，训练失败 ✗ |
| **pi05_flatten_fold_split_0/split_0_v1** | `checkpoints/pi05_flatten_fold_split_0/split_0_v1/` | **仅 norm_stats.json**，训练失败 ✗ |
| **pi05_flatten_fold_split_1/split_1_v1** | 同上 | **仅 norm_stats.json**，训练失败 ✗ |
| **pi05_flatten_fold_split_2/split_2_v1** | 同上 | **仅 norm_stats.json**，训练失败 ✗ |
| **pi05_flatten_fold_split_3/split_3_v1** | 同上 | **仅 norm_stats.json**，训练失败 ✗ |

### 训练失败原因

`run_gf0.sh`（normal 训练）和 `run_gf1.sh`（split 训练）均在 `data_loader.py` 创建数据集时报错：

```
NameError: name 'load_dataset' is not defined
```

出错位置：`lerobot/common/datasets/lerobot_dataset.py:622`，`load_dataset` 函数虽然在文件头 `from datasets import load_dataset` 导入，但运行时未定义。当前在交互式 `uv run python` 中测试可以正常导入，可能是：
- 运行时 `datasets` 包版本不兼容（当前已安装 4.8.4）
- 某种环境冲突已在后续操作中被修复

**需要重新运行训练验证是否已修复。**

---

## 完整复现步骤

### 阶段 0：环境与数据准备

#### 0.1 修复/验证环境
```bash
cd /home/tim/workspace/deepdive_kai0/kai0
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
验证 `load_dataset` 可用：
```bash
uv run python -c "from datasets import load_dataset; print('OK')"
```

#### 0.2 补全 advantage 视频
advantage 和 base 共享相同的 3,055 个 episode。两种方案：

**方案 A：从 base 建符号链接**（如果视频内容确实相同）
```bash
# 为缺失的视频建立链接
for chunk in chunk-002 chunk-003; do
  for cam in observation.images.top_head observation.images.hand_left observation.images.hand_right; do
    src="data/Task_A/base/videos/$chunk/$cam"
    dst="data/Task_A/advantage/videos/$chunk/$cam"
    mkdir -p "$dst"
    ln -sf "$src"/*.mp4 "$dst/" 2>/dev/null
  done
done
```

**方案 B：重新下载**
```bash
python scripts/download_dataset.py  # 可能需要加参数只下载 advantage
```

#### 0.3（可选）下载 Task_B / Task_C 数据
仅在需要复现全部三个 Task 时需要。

---

### 阶段 1：Normal π₀.₅ Fine-tuning

**目标**：在 Task_A/base 上全参数微调 π₀.₅，获得基础模型。

#### 1.1 确认 config 路径正确
`config.py` 中 `pi05_flatten_fold_normal` 的路径已配置为当前环境：
- `repo_id`: `/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/base` ✓
- `weight_loader`: `/vePFS/tim/workspace/openpi_cache/.../pi05_base/params` ✓

#### 1.2 计算 norm stats（已有）
`data/Task_A/base/norm_stats.json` 已存在，checkpoint 目录中也已生成 `norm_stats.json`。如需重算：
```bash
uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_normal
```

#### 1.3 训练
```bash
# 使用 run_gf0.sh 或直接运行：
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train.py pi05_flatten_fold_normal \
  --exp_name=normal_v1 --no-wandb-enabled --fsdp-devices 8 --batch-size 128
```
- 100K steps，batch_size 128（8 GPU FSDP）
- checkpoint 每 5000 步保存一次

---

### 阶段 2：Model Arithmetic（模型融合）

**目标**：将数据按 episode 分成 4 份，各训一个子模型，再混合权重。

#### 2.1 Split 子模型训练
episode 分组已就绪（`split_episodes_[0-3].json`，764/764/764/763），splits 数据也已生成。

`run_gf1.sh` 中的训练方式：使用 `--data.episodes` 参数从 base 数据集中过滤 episode（而非使用 splits/ 子目录）。

```bash
# 逐个训练（每个 25K steps）
for i in 0 1 2 3; do
    EPISODES=$(python3 get_episodes.py $i)
    uv run scripts/train.py pi05_flatten_fold_split_$i \
      --exp_name=split_${i}_v1 --no-wandb-enabled \
      --fsdp-devices 8 --batch-size 128 \
      --data.episodes $EPISODES
done
```

#### 2.2 Dump 验证数据
```bash
python model_arithmetic/dump_data.py --dataset pi05_flatten_fold_normal --output flatten_fold_val.pkl
```

#### 2.3 混合 Checkpoint
```bash
python model_arithmetic/arithmetic.py \
  --config pi05_flatten_fold_normal \
  --data-path flatten_fold_val.pkl \
  --checkpoints \
    checkpoints/pi05_flatten_fold_split_0/split_0_v1/<best_step> \
    checkpoints/pi05_flatten_fold_split_1/split_1_v1/<best_step> \
    checkpoints/pi05_flatten_fold_split_2/split_2_v1/<best_step> \
    checkpoints/pi05_flatten_fold_split_3/split_3_v1/<best_step> \
  --output checkpoints/mixed_flatten_fold \
  --optimize_method inverse_loss --use_gpu --gpu_ids "0"
```

---

### 阶段 3：Stage Advantage + AWBC

**目标**：训练优势估计器，标注优势标签，用于 AWBC 训练。

> **捷径**：已释出的 `Task_A/advantage/` 已包含完整的 advantage 标注（`stage_progress_gt`, `relative_advantage`, `absolute_advantage`, `task_index`）和 `tasks.jsonl`（binary: negative/positive）。**可直接跳到 Step 4 AWBC 训练**。

#### 3.1 (可跳过) Step 1 — 训练 Advantage Estimator
```bash
uv run python scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD \
  --exp_name=run1 --save_interval 10000
```
- PyTorch 训练，100K steps
- 需要 advantage 数据中的 `stage_progress_gt` 列 ✓（已有）
- `skip_norm_stats=True`，无需计算 norm stats

#### 3.2 (可跳过) Step 2 — 预测 Advantage
```bash
# 先修改 stage_advantage/annotation/eval.py 中的 MODELS_CONFIG_MAP
uv run python stage_advantage/annotation/eval.py Task-A KAI0 data/Task_A/advantage
```

#### 3.3 (可跳过) Step 3 — 离散化 Advantage
```bash
cd stage_advantage/annotation
python discretize_advantage.py /path/to/dataset \
    --threshold 30 --chunk-size 50 --discretion-type binary \
    --advantage-source absolute_advantage
```

#### 3.4 Step 4 — AWBC 训练
前置条件：advantage 数据需要完整视频（当前缺失 27%，需先补全）。

```bash
uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_awbc
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_awbc --exp_name=run1
```
- `repo_id` 指向 `data/Task_A/advantage/`
- `prompt_from_task=True`：训练时 prompt 取自 `tasks.jsonl`（"fold the cloth, Advantage: positive/negative"）
- 100K steps，batch_size 256

---

### 阶段 4：Train-Deploy Alignment（需实体机器人，可选）

#### 4.1 数据增强
```bash
# 时间缩放
python train_deploy_alignment/data_augment/time_scaling.py \
  --src_path data/Task_A/base --tgt_path data/Task_A/time_scaled --extraction_factor 2

# 空间镜像
python train_deploy_alignment/data_augment/space_mirroring.py full \
  --src-path data/Task_A/base --mirror-path data/Task_A/mirrored --merge-path data/Task_A/merged
```

#### 4.2 使用增强数据训练
更新 config 的 `repo_id` 指向增强后的数据集，重新训练。

#### 4.3 DAgger 采集 + 推理部署（需实体机器人和硬件）
参见 `train_deploy_alignment/dagger/` 和 `train_deploy_alignment/inference/` 的 README。

---

## 当前阻塞项汇总

| 编号 | 阻塞项 | 优先级 | 说明 |
|------|--------|--------|------|
| **B1** | `load_dataset` NameError | **高** | 所有训练均因此失败，需验证是否已修复并重跑 |
| **B2** | advantage 视频不完整 | **中** | 缺失 2,496/9,165 (27%)，阻塞 AWBC 训练和 advantage estimator 训练 |
| **B3** | 损坏视频文件 | **低** | 8-12 个 mp4 损坏，可能导致个别 episode 训练出错 |
| **B4** | Task_B/C 数据未下载 | **低** | 仅影响非 Task_A 的复现 |

## 推荐执行顺序

1. **修复 B1**：验证 `load_dataset` 问题是否已解决，做一个小 3-step 测试训练
2. **启动阶段 1**：跑 `pi05_flatten_fold_normal` 全量训练 (100K steps)
3. **并行修复 B2**：补全 advantage 视频（从 base 符号链接或重新下载）
4. **启动阶段 2**：normal 训练完成后，跑 4 个 split 训练 + model arithmetic
5. **启动阶段 3**：advantage 视频补全后，跑 AWBC 训练
6. 阶段 4 数据增强可与上述并行进行
