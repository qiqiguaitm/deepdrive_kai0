# Stage Classifier 训练方案（gf1 on Task_A/advantage）

**创建时间**: 2026-04-19
**状态**: 🟢 **Phase 2 进行中 — Phase 1 目标已达标**
**目标**: 训练一个视频级 stage 分类器，用于自动标注 `dagger/` 数据的 `stage_progress_gt`，解锁 dagger 进入 AWBC 训练

---

## ⭐ 当前进展 Snapshot (2026-04-19 16:45 UTC)

### ✅ Phase 1: Feature Precompute
| Backbone | 节点 | 状态 | 备注 |
|----------|------|------|------|
| **V-JEPA 2.1 Large** (4.8 GB, 384 res) | gf1 | ✅ **done** (3055/3055) | 11:11→15:28，4h17m, 8-GPU |
| VideoMAE v2 Giant (4.1 GB, 224 res) | gf0 | ⏸ **停止** | 142/3055 已计算后停；Large 已达标，不再需要 A/B |
| V-JEPA 2 ViT-g (15.3 GB, 384 res) | - | ⏸ **跳过** | - |
| V-JEPA 2.1 Giant (15.7 GB, 384 res) | - | ⏸ **跳过** | - |
| V-JEPA 2.1 Gigantic (28.2 GB, 384 res) | - | ⏸ **跳过** | - |

**结论**：V-JEPA 2.1 Large 单 backbone 已完全满足目标指标，**跳过其他 backbone A/B**，节省 ~30h precompute。

**排障记录**：
- V-JEPA 2.1 下载：HF 初始 snapshot_download 卡死 → 改 `aria2c --all-proxy=8888` + 解析 302 后直连 xet CDN → 14 MB/s（提速 36×）
- Meta 研究模型走 torch.hub (dl.fbaipublicfiles.com) 可直连，无需代理
- VideoMAE 缺 `easydict` → `uv pip install easydict` 装到共享 venv 解决
- VideoMAE aria2 直传后 HF snapshot 缺 `model.safetensors` 符号链 → 手工 `ln -s` 修复
- **CUDA_VISIBLE_DEVICES bug**：`precompute_features.py` 原代码在 Python 内覆盖 env → 8 workers 都跑到 GPU 0。Fix: 改 `--gpu` 默认 `None`，shell 的 `CUDA_VISIBLE_DEVICES=$i` 才生效
- **VideoMAE 输入格式**：`(B, C, T, H, W)` 需要 permute；且 VideoMAEv2 `forward` 默认 pool 到 `(B, D)`，需 monkey-patch 取 per-token

### ✅ Phase 2: Head Training（V-JEPA 2.1 Large cache）

**4 并行 2-GPU 探索实验，15:40:51 启动，6-9 min/实验 全部完成。**

| Exp | 改动 | Best frame_acc | Best boundary_mae | 训练用时 |
|-----|------|----------------|-------------------|---------|
| **🏆 E1 baseline** | default (hidden=384, n_layers=2, lr=5e-4) | **0.9960** @step 4K | **3.5 frames (0.12s)** @step 4K | 6 min |
| E2 capacity | hidden=512, n_layers=3 | 0.9957 @step 7K | 3.9 frames | 8 min |
| E3 strong_mono | mono=0.5, smooth=0.3, fold_weight=5, boundary=0.9 | 0.9945 @step 2K | 5.4 frames | 7 min |
| E4 long_train | 40K steps, warmup=2000, lr=3e-4 | 0.9956 @step 15K | 4.1 frames | 9 min |

**🏆 选定最终模型**（**按 boundary_mae 最低准则**）：
```
/vePFS/.../kai0/checkpoints/stage_classifier_vjepa2_1_large/E1_baseline/best.pt
  (step 4K, frame_acc=0.9960, boundary_mae=3.5 frames)
```

**关键观察**：
1. 任务偏 "简单"：V-JEPA 2.1 Large 冻结 backbone + 4M 参数 head，**4 分钟** 达 99.6% frame_acc
2. E1 baseline 赢双冠（frame_acc 最高 + boundary_mae 最低）
3. E3 强 mono loss 反而害 frame_acc — 放弃该方向
4. E4 长训得到最高 mono_rate (0.667) 和 conf (2010)，但 boundary_mae 略差
5. **10K steps 已足够**（20K/40K 过度；E1 在 step 4K 就 best）

### ✅ Phase 3: Dagger 推理 + 可视化验证

**3457 dagger ep 推理**：gf1 8-GPU 并行，16:10 启动，用 E1 best head。
- 增量 metrics flush（每 ep 写 JSON）+ resume（跳过已标 parquet）
- 输出 `dagger_with_stage/` 数据集（添加 `stage_progress_gt` 列，值 ∈ {0.25, 0.75}）

**Sanity-check 可视化（advantage 306 val ep 全量）**：
- `eval_val_predictions.py` → 生成 `val_predictions.json` 每 ep GT vs PRED
- `sanity_check_labels.py` → 渲染 170 eps（按 abs_offset 倒序，worst first）
- 产物：`/vePFS/.../kai0/label_sanity_viz/overview.html`

**结果 (val 306 ep 全量)**：

| 指标 | 值 |
|------|---|
| boundary_mae | **3.97 frames** (0.132s) |
| median \|offset\| | 3 frames |
| p75 / p90 / p95 \|offset\| | 5 / 10 / 14 |
| max \|offset\| | 30 frames |
| `>10 frames` 比例 | 29/306 (9.5%) |
| **signed mean offset** | **−3.18** (model 系统性偏早 ~3 frames) |

**Signed mean ≠ 0 的主因**：训练用 `class_weight_fold=3.0`（3× 权重），模型为避免"漏报 fold"更激进预测 fold → 系统性偏早。

**人工 check 结论 (用户)**：人工标签本身有一定噪声（boundary 标注会 ±几帧），影响**不大** → **不再优化新模型**，使用 E1 best 继续推进下游。

### 方案 A/B/C/D 评估（**不执行**）
用户判断：当前 3.97 frames mae 已够，人工标噪声是主要剩余误差源。
- ~~方案 A (two-stage refine)~~
- ~~方案 B (boundary-aware loss)~~
- ~~方案 C (state-velocity fusion)~~
- ~~方案 D (3-camera)~~
- 跳过，直接进入 Phase 4 下游 pipeline。

---

## 📋 下一步计划（**Phase 4-7：下游 AWBC v2 pipeline**）

### Phase 4: Dagger pseudo-label 完成 (进行中，~3h)
- 16:10 启动，预计 19:00 UTC 完成 3457 ep 推理
- 产物: `dagger_with_stage/` with `stage_progress_gt ∈ {0.25, 0.75}`

### Phase 5: AE 推理 + Discretize

#### 决策：沿用现有 AE（不在 dagger 上重训）

**AE 训练原理回顾**
- 架构：`AdvantageEstimator` (Pi0.5 PyTorch + value_head MLP)
- 输入：当前 + `his_-100` 两时间步的 3 相机图像 + state + prompt
- 训练标签：`progress = stage_progress_gt(t) − stage_progress_gt(t-100)` (100 帧窗口进度差分)
- 损失：`MSE(value_pred, progress)`，`loss_action_weight=0`（纯值头训练）
- ckpt：`ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1/100000`（100K steps）

**是否应该用 dagger 重训？— 不**

| 维度 | 用 dagger 重训 | 保持原 AE（**采用**） |
|------|---------------|---------------------|
| 数据量 | 6512 ep (2.1×) | 3055 ep |
| 分布 | 含失败/卡顿轨迹 → **稀释 advantage 区分度** | 只见典型专家节奏 |
| 标签噪声 | dagger sp_gt σ≈3 帧（classifier 误差） | advantage sp_gt σ≈1-2 帧（人工） |
| 训练成本 | +2-3 天 on 8×A100 | 0 |
| 收益 | log(N) 尺度，~10-20% loss 下降 | - |

**核心论据**：AWBC advantage 语义是"**典型专家节奏** vs 当前执行"。AE 应学"典型专家"，再对 dagger 的非典型/失败评价。重训会把 dagger 失败也当典型 → 削弱 advantage 信号。符合 χ₀ 论文做法。

**naive 替代方案被验证不可行**：
用 `sp_gt(t+50) - sp_gt(t)` 替代 AE 输出理论上可行，但实测在 advantage ep 20 上：
- AE abs_adv range `[-0.10, 0.17]`, **std=0.044**（正/负都有）
- Naive sp_gt diff range `[0.023, 0.052]`, **std=0.013**（近似常数）
- 相关性 0.34 — AE 捕获 **视觉执行质量**，sp_gt diff 只是**线性分段**
- 用 naive 法 discretize 退化（stage 内常数 → 30 百分位无意义）

#### 执行步骤

```bash
# 1. AE inference on dagger_with_stage (gf1 单 GPU ~1-2h)
bash train_scripts/launch/run_ae_infer_dagger.sh
# → dagger_with_stage/data_KAI0_100000/*.parquet (含 absolute_advantage 列)

# 2. Sanity check: AE 输出分布 vs advantage 的分布
python -c "
import pyarrow.parquet as pq, numpy as np
from pathlib import Path
for name, path in [
    ('advantage', '.../advantage/data'),
    ('dagger',    '.../dagger_with_stage/data_KAI0_100000'),
]:
    vals = []
    for p in list(Path(path).rglob('*.parquet'))[:50]:
        vals.extend(pq.read_table(p)['absolute_advantage'].to_numpy())
    v = np.array(vals)
    print(f'{name}: std={v.std():.3f} range=[{v.min():.3f}, {v.max():.3f}]')
"
# 若 dagger std 远小于 advantage → AE 在 dagger 上 generalize 差 → 考虑 warm-start fine-tune (6h)
# 若分布相近 → 继续

# 3. 复制 AE 预测结果到 dagger_advantage/ 作为 discretize 输入
cp -r dagger_with_stage/data_KAI0_100000 dagger_advantage/data
# copy/symlink videos + meta

# 4. discretize_advantage.py --stage-nums 2 --threshold 30
python kai0/stage_advantage/annotation/discretize_advantage.py \
  dagger_advantage --stage-nums 2 --threshold 30 --discretion-type binary \
  --advantage-source absolute_advantage
# → 输出 dagger_advantage/ 带 task_index ∈ {0, 1} 和更新后的 tasks.jsonl
```

#### 兜底：AE 在 dagger 上 generalize 差时

若 Sanity Check 发现 `absolute_advantage` 分布退化（std 远小于 advantage），降级方案：

```bash
# Warm-start fine-tune (20K additional steps, ~6h on 8 GPU)
# 从 ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1/100000 继续
# 训练数据：advantage (主) + dagger_with_stage (辅，低权重)
# 目标：让 AE 学会在 dagger OOD 输入上稳定预测，同时保持 advantage 上的区分度
```
仅当必要时执行，优先保持纯 advantage 训练的 AE。

### Phase 6 (简化): Merge（无 mirror）
**决策**: 基于 `task_e_master_plan §2.3.1` 内部 A/B（mirror 中性/略负收益）+ D405 摄像头不对称风险 → **跳过 space_mirror**，回收 247 GB 数据。
```bash
# 直接 merge: advantage (3055) + dagger_advantage (3457) = awbc_v2_full (6512 ep)
python kai0/scripts/merge_lerobot.py \
  --src-paths kai0/data/Task_A/advantage kai0/data/Task_A/dagger_advantage \
  --tgt-path  kai0/data/Task_A/awbc_v2_full \
  --repo-id   awbc_v2_full

# compute_norm_stats_fast
python kai0/scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_awbc_v2
```

**已删数据集 (~247 GB)**：`advantage_mirror`, `advantage_sym`, `dagger_labeled`（fake task_index=1 hack）, `dagger_labeled_mirror`, `dagger_sym`

### Phase 7: AWBC v2 训练（两节点并行 A/B）

**策略**：gf1 跑 vanilla 版 baseline，gf0 跑 **部署鲁棒性** 变体，对照"精度"与"泛化"。

#### 为什么需要 robust 变体？

部署 sim01 与训练数据源有差异：
| 变化 | 偏差 | 影响 |
|------|------|------|
| **手腕相机 D435 → D405** | FOV 87°→95°，色彩响应不同 | 图像色调/畸变偏移 |
| top_head **高度/角度** | 高度 ±5 cm, pitch/yaw ±3° | 视角几何变化 |
| 机械臂 **间距** | ±2-3 cm | state 映射到末端位置偏差 |

当前 `model.py:196-212` JAX 增强**过于温和**：`RandomCrop(0.95) + Rotate(±5°) + ColorJitter(0.3, 0.4, 0.5)`。

#### 方案对照

| 节点 | Config | Visual Augmentation | State Noise | 目的 |
|------|--------|-------------|------------|------|
| **gf1** | `pi05_flatten_fold_awbc_v2` | default (mild) | 无 | 训练域 baseline |
| **gf0** | `pi05_flatten_fold_awbc_v2_robust` | **aggressive** ↓ | **无**（与现状对齐） | 跨设备鲁棒（仅图像层增强） |

**aggressive aug 配置**：
```python
# Color (所有 3 相机)
ColorJitter(brightness=0.5, contrast=0.6, saturation=0.8, hue=0.08)
+ GaussianNoise(σ=0.02)      # 传感器噪声
+ RandomGamma(0.7, 1.3)      # 曝光差异

# Geometric - top_head
RandomCrop(0.85)             # 0.95 → 0.85
Rotate(±10°)                 # ±5° → ±10°
RandomPerspective(0.1)       # 新增，pitch/yaw 鲁棒
RandomScale(0.9, 1.1)        # 新增，相机高度变化

# Geometric - wrist (D435↔D405 切换)
Rotate(±8°)                  # 新增
RandomCrop(0.90)             # 新增
```

**关于 state noise**：**不采用**。与当前 codebase 对齐（无 state noise augmentation），避免 state-action 不匹配与 gripper 维度风险。鲁棒性提升**纯靠图像层 augmentation**。

#### 启动命令

```bash
# gf1 vanilla (20:20 启动)
bash train_scripts/launch/run_awbc_v2_gf1.sh
# Config: pi05_flatten_fold_awbc_v2，data=awbc_v2_full 6512ep，30K steps

# gf0 robust (并行，20:20 启动)
bash train_scripts/launch/run_awbc_v2_robust_gf0.sh
# Config: pi05_flatten_fold_awbc_v2_robust，data=awbc_v2_full 6512ep，30K steps
```

#### 评测（训练完成后）

1. **训练域 val MAE**：gf0 robust 预期略差于 gf1 vanilla（robustness trade-off），但 <10% 掉分为可接受
2. **sim01 真机 success rate（核心指标）**：
   - D435（训练设备）：gf1 ≈ gf0
   - **D405（新设备）**：gf0 ≫ gf1 ← 目标
   - 故意 ±5 cm 高度扰动：gf0 保持 vs gf1 掉分
3. 综合指标：选**真机 success rate 最高**的 checkpoint 作最终部署版

#### Phase 8 (可选): Bridge DAgger TDA Fine-tune

若 gf0 robust 仍不足，追加：
- sim01 用 **D405 teleop 采 20-50 episodes**（~1-2h 采集）
- 对 gf0 robust checkpoint fine-tune 5K steps（~2h）
- 这就是 χ₀ 论文 **Train-Deploy Alignment** 模块的完整实现

### 关键里程碑 & 预计用时（Phase 4-8 完整流程）
| Phase | 节点 | 预计用时 | ETA (from 17:07) |
|-------|------|---------|------------------|
| 4 Dagger inference (16 worker gf0+gf1) 收尾 | gf0+gf1 | ~1h (剩余) | 18:10 UTC |
| 5a AE inference on dagger (单 GPU) | gf1 | ~1-2h | 20:00 UTC |
| 5b Sanity check abs_adv 分布 | — | ~1 min | 20:01 UTC |
| 5c (可选) AE warm-start fine-tune | gf1 8 GPU | 6h，仅必要时 | 02:00 次日 |
| 5d discretize --stage-nums 2 | — | ~5 min | 20:05 UTC |
| 6 Merge + norm_stats | — | ~15 min | 20:20 UTC |
| **7a gf1: awbc_v2 vanilla 30K** | gf1 8 GPU | ~15h | **20 号中午** |
| **7b gf0: awbc_v2 robust 30K** | gf0 8 GPU (**并行**) | ~15h | **20 号中午** |
| 8 (可选) Bridge DAgger TDA fine-tune | gf0/gf1 | 2h (采集 1-2h + 训 2h) | 20 号下午 |

**节点使用总时长**：gf0 24h (vanilla 停训后从 17:00 到 20 号 12:00 UTC), gf1 同。

### 验收指标
- Phase 4: dagger_with_stage parquet 3457 个文件齐，pseudo labels 合理（对齐人工标注的 t*/N 分布）
- Phase 5: dagger 的 task_index 分布接近 50/50（binary discretize）
- Phase 6: awbc_v2_full 的 episodes.jsonl 有 **6512** 条记录（advantage 3055 + dagger_advantage 3457）
- Phase 7: awbc_v2 val eval MAE **≤ gf0_awbc_baseline_v2** (joint_50=0.012 @step 20K)

---

## 一、背景与动机

### 为什么需要这个模型

`Task_A/advantage/`（3055 ep）有人工标注的 `stage_progress_gt`。
`Task_A/dagger/`（3457 ep）**没有**。

**paper χ₀ 的 SA 模块需要 stage_progress_gt** 做 `--stage-nums 2` stage-aware discretize。没有它，dagger 3457 ep 数据**不能以完整 paper 精神参与 AWBC 训练**。

**人工标 dagger stage ≈ 115 小时**，不现实。

**替代方案**：训练一个 video-level stage classifier，用 advantage/ 的 GT 监督，推理出 dagger 的 pseudo stage labels。

### 最终目标

```
完整 pipeline:
  stage_classifier (训练) on advantage/
          ↓
  stage_classifier (推理) on dagger/
          ↓
  dagger with pseudo_stage_progress_gt
          ↓
  AE 推理 on dagger_with_stage → dagger 获得 absolute_advantage
          ↓
  discretize_advantage.py --stage-nums 2 → dagger 获得 task_index
          ↓
  merge advantage_sym + dagger_with_labels → awbc_v2_full (13K ep)
          ↓
  gf1 AWBC 训练 → paper χ₀ 路径的 offline 版本
```

---

## 二、任务定义

### 输入 / 输出

```
任务: Episode-level binary classification per frame
输入: 一个 episode 的视频（任意长度 N frames，30 fps）
输出: per-frame label ∈ {0 (flat), 1 (fold)}
约束: 输出序列必须严格单调不减:
       [0, 0, 0, ..., 0, 1, 1, 1, ..., 1]
       存在且仅存在一个 boundary frame t*
下游: 每帧 label → pseudo stage_progress_gt → discretize --stage-nums 2
```

### 评测指标

| 指标 | 定义 | 目标 |
|---|---|---|
| **frame_accuracy** | 逐帧分类正确率 | ≥ 97% |
| **boundary_mae** | `\|pred_boundary - gt_boundary\|` (frames) | ≤ 15 frames (0.5s @ 30fps) |
| **boundary_mae_sec** | 同上按秒 | ≤ 0.5 sec |
| **monotonic_rate** | raw output 自然单调比例 | ≥ 90% |
| **dp_agreement** | DP 修正前后标签一致率 | ≥ 95% |
| **confidence_mean** | DP score margin 均值 | 主要用于 OOD 检测 |

---

## 三、方案设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│  Input: Episode (任意长度 N 帧, 变长)                        │
│         → 滑动窗口切成 T=16 帧 clips, stride=8 (50% overlap) │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Video Backbone (FROZEN)                           │
│    首选: V-JEPA 2.1 (facebook/vjepa2.1-vit-large)           │
│    备选: V-JEPA 2 / VideoMAE v2 (fallback)                  │
│    输入: (B, T=16, 3, 224, 224)                             │
│    输出: tube tokens (B, 8 tubes × 196 spatial, 1024)       │
│         (tubelet_size=2 → T/2=8 temporal tubes)             │
│  处理: reshape + spatial mean pool → (B, 8 tubes, 1024)    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Tube → Frame Cross-Attention (TRAINABLE)          │
│    Tube features: (B, 8, 1024) → Linear → (B, 8, 384)      │
│    Frame queries: (16, 384) learnable                       │
│    2-layer cross-attention:                                  │
│      Q = frame_queries (16 positions)                        │
│      K, V = tube features (8 positions)                      │
│      每 frame query 自动从 8 个 tubes 聚合信息                │
│    输出: (B, 16, 384) frame-aligned features                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Per-frame Classification Head                      │
│    MLP: 384 → 128 → 2 (binary logits)                        │
│    输出: (B, 16, 2) logits per clip                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Episode-level Aggregation (推理时)                 │
│    滑动窗口重叠区域 → logits 平均                            │
│    得到完整 N 帧的 per-frame logits (N, 2)                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 5: DP Boundary Detection (推理时，硬约束)             │
│    动态规划找最优单一 boundary t*:                           │
│      t* = argmax[sum log_p0[0..t*] + sum log_p1[t*+1..N-1]] │
│    保证输出满足 [0..0 1..1] 单调格式                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
                   Per-frame labels (N,) ∈ {0, 1}
```

### 3.2 Backbone 选型

| 候选 | HF repo | 参数量 | 预训练 | T | 我们任务适配 |
|---|---|---|---|---|---|
| **V-JEPA 2 (SSv2 finetuned)** ⭐ | `facebook/vjepa2-vitl-fpc16-256-ssv2` | ~300M | Self-sup video + SSv2 fine-tune | **16** | ⭐⭐⭐⭐⭐ 精细动作专家 |
| V-JEPA 2 (pretraining) | `facebook/vjepa2-vitl-fpc64-256` | ~300M | Self-sup video | 64 | ⭐⭐⭐⭐ longer context |
| V-JEPA 2 ViT-Giant | `facebook/vjepa2-vitg-fpc64-256` | ~1B | Self-sup video | 64 | ⭐⭐⭐⭐⭐ 但大 |
| VideoMAE v2 Base | `OpenGVLab/VideoMAEv2-Base` | ~90M | Self-sup + Kinetics | 16 | ⭐⭐⭐ 小模型 baseline |
| VideoMAE Large | `MCG-NJU/videomae-large-finetuned-kinetics` | ~304M | MAE + Kinetics | 16 | ⭐⭐⭐ human action |

**注意**：`V-JEPA 2.1` 尚未公开发布（2026-04 HF 查无此仓库）。当前最佳是 **V-JEPA 2 ViT-L SSv2-finetuned**（`facebook/vjepa2-vitl-fpc16-256-ssv2`）：
- T=16 与我们设计匹配
- SSv2 fine-tuning = **"Something-Something v2" 数据集**（精细手部操作动作，如"folding", "pushing"）
- 比通用 Kinetics 预训练更适配衣物折叠任务

**确认的 HF repo**（2026-04-19 验证）：
- ✅ `facebook/vjepa2-vitl-fpc16-256-ssv2` — 首选
- ✅ `facebook/vjepa2-vitl-fpc64-256` — 备选（T=64）
- ✅ `facebook/vjepa2-vitg-fpc64-256` — 更大（ViT-G）
- ✅ `OpenGVLab/VideoMAEv2-Base` — VideoMAE v2 fallback
- ❌ `facebook/vjepa2.1-*` — **不存在**（未发布）

**所有 backbone 都 FROZEN**（无梯度、无 LoRA），只训顶层。

**选择依据**：V-JEPA 系列基于 predictive representation，对"动作断点"（flat→fold 切换）更敏感。若 2.1 HF 不可用，fallback 到 V-JEPA 2，再 fallback 到 VideoMAE v2。

### 3.3 可训参数量

| 组件 | 参数 |
|---|---|
| `tube_proj` (Linear 1024→384) | ~400K |
| `frame_queries` (16, 384) | 6K |
| `tube_pos_emb` (8, 384) | 3K |
| 2× cross-attn layers | ~2.4M |
| MLP head (384→128→2) | ~50K |
| **合计可训** | **~2.9M** |
| Backbone frozen | ~600M（不训）|

---

## 四、数据集划分

### 4.1 Train / Val Split（Episode-level，不是 frame-level）

```
Source: Task_A/advantage/  (3055 episodes)

Split 策略:
  - Deterministic, seed=42
  - 90% train (2750 eps), 10% val (305 eps)
  - 按 episode_index 保留连续 episodes in val (便于复现)
  - save split indices to data/Task_A/stage_classifier_split.json

Val 不参与训练，作为 held-out eval 集
```

### 4.2 Train Clip Sampling

```python
策略: 
  70% clips 采自 boundary ±8 帧 (强化边界学习)
  30% clips 全局均匀采样

每 training clip:
  T=16 frames
  label per frame = (stage_progress_gt >= 0.5).long()
  来自 advantage/ 的 train split
```

### 4.3 Val Evaluation

```python
对 val 中每个 episode:
  - 完整 episode 作滑动窗口推理（任意长度 N）
  - DP boundary 修正
  - 对比 pred_boundary vs gt_boundary
```

---

## 五、训练详情

### 5.1 预提取 Tube Features（一次性，可缓存）

```python
# precompute_features.py
"""
对 advantage/ 所有 episodes, 生成 all overlapping T=16 clips 的 tube features
保存到 disk 作 training cache
"""

for ep_idx in range(3055):
    ep_frames = load_episode_frames(ep_idx)  # (N, 3, 224, 224)
    
    # Overlapping clips
    for clip_start in range(0, N - 16 + 1, 8):
        clip = ep_frames[clip_start:clip_start+16]  # (16, 3, 224, 224)
        
        with torch.no_grad():
            tokens = backbone(clip.unsqueeze(0))     # (1, 1568, D)
            tube_feats = tokens.reshape(1, 8, 196, D).mean(dim=2)  # (1, 8, D)
        
        save({
            "tube_feats": tube_feats.squeeze(0).cpu(),  # (8, 1024)
            "labels": (stage_progress_gt[clip_start:clip_start+16] >= 0.5).long(),
            "ep_idx": ep_idx,
            "clip_start": clip_start,
        }, f"cache/clip_{ep_idx:04d}_{clip_start:04d}.pt")
```

**预计时间**：3055 ep × avg 100 clips/ep = 305K clips，batched 8 clips/batch on 8 GPU:
- V-JEPA 2.1 Large forward: ~50ms/clip (batch 8)
- Total: 305K / 8 × 50ms × 8 GPU parallel = 约 **4h on 8 GPU**

**Cache size**: 305K clips × (8 tubes × 1024 float16) = 305K × 16KB = **~5 GB**

### 5.2 训练 Cross-Attn + MLP

```python
# trainer.py
TrainingConfig = {
    "backbone_frozen": True,
    "use_cached_features": True,  # 跳过 backbone forward
    "hidden_dim": 384,
    "n_cross_attn_layers": 2,
    "n_heads": 8,
    "clip_len": 16,
    
    "batch_size": 128,          # features 小，可大 batch
    "num_epochs": 20,
    "num_train_steps": 20_000,  # ~20K steps
    "lr": 5e-4,                  # 小网络可大 lr
    "weight_decay": 1e-3,
    "scheduler": "cosine_warmup_1000",
    
    "class_weights": [1.0, 3.0],  # flat:fold 不平衡（~75:25）
    "loss_smooth_weight": 0.1,
    "loss_mono_weight": 0.2,
    
    "val_every_steps": 1000,
    "save_every_steps": 5000,
}
```

### 5.3 Loss 组合

```python
def compute_loss(logits, labels):
    """
    logits: (B, 16, 2)
    labels: (B, 16) ∈ {0, 1}
    """
    # ── (a) Weighted CE ──
    weights = torch.tensor([1.0, 3.0])
    loss_ce = F.cross_entropy(
        logits.reshape(-1, 2),
        labels.reshape(-1),
        weight=weights,
        reduction='mean'
    )
    
    # ── (b) Smoothness: 相邻帧 logit 差小 ──
    delta = logits[:, 1:] - logits[:, :-1]  # (B, 15, 2)
    loss_smooth = delta.pow(2).mean()
    
    # ── (c) Monotonicity: P(fold) 不应下降 ──
    probs = logits.softmax(-1)[:, :, 1]     # (B, 16)
    delta_p1 = probs[:, 1:] - probs[:, :-1] # (B, 15)
    loss_mono = F.relu(-delta_p1).mean()    # 只罚下降
    
    return loss_ce + 0.1 * loss_smooth + 0.2 * loss_mono
```

### 5.4 训练时间估算

```
Cached features → 跳过 backbone → 超快
Cross-attn + MLP 前向: ~5ms / batch
Batch size 128, 20K steps:
  - 20K × 5ms = 100s forward (trivial)
  - + backward + optim → ~2-3h total
  - on 1 GPU 足矣
```

**gf1 8-GPU 训练 2-3h 完成**。

---

## 六、推理流程（应用到 dagger）

### 6.1 整个 Episode 推理（任意长度）

```python
def infer_episode(model, backbone, ep_frames):
    """
    ep_frames: (N, 3, 224, 224)
    returns: pred_labels (N,), boundary_t_star, confidence
    """
    N = ep_frames.shape[0]
    T = 16
    stride = 8
    
    # Step 1: 滑动窗口提取 tube features
    logits_sum = torch.zeros(N, 2)
    count = torch.zeros(N)
    
    for start in range(0, max(N - T + 1, 1), stride):
        end = min(start + T, N)
        clip = ep_frames[start:end]
        if clip.shape[0] < T:  # pad last if needed
            padding = T - clip.shape[0]
            clip = torch.cat([clip, clip[-1:].expand(padding, -1, -1, -1)])
        
        # Backbone forward
        with torch.no_grad():
            tokens = backbone(clip.unsqueeze(0))
            tube_feats = tokens.reshape(1, 8, 196, -1).mean(dim=2)  # (1, 8, D)
        
        # Model forward (cross-attn + MLP)
        with torch.no_grad():
            clip_logits = model.forward_from_tubes(tube_feats)[0]  # (16, 2)
        
        # Accumulate (drop padding)
        real_len = end - start
        logits_sum[start:end] += clip_logits[:real_len]
        count[start:end] += 1
    
    # Step 2: Average overlapping predictions
    per_frame_logits = logits_sum / count.unsqueeze(-1)  # (N, 2)
    
    # Step 3: DP boundary detection (保证单调)
    labels, t_star, confidence = best_boundary_dp(per_frame_logits)
    
    return labels, t_star, confidence
```

### 6.2 DP Boundary Detection

```python
def best_boundary_dp(logits):
    """
    logits: (N, 2)
    returns: labels (N,), t_star (int), confidence (float)
    """
    N = logits.shape[0]
    log_p = F.log_softmax(logits, dim=-1)
    
    cum_log_p0 = torch.cumsum(log_p[:, 0], dim=0)
    cum_log_p1 = torch.cumsum(log_p[:, 1], dim=0)
    total_log_p1 = log_p[:, 1].sum()
    
    # score(t*) = sum log_p0[0..t*] + sum log_p1[t*+1..N-1]
    scores = cum_log_p0 + (total_log_p1 - cum_log_p1)
    t_star = scores.argmax().item()
    
    labels = torch.zeros(N, dtype=torch.long)
    labels[t_star+1:] = 1
    
    confidence = (scores.max() - scores.mean()).item()
    
    return labels, t_star, confidence
```

### 6.3 推理时间估算（dagger）

```
dagger: 3457 episodes, avg 700 frames/episode = 2.4M frames
每 episode ≈ 87 clips (N/stride)
V-JEPA 2.1 Large forward: 80ms/clip (batch 8) on 1 A100

Per episode: 87 / 8 × 80ms × 8 GPU parallel = ~80ms
Total: 3457 × 80ms ≈ 5 min on 8 GPU 并行

加 cross-attn + head + DP: 几乎免费

完整 dagger 推理: ~10-15 分钟
```

### 6.4 写回 dagger parquet

```python
# 每 episode 处理完:
pseudo_sp_gt = torch.where(labels == 0, 0.25, 0.75)  # 伪 stage_progress_gt
ep_parquet['stage_progress_gt'] = pseudo_sp_gt.numpy()
ep_parquet.to_parquet(output_path)
```

### 6.5 下游链路

```bash
# Step 1: dagger_with_stage 产生完毕

# Step 2: AE 推理（现有 ckpt）
python stage_advantage/annotation/eval.py Flatten-Fold KAI0 \
    /vePFS/.../data/Task_A/dagger_with_stage

# Step 3: Discretize
python stage_advantage/annotation/discretize_advantage.py \
    /vePFS/.../data/Task_A/dagger_with_stage_KAI0_abs_binary \
    --threshold 30 --discretion-type binary \
    --advantage-source absolute_advantage --stage-nums 2

# Step 4: 合并
python merge_lerobot.py \
    --src_paths .../advantage_sym .../dagger_with_stage_KAI0_abs_binary_stage_sym \
    --tgt_path .../awbc_v2_full

# Step 5: 启动 AWBC 训练
```

---

## 七、评测方案

### 7.1 Val Split 评测指标详细

```python
def evaluate(model, val_episodes):
    metrics = {
        "frame_accuracy": [],
        "boundary_mae_frames": [],
        "boundary_mae_sec": [],
        "monotonic_rate_raw": [],
        "dp_agreement": [],
        "confidence_mean": [],
    }
    
    for ep in val_episodes:
        gt_labels = (ep.stage_progress_gt >= 0.5).long()
        gt_boundary = find_first_transition(gt_labels)
        
        # Inference
        raw_logits = model.infer_episode(ep)
        raw_labels = raw_logits.argmax(-1)
        
        # Raw output monotonic?
        raw_is_mono = (raw_labels.diff() >= 0).all().item()
        metrics["monotonic_rate_raw"].append(float(raw_is_mono))
        
        # DP correction
        dp_labels, dp_boundary, confidence = best_boundary_dp(raw_logits)
        
        # Metrics
        metrics["frame_accuracy"].append((dp_labels == gt_labels).float().mean().item())
        metrics["boundary_mae_frames"].append(abs(dp_boundary - gt_boundary))
        metrics["boundary_mae_sec"].append(abs(dp_boundary - gt_boundary) / 30.0)
        metrics["dp_agreement"].append((dp_labels == raw_labels).float().mean().item())
        metrics["confidence_mean"].append(confidence)
    
    return {k: np.mean(v) for k, v in metrics.items()}
```

### 7.2 验收标准

| 指标 | 达标值 | 理由 |
|---|---|---|
| frame_accuracy | ≥ 97% | 下游 `--stage-nums 2` discretize 对单帧噪声容忍 70% 分位 |
| boundary_mae_frames | ≤ 15 (0.5s) | paper stage 标注也有 ~±5-10 帧人工误差 |
| monotonic_rate_raw | ≥ 90% | 证明模型学到了 task 结构（非单独决策每帧）|
| dp_agreement | ≥ 95% | 高则说明 raw 预测已近单调，DP 只是兜底 |
| confidence_mean | > 1.0 | 大部分 episode 有清晰 boundary（log-space margin）|

### 7.3 失败判据

若以下任一发生，说明方案有问题：
- frame_accuracy < 93% → backbone 不够强或 head 过拟合
- boundary_mae > 30 frames → clip sampling 策略 / temporal context 不足
- monotonic_rate < 70% → smoothness/mono loss 未起作用
- confidence_mean < 0.3 → logits 过度均匀，模型没学到决策边界

### 7.4 V-JEPA 2.1 vs VideoMAE v2 A/B 对比

```python
# 训两个版本（并行 or 串行）
model_vjepa = StageClassifier(backbone="facebook/vjepa2.1-vit-large")
model_vmae  = StageClassifier(backbone="MCG-NJU/videomae-v2-huge")

# Eval on same val split
results_vjepa = evaluate(model_vjepa, val_eps)
results_vmae  = evaluate(model_vmae, val_eps)

# 基于 frame_accuracy + boundary_mae 综合选优
selected = vjepa if results_vjepa['frame_accuracy'] > results_vmae['frame_accuracy'] + 0.5 else ...
```

### 7.5 Smoke Test: Sanity Check

训完后抽样检查 5 个 val episodes：
- 手工对比 pred boundary 和 video 中实际的 flat→fold 切换时刻
- 验证 pred curve 看起来单调合理
- 画 pred vs GT stage_progress 曲线（可视化）

---

## 八、目录结构（新增）

```
/vePFS/tim/workspace/deepdive_kai0/
├── kai0/src/openpi/models/video/
│   ├── __init__.py
│   ├── stage_classifier.py      # VideoStageClassifier 模型
│   └── backbones.py             # V-JEPA 2.1 / VideoMAE v2 loader
├── train_scripts/
│   ├── data/
│   │   └── split_advantage_stage.py   # 生成 train/val split json
│   ├── stage_classifier/              # 新目录
│   │   ├── precompute_features.py     # 预提取 tube features
│   │   ├── train.py                   # 训练主脚本
│   │   ├── evaluate.py                # Val 评测
│   │   ├── infer_dagger.py            # 推理 dagger
│   │   └── write_pseudo_labels.py     # 写回 dagger parquet
│   └── launch/
│       └── run_stage_classifier_gf1.sh  # gf1 启动脚本
└── docs/training/
    └── stage_classifier_plan.md       # 本文档
```

---

## 九、实施时间线

### 原计划 vs 实际 (2026-04-19 更新)

| Day | 原计划 | 实际 |
|-----|--------|------|
| 0 | 写文档 + 代码 | ✅ 完成 |
| 1 | V-JEPA 2.1 Large precompute (4h) | ✅ 11:11→15:28 UTC (4h17m, 8-GPU)；首轮 4 head 实验全达标 |
| 2 | 训练 + val 评测 | ✅ **提前到 Day 1**：E1 baseline 在 step 4K 达 frame_acc=0.996 |
| 3 | dagger precompute + 推理 | ⏳ 待 backbone A/B 完成后启动 |
| 4 | AE + discretize + merge + awbc_v2 | ⏳ 规划不变 |

### 剩余 Phase 时间线

```
Phase 1 尾 (~T+3h from 15:46, ETA 18:30):
  - gf0: VideoMAE v2 Giant precompute done (224 res, 快)
  - gf1: 4 head experiments done (长的 E4 ~30 min, E1-3 < 10 min)
  - 决策点: E1 baseline 是否为最终 head config（预期是）

Phase 3 — Backbone A/B (~T+3h → T+18h):
  - gf0 queue: V-JEPA 2 ViT-g → V-JEPA 2.1 Giant
  - gf1 queue: V-JEPA 2.1 Gigantic（如果需要）
  - 每个 backbone 训练 E1 config 10K steps ~5 min
  - 产物: 5-backbone head_metrics.json 表

Phase 4 — Dagger Inference (~T+20h):
  - 最佳 backbone 对 3457 dagger ep 做 precompute（~3h on 8 GPU）
  - infer_dagger.py 写 pseudo_stage_progress_gt → dagger_with_stage/
  - Smoke: 随机 20 ep 渲染 progress 曲线确认

Phase 5 — Downstream → awbc_v2 (~T+24h+):
  - AE 推理 on dagger_with_stage
  - discretize --stage-nums 2
  - space_mirror on both → 合并成 advantage_v2 (13K ep)
  - compute_norm_stats
  - gf1 启动 pi05_flatten_fold_awbc_v2（8×A100 FSDP）
```

### 遗留 Day 4 原计划（不变）

```
Day 4:
  - 下游 pipeline:
    - AE 推理 on dagger_with_stage
    - discretize --stage-nums 2 on dagger
    - merge advantage_sym + dagger_stage → awbc_v2_full (13K ep)
    - compute_norm_stats
  - 启动 gf1 AWBC training

总时间: 4 天
```

---

## 十、风险与应对

| 风险 | 概率 | 应对 |
|---|---|---|
| V-JEPA 2.1 HF 未发布 | 中 | Fallback 到 V-JEPA 2 or VideoMAE v2 |
| dagger OOD 严重（姿态异常）| 中 | confidence 低的 episode 打 "unknown" flag, 从训练中排除 |
| boundary 精度差 > 30 frames | 低 | 加大 boundary-focused sampling 比例; 增加 temporal layers |
| 训练不收敛 | 极低 | 减小 lr; 检查 class weight |
| 下游 AWBC 反而变差 | 中 | 做 A/B: advantage_sym only vs + dagger_auto_labeled, 择优 |
| 伪标签噪声传导 | 中 | discretize 的 70% 分位对单帧噪声鲁棒, 整体影响小 |

---

## 十一、决策依据与当前路径对比

| 路径 | 描述 | 数据量 | 预期 AWBC 效果 | 实施成本 |
|---|---|---|---|---|
| **Path X** | 只 advantage_sym (6110 ep) | 6110 | ~65-70% | 已就位，0 额外 |
| **Path X + stage_classifier** ⭐ | + dagger_auto_labeled (13K ep) | 13024 | **~70-75%** | **+4 天** |
| Path 人工标 dagger | 严格 paper 方法 | 13024 | ~75-80% | ❌ 115 h 人工 |

**本方案 = 用 4 天工程代换 115h 人工 + 100% 额外数据**。

---

## 十二、gf1 启动命令预览

```bash
# 主流程（待写完代码后运行）:
cd /vePFS/tim/workspace/deepdive_kai0

# Step 1: 生成 split
python train_scripts/data/split_advantage_stage.py

# Step 2 (在 gf1 执行): 预提取 tube features
ssh ... "bash train_scripts/stage_classifier/precompute_gf1.sh"

# Step 3 (gf1): 训练
ssh ... "bash train_scripts/launch/run_stage_classifier_gf1.sh"

# Step 4 (gf1): 推理 dagger
ssh ... "bash train_scripts/stage_classifier/infer_dagger_gf1.sh"

# Step 5: 下游 AWBC
# ... (复用之前的 AE inference + discretize + merge pipeline)
```

---

## 十三、验证 V-JEPA 2.1 可用性（执行前）

```python
# 启动前先跑一次验证:
from transformers import AutoModel
try:
    m = AutoModel.from_pretrained("facebook/vjepa2.1-vit-large")
    print("✅ V-JEPA 2.1 available")
except Exception as e:
    print(f"❌ V-JEPA 2.1 not found: {e}")
    try:
        m = AutoModel.from_pretrained("facebook/vjepa2-vit-large")
        print("✅ V-JEPA 2 available (fallback)")
    except:
        m = AutoModel.from_pretrained("MCG-NJU/videomae-v2-huge")
        print("✅ VideoMAE v2 (final fallback)")
```

---

## 十四、关键参考

- V-JEPA 2 paper: [arXiv 2506.09985](https://arxiv.org/abs/2506.09985)
- VideoMAE v2 paper: [arXiv 2303.16727](https://arxiv.org/abs/2303.16727)
- kai0 paper: [arXiv 2602.09021](https://arxiv.org/abs/2602.09021)
- 本项目已有 AE pipeline: `kai0/stage_advantage/annotation/`
- 本项目已有 discretize 脚本: `kai0/stage_advantage/annotation/discretize_advantage.py`
