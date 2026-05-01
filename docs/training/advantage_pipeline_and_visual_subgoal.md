# kai0 Advantage Pipeline 与视觉 Subgoal 增强方案

本文分两部分：
1. **Part 1** — 当前 kai0 仓库中 Advantage 是如何**标注、训练、推理、离散化、下游使用**的完整 pipeline。
2. **Part 2** — 如何通过引入 **goal / stage subgoal 视觉相似度**增强 Advantage 的信号质量。

---

## Part 1：现有 Advantage Pipeline

### 1.0 流水线概览

```
Step 0: 人工阶段标注  →  parquet 新增 stage_progress_gt 列  (offline, 手工)
             │
             ▼
Step 1: 训练 Advantage Estimator (PyTorch DDP)
             │  config.py: ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD
             │  script:    scripts/train_pytorch.py
             ▼
Step 2: 推理全量数据 → parquet 写入 absolute_value / absolute_advantage / relative_advantage
             │  stage_advantage/annotation/eval.py + evaluator.py
             ▼
Step 3: 离散化 advantage → task_index + tasks.jsonl
             │  stage_advantage/annotation/discretize_advantage.py
             ▼
Step 4: AWBC 训练 (JAX FSDP)
                config.py: pi05_flatten_fold_awbc
                script:    scripts/train.py
                prompt:    "<task>. Advantage: positive/negative"
```

---

### 1.1 Step 0 — 阶段 GT 标注（`stage_progress_gt`）

**定义**（`kai0/stage_advantage/README.md:42-46`）：

将任务分成 K 个子阶段，每条 episode 人工标注每段的起止时间戳。对第 k 个 stage（0-indexed）内某一帧：

$$
\text{stage\_progress\_gt}
= \frac{k}{K} + \frac{1}{K} \cdot \frac{\text{frame\_pos\_within\_stage}_k}{\text{segment\_length}_k}
$$

- 取值范围 `[0.0, 1.0]`，**整集单调递增**；
- 每个 stage 占等宽 1/K 区间，段内按**时间线性插值**；
- stage 边界处连续无跳变（第 k 段末 = `(k+1)/K` = 第 k+1 段首）；
- 写入 parquet，与 `observation.state`、`action` 并列。

**Task A（flatten & fold）** 典型 K=2：stage 0 = 把衣服摊平到桌面，stage 1 = 折叠。

---

### 1.2 Step 1 — 训练 Advantage Estimator

#### 数据：`AdvantageLerobotDataset`（`kai0/src/openpi/training/advantage_dataset.py:10`）

每次 `__getitem__` 产出一对帧：

- `t = idx`（当前帧）
- `t' = uniform(ep_start, ep_end)`（**同 episode 均匀随机**一帧；变量名 `his_-100_*` 是历史残留，**不是固定往回 100 步**），见 `handle_timestep_difference_mode` (行 151–178)，16 次重试避开 `t == t'` 或 timestamp 相同

**回归 label**（advantage_dataset.py:133-135）：

```python
progress = stage_progress_gt[t] - stage_progress_gt[t']   # ∈ [-1, 1]，有符号
```

**跨 stage 处理**：因为 `stage_progress_gt` 是贯穿全集的单调量，**无需特殊处理**——跨段就是"经过的完整 stage 数 × (1/K) + 两端段内贡献"之和。

#### 模型：`AdvantageEstimator`（`kai0/src/openpi/models_pytorch/pi0_pytorch.py:464`）

继承 `PI0Pytorch`，复用 PaliGemma 视觉 + language + action expert 主干，唯一新增：

```python
self.value_head = nn.Sequential(
    nn.Linear(width, width), nn.SiLU(),
    nn.Linear(width, width), nn.SiLU(),
    nn.Linear(width, 1),      nn.Tanh(),   # 输出 ∈ [-1, 1]
)
```

- 输入：3 路相机（top_head / hand_left / hand_right）+ 14D 状态 + task prompt
- Head 作用在 transformer 最后一层的 **state token**（`suffix_out_full[:, 0, :]`）
- 初始化：加载 `pi05_base` 预训练权重

#### 损失（pi0_pytorch.py:562-581）

```python
loss = loss_action * loss_action_weight + value_loss * loss_value_weight
value_loss = MSE(value_pred(t) - value_pred(t'),  progress)
```

Advantage training config 里：
- `loss_value_weight = 1.0`（启用 value loss）
- `loss_action_weight = 0.0`（关闭 flow-matching action loss）

即**只训 value head**，主干被 progress 信号反向传播微调。

#### 训练启动

```bash
cd /home/tim/workspace/deepdive_kai0/kai0

# 无需预计算 norm_stats (skip_norm_stats=True)
uv run torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py \
  ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD \
  --exp_name=adv_est_v1 --save_interval 10000 --batch-size 144
```

Config：
- `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD`（单卡 bs=16，100K step）
- `ADVANTAGE_TORCH_PI06_FLATTEN_FOLD`（8 卡 bs=144，100K step，用 pi06_base）

产物：`checkpoints/.../model.safetensors`。

---

### 1.3 Step 2 — 推理：写回 `absolute_value` / `*_advantage`

脚本：`kai0/stage_advantage/annotation/evaluator.py` → 供 `eval.py` 调用。

对每条 episode，按 **batch + prefetch** 遍历所有帧 `n`，调用 `model.sample_values` 产出三列：

| 列名 | 含义 | 计算式（evaluator.py:264-266, 476） |
|------|------|-------------------------------------|
| `absolute_value` | 绝对进度估计 | `V(frame_n, frame_0)` |
| `relative_advantage` | 局部 50 帧 advantage | `V(frame_n, frame_{n+50})` |
| `absolute_advantage` | 相邻 relative_interval 差分 | `(absolute_value[n+50] - absolute_value[n]) / interval × 50` |

- `relative_interval = 50`（`eval.py:149` 默认）
- 输出统一 clip 到 `[-1, 1]`
- 结果通过 `add_advantage_to_parquet.py`（或同等脚本）**追加列写回 parquet**

---

### 1.4 Step 3 — 离散化：连续 advantage → 二值 `task_index`

脚本：`kai0/stage_advantage/annotation/discretize_advantage.py`

两种模式（可组合）：

#### (a) 简单百分位分箱

```
per-episode percentile threshold (默认 30%):
    task_index = 1  if absolute_advantage >= percentile_30 else 0
```

即每条 episode 里分数前 70% 的帧标 positive，剩余 negative。

#### (b) 按 stage 独立分箱（`--stage-nums K`）

- 用每帧的 `stage_progress_gt` 判定所属 stage（`get_stage_index`，行 60-77）
- **每个 stage 内独立算百分位**再分箱
- 结果：一个 stage 内部也有 pos/neg 对比，避免"stage 0 整体 advantage 低 → 全被判 negative"的失衡

#### 产物

- parquet 的 `task_index` 列被**覆盖**为 0/1（或 0..K-1 用于 q5 变体）
- `tasks.jsonl` 扩为 2 条（binary 二元模式，由 `discretize_advantage.py:200-201` 生成，**句号分隔**）：
  ```json
  {"task_index": 0, "task": "Flatten and fold the cloth. Advantage: negative"}
  {"task_index": 1, "task": "Flatten and fold the cloth. Advantage: positive"}
  ```

---

### 1.5 Step 4 — AWBC 训练（下游消费）

本节详述 kai0 里 AWBC（Advantage-Weighted Behavior Cloning）的实现机制，并与纯 BC 做对照。

#### 1.5.1 AWBC vs Normal BC 的 Config Diff

`pi05_flatten_fold_awbc` 与 `pi05_flatten_fold_normal` 的**唯一实质差别**：

| 字段 | `pi05_flatten_fold_normal`（BC）<br>(config.py:1230) | `pi05_flatten_fold_awbc`<br>(config.py:2256) | 说明 |
|---|---|---|---|
| `repo_id` | `Task_A/base`（3055 ep，原始） | `Task_A/advantage`（3055 ep，已 discretize） | 帧内容相同，只是 `task_index` 列被重写、`tasks.jsonl` 扩成 2 条 |
| `prompt_from_task` | `False`（用 `default_prompt`） | **`True`**（查 tasks.jsonl） | **唯一真正的开关** |
| `default_prompt` | `"Flatten and fold the cloth."` | 同左（但被 `prompt_from_task` 覆盖） | |
| `num_workers` | 8 | 16 | 次要 |
| model / weight / steps / bs / lr / augment / loss | 完全一致（pi05_base → 100k × bs256） | | |

一句话：**AWBC = normal BC + 打开 `prompt_from_task` + 换一个经过 discretize 的数据集**。

#### 1.5.2 Prompt 注入：两条不同的 path

kai0 里有两条把 advantage 信号拼进 prompt 的 path，**互斥、不要混用**：

**Path A — 离散 + `PromptFromLeRobotTask`（`src/openpi/transforms.py:376`）** ← **AWBC 走这条**

```python
task_index = int(data["task_index"])
data["prompt"] = self.tasks[task_index]     # 查 tasks.jsonl
```

- 要求：数据集 `tasks.jsonl` 必须已由 `discretize_advantage.py` 生成（句号格式）
- Prompt 字面量：`"Flatten and fold the cloth. Advantage: positive"` / `"Flatten and fold the cloth. Advantage: negative"`
- **二值分桶**（或 q5drop 变体下是 5 档 Quality：`". Quality: 1/5"` ... `". Quality: 5/5"`，config.py:2312）

**Path B — 连续 + `InsertAdvantageIntoPrompt`（transforms.py:114-121）** ← **AWBC 不走这条**

```python
data["prompt"] = data["prompt"].rstrip(".,") + f", Advantage: {advantage:.4f}"
```

- 要求：parquet 必须有 `advantage` 列（`absolute_advantage` 的浮点值）
- Prompt 字面量：`"Flatten and fold the cloth, Advantage: 0.5234"`（**逗号**分隔，4 位小数）
- 仅用于实验变体（把连续 advantage 作为条件信号的消融）

**⚠️ CLAUDE.md 反复提醒的陷阱**：推理时 prompt 必须**字面量级**匹配训练时的格式（逗号 vs 句号、是否 .4f、"positive" vs 数值）。弄混就掉点。

#### 1.5.3 损失：就是 BC 的 loss，**无重加权**

`src/openpi/models/pi0.py:260`：

```python
main_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)   # flow-matching MSE，所有样本等权
```

- **无** `exp(β·A)` 样本级重加权（非传统 AWR）
- **无** 负样本过滤
- positive 帧和 negative 帧**同在一个 batch、同一个 MSE**，唯一差别是 prompt 里多了 3–4 个 token

这是 kai0 的 AWBC 最反直觉的点：它**不在 loss space 做 advantage 权重**，而是把 advantage 信号通过 PaliGemma 的 text tokenizer 注入，让 cross-attention 去解耦两种条件分布。

#### 1.5.4 训练启动

```bash
cd /home/tim/workspace/deepdive_kai0/kai0

uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_awbc
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_awbc \
  --exp_name=awbc_v1 --fsdp-devices 8 --batch-size 256
```

产物：`checkpoints/pi05_flatten_fold_awbc/awbc_v1/{5000,...,100000}/`。

Launchers 参考 `train_scripts/launch/run_awbc_{baseline,q5drop,v2_*,daggeronly}_*.sh`。

#### 1.5.5 推理时 prompt 硬性要求

Serving 端（`start_scripts/start_policy_node.sh` → `serve_policy.py`）**不硬编码** advantage 标签，由调用方通过 `task` 字段传入。要激活"高 advantage"策略必须传：

```
"Flatten and fold the cloth. Advantage: positive"
```

（与 `tasks.jsonl[task_index=1]` 逐字节一致：句号+空格+大写 "Advantage:"+空格+小写 "positive"）

**漏个句号或换成逗号 → 模型落回到它没见过的 prompt 分布 → 掉点。**

#### 1.5.6 相比 BC 的优势

| 维度 | 纯 BC | AWBC |
|---|---|---|
| 低质演示帧处理 | 直接混入，拉平动作分布，学出"演示均值策略" | 进 negative prompt 分支，推理时被切掉 |
| 样本利用率 | 若手动过滤低质 → 样本流失；不过滤 → 拖拽质量 | **100%** 利用，pos/neg 都训 |
| 推理时行为可调 | 只能固定策略 | 改 prompt 可切保守/激进 |
| 训练开销 | baseline | **几乎相同**（prompt 多几个 token，吞吐不变） |
| 需要额外标注 | 无 | 需要跑完 Step 0–3 拿到 task_index |
| 与 AWR 对比 | — | **无 `exp(β·A)` 方差爆炸、无 β 调参**；负样本不被 `exp(·)` 压到 0，仍贡献梯度 |

**本质差别**：把 AWR 的 loss-space 重加权迁移到 **prompt-space 的语言条件化**。在 VLA（PaliGemma + action head）架构下，text encoder 天然会做这件事，不用改 loss、不用改优化器、不用调 β，**成本最低**。

#### 1.5.7 代价与注意事项

1. **强依赖信号链**：Stage 标注质量 → Advantage estimator 拟合 → 离散化阈值，任一环出错整条链失效。`awbc_v2_robust` / `awbc_daggeronly` 等变体都是在调这个链路的鲁棒性。
2. **Prompt 字面量敏感**：训练和推理必须逐字符一致（句号 vs 逗号、大小写、空格）。CLAUDE.md 专门警告。
3. **仍然是 offline**：和 AWR/IQL 同样没有 online rollout，策略覆盖受限于演示数据；pos/neg 比例失衡（默认 70%/30%）会让模型偏向多数类。
4. **Metadata dropout（π0.7 技术）**：若想推理时允许 prompt 省略 advantage 后缀，训练时要加 `DropPromptSuffix`（transforms.py:126+）随机删后缀。`pi05_flatten_fold_awbc_q5drop` 正是此类变体（Option A 里 dropout 被关闭，因为单源 demo 的 advantage 方差本就很小，再 dropout 信号更弱）。

---

### 1.6 现有 pipeline 的已知弱点（为 Part 2 铺垫）

1. **label 在"示教者停顿/犹豫"帧上有偏**：时间在走但场景未变，progress 仍被记作 `Δt / segment_length`，模型被迫从伪特征（光照、人手背景）找信号。
2. **段内线性假设过强**：真实操作有 reach / grasp / transport / place 等子相，视觉变化速率并不均匀。
3. **跨 stage 跳跃对监督的贡献不稳**：跨 stage 的对 `(t, t')` 提供"经过了多少个 stage"的粗粒度信号，但段内信号与段间信号**混在同一个 MSE 里**，权重失衡。
4. **纯依赖时间-位置的标注**：无法反映"做完了什么"——例如折叠失败回到起点，时间继续走、progress 继续涨，但真实 value 应该下降。

---

## Part 2：用 Goal / Stage Subgoal 视觉相似度增强 Advantage

### 2.1 核心思路

**布料状态变化 = 任务进度**（尤其 Task A）。top_head 直接看布料，是"任务状态"的最直接观测。引入一个**视觉距离函数** `d(x, ref)`（`ref` = goal 或 stage subgoal image），与现有时间线性 label 形成**多信号监督**，校准时间 label 的失真点。

---

### 2.2 Goal vs Stage Subgoal — 推荐 **stage subgoal**

| 维度 | 单一 Goal | Stage Subgoal（每 stage 一个 anchor） |
|------|-----------|--------------------------------------|
| 单调性 | ❌ Task A 中 stage 0 摊布会**远离**folded goal | ✅ 每 stage 内 `d(t, anchor_k)` 单调下降 |
| 结构先验 | 丢掉 stage 标注 | 复用已有 `stage_progress_gt` 边界 |
| 失败恢复鲁棒性 | 任何失败状态都被当作"远离 goal" | 可判定"退回上一 stage" |
| 实现复杂度 | 最简单 | 中等（每 stage 一个 anchor） |
| Task A 适配 | 差（多阶段非单调） | 好 |
| Task B/C 单阶段 | OK | 退化为 K=1，等价于 Goal |

**结论**：有 `stage_progress_gt` 时用 **stage subgoal**；单阶段任务退化为 goal 是特例。

---

### 2.3 Anchor 构造

**每 episode 独立**（吸收布料/光照/视角的 episode 间差异）：

```python
# 对 episode e，找到每个 stage 的"末尾帧"作 anchor
for k in range(K):
    boundary_frames = np.where(
        (stage_progress_gt >= (k+1)/K - 1e-6) &
        (stage_progress_gt <  (k+1)/K + 0.01)
    )[0]
    # 取段末 5-10 帧的视觉特征均值，抗遮挡/模糊
    anchor_feat[e, k] = mean(feature(top_head[boundary_frames[-10:]]))
```

**视觉特征选择**（按稳健性排序）：
1. **DINOv2 ViT-L/14** — 通用、已在生态中可用（见 `memory/feedback_model_availability.md`）
2. **LPIPS**（AlexNet/VGG backbone）— 适合 pixel-level 变化
3. **自训 cloth encoder**（MAE/SimCLR on top_head 帧）— 最定制化但要额外训练

**离线预计算**：anchor 特征 + 每帧 top_head 特征一次性跑完存成 `.npz`，训练时直接 lookup。

---

### 2.4 三种集成方式（按实现成本排序，推荐先试 A）

#### A. 辅助正则（加 loss 项，**推荐起步**）

> 核心：**视觉几乎没变 → 压扁模型预测的 value 差**。

```python
# 离线算好
d_top = cosine_dist(feat_top[t], feat_top[t'])     # ∈ [0, 2]

# 训练时
v_diff   = V(t) - V(t')
main_loss = MSE(v_diff, progress)                  # 现有 loss
aux_loss  = (v_diff ** 2) * torch.exp(-alpha * d_top)
loss      = main_loss + lam * aux_loss
```

- `lam ≈ 0.05–0.1`、`alpha ≈ 2.0`（视 DINOv2 距离分布调）
- `exp(-α·d_top)` 是"视觉越像，权重越大"的衰减核
- 只加正则，不替换主 loss → 风险最低，可热插拔

#### B. Label 重加权（调权不调目标）

> 核心：**视觉变化小的对降权**（不是删除）。

```python
weight = torch.sigmoid((d_top - tau) / sigma)      # 视觉变化越大权重越接近 1
loss   = weight * MSE(v_diff, progress)
```

- `tau` 取 `d_top` 分布中位数，`sigma` 取 IQR/2
- 比 A 更激进：真正"欺骗性"的停顿帧对会被**显著降权**
- 代价：失去部分样本的监督信号

#### C. 重定义 label（更改 progress 语义，**最激进**）

> 核心：用**相对 subgoal 的视觉距离差**替代时间插值 label。

```python
# t 所在 stage k
progress_visual = d(t', anchor_k) - d(t, anchor_k)   # 越接近 anchor → 越正

# 跨 stage 的 (t, t'):
# 方式 1：用 t 所在段 anchor 统一算（简单）
# 方式 2：按 1/K 权重累加各段贡献（严谨）
progress_label  = beta * progress_visual + (1 - beta) * progress_gt
```

- `beta` 从 0 开始逐步增加（课程式），`beta = 0.3` 往往够
- **风险**：完全脱离手工 stage 标注后，下游 `discretize_advantage.py` 的 `--stage-nums` 分桶会失效，需要同步改
- 适合现有 pipeline Spearman 已经不佳、想推翻重来的情形

---

### 2.5 改动落地点（在现有代码中插入）

| 文件 | 改动 |
|------|------|
| `kai0/src/openpi/training/advantage_dataset.py` | `__getitem__` 里 lookup 预计算的 `d_top(t, t')`，放进 `final_item['d_top']` |
| `kai0/src/openpi/models_pytorch/pi0_pytorch.py` | `AdvantageEstimator.forward` 里按方案 A/B 构造 loss；新增 `loss_visual_weight` 字段 |
| `kai0/src/openpi/training/config.py` | 新 config `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD_VISUAL`：`loss_visual_weight=0.1`、`visual_feat_path=...`、`stage_nums=2` |
| 新增脚本 `scripts/precompute_top_cam_features.py` | 跑 DINOv2 on top_head，输出 `{ep_idx: ndarray[N_frames, D]}` 存 `.npz` / parquet 附加列 |
| 新增脚本 `scripts/compute_stage_anchors.py` | 从 `stage_progress_gt` 反推边界，算每 stage anchor 特征 |

---

### 2.6 风险清单

| 风险 | 缓解 |
|------|------|
| top_head 被手臂/夹爪遮挡 | (1) 三摄特征融合 (2) 异常帧检测（特征 L2 范数骤降）跳过 |
| DINOv2 对"布料折叠构型"区分度不足 | 先跑 EDA：算 anchor 特征 vs 各帧距离随 stage_progress_gt 的变化曲线；不行换 LPIPS 或自训 encoder |
| anchor 质量对单 episode 敏感 | 段末 5–10 帧特征平均，或做鲁棒中位数 |
| 加了视觉正则后，Spearman 没涨甚至下降 | 说明当前瓶颈是模型容量而非 label 噪声——先跑 baseline eval 再决策 |
| 方案 C 改 label 语义后打破下游 discretize | 若走 C，需同步调整 `discretize_advantage.py` 的 `--stage-nums` 逻辑或跳过 |

---

### 2.7 验证计划（在决定是否投入前先做）

1. **Baseline 定位**：跑 `kai0/stage_advantage/eval_adv_est.py`，记录当前 model 的 `Spearman(absolute_value, stage_progress_gt)` 和 per-episode 分布——判断瓶颈是否在 label 噪声；
2. **EDA**：在 200 条采样 episode 上预计算 DINOv2(top_head) 特征，画：
   - 每 episode `d(t, anchor_k)` 随 t 的曲线（应近单调）
   - "teleop 停顿帧"（action 近零 N 帧）上 progress_gt 与 d_top 的散点
3. **消融 A**：`loss_visual_weight ∈ {0, 0.05, 0.1, 0.2}`，同步看 Spearman 与下游 AWBC success rate；
4. **消融 goal vs subgoal**：`stage_nums ∈ {1(=goal), 2}` 对比，验证本文结论。

---

## 附录：关键引用

- 人工标注 spec：`kai0/stage_advantage/README.md:32-50`
- Dataset：`kai0/src/openpi/training/advantage_dataset.py:100-178`
- Model / Loss：`kai0/src/openpi/models_pytorch/pi0_pytorch.py:464-581`
- 推理：`kai0/stage_advantage/annotation/evaluator.py:255-481`
- 离散化：`kai0/stage_advantage/annotation/discretize_advantage.py:60-321`
- 训练 config：`kai0/src/openpi/training/config.py`（搜 `ADVANTAGE_TORCH_`）
- Eval：`kai0/stage_advantage/eval_adv_est.py`
