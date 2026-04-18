# AWBC π0.7-style 实验方案与跟进

**创建时间**: 2026-04-18
**状态**: 训练中（gf0 baseline + gf1 **Option A: Quality only, no dropout** 并行）
**负责人**: Tim

> **[2026-04-18 更新]** Step 5000 eval 显示 dropout=15% 版本在多数指标上落后 gf0 baseline。
> 已 kill gf1, dropout 0.15 → 0.0，resume from step 5000。详见第十节。

---

## 一、实验目标

在 Task_A (flatten fold) 上对比两种 AWBC 设计的效果：

- **Baseline (gf0)**：当前 `pi05_flatten_fold_awbc` — binary advantage discretization（positive/negative）
- **π0.7-style (gf1)**：新增 `pi05_flatten_fold_awbc_q5drop` — 5-bin Quality + stage-aware + metadata dropout

灵感来源：[π0.7 paper](https://www.pi.website/download/pi07.pdf) Sec V ("Diversifying the prompt")。核心思路是**多维度 metadata prompting + dropout**。

### 预期改进（基于 π0.7 paper + 本实验离线分析）

| 指标 | gf1 vs gf0 预期 |
|------|----------------|
| Rollout 速度 | **+10-20%** |
| Rollout 成功率 | +2-5% |
| Action MAE@50 | -5-10% |
| Episode 长度 | 缩短 19-39%（Oracle 上限 64%）|

---

## 二、架构对比

### gf0 Baseline

```
Dataset: data/Task_A/advantage (binary labels, threshold=70%)
  task_index=0 → "Flatten and fold the cloth. Advantage: negative"
  task_index=1 → "Flatten and fold the cloth. Advantage: positive"

Pipeline:
  PromptFromLeRobotTask → RepackTransform → AgilexInputs → ModelTransforms
  (无 dropout)
```

### gf1 π0.7-style (三个改动)

```
Dataset: data/Task_A/advantage_q5 (n_slices=5, stage-nums=2)
  task_index=0 → "Flatten and fold the cloth. Quality: 1/5"
  task_index=1 → "Flatten and fold the cloth. Quality: 2/5"
  task_index=2 → "Flatten and fold the cloth. Quality: 3/5"
  task_index=3 → "Flatten and fold the cloth. Quality: 4/5"
  task_index=4 → "Flatten and fold the cloth. Quality: 5/5"

Pipeline:
  PromptFromLeRobotTask → RepackTransform
  → ★ DropPromptSuffix(rate=0.15, marker=". Quality:")  ← NEW
  → AgilexInputs → ModelTransforms
```

**三个关键改动**：
1. **n_slices=5 Quality 分箱**：信息量 log2(5)/log2(2) = 2.3x
2. **Stage-aware percentile (stage-nums=2)**：每阶段独立算 percentile，避免 fold 阶段过度代表
3. **15% prompt dropout**：推理时对 prompt 缺失的鲁棒性

### 推理 prompt

两边推理时都用"最好"的 prompt 得到最优动作：

- gf0: `"Flatten and fold the cloth. Advantage: positive"`
- gf1: `"Flatten and fold the cloth. Quality: 5/5"`

---

## 三、代码改动（全部向后兼容）

### 3.1 新增 transform (`src/openpi/transforms.py`)

```python
@dataclasses.dataclass(frozen=True)
class DropPromptSuffix(DataTransformFn):
    """π0.7-style metadata dropout: randomly strip configurable suffix.
    dropout_rate=0.0 = no-op（默认）.
    """
    dropout_rate: float = 0.0
    suffix_marker: str = ". Quality:"

    def __call__(self, data):
        if self.dropout_rate <= 0.0 or "prompt" not in data:
            return data
        if np.random.random() >= self.dropout_rate:
            return data
        prompt = data["prompt"]
        if not isinstance(prompt, str):
            prompt = prompt.item() if hasattr(prompt, "item") else str(prompt)
        if self.suffix_marker in prompt:
            base = prompt.split(self.suffix_marker)[0]
            return {**data, "prompt": np.asarray(base + ".")}
        return data
```

### 3.2 `LerobotAgilexDataConfig` 加字段 (`src/openpi/training/config.py`)

```python
# π0.7-style metadata dropout
prompt_suffix_dropout_rate: float = 0.0   # 默认 0.0 = 不触发
prompt_suffix_marker: str = ". Quality:"
```

`create()` 中（仅当 rate > 0 才插入）：
```python
if self.prompt_suffix_dropout_rate > 0.0:
    data_transforms.inputs.insert(0, _transforms.DropPromptSuffix(...))
```

### 3.3 新 TrainConfig

```python
TrainConfig(
    name="pi05_flatten_fold_awbc_q5drop",
    model=pi0_config.Pi0Config(pi05=True),
    data=LerobotAgilexDataConfig(
        repo_id=".../data/Task_A/advantage_q5",
        default_prompt="Flatten and fold the cloth.",
        use_delta_joint_actions=False,
        prompt_suffix_dropout_rate=0.15,
        prompt_suffix_marker=". Quality:",
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=.../pi05_base/params,
    num_train_steps=100_000,
    keep_period=5000,
    num_workers=16,                # 优化（原 8）
    batch_size=256,
),
```

也把 baseline `pi05_flatten_fold_awbc` 的 `num_workers` 同步提升到 16（优化 DataLoader）。

### 3.4 DataLoader 优化 (`src/openpi/training/data_loader.py`)

```python
self._data_loader = torch.utils.data.DataLoader(
    ...,
    prefetch_factor=8 if num_workers > 0 else None,  # 新增（默认 2）
)
```

### 3.5 Checkpoint bug 修复 (`src/openpi/training/checkpoints.py`)

```python
options=ocp.CheckpointManagerOptions(
    ...,
    single_host_load_and_broadcast=True,  # 修复多主机 device mismatch
),
```

### 3.6 新建 scripts（`/vePFS/tim/workspace/deepdive_kai0/`）

- `run_awbc_baseline_gf0.sh`：gf0 单机 baseline 启动
- `run_awbc_q5drop_gf1.sh`：gf1 单机 π0.7-style 启动
- `prepare_advantage_q5.sh`：离线生成 `advantage_q5` 数据集

---

## 四、数据预处理（advantage_q5 生成）

```bash
bash /vePFS/tim/workspace/deepdive_kai0/prepare_advantage_q5.sh
```

流程：
1. **拷贝** `advantage/` 的 meta + parquets 到 `advantage_q5/`（videos 用 symlink，省 60GB）
2. **复用** `norm_stats.json`（state/action 分布不变）
3. **标记** `.kai0_ts_validated`（跳过 20 分钟 LeRobot 初始化校验）
4. **Discretize** 运行 `discretize_advantage.py --n-slices 5 --stage-nums 2`
5. **覆盖** `tasks.jsonl` 为 π0.7 Quality 格式

预期耗时：~3 分钟（幂等，可重跑）。

### 验证分箱结果

```
Stage 0 (flat, 73.6% 帧) task_index 分布:  19/20/20/20/21%  ← 均匀
Stage 1 (fold, 26.4% 帧) task_index 分布:  18/18/20/22/22%  ← 稍高 quality
```

**Stage-aware rebalance 成功**：避免 flat percentile 在跨阶段聚集的问题。

---

## 五、深度定量分析（实验前预判成功概率）

### Q_A: Quality 信号质量（vs 噪声）✅

| 指标 | 实测 | 判定 |
|------|------|------|
| 相邻帧 label 变化率 | 0.26 | 远低于 random 0.80 → **信号稳定** |
| Episode 内最长 same-label run | 平均 56 帧 | 动作连续聚集 |
| Episode 内 label entropy | 1.55 / log(5)=1.61 | 分布均匀 |

### Q_B: Quality 独立于 state ✅

- Logistic(state → label): 28.4% 准确率
- Majority baseline: 21.1%
- **信息增量 +7.3pp** → Quality 提供 state 之外的新信息

### Q_C: Quality 对 action 区分度（控制 state 后）✅

- 聚类 state 为 50 簇，组间 action 方差 / 组内方差 = **η² = 3.1%**
- 参考：弱 <1%, 弱-中 1-5%, 中 5-15%, 强 >15%
- 3.1% 属于"弱-中"——AWBC 有空间但上限有限

### Q_D: Oracle 上限 🔥

| | 帧数 | Q=4 比例 | final progress |
|---|------|---------|---------------|
| Bottom 20% episodes | 1534 | 11% | 0.999 |
| Top 20% episodes | **548** | 50% | 0.997 |

**Oracle 缩短上限 64.3%** → 实际 AWBC 改进预期 19-39%。

### AdvantageEstimator 质量

`absolute_value`（估计器直接预测）vs GT `progress(t)-progress(t-100)`:
- **Per-episode median corr = 0.896**
- 97.3% episodes > 0.7, 45.7% > 0.9
- Pooled corr = 0.74, R² = 0.55

**估计器本身很好**。但我们用的 `absolute_advantage = absolute_value(t+50) - absolute_value(t)` 是**二阶导数**，噪声被放大（corr ~0.3）。这是当前设计的主要短板之一。

### 综合判定：4/4 ✅ **AWBC q5drop 预期明显优于 baseline**

---

## 六、训练运行状态

### 6.1 配置

| 机器 | IP | 配置 | batch | num_workers | prefetch_factor |
|------|-----|------|-------|-------------|------------------|
| **gf0** | 192.168.0.144 | `pi05_flatten_fold_awbc` | 256 | 16 | 8 |
| **gf1** | 192.168.0.161 | `pi05_flatten_fold_awbc_q5drop` | 256 | 16 | 8 |

### 6.2 启动记录

- **第一次启动（未优化 DataLoader）**：2026-04-18 07:41 (gf0), 08:10 (gf1)
  - gf0 跑到 Step 400 (loss 0.0171)，rate 波动 4.3-9.7 s/it，GPU idle 50%
  - gf1 跑到 Step 100 (loss 0.3108)
  - Kill：发现 DataLoader prefetch 太小导致 GPU 饥饿
- **第二次启动（优化后）**：2026-04-18 08:30
  - DataLoader: num_workers 8→16, prefetch_factor 默认 2→8
  - GPU idle 降到 ~5%，rate 稳定 3.5 s/it

### 6.3 性能

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| GPU 100% 时间比例 | ~50% | **~90%** |
| Rate | 4.3-9.7 s/it 波动 | **3.5 s/it 稳定** |
| 单次训练 ETA | 5-6 天 | **~4 天** |

### 6.4 训练曲线

| Step | gf0 loss | gf1 loss | gf0 grad_norm | gf1 grad_norm |
|------|----------|----------|---------------|---------------|
| 0 | 0.8036 | 0.8180 | 5.19 | 5.31 |
| 100 | 0.3013 | 0.3108 | 2.05 | 2.05 |

**观察**：gf1 loss 略高（~3%）。这是**预期**的——dropout 让模型不能"作弊"全靠 prompt。关键看推理时差异。

---

## 七、关键诊断与纠错（避免重复踩坑）

### 7.1 GPU 周期性空闲 40 步 spike

**现象**：每 40 步 rate 从 3.5 spike 到 9.7 s/it，GPU 空闲 5-7 秒。

**根因**：DataLoader prefetch buffer (num_workers=8 × prefetch_factor=2 = 16 batches) 用完后，
workers 来不及解码视频 → 主线程卡 `queue.get`。

**修复**：num_workers 8→16, prefetch_factor 2→8 → buffer 从 16 到 128 batches，优化后 GPU idle <5%。

### 7.2 Advantage 数据集有坏视频？

**误报**：`data_loader.py:61` 注释说 "311/3055 (10%) missing videos"，但 vePFS 实际没坏视频。

**真相**：80-byte 的"视频"实际上是 **symlink** 指向 `base/` 复用。2496 symlinks 全部有效。

### 7.3 2.0 → 3.5 s/it 不是"变慢"

**错误推理**：我曾说"现在比历史慢 75%"。

**真相**：
- 历史 2.0 s/it = batch_size=128, Task_A/base dataset
- 现在 3.5 s/it = batch_size=256（2x）, Task_A/advantage
- **Samples/sec**：历史 64, 现在 73 → 当前实际**快 14%**

### 7.4 vePFS I/O 竞争？

**观测**：gf0 和 gf1 并行时 rate 完全相同（3.5 s/it），spikes 同步。

**结论**：**没证据**说明存在明显 I/O 竞争（至少在优化后）。我之前说的 "+40% 竞争" 没有数据支撑。

### 7.5 多主机 checkpoint bug

**现象**：Step 1000 save 时 `INVALID_ARGUMENT: Buffer on cuda:1, replica assigned to cuda:0`。

**根因**：Orbax 多主机 checkpoint 的 device assignment 逻辑 bug。

**修复**：`CheckpointManagerOptions(single_host_load_and_broadcast=True)`。单机不触发此 bug，但作为防御保留。

### 7.6 首次训练 "5+ 小时未到 Step 0"

**现象**：pi05_flatten_fold_awbc 双机 TCP 首次启动后 5 小时仍在"XLA 编译"。

**根因**：`scripts/train.py:247` 的 `img[i]` 对多主机 sharded 数组做 Python int 索引 → `apply_primitive(gather)` 路径 → 每次 5+ 分钟编译 + NCCL clique 初始化。

**修复**：在 `config.wandb_enabled=False` 时跳过该 image-logging 段；始终用 `np.asarray()` 先把整体 batch 拉到 host。

### 7.7 RoCE 跨子网不可行

**尝试**：各种 NCCL env var（`NCCL_IB_USE_RDMA_CM`, `NCCL_IB_ROUTABLE_FLID_GID_INDEX`, VolcEngine 平台 config）。

**结论**：两节点 mlx5_1~4 在不同 /27 子网，NCCL 的 verbs 路径在跨子网下 `ibv_modify_qp` 失败（kernel 不做 L3 routing 的 DMAC 解析）。`ib_write_bw -R`（rdma_cm）可以跨子网工作，但 NCCL 数据面不用 rdma_cm。

**结论**：**容器内无法修复**，需运维配置 SR-IOV 或同子网。当前转为单机并行训练。

---

## 八、后续计划

### Phase 1：当前（~4 天）
- gf0 baseline + gf1 **Option A**（Quality only, no dropout）并行训练到 100K steps
- Monitor 追 Step 10000/50000/100000 checkpoint 的 eval MAE 对比

### Phase 2：评估（训练完成后 ~1 天）
- 离线评估（evaluate_heldout.py 或 rollout）
- 对比指标：
  - Action MAE@1/10/50 关节
  - gripper MAE
  - Rollout 成功率（sim 或 real）
  - Episode 完成帧数

### Phase 3：若 Option A 仍≤ gf0，Tier 2 实验
- **Option D（Mistake flag）**：从失败回合 / DAgger 数据引入 bi-modal 信号
  （当前 demo-only 数据 η²=3% 天花板低，Mistake 提供真正的 good/bad 差异）
- **Option E（Speed 标签）**：对演示速度分桶 — 速度方差通常 > 质量方差
- **GT Upper Bound**：用 `progress(t)-progress(t-100)` 直接当 Quality（零噪声对照）
- **Continuous advantage**：直接把 float 写进 prompt（无量化损失）

---

## 九、操作命令速查

```bash
# 准备数据集（幂等）
bash /vePFS/tim/workspace/deepdive_kai0/prepare_advantage_q5.sh

# 启动 gf0 baseline（在 gf0 本机执行，或 ssh 到 gf0）
bash /vePFS/tim/workspace/deepdive_kai0/run_awbc_baseline_gf0.sh

# 启动 gf1 π0.7-style（通过 gf0 跳板 ssh 到 gf1）
bash /vePFS/tim/workspace/deepdive_kai0/run_awbc_q5drop_gf1.sh

# 查看进度
tail -f /vePFS/tim/workspace/deepdive_kai0/logs/gf0_awbc_baseline_*.log
tail -f /vePFS/tim/workspace/deepdive_kai0/logs/gf1_awbc_q5drop_*.log

# Kill 训练
ssh -p 2222 root@192.168.0.144 "pkill -9 -f train.py"
ssh -p 2222 root@192.168.0.144 "ssh -p 2222 -i /root/.ssh/ssh_worker_rsa_key -o StrictHostKeyChecking=no root@192.168.0.161 'pkill -9 -f train.py'"
```

---

## 十、Option A 切换（2026-04-18）

### 10.1 触发原因

Step 5000 in-training eval 对比（6 个 eval point: Step 2101/2201/2301/3000/4000/5000）显示
gf1（dropout=15%）在大多数 MAE 指标上落后 gf0 baseline 1-18%，且差距随 step 增大不收敛。

### 10.2 根因分析（对第五节的修正与深化）

复盘时发现 Option B "用 `absolute_value` 分桶" 设计缺陷：

- `absolute_value` = 预测的 **累计 progress** (corr=0.896 vs GT progress)
- 按 progress 分桶 → Quality 5 = episode 末尾，Quality 1 = episode 开头
- 这等价于**位置编码**而非**质量信号** → 无 AWBC 意义

正确的质量信号来自 `absolute_advantage = absolute_value(t+50) - absolute_value(t)`
（progress 变化率），但差分放大噪声 → corr 降至 ~0.4。

**gf1 原设计（dropout=15%）失败的根本原因**：
1. Demo-only 数据的 advantage 方差本就很小（Q_C 测 η²=3.1%，弱-中）
2. Quality 信号是 AdvantageEstimator 输出的差分 → 噪声被放大
3. 再叠加 15% dropout → 进一步稀释已经很弱的信号
4. π0.7 paper dropout 有效的前提是**多模态 prompt 冗余**（Quality + Speed + Mistake + Subgoal），
   单一 Quality 维度下 dropout 没有冗余来弥补

### 10.3 Option A 设计

**保留**：
- n_slices=5 Quality 1-5 分桶（log2(5)/log2(2)=2.3× 信息量）
- Stage-aware rebalance（stage-nums=2）
- DataLoader 优化 (num_workers=16, prefetch_factor=8)
- In-training eval + 90/10 split

**移除**：
- Prompt suffix dropout（0.15 → 0.0）

**保留 config 名 `pi05_flatten_fold_awbc_q5drop`**：checkpoint dir 与 exp_name
绑定，保留原名以 resume from step 5000。config 内注释标明 Option A 切换。

### 10.4 实施

- 时间：2026-04-18
- Kill: `ssh ... kill 1776525`（gf1 at step 5000）
- 改动: `config.py:1510` `prompt_suffix_dropout_rate: 0.15 → 0.0`
- Resume: `bash run_awbc_q5drop_gf1.sh`（config 路径不变，自动 load step 5000 ckpt）

### 10.5 预期

Option A 是"最小改动"消融：剥离 dropout，检验剩余三个改动（n_slices=5 + stage-aware
+ Quality prompt）是否单独能带来 gf1 > gf0。

- **若 Option A > gf0**：验证多分桶 + stage-aware 有效，dropout 是拖累
- **若 Option A ≈ gf0**：demo-only 数据已触及天花板 → 转 Option D (Mistake) / E (Speed)
- **若 Option A < gf0**：说明连多分桶都比 binary advantage 差 → 需深度诊断 prompt encoder
  是否真的在利用 Quality 信号（可做 embedding 可视化）
