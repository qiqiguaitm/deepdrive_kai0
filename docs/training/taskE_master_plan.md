# kai0 Task E 主规划 — 扶起倒箱 (stand up the fallen box)

> **作用**: Task E 全流程方案 —— 训练（本机 4×5090，差异化实验并行）+ 离线评测（inline + archive）+ 真机部署。
> **硬件**: sim01 本地 4× RTX 5090 32GB（driver 580 / CUDA 13）；数据盘 `/data1` 可用 5.5 TB；62 GiB RAM。
> **关联文档**: [taskA_master_plan.md](../deployment/taskA_master_plan.md), [sim01_deployment.md](../deployment/sim01_deployment.md)。
> **创建 / 最近更新**: 2026-04-19（v5：Phase 2 vision MLP LoRA 验证通过 @1=0.0260，展开 Pipeline B）

---

## TL;DR — 当前状态（2026-04-19）

**最佳单模 single-sample MAE@1 = 0.0262**（E2 v3+lowLR, Phase 1），相对 v2 基线 0.0411 改善 **-36%**。

**已完成阶段**：
| 阶段 | 结论 |
|---|---|
| **Phase 0** v2 基线（pi05_base + base, 15k 步）| ✅ 完成 @1=0.0411 |
| **Phase 1 - 差异化 2×2**（v3/v4/v5/v8，init × data）| ✅ 完成 —— **v3 kai0+base** 胜出 @1=0.0333，**time_scaling 负贡献**，aug 在小数据上 marginal |
| **Phase 1 - v3 续训 fine-tune**（E1/E2/E3/E4 四手段并行）| ✅ 完成 —— **E2 v3+lowLR** 胜出 @1=0.0262，**lowLR 是主力，EMA 次之** |

**Phase 2 结果 & 进行中**：

| 实验 | best @1 | vs E2 | 结论 |
|---|---:|---:|---|
| **T1-1** vision MLP LoRA r=16（E2 init）| **0.0260** @ 15k | **-0.8%** | ✅ LoRA 微幅超过 E2，但 init bug 限制潜力 |
| **T1-2** vision MLP LoRA r=32（E2 init）| 0.0261 @ 15k | -0.4% | rank 增加无显著差异 |
| **T2** E2 + ultra-lowLR + EMA | 0.0261 @ step 1-2k | ~flat | 饱和（E2 已收敛，无空间 fine-tune）|
| **T6** kai0_mixed_1 + allgood 从头 | 🟢 GPU 3 跑中 | — | 独立 baseline check |
| **T7** LoRA r=16 + **fixed init(w_b=0)** + **25k 步** + **base_merged(128 ep)** | 🟢 GPU 0 跑中 | 预期 -15~25% | Pipeline B phase-a |
| ~~T1 vision 全解冻 + FSDP~~ | — | — | ❌ 32 GB 装不下，LoRA 是唯一可行路径 |

**Pipeline B (2026-04-19 进行)** —— 目标压 @1 < 0.024，不用 ensemble/MA/DAgger：

| 阶段 | 实验 | 预期 @1 | 成本 | 状态 |
|---|---|---|---|---|
| B-1 | **T7** fixed-init + 25k + base_merged | 0.022-0.025 | 4.5h 训 | 🟢 进行中 GPU 0 |
| B-2 | **T8** DoRA + MLP LoRA（T7 ckpt / E2 起） | 0.021-0.023 | 1-2h 代码 + 3h 训 | ⏸️ B-1 后 |
| B-3 | **T9** MHA LoRA + MLP LoRA（attention 补覆盖）| 0.020-0.023 | 2-3h 代码 + 3h 训 | ⏸️ B-1 后（可选）|

**B-1 的 3 个关键升级**（核心新发现 2026-04-19）：
1. **LoRA init 修复**：w_b = zeros（原 normal(0.01) 导致 step 0-4k 扰动，浪费 ~5% 潜力）
2. **数据扩到 base_merged**：mirror aug 第一次真正流入 vision（之前 vision 冻结时 mirror 信号无处可去）
3. **25k vs 15k**：T1-1 step 14k→14999 仍在 -1.2%/2k 下降，没触底

**硬件墙 & 绕法**（全部踩过）：
- 5090 32 GB：**全 vision 解冻 OOM**，所有组合（FSDP=2 bs=2 MEM_FRACTION=0.98）都差 72 MB 越不过；LoRA 是唯一可行路径
- sim01 **NUMA 0/2 无 DIMM**：GPU 1/2 在 inline-eval 后 **SIGSEGV**，user-space 修复（numactl, cgroup v2 cpuset.mems, --interleave --strict）**全部拦不住**；faulthandler 能捕 C 栈但不能阻止
- 路径：只用 **GPU 0 + GPU 3**（健康 NUMA）；坏卡只能跑**无 inline-eval** 的 smoke

**下一步目标**：LoRA 若 @1 < 0.024 → 部署；否则考虑 DAgger（用户已排除）或接受 @1 ≈ 0.025 做真机


---

## 0. 数据摘要（已采集）

**路径**: `/data1/DATA_IMP/KAI0/Task_E_2026-04-17/base/`

| 项 | 值 |
|---|---|
| Prompt | `stand up the fallen box` |
| Episodes | 73（全部 success，操作员 ztm）|
| 总帧数 | 77 484（≈ 43 min @30 fps）|
| Episode 长度（帧）| min 134 / p25 537 / p50 840 / p75 1270 / max 4666 |
| 相机 | `top_head`, `hand_left`, `hand_right` (RGB 480×640 + depth) |
| State / Action 维度 | 14（双臂 6+1 × 2；含 gripper），float32 |
| 数据格式 | LeRobot v2.1，1 chunk |

**关键观察**：
- Task A（对照）3 055 ep / 3.36 M 帧 → Task E 小约 **43 倍**，抗过拟合是设计重点。
- 所有 ep 都是正样本 → **AWBC 第一版不启用**。
- 仅 1 位操作员（ztm）+ 末 9 条全是长尾长轨迹 → val 必须**打散**。
- **机器人本体 / 相机在采数前有更换**（同型号 Piper + D435/D405 单体互换）—— 动作分布要 re-adapt，视觉域漂移在 SigLIP 鲁棒范围内。

数据已切分完成（脚本 `scripts/prepare_task_e_splits.py` 已跑）：
- 64 ep train / 9 ep val（seed=42 打散，val span 100% 采集周期）
- timestamp 重写成精确 1/30s（源数据 timestamps 是墙钟抖动的）
- 视频目录重命名为 `observation.images.{top_head,hand_left,hand_right}`（LeRobot v2.1 模板要求）
- 生成了 LeRobot v2.1 所需的 `meta/episodes_stats.jsonl`（`scripts/generate_episodes_stats.py`）

---

## 1. 系统预检查 & 已知阻塞

### 1.1 关键发现：**sim01 NUMA 拓扑 BIOS/硬件级故障（已绕过）**

`numactl --hardware` 显示：
```
NUMA node 0: 32 GB RAM, CPUs 0-7,32-39      ✅
NUMA node 1:  0 MB RAM, CPUs 8-15,40-47     ❌ 空插槽，无 DIMM
NUMA node 2:  0 MB RAM, CPUs 16-23,48-55    ❌ 空插槽，无 DIMM
NUMA node 3: 32 GB RAM, CPUs 24-31,56-63    ✅
```

CUDA 驱动按 PCIe 拓扑把 GPU1/GPU2 绑到 node 1/2。JAX/XLA 做 pinned memory 分配时在 0-byte 节点悄悄失败 → **inline-eval 后 SIGSEGV/SIGBUS**（Phase 1 E3/E4、Phase 2 T2 多次重现）。

**用户态修复尝试（全部失败）**：
| 防御层 | 结果 | 备注 |
|---|---|---|
| `numactl --membind=0,3 --cpunodebind=0,3` | 死 step 4000 | 同挂 |
| `numactl --interleave=0,3 --strict` | 死 step 8000 | 推迟 1 轮 eval |
| **cgroup v2 `cpuset.mems=0,3`** | 死 step 1000 | 内核级也拦不住 |
| `faulthandler.enable()` | **能捕 C 栈** | 诊断有效但不阻止；显示死在 `_run_inline_eval` 返回后 train loop resume |

**根因**：JAX/XLA/CUDA driver 的 `cudaHostAlloc(..., numa_local)` 带**显式 NUMA hint** 直接 syscall，kernel 按命令去 0-byte 节点分配 → SIGBUS。numactl 的 policy 优先级在这些 syscall 下面，cgroup cpuset.mems 也绕不过 driver 显式 hint。

**唯一可靠路径**：**只用 GPU 0 + GPU 3**（NUMA 3 / NUMA 0 健康）。GPU 1/2 仅限**无 inline-eval** 的 smoke。DIMM 补齐后可回到 4 卡全用。

**每卡一个实验**（不做跨卡 FSDP）：
- 每卡独立 JAX 进程，无跨卡 NCCL collectives
- 单卡 batch=4（冻 backbone）~21 GB 稳
- 两个健康 GPU → 每轮最多 2 个实验并行，其他排队

**DIMM 补齐后**可回到 4-GPU FSDP 单实验 batch=16 的高吞吐路线，参见 §附录 B。

### 1.2 RAM 预算（62 GiB 总）

4 个 JAX 进程 + data loader workers：
- 每 JAX 进程常驻 ~4.5 GB host
- 每 data loader worker（pi05，num_workers=2）~2.5 GB
- 总计：4 × (4.5 + 2 × 2.5) = 38 GB，剩 ~24 GB 给 OS / buffer —— **必须 `num_workers=2`，别用默认 4**
- 并发 compute_norm_stats_fast 也会吃 2–3 GB；大数据集上避免和训练同时跑

### 1.3 其它已处理的问题

| 项 | 状态 | 动作（已完成） |
|---|---|---|
| Blackwell XLA autotuner | ⚠️ SIGSEGV | `XLA_FLAGS=--xla_gpu_autotune_level=0`（start_train.sh 默认 export） |
| pi05_base GCS 下载 | ⚠️ 只有 2.7 MB/s | gf0 打包 → Volcano TOS `/transfer-shanghai` → sim01 下载（20 min 上 + 19 min 下） |
| `start_server_xla_cache.sh` | ⚠️ `unset XLA_FLAGS` 覆盖补丁 | 已改为 `${XLA_FLAGS:-"--xla_gpu_autotune_level=0"}` |
| `collection_templates.yml` 缺 dagger 模板 | ⚠️ | 已补 `task_e_dagger` |
| 原始数据 episodes.jsonl 用 `episode_id` 而非 LeRobot 标准 `episode_index` | ⚠️ | 切分脚本已统一为 LeRobot v2.1 schema |
| 原始数据 timestamp 抖动（mean 0.048 s vs fps=30 → 应 0.033 s）| ⚠️ | 切分脚本统一写成 `frame_index / 30` |
| 原始数据缺 `episodes_stats.jsonl` | ⚠️ | `scripts/generate_episodes_stats.py` 已产出 |
| `train.py` save_state 多传参 | ⚠️ crash | 删掉多余 `config.save_train_state` 参数 |
| v2 跑到 16.4k 被 OOM killer 砍 | ⚠️ 62 GB 耗尽 | 降 `num_workers` 4→2；并发实验 ≤ 4 |

---

## 2. 训练方案

**核心设计**：四路**单卡差异化实验并行**，共同约束为 **abs 动作 + 冻 PaliGemma（只训 Action Expert）**；差异化因子为 `init × 数据`。

### 2.1 为什么冻 backbone + abs

- **数据小（64 train ep）**：全量训 3.3B 参数模型极易过拟合。冻 backbone 后 trainable ≈ 800 M，train_state 从 ~65 GB 降到 ~22 GB。
- **显存（5090 32 GB/卡 单卡运行）**：冻后单卡 batch=4 稳在 ~21 GB，留 ~10 GB 给 inline eval 的 KV cache。
- **action expert 已经足够适配 Task E 的动作分布**（v2 基线 step 14k 时 MAE@1=0.0411 已接近 Task A 规模基线）。
- **abs vs delta**：v14 delta（batch=8）loss 曲线好看但 val MAE 劣于 abs 同步数；且部署时 delta 需要在每步用当前观测 state 加回去，存在误差积累。**所有差异化实验统一用 abs**。

### 2.2 冻结策略（代码落地）

pi05 子模块路径规则：
- `PaliGemma.img.*` → 视觉塔（SigLIP）
- `PaliGemma.llm.layers.*`（不带 `_1`） → PaliGemma LLM 主干
- `PaliGemma.llm.layers.*_1` → Action Expert（LLM 内的并行塔）
- 顶层 `action_in_proj / action_out_proj / time_mlp_*` → Action 投影层（不在 `PaliGemma` 下）

```python
from openpi.shared import nnx_utils
import flax.nnx as nnx

freeze_filter=nnx.All(
    nnx_utils.PathRegex(".*PaliGemma.*"),            # 冻整个 PaliGemma
    nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),     # 但放开 Action Expert 的 LLM 塔
),
# 顶层 action_in_proj / action_out_proj / time_mlp_* 不匹配 PaliGemma，天然不被冻 ✅
ema_decay=None,   # 冻 backbone 下 EMA 意义不大，省 ~6.5 GB/卡
```

### 2.3 四路差异化实验矩阵

| 实验 | config name | GPU | init | 数据 repo_id | 数据量 | 备注 |
|---|---|---|---|---|---|---|
| **v3** | `pi05_stand_box_kai0init` | 1 | kai0_mixed_1 | `Task_E/base` | 64 ep | kai0 init 的纯迁移效果 |
| **v4** | `pi05_stand_box_aug` | 2 | pi05_base | `Task_E/base_full_aug` | 256 ep | 全增强（mirror + time × mirror×time） |
| **v5** | `pi05_stand_box_kai0_aug` | 3 | kai0_mixed_1 | `Task_E/base_full_aug` | 256 ep | v3+v4 组合，期待双增益叠加 |
| **v8** | `pi05_stand_box_mirror` | 0 | pi05_base | `Task_E/base_merged` | 128 ep | v4 消融：仅 mirror，验证 time_scaling 边际 |

共同超参：`batch_size=4 / num_workers=2 / fsdp_devices=1 / num_train_steps=15000 / save_interval=2000 / keep_period=5000 / seed=42 / ema_decay=None`。

**kai0_mixed_1 init 路径**：`/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/Task_A/mixed_1/params`。这是 Task A 上用等权 Model Arithmetic 合并的 4 个 split 的权重，在同硬件平台上 Warmed up，理论上迁移到 Task E 优于 GCS 裸 pi05_base。

### 2.3.1 Phase 1 结果（2×2 差异化 2026-04-18）

v3/v4/v5/v8 最终 single-sample MAE@1 / @50：

| 排名 | exp | init | 数据 | best step | **@1** | @10 | @25 | @50 |
|---|---|---|---|---:|---:|---:|---:|---:|
| 🥇 | **v3 kai0+base** | kai0_mixed_1 | base (64) | **12000** | **0.0333** | 0.0514 | 0.0775 | 0.1164 |
| 🥈 | v5 kai0+aug | kai0_mixed_1 | full_aug (256) | 14999 | 0.0351 | 0.0552 | 0.0881 | 0.1329 |
| 🥉 | v8 pi05+mirror | pi05_base | mirror (128) | 14000 | 0.0421 | 0.0598 | 0.0876 | 0.1283 |
| 4 | v4 pi05+aug | pi05_base | full_aug | 10000 | 0.0452 | 0.0677 | 0.0978 | 0.1350 |
| —— v2 基线（pi05+base）| | | | 14000 | 0.0411 | — | — | — |

**核心结论**：
1. **init 因子主导**：kai0_mixed_1 比 pi05_base 提升 **20–26% @1**（v3 vs v4；v5 vs v8）。Task A warmed-up backbone 对 Task E 的迁移效果显著。
2. **time_scaling 负贡献**：v8 (mirror only) > v4 (full aug)，v3 (base) > v5 (aug)。time_scaling 稀释信号、在 64 ep 小数据上伤害长 horizon。
3. **kai0 init 5× 加速**：v3 在 step 2k 就达 @1=0.0427，接近 v2 step 14k 的 0.0411。

### 2.3.2 Phase 1 v3 续训（E1/E2/E3/E4 微调 trick 2026-04-18）

从 v3/12000 init 再训 15k 步，测 4 个 fine-tune 手段：

| exp | 变化 | FINAL @1 | vs v3/12000 基线 | 趋势 |
|---|---|---:|---:|---|
| 🥇 **E2** v3+**lowLR**（peak_lr 2.5e-5 → 1.25e-5） | 0.0262 | **-21.3%** | 连续 6 ckpt 单调下降 |
| 🥈 E1 v3+**EMA**（decay=0.9999） | 0.0297 | -10.8% | 单调缓慢收敛 |
| E3 v3+EMA+lowLR combo | 0.0290 (step 8k, dead) | -13% | EMA 后期起效但 GPU 1 NUMA 崩 |
| E4 v3+**long**（纯续训 default LR）| 0.0327 | -2% | 平台震荡（默认 LR 对已收敛权重是破坏）|

**核心结论**：
1. **lowLR 是最有效 fine-tune**：默认 LR 在已收敛权重上会 overshoot；半 LR 稳推。
2. **EMA 单独只有 10% 增益**，combo 中 EMA 加持 lowLR 但不显著超越。
3. **E2 的 @1=0.0262 是当前 single-sample 最强**，成为 Phase 2 起点。

### 2.3.3 Phase 2 训练端突破（进行中 2026-04-19）

目标：压 @1 < 0.025，不使用 ensemble/MA/DAgger。

| exp | 策略 | 可训参数 | 预期 @1 | GPU | 状态 |
|---|---|---|---|---|---|
| **T1-1** | vision MLP LoRA r=16（E2 init） | AE 750M + LoRA ~4.5M | -4 到 -7% | GPU 0 | 🟢 跑中 |
| **T1-2** | vision MLP LoRA r=32（E2 init） | AE 750M + LoRA ~9M | -5 到 -8% | GPU 3 | 🟢 跑中 |
| **T2** | E2 + ultra-lowLR 5e-6 + EMA (5k 步) | 750M | -2 到 -5% | ⏸️ 排队 | 待 GPU 0/3 空 |
| **T6** | kai0 + base + lowLR + EMA 从头 15k | 750M | -3 到 -7% | ⏸️ 排队 | 同上 |
| ~~T1 vision 全解冻 FSDP~~ | 1.15B trainable | -5 到 -12% | — | ❌ 放弃 | 32 GB 装不下 |

**关键设计**：
- **LoRA 只加在 MLP 两个 Dense**（不加 attention）。rank=16 / 32 覆盖 27 个 SigLIP block，每 block 2 条 LoRA 分支。
- **freeze_filter**: `All(PaliGemma, Not(llm.*_1), Not(lora_[ab]))` → 冻原 PaliGemma，放 AE + LoRA。
- **init from E2/14999 params**：LoRA 权重全零初始化（接近 identity），接力 E2 继续训。

**决策门**：
- LoRA r=16 step 4k @1 < 0.025 → 继续跑完
- r=16 无改善但 r=32 有 → rank 有效，r=32 跑完
- 两者都 < 2% 增益 → vision 非瓶颈，停在 E2，考虑 DAgger/真机

### 2.4 数据增强 pipeline

原始 64 ep → 通过两套变换各自扩充一倍，再组合：

| 数据集 | 生成方式 | ep 数 |
|---|---|---|
| `base` | 原始切分 train 部分 | 64 |
| `base_mirror` | 空间镜像（L/R 臂数据互换 + 视频水平翻转） | 64 |
| `base_merged` | `base ∪ base_mirror` | **128** |
| `base_time2` | 时间降采样（每 2 帧取 1，仿真 2× 速度） | 128 |
| `base_full_aug` | `base ∪ base_mirror ∪ base_time2 ∪ base_mirror×time` | **256** |

**空间镜像**（`train_deploy_alignment/space_mirroring.py`）：
- 左右臂 state/action 维度 swap（0..6 ↔ 7..13）
- 相机 `hand_left` ↔ `hand_right` 视频互换
- 所有视频 horizontal flip（沿 W 轴）
- gripper 维度（6, 13）只 swap 不 flip

**时间缩放**（`train_deploy_alignment/time_scaling.py`）：
- `extraction_factor=2`：从 30 fps 序列按步长 2 抽样成两条 15 fps 序列
- timestamp 仍按 fps=30 写入（保持 norm_stats 兼容）

### 2.5 Inline eval 基础设施（in-process，共享 `train_state.params`）

**问题**：subprocess eval 会在同 GPU 上复制一份 ~18 GB params → 36 GB > 32 GB OOM；另启 GPU 则浪费一张 5090；训前单独 eval 不能实时反馈。

**解法**：在 `train.py` 主循环内用 `flax.nnx.merge(graphdef, state)` 构造 Policy，**params 指针共享同一批 `jax.Array`**（已用 id() 验证），额外显存 ~3 GB（只有推理激活），峰值 ~21 GB 稳在 32 GB 内。

关键实现：

```python
# kai0/scripts/train.py
def _build_eval_policy(train_state, config, data_config):
    model = nnx.merge(train_state.model_def, train_state.params)  # 共享 Array 指针
    return _policy_config.PolicyFromModel(model, config, data_config, ...)

def _run_inline_eval(train_state, config, data_config, step, mesh):
    val_root = os.environ.get("INLINE_EVAL_VAL_ROOT")
    if not val_root:
        return
    samples = _load_val_data(val_root, n_frames)    # 缓存一次后 O(1)
    policy = _build_eval_policy(train_state, config, data_config)
    # ... compute MAE@{1,10,25,50} ...
    wandb.log({"val/mae_1": ..., "val/mae_10": ..., "val/mae_25": ..., "val/mae_50": ...}, step=step)

# 主循环 save_state 之后：
if os.environ.get("INLINE_EVAL_VAL_ROOT"):
    _run_inline_eval(train_state, config, data_loader.data_config(), step, mesh)
```

每 2k 步额外耗时 ~30–60 s（9 val ep × 20 帧 × 推理）。

### 2.6 启动（`scripts/start_train.sh` 统一入口）

```bash
# Usage: start_train.sh <config> <exp_name> <gpu> [--bs N --steps N --seed N --no-eval]
cd /data1/tim/workspace/deepdive_kai0

./scripts/start_train.sh pi05_stand_box_mirror      v8_pi05_mirror 0  # v8 on GPU0
./scripts/start_train.sh pi05_stand_box_kai0init    v3_kai0_base   1  # v3 on GPU1
./scripts/start_train.sh pi05_stand_box_aug         v4_pi05_aug    2  # v4 on GPU2
./scripts/start_train.sh pi05_stand_box_kai0_aug    v5_kai0_aug    3  # v5 on GPU3
```

`start_train.sh` 自动设置的环境变量：
- `CUDA_VISIBLE_DEVICES=<gpu>`
- `XLA_FLAGS="--xla_gpu_autotune_level=0"`（Blackwell 必需）
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 + PREALLOCATE=false`
- `OPENPI_DATA_HOME=<project_root>/openpi_cache`
- `WANDB_MODE=offline`
- **inline eval**: `INLINE_EVAL_VAL_ROOT=<Task_E/val>, INLINE_EVAL_N_FRAMES=20, INLINE_EVAL_EVERY=1`（`--no-eval` 关闭）

启动后 `nohup` + `disown` 脱离 shell，日志到 `logs/train_<exp_name>.log`。

### 2.7 时长与步数

冻 backbone + batch=4 单卡 5090：约 **2.2 steps/s**（base 数据），aug 数据 I/O 更重约 **1.8 steps/s**：

| 步数 | base 单卡 | aug 单卡 |
|---:|---|---|
| 首次编译 | 3–5 min | 3–5 min |
| 10 k | 1.5 h | 1.8 h |
| **15 k** | **1.9 h** | **2.3 h** |

**早停**：val MAE@1 连续 2 个 ckpt（≥4k 步）不降则停。

### 2.8 训练期约束

- **不得并跑** `start_autonomy.sh` / 任何推理服务：显存 + CUDA 冲突
- 并发 eval/compute_norm_stats_fast 进程会吃 RAM —— 必要时先停
- W&B offline：训完后 `wandb sync wandb/offline-run-<id>`

---

## 3. 离线评测方案

### 3.1 主指标：Val action MAE

两条通路：

**A. Inline（训练期实时）** —— `train.py` 内 `_run_inline_eval`（§2.5）
- 每 save_interval（2k 步）一次
- 9 val ep × 20 帧 query
- 写 wandb keys：`val/mae_1`, `val/mae_10`, `val/mae_25`, `val/mae_50`
- 用于训练期 live 监测，**不**落盘单独 json（省 ckpt 目录污染）

**B. Archive（离线）** —— `scripts/eval_val_action_mse.py` + `scripts/auto_eval_v2.sh` 风格守护
- 跑完训练后对 `keep_period`（5k 步）留下的 ckpt 做全量评测（n_sample_frames=200）
- 落盘到 `<ckpt>/eval_val.json`，然后由 auto_eval 类脚本归档到 `logs/eval_history_<run>/step_<N>.json`（orbax 轮换会吃掉 ckpt 下的 json，必须外部归档）

**放行阈值**（建议，来自 Task A 基线经验）：

| 指标 | 用途 | 阈值 |
|---|---|---|
| 1-step MAE（关节）| 主 | < 0.02 rad |
| 10-step MAE | 辅 | < 0.05 rad |
| 50-step MAE | 辅 | < 0.12 rad |
| gripper MAE (dim 6, 13) | 主 | < 0.005 m |

**v2 基线参考**：step 14k MAE@1=0.0411（还差 2× 到阈值，v3/v4/v5/v8 目标往下压）。

### 3.2 JAX ↔ 推理服务一致性

```bash
uv run scripts/test_inference_parity.py \
  --config <best_config> \
  --checkpoint kai0/checkpoints/<best_config>/<best_exp>/<best_step>
```
阈值 `max_abs_diff < 1e-3`。

### 3.3 推理延迟 & 质量

```bash
uv run python scripts/test_inference_server.py --check all \
  --config <best_config> \
  --checkpoint kai0/checkpoints/<best_config>/<best_exp>/<best_step>
```
sim01 单卡 5090 目标：chunk(50 步) p50 < 40 ms，p99 < 70 ms。

### 3.4 Rerun 定性回放

对 9 条 val ep 跑预测 vs GT 动作叠加可视化（`docs/deployment/inference_visualization.md`）。挑 MAE 靠前 + 靠后各 1 条对比。

---

## 4. 真机测试方案（sim01）

### 4.1 切换前检查

```bash
pkill -f "scripts/train.py" || true             # 训练停
# 相机和 Piper 换过硬件，calibration 必须重做：
./piper_tools/calibrate_can_mapping.py
# 更新 config/calibration.yml 里的 T_world_cam*、T_link6_cam*
./scripts/test_hardware.py                       # 相机 + CAN + 回零
./scripts/test_cameras.py                        # 三路 RealSense 序列号（新 unit 要核对）
```

### 4.2 启动

```bash
# Terminal A — 策略服务（GPU0）
cd /data1/tim/workspace/deepdive_kai0
CUDA_VISIBLE_DEVICES=0 \
  XLA_FLAGS="--xla_gpu_autotune_level=0" \
  ./scripts/start_server_xla_cache.sh \
    --config pi05_stand_box_normal \
    --checkpoint kai0/checkpoints/pi05_stand_box_normal/stand_box_v1/<best>

# Terminal B — 自主栈（先空跑）
./scripts/start_autonomy.sh --mode ros2
# Rerun 下看 2–3 个 chunk 方向合理后再：
./scripts/toggle_execute.sh on
```

### 4.3 评测矩阵

| 维度 | 取值 |
|---|---|
| 倒向 | 前 / 后 / 左 / 右（4）|
| 桌面位置 | 近 / 中 / 远（3）|
| 每组重复 | 3 |
| **主矩阵合计** | **36 trials** |
| **OOD 补充** | 6 trials（换尺寸 / 花色 / 轻微遮挡） |

### 4.4 成功判定

- **成功**：松手后底面朝下稳立 ≥ 5 s，双臂无碰撞、无急停，单次 ≤ 90 s。
- **部分成功**：立起后自行倒塌 / 需轻扶。
- **失败**：卡死 / 碰撞急停 / 动作发散 / 箱子被推飞。
- 每 trial 记录：初始姿态、耗时、类别、失败模式（抓不住 / 错侧 / 力度不足 / 抖振 / 其它）。

### 4.5 指标

- **主**：整体成功率 + 分层成功率（倒向 × 位置），95 % Wilson CI。
- **辅**：完成时间、人工干预次数、chunk 延迟（均值 / p99）、ROS2 时间戳对齐。

### 4.6 安全

- Piper 硬件急停按钮 + `./scripts/toggle_execute.sh off` 软停。
- 超 90 s 强制人工终止，计失败。
- **失败 trial 存 DAgger**：`collection_templates.yml` 的 `task_e_dagger`（已添加）+ `./scripts/start_data_collect.sh`。

---

## 5. 迭代路线

| 版本 | 状态 | 触发 | 动作 |
|---|---|---|---|
| **v2** `stand_box_v2_pi05base` | ✅ 完成 step 14k MAE@1=0.0411 | — | pi05_base + base + abs + freeze 基线 |
| **v3/v4/v5/v8** 并行 | 🟢 运行中（2026-04-18）| v2 基线 | 四卡差异化实验（见 §2.3）|
| **v2.5** 更多 step / 更小 LR | v3–v8 所有 MAE 仍下降时 | v3–v8 收敛 | best-of-4 挑 init+data，续训 15k→30k |
| **v9** vision 放开 | v2.5 val MAE 明显劣于 Task A 基线 | v2.5 结果 | filter 改成只冻 `.*PaliGemma\\.llm.*`（放开 SigLIP） |
| **v10** DAgger | 真机总体 <60% 或单倒向 <40% | 真机反馈 | 采 20–30 条失败姿态 → 并入 base 重训 best |
| **v11** AWBC + MA | 积累正/负样本后 | DAgger 覆盖后 | 阶段优势标注 → `pi05_stand_box_awbc` → MA 合并 |

---

## 6. 待办清单

### 6.1 已完成 ✅

1. ~~修 pi05_base 本地缓存~~ → 改用 TOS 传输（gf0 打包 → `/transfer-shanghai` → sim01）
2. ~~`scripts/prepare_task_e_splits.py`~~（64 train / 9 val，timestamp fix，video dir rename）
3. ~~`scripts/generate_episodes_stats.py`~~（LeRobot v2.1 必需）
4. ~~`scripts/compute_delta_norm_stats_fast.py`~~ 快速产出 delta norm_stats
5. ~~`scripts/eval_val_action_mse.py`~~（独立评测）
6. ~~`scripts/auto_eval_v2.sh`~~（keep_period ckpt 归档守护）
7. ~~`scripts/start_train.sh`~~（统一训练入口 + inline eval env vars）
8. ~~`train.py` inline eval（nnx.merge 共享 params）~~
9. ~~config.py 加 v2/v3/v4/v5/v8 + freeze_filter 模板~~
10. ~~`start_server_xla_cache.sh` XLA_FLAGS 修复~~
11. ~~`collection_templates.yml` 补 `task_e_dagger`~~
12. ~~诊断 sim01 NUMA 故障~~（记入 memory）
13. ~~v2 基线训练完成~~（step 14k MAE@1=0.0411）
14. ~~空间镜像 + 时间缩放数据集生成~~（base_mirror / base_merged / base_time2 / base_full_aug）

### 6.2 待做（按优先级）

1. **P0（监控）** 每 2k 步检查 v3/v4/v5/v8 的 inline eval `val/mae_1` 曲线，早停劣势实验（连续 2 个 ckpt 不降）
2. **P0（归档）** 跑完后对 keep_period ckpt（5k/10k/15k）做 auto_eval 归档到 `logs/eval_history_<run>/`
3. **P1（best ckpt 选择）** v3–v8 + v2 对比，挑 MAE@1 最低 ckpt 做真机测试
4. **P1（真机）** 相机/Piper 换件后**重做 hand-eye 标定**
5. **P1（硬件）** socket 1 / socket 2 补 DIMM（长期解，见附录 B）
6. **P2** v2.5（挑 best 续训）/ v9（vision 放开）/ v10（DAgger）/ v11（AWBC+MA）

---

## 附录 A：关键文件索引

| 用途 | 路径 |
|---|---|
| 训练配置注册 | `kai0/src/openpi/training/config.py` |
| 训练主循环（含 inline eval） | `kai0/scripts/train.py` |
| 训练启动入口 | `scripts/start_train.sh` |
| 独立评测脚本 | `scripts/eval_val_action_mse.py` |
| Ckpt 自动归档守护 | `scripts/auto_eval_v2.sh`（模板） |
| delta norm_stats（快）| `scripts/compute_delta_norm_stats_fast.py` |
| 数据切分 / 修复 | `scripts/prepare_task_e_splits.py`, `scripts/generate_episodes_stats.py` |
| 空间镜像 | `kai0/train_deploy_alignment/space_mirroring.py` |
| 时间缩放 | `kai0/train_deploy_alignment/time_scaling.py` |
| 模型 freeze filter 工具 | `kai0/src/openpi/models/pi0_config.py`（`get_freeze_filter`）|
| Agilex policy 变换 | `kai0/src/openpi/policies/agilex_policy.py` |
| 本地数据工作副本 | `kai0/data/Task_E/{base,base_mirror,base_merged,base_time2,base_full_aug,val}/` |
| 原始数据（只读）| `/data1/DATA_IMP/KAI0/Task_E_2026-04-17/base/` |
| pi05_base init | `openpi_cache/openpi-assets/checkpoints/pi05_base/params/` |
| kai0_mixed_1 init | `kai0/checkpoints/Task_A/mixed_1/params/` |
| Ckpt 输出 | `kai0/checkpoints/<config>/<exp>/<step>/` |
| 手眼标定（换件后必改）| `config/calibration.yml` |

## 附录 B：sim01 NUMA 故障参考

已存入 Claude 持久 memory（`sim01_numa_broken.md`）。BIOS / DIMM 层面问题，无用户态绕法。

**现状（2026-04-18）**：通过"单卡一个实验、四卡跑四个实验"绕过 NCCL 崩溃，吞吐按实验数补偿。
**DIMM 补齐路线**：socket 1 + socket 2 各补 1 根 DIMM（≥8 GB 任意频率），`numactl --hardware` 确认 4 个 node 都有 MemTotal > 0 后可回到 4-GPU FSDP batch=16 单实验高吞吐模式。

## 附录 C：后续改进路线候选（深度调研 2026-04-19）

按 **EV / 成本** 排序，从 E2 = 0.0262 baseline 继续压精度的方案（**不用 ensemble / MA / DAgger**）：

### Tier 1（当前 Pipeline B）—— 立即可做

| ID | 手段 | 预期 Δ@1 | 代码 | 训练 | 状态 |
|---|---|---|---|---|---|
| **T7** | fixed-init LoRA r=16 + base_merged + 25k | -15~25% | 已完成 | 4.5h | 🟢 进行中 |
| **T8** | **DoRA**（weight-decomposed LoRA）on MLP | -3~5% 额外 | 1-2h | 3h | ⏸️ T7 后 |
| **T9** | **MHA LoRA**（给 Q/K/V/O 加 rank=16） | -2~5% 额外 | 2-3h | 3h | ⏸️ 可选 |
| T10 | **AdamW weight_decay 1e-4 → 0.01** | -1~3% | 1 行 | 3h | 可叠加 |

### Tier 2（需更大算力）

| ID | 手段 | 预期 Δ@1 | 成本 |
|---|---|---|---|
| T11 | **远程 gf0/gf1 (8×A100 80GB)** 全 vision 解冻 + 25k | -5~12% | 数据传输 1h + 5h 训 |
| T12 | **Horizon-weighted loss**（@1-@10 加权 1.5×）| -1~3% | 5 行 + 3h |

### Tier 3（边际 / 长期）

- T13 **rsLoRA**：alpha=r·√r（r≥64 时显优，r=16 无差）
- T14 **LoRA dropout**：p=0.1 在 LoRA 分支
- T15 **Curriculum learning**：eps 按难度排序
- T16 **多任务 co-train** with Task A 数据（kai0_mixed_1 已部分吸收）

### Pipeline B 决策树

```
T7 完成 →
  @1 < 0.024 → 停，真机验证
  0.024 ≤ @1 < 0.028 → T8 (DoRA) 叠加
  @1 ≥ 0.028 → T9 (MHA LoRA) + T10 (wd) 联合
所有 Tier 1 跑完仍 > 0.020 → T11 远程全解冻（最后大杀器）
```

### 核心认知（2026-04-19 发现）

1. **LoRA init 至关重要**：w_b = zeros 是 identity start 的前提。normal init 导致 step 0-4k 扰动 E2 已收敛权重，损失 ~5% 潜力。
2. **Mirror aug 的激活条件**：vision 冻结时，mirror 对 SigLIP 没有信号（输入改但 feature 不变），AE 学的只是 action swap；LoRA 解冻后 mirror 才真正帮 vision 学左右对称。
3. **rank 不是瓶颈**（Task E 场景）：r=16 vs r=32 在 T1-1/T1-2 同步到 0.0001 差异。说明 Task A→Task E 的视觉 delta 用 r=16 就够，**升 rank 不如加 MHA coverage 或换 DoRA**。
4. **单卡 5090 32 GB 是硬墙**：全 vision 解冻（1.15B trainable opt state ~9 GB + params ~6.6 GB + 其它 ~10 GB = 峰值 > 32 GB）OOM 不可绕；LoRA / DoRA / 部分解冻是唯一软件路径。
