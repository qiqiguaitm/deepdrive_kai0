# Realtime-VLA 三论文对 deepdive_kai0 部署架构的优化分析与实施路线 (选项 X)

> 生成时间: 2026-05-19, 最后更新: v0.6 (2026-05-19)
> 决策状态: **已选定选项 X (双推理架构并存, 新 ckpt 走 PyTorch+Triton)**
> 输入材料: realtime-vla / realtime-vla-flash / realtime-vla-v2 三仓 + docs
> 基线: deepdive_kai0 sim01 部署 (JAX, RTC 已实装)

---

## 目录

- [0. 决策摘要](#0-决策摘要)
- [1. 上下文与基线](#1-上下文与基线)
  - [1.1 deepdive_kai0 现状](#11-deepdive_kai0-现状)
  - [1.2 三论文核心思想速览](#12-三论文核心思想速览)
  - [1.3 关键洞察](#13-关键洞察)
  - [1.4 选项 X 决策依据](#14-选项-x-决策依据)
- [2. 优化项目排序与复杂度评估](#2-优化项目排序与复杂度评估)
  - [2.1 成本维度 (agent 时代 4 维)](#21-成本维度-agent-时代-4-维)
  - [2.2 8 项优化排名表](#22-8-项优化排名表)
  - [2.3 兼容性矩阵](#23-兼容性矩阵)
- [3. 实施路线图 — 5 阶段](#3-实施路线图--5-阶段)
  - [3.1 总体阶段图](#31-总体阶段图)
  - [3.1.1 备选路径 PI0Pytorch fix 清单](#311-已知前置工程清单)
  - [3.1.2 子任务清单](#312-子任务清单-阶段-0-1)
  - [3.2 阶段 1 — 短期热身 (1 周)](#32-阶段-1--短期热身-1-周)
  - [3.3 阶段 2 — 任务速度主线 (1-2 周)](#33-阶段-2--任务速度主线-1-2-周)
  - [3.4 阶段 3 — 选项 X 落地](#34-阶段-3--选项-x-落地)
    - [3.4.5 TensorRT 路径回顾](#345-tensorrt-路径回顾-2026-05-20)
  - [3.5 阶段 4 — 任务质量 (3-6 周)](#35-阶段-4--任务质量-3-6-周)
  - [3.6 阶段 5 — 推理极致 (可选)](#36-阶段-5--推理极致-可选)
- [4. 真机测试方案](#4-真机测试方案)
  - [4.1 测试 1: sim01 模型实际推理延迟 ✅](#41-测试-1-sim01-模型实际推理延迟--完成-2026-05-20)
    - [4.1.1 测量方法](#411-测量方法)
    - [4.1.2 实验配置](#412-实验配置)
    - [4.1.3 原始数据样本](#413-原始数据样本-前-5-条)
    - [4.1.4 实测分位数](#414-实测分位数)
    - [4.1.5 分布直方图](#415-分布直方图-cleaned-n1299)
    - [4.1.6 与 V1 路径对比](#416-与-v1-路径对比)
    - [4.1.7 决策映射](#417-决策映射)
    - [4.1.8 测量复跑](#418-测量复跑)
  - [4.2 测试 2: Piper 关节 t_motion 滞后](#42-测试-2-piper-关节-t_motion-滞后)
- [5. Fallback 方案 (选项 Y)](#5-fallback-方案-选项-y)
- [6. V1 Triton 推理优化实施日志 (2026-05-20)](#6-v1-triton-推理优化实施日志-2026-05-20)
  - [6.1 总进度表 (Step 0-9)](#61-总进度表-step-0-9)
  - [6.2 Sweep 方法论](#62-sweep-方法论)
  - [6.3 PyTorch baseline 路径 (Step 0-4)](#63-pytorch-baseline-路径-step-0-4)
  - [6.4 V1 Triton 路径 (Step 5-9)](#64-v1-triton-路径-step-5-9)
  - [6.5 累积分析与硬件下限](#65-累积分析与硬件下限)
  - [6.6 待实施 (Step 11+, 结构性优化)](#66-待实施-step-11-结构性优化)
- [7. Layer B 系统级优化 plan (next phase)](#7-layer-b-系统级优化-plan-next-phase)
  - [7.1 范围 + 约束](#71-范围--约束)
  - [7.2 B4 V1 serve 包装](#72-b4-v1-serve-包装-3-5-天-主线)
  - [7.3 B1 全链路 latency profiling](#73-b1-全链路-latency-profiling-2-3-天)
  - [7.4 B2 Preprocess 全 GPU 化](#74-b2-preprocess-全-gpu-化-1-3-天-数据驱动)
  - [7.5 时序 + 风险](#75-时序--风险)
- [8. 修订历史](#8-修订历史)

---

## 0. 决策摘要

### 已选方案: 选项 X — 双推理架构并存

| 维度 | 决策内容 |
|---|---|
| 训练侧 | JAX (现状) + PyTorch (新增, 用 `train_pytorch.py`) 并存, 新 fine-tune 按需选 |
| 推理侧 | JAX backend (`serve_policy.py:8000`) + PyTorch backend (新增 `serve_policy_pytorch.py:8001`), sidecar `framework` 字段分发 |
| 旧 JAX ckpt | 不迁, 走 JAX 推理 + #6 浅层优化 (1.5-2× baseline) |
| 新 ckpt | 走 V1 Triton 路径 (复用 + 5090 重 autotune), **实测 P50=32 ms / 8.0× vs eager**, 见 §6 |
| 痛点状态 | 双实现痛点 (数值对齐 / 双实现同步 / bug 翻倍 定位) **不存在**; 剩余痛点是"两个独立项目"成本 |

### 5 阶段实施一览

| 阶段 | 目标 | 主要项目 | 期望收益 | 触发条件 |
|:---:|---|---|---|---|
| **1** | 短期热身 | #8 延迟标定 + #6 JAX 浅层 + 真机测试 1/2 | 抖动 -30-50%, 推理 1.5-2× | 立即可启动 |
| **2** | 任务速度 | #5 timeaxis_smooth QP | 任务耗时 1.5-2× | 阶段 1 #8 完成 |
| **3** | 选项 X 落地 | PyTorch 训等效 POC + V1 Triton 推理 (✅ 完成) + sidecar | **P50=32 ms (8.0×)** ✅ | 阶段 1 真机测试 1 完成 |
| **4** | 任务质量 | #4 速度自适应学习 (油门数采 + 回归 head) | 精细阶段成功率 ↑, 耗时 -50% | 阶段 2 QP 落地 |
| **5** | 推理极致 (可选) | #3 Flash 推测推理 + #7 客户端 MPC | 复合 10-20× / 跟踪误差 -50% | 阶段 3 完成; #7 需 t_motion > 50ms |

### 文档使用方法

- 按 §3 各阶段文档逐步实施, 每阶段都给出具体改动文件 + 步骤 + 验证标准
- §2 排序表是排序与复杂度参考, 阶段顺序并非简单按"#编号"
- §4 是真机测试脚本说明, 阶段 1 用
- §5 是 fallback (Y 选项) 触发条件 + 切换方法
- 任何决策点变更, 在 §6 修订历史里追加版本

---

## 1. 上下文与基线

### 1.1 deepdive_kai0 现状

#### 推理栈
- **框架**: JAX + Flax + Orbax, sim01 上 `serve_policy.py` WebSocket 服务 (`:8000`)
- **模型**: π₀.₅, action_horizon=50, joint_dim=14 (双臂)
- **ROS2 节点**: `policy_inference_node.py` 拉取相机 + 关节, 走 WebSocket 推理, 返回 chunk
- **双时钟**:
  - publish_rate = 30 Hz
  - inference_rate = 3.0 Hz (≈ 333 ms/cycle)
  - chunk_size = 50, latency_k = 8 (头部裁剪步)
- **RTC 已实装**:
  - 模型层: `Pi0RTC.sample_actions` 在去噪迭代内对 `[d, exec_h)` 区间引导 (`rtc_execute_horizon=16`, `max_guidance_weight=0.5`)
  - 运行时层: `StreamActionBuffer.integrate_new_chunk` latency 裁剪 + 8 步线性 overlap 平滑
- **XLA**: 已有 `start_server_xla_cache.sh` 预编译缓存

#### 硬件
- **部署机 sim01**: Ubuntu 24.04, **2× RTX 5090 32GB** (Blackwell sm_120)
- **训练机 gf0**: 8× A100 80GB; uc01-03 / js01-04 GPU 集群
- **机械臂**: 双 Piper, CAN slave 控制
- **相机**: 1× D435 (top) + 2× D405 (左右腕), 三路 RGB @ 30fps (depth 关闭)

#### 现状弱点
- RTC `latency_k=8` 是手调常量, 没数据校准
- 系统延迟未量化 (相机曝光 / 读出 / proprio / motor 滞后均无测量)
- 速度全局一致 (30Hz 一档), 折叠/对位等精细阶段无降速逻辑
- chunk 边界靠 8 步线性平滑兜底, 物理量 (速度/加速度) 无约束

#### 已存在的 PyTorch 训练侧 (影响 §1.4 决策)
deepdive_kai0 **已在维护双栈**:
- `kai0/scripts/train_pytorch.py` (646 行, DDP/torchrun 支持)
- `kai0/src/openpi/models_pytorch/`:
  - `pi0_pytorch.py` (646 行): `PI0Pytorch` 类支持 pi05 (`PI0Pytorch.pi05` 字段, train_pytorch.py:410 读 `config.model.pi05`); 含 `AdvantageEstimator`
  - `gemma_pytorch.py` (281 行)
  - `preprocessing_pytorch.py` (358 行)
- 实际使用: `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD` config 走 PyTorch 训练 advantage/stage 模型
- **主流 pi05 fine-tune 仍走 JAX `scripts/train.py`** — 选项 X 阶段 3.1 需 POC 验证等效性

### 1.2 三论文核心思想速览

| 论文 | 核心技术 | 论文报告收益 | 与 deepdive_kai0 关系 |
|---|---|---|---|
| **V1** *Running VLAs at Real-time Speed* | PyTorch + Triton 内核全栈优化 (CUDA Graph + GEMM 分块 + QKV/RMSNorm/RoPE fusion + Stream 并发) | 105ms → 27.3ms 双视角 @ 4090 | 阶段 3.2 推理 serve 直接套用 |
| **Flash** *Realtime-VLA FLASH* | 110M draft + flow-matching 端点重建并行验证 + 阶段感知 fallback | 3.04× 加速, 成功率 -0.3pp | 阶段 5 #3, 与阶段 3 复合 |
| **V2** *Learning to Run VLAs Fast, Smooth, and Accurate* | 4 类延迟标定补偿 + 服务端时间轴 QP + 客户端 MPC + 速度自适应学习 | 任务耗时 2-4×, 成功率保持 | 阶段 1 #8, 阶段 2 #5, 阶段 4 #4, 阶段 5 #7 |

### 1.3 关键洞察

#### A. 333ms 是 timer 节流, 不是模型上限
`inference_rate=3.0` 是 ROS2 timer 参数, 推理线程每 333ms tick 一次。模型实际推理时间估计 100-200ms (5090 上, 50 步去噪 × 2-4ms/step + VLM prefill ~30ms)。

**影响**:
- 加速模型本身不能让推理"看起来更快" — timer 还是 333ms 一发
- 但加速模型可以**让你提高 inference_rate** (例如 6-10 Hz), 让 RTC 收敛更稳
- 真机测试 1 (§4.1) 是后续推理优化路线决策的关键依据

#### B. deepdive_kai0 双栈已存在
PyTorch 训练侧 (`train_pytorch.py` + `models_pytorch/`) 已在维护, advantage / stage classifier 已在用。"双栈维护"对 deepdive_kai0 不是 0→1 的新增, 而是 0.5→1 的扩张。

#### C. 双推理架构下"双实现痛点"消失
关键认识: 7 个双栈痛点中, 第 1-3 项 (数值对齐 / 双实现同步 / bug 定位翻倍) 都是"**同一模型两份实现要同步**"的成本。

如果每个 ckpt 只活在一个框架里 (JAX ckpt → JAX 推理, PyTorch ckpt → PyTorch 推理), 这部分成本完全没有。剩余痛点 4-7 (sidecar 扩展 / 工具链分裂 / AI 上下文翻倍 / upstream 同步) 是"维护两个独立项目"成本, 数量级低于"双实现"。

| # | 痛点 | 双推理架构下 |
|:---:|---|---|
| 1 | 数值对齐 | **消失** |
| 2 | 架构改动双实现 | **大部分消失** |
| 3 | Bug 定位翻倍 | **消失** |
| 4 | sidecar 扩展 | 保留 (加 framework 字段) |
| 5 | 工具链分裂 | 弱化 (仅交叉对比时显现) |
| 6 | AI/人上下文翻倍 | 弱化 (per-ckpt 上下文固定) |
| 7 | upstream openpi 同步 | 可选 (哪边活跃跟哪边) |

#### D. agent 时代成本评估标准变化
不再用"人·周"度量, 改用 4 维任务本征复杂度。代码量大但纯逻辑的项 (V1 全栈端口) AI 可加速; 需要真机/数据迭代验证的项 (Flash 阈值调参、速度自适应数采、MPC 硬件辨识) 卡物理时间。详见 §2.1。

### 1.4 选项 X 决策依据

#### 用户优先级 (Q1+Q5)
- Q1: 探索性优化, 不指向单一痛点
- Q5: 推理速度 + 任务速度都要, **任务速度优先**

#### 同架构选项对比 (双推理架构假设 + 忽略 100+ JAX ckpt 迁移)

| 选项 | 训练 | 推理 | 旧 ckpt | 新 ckpt | 选定状态 |
|---|---|---|---|---|:---:|
| **X** | JAX (现状) + PyTorch (新增) | JAX backend + PyTorch+Triton backend | 1.5-2× (#6) | **5-10×** | ✅ |
| Y | JAX only | JAX + JAX→ONNX→TRT (按 ckpt 选) | 1.5-2× / 3-5× | 3-5× | Fallback |
| Z | JAX + PyTorch 都活跃 | 三 backend | 自选 | 自选 | 过度设计, 不选 |

#### 为什么选 X
1. **旧 ckpt 不迁** — 走 JAX backend 拿 1.5-2× 足够日常
2. **新 ckpt 拿满 5-10×** — 走 PyTorch+Triton 路径 A
3. **`train_pytorch.py` + `PI0Pytorch` 代码层已支持 pi05**, 非从零搭
4. **避开 ONNX 导 flow loop 的技术风险点** (Y 的主要技术风险)
5. **双实现痛点 1-3 不存在** (双推理架构使然)

#### Y 作为 fallback (§5)
若阶段 3.1 PyTorch 训练等效性 POC 失败, 退回 Y。

---

## 2. 优化项目排序与复杂度评估

### 2.1 成本维度 (agent 时代 4 维)

| 维度 | 含义 | AI 可压缩? |
|---|---|:---:|
| **逻辑复杂度** | 算法/数学/工程难度 (代码量 + 框架端口规模) | ✅ 大幅 |
| **决策密度** | 需要人工拍板的关键架构 + 阈值选择数量 | △ 部分 |
| **实验迭代周期** | 真机/数据迭代验证的轮次 × 单轮耗时 | ❌ 卡物理时间 |
| **架构持续耦合** | 长期维护成本 + 是否锁死未来选项 | ❌ 常量 |

**关键认识**: agent 时代真正的瓶颈是"需要真机/数据迭代验证的轮次", 不是代码量。

### 2.2 8 项优化排名表

按期望收益 (gain × 兑现概率) 排序:

| 排名 | 优化项 | 来源 | 期望收益 | 逻辑复杂度 | 决策密度 | 实验迭代 | 架构耦合 | 阶段 |
|:---:|---|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **#1** | V1 PyTorch+Triton 全栈端口 | V1 | 推理 333ms → 30-60ms (**5-10×**) | 中 | 中 | 高 | 中 (X 下) | 3.2 |
| **#2** | V1 JAX→ONNX→TensorRT 半端口 | V1 思想 | 推理 60-100ms (**3-5×**) | 中 | 低 | 中 | 低-中 | Fallback Y |
| **#3** | Flash 草稿 + 推测推理 | Flash | 推理 **2-3×** (与 #1/#2 复合) | 高 | 高 | 高 | 中-高 | 5 |
| **#4** | V2 速度自适应学习 | V2 §4.4 | 关键阶段任务耗时 **-50%, 成功率 +10-20%** | 中 | 中 | **极高** (数采) | 低 | 4 |
| **#5** | V2 时间轴 QP 重参数化 | V2 §4.3.1 | 任务耗时 **1.5-2×** | 低 | 低 | 低 | 低 | 2 |
| **#6** | JAX/XLA 推理微优化 | V1 思想 | 推理 **1.5-2×** (333→150-200ms) | 低 | 低 | 低 | 极低 | 1 |
| **#7** | V2 客户端 MPC + 1 阶滞后辨识 | V2 §4.3.2 | 跟踪误差 -50% (若 t_motion > 50ms) | 中 | 中 | 高 | 中 | 5 (条件) |
| **#8** | V2 延迟标定 + 感知对齐 | V2 §4.2 | 抖动 -30-50%, 为 #5/#7 提供准数 | 极低 | 极低 | 低 | 极低 | 1 |

#### Pareto 分布

| 象限 | 优化项 |
|---|---|
| 高收益 × 低复杂度 (最佳 ROI) | #5 (QP), #2 (Y fallback) |
| 高收益 × 高复杂度 (值得投入) | #1 (X 阶段 3.2), #3 (Flash), #4 (速度自适应) |
| 中收益 × 极低复杂度 (首发热身) | #6 (浅层 JAX), #8 (延迟标定) |
| 中收益 × 高复杂度 (条件投入) | #7 (MPC, 需 #8 数据验证) |

### 2.3 兼容性矩阵

| 优化项 | RTC | sidecar | 训练管线 | ROS2 节点 |
|---|:---:|:---:|:---:|:---:|
| #1 V1 全栈端口 | 不冲突 (算法层) | 推理 sidecar 加 framework 字段 | 训练侧选 X 时需切 PyTorch | 服务端新增 `serve_policy_pytorch.py` |
| #2 V1 TRT 半端口 (Y) | 不冲突 | sidecar 加 ONNX/engine 路径 | 训练 JAX 不变 | 服务端逻辑微调 |
| #3 Flash | 与 RTC 重叠 (都在去噪过程) | 需 draft sidecar | 加 draft 训练 job | 改 inference 调用图 |
| #4 速度自适应 | 通过 dt 联动 | 需新 head 权重 | 加 head 训练 | 改 teleop + 推理后处理 |
| #5 时间轴 QP | 正交 | 不影响 | 不影响 | WebSocket 后处理 + 客户端定时器改造 |
| #6 JAX 浅层 | 不影响 | 不影响 | 不影响 | 不影响 |
| #7 MPC | 不影响 | 不影响 | 不影响 | 新 `mpc_tracker_node` |
| #8 延迟标定 | 增强 | 不影响 | 不影响 | 改 `policy_inference_node._get_synced_frame` |

**冲突点提醒**:
- **#5 QP 与 RTC `latency_k` 单位变化**: 从"步数"改"时间", 阶段 2 主要工程量
- **#3 Flash 的 K 验证步与 RTC prefix weights**: 两者都在去噪迭代内部, 实施时建议 RTC 先生成完整 chunk 候选 → Flash 验证哪些步可执行
- **#1 / #2 与 #3 复合**: V1 的 Triton kernel 是为 pi05 完整去噪写的, Flash 的并行验证调用模式不同, 内核可能要分叉两套

---

## 3. 实施路线图 — 5 阶段

### 3.1 总体阶段图

> **当前推理基线 (2026-05-20, v0.11)**: V1 Triton 路径已落地, P50 = **32.05 ms** (8.00× vs eager, 详见 §6). 阶段 3.2 工程量从原计划"自写 PyTorch+Triton (1-2 周)"实际收敛到 3 天 (V1 复用 + sentencepiece adapter + 5090 重 autotune). 阶段 3 剩余主要是: §3.4.1 PyTorch 训练等效性 POC (1-2 周), §3.4.3 sidecar framework 字段 (1 天).

```
阶段 0 (已完成):
  ✅ venv_5090 / .venv_5090_trt   ← sm_120 兼容
  ✅ benchmark_pi05_inference.py  ← PyTorch 5-backend baseline (E=41ms)
  ✅ V1 Triton 集成               ← P50=32.05ms (§6)
       │
       ↓
阶段 1 (短期, 1 周, 并行可做):
  ⏳ #8 延迟标定        ← 1-2 个 autonomy session
  ⏳ #6 JAX 浅层优化     ← AOT/cache/bf16, 现有 serve_policy 拉满
  ⏳ 真机测试 2         ← Piper t_motion (决定阶段 5 #7 是否上)
  ⏳ inference_rate 调参 ← 3Hz → 8-10Hz, 看 RTC 是否更稳
       │
       ↓
阶段 2 (任务速度主线, 1-2 周):
  ⏳ #5 时间轴 QP       ← 任务速度 1.5-2× (Q5 主要诉求)
       │
       ↓
阶段 3 (选项 X 落地, 剩余 1-2 周):
  ⏳ 3.1 PyTorch 训练等效性 POC (1-2 周)
  ✅ 3.2 推理 serve (V1 路径已完成, P50=32ms)  — 详 §6
  ⏳ 3.3 sidecar framework 字段 (1 天)
  ⏳ 3.4 新 fine-tune 切 PyTorch (持续)
       │
       ↓
阶段 4 (任务质量, 3-6 周):
  ⏳ #4 速度自适应      ← 油门数采 + 速度回归 head
       │
       ↓
阶段 5 (可选, 已降级):
  ⏳ #3 Flash 推测推理   ← 研究性: baseline 32ms × 3× = ~11ms, 边际有限
  ⏳ #7 客户端 MPC       ← 仅 真机测试 2 测出 t_motion > 50ms 时上
```

### 3.1.1 已知前置工程清单

> v0.11 后状态: 走 V1 路径无需修 PI0Pytorch (因为不再用 PI0Pytorch 跑 inference, V1 直接 dict-of-tensors). 下表保留作为"备选路径 max-autotune 实施前置"参考.

阶段 3.2 备选路径 (PyTorch + max-autotune) 的 model code fix:

| # | 文件:行 | 现状 | 需改成 | 影响 |
|:---:|---|---|---|---|
| 1 | `pi0_pytorch.py:172` | `sample_noise` 硬编码 `dtype=torch.float32` | `dtype=next(self.parameters()).dtype` | sample_actions bf16 闭环 |
| 2 | `pi0_pytorch.py:402,405` | `sample_actions` 的 `dt`/`time` 硬编码 fp32 | 跟随 model dtype | bf16 denoising loop |
| 3 | `pi0_pytorch.py:461` | `denoise_step` 返回 `action_out_proj(suffix_out)` 但 suffix_out 是 fp32 | 加 `suffix_out = suffix_out.to(model.dtype)` cast | 修 fp32/bf16 mismatch |
| 4 | `pi0_pytorch._prepare_attention_masks_4d` | `torch.where(mask, 0.0, -inf)` 输出 fp32 | 用 `torch.tensor(..., dtype=model.dtype)` 包 literal | Inductor SDPA 严格 dtype 检查 |
| 5 | `pi0_pytorch.py:229 embed_prefix` | `torch.tensor(att_masks_list, ...)` host→device copy | 预分配 GPU tensor | manual CUDA Graph (backend C) 需要 |
| 6 | `preprocessing_pytorch.py:160` | `class SimpleProcessedObservation:` nested class | 提到模块级别 | torch.compile fullgraph=True 才需要 |

工时估算: 1-3 天 (备选路径才需要做).

### 3.1.2 子任务清单 (阶段 0-1)

| 子任务 | 阶段 | 状态 |
|---|:---:|---|
| ✅ kai0/.venv_5090 / .venv_5090_trt (PyTorch nightly cu128) | 0 | 完成 |
| ✅ benchmark_pi05_inference.py (5-backend) | 0 | 完成 |
| ✅ 真实 JAX ckpt → V1 pickle 转换 (`convert_kai0_to_v1.py`) | 3.2 | 完成 |
| ✅ V1 Triton 集成 + 5090 重 autotune (5 kernel) | 3.2 | 完成 (P50=32.05ms, 见 §6) |
| ⏳ ROS2 推理节点 latency 实测 (#8 标定) | 1 | 待做 |
| ⏳ Piper t_motion chirp 测试 | 1 | 待做 |
| ⏳ inference_rate 3→10 Hz 调参 + RTC 行为观察 | 1 | 待做 |
| ⏳ sidecar framework 字段 + 启动脚本分发 | 3.3 | 待做 |
| ⏳ V1 推理路径包装为 serve_policy_v1.py + WebSocket | 3.2 后 | 待做 |

### 3.2 阶段 1 — 短期热身 (1 周)

#### #8 延迟标定 + 感知对齐

**目标**: `latency_k=8` 从经验值改为数据驱动值, RTC chunk 边界抖动 -30-50%

**改动文件**:
- `ros2_ws/src/piper/scripts/policy_inference_node.py::_get_synced_frame` — 接收图像时用 `header.stamp` 反查 joint deque, 而非"图像到达时"的关节
- 新增 `start_scripts/diag/measure_latencies.py` — 一次性 autonomy session 标定脚本

**步骤**:
1. 跑一次完整 autonomy 周期, 记录:
   - 相机 ROS timestamp vs `header.stamp` (D435/D405 曝光延迟)
   - 关节 state ROS timestamp vs CAN 接收时刻
   - `_publish_action` 发出时刻 vs Piper `/joint_state` 回环响应
2. 落表: `t_camera`, `t_readout`, `t_proprio`, `t_motion` (期望与 V2 数量级一致: 50/33/50/150 ms)
3. 改 `_get_synced_frame`: 用 timestamp 从 joint state deque 反查"曝光那一刻"的关节
4. `latency_k = round((t_camera + t_readout + t_motion) × publish_rate)`, 推测 ~7-10

**验证**: 重启 autonomy, 看 chunk 边界 RMS 抖动 (`StreamActionBuffer` 切换时刻的关节速度跳变) 是否下降。

#### #6 JAX 浅层优化

**目标**: 把现有 JAX serve_policy 推理时间压到该栈下最低 (期望 333ms → 150-200ms)

**改动文件**: `kai0/scripts/serve_policy.py`

**4 项改动**:
- (a) **AOT compile**: 服务启动时 `jit(fn).lower(args).compile()` 预编译, 不靠首次 trace 触发编译
- (b) **持久化 XLA cache 命中确认**: 已有 `start_server_xla_cache.sh`, 但需确认每次重启服务都命中 (常见坑: cache 路径变 / JAX 版本变 / device 列表变都会 invalidate)
- (c) **局部 bf16/fp16**: pi05 base 已 bf16, 但 normalization apply / tokenizer embedding lookup / image preprocessing 多半还在 fp32, 局部转 bf16 省 10-15%
- (d) **VLM prefill 与 AE 第 1 步并发**: 论文 V1 §5.2 数据显示 stream overlap 提升约 3.7%; JAX 下拆两个 jit 函数 + multi-stream

**验证**: WebSocket 服务 stress test, 推理 RTT P50/P95/P99 分布对比改动前后。

#### 真机测试 1 + 2
见 §4。期望产出:
- 测试 1: 模型实际推理时间 P50/P95/P99 分布表
- 测试 2: Piper t_motion 数值, 决定阶段 5 #7 是否做

### 3.3 阶段 2 — 任务速度主线 (1-2 周)

#### #5 时间轴 QP 重参数化 (V2 §4.3.1 + `realtime-vla-v2/server/optimizer.py`)

**目标**: chunk 几何不变, 重分配 Δt_i, 任务耗时 1.5-2×

**改动文件**:
- `kai0/src/openpi/serving/websocket_policy_server.py` — 推理后处理添加 OSQP 调用
- 新增 `kai0/src/openpi/policies/timeaxis_optimizer.py` — 复用 `realtime-vla-v2/server/optimizer.py::TimeParameterizationMPC`
- `ros2_ws/src/piper/scripts/policy_inference_node.py::_publish_action` — 定时器从 `1/publish_rate` 改为 `dt[k]` 驱动

**步骤**:
1. 集成 OSQP 到服务端推理后处理, V2 cloth config 参考参数:
   ```yaml
   dt_ref: 0.016    # 62.5 Hz 参考
   dt_min: 0.008    # 125 Hz 上限
   dt_max: 0.025    # 40 Hz 下限
   lambda_acc: 10.0
   horizon: 50
   v_max: <Piper 关节最大速度>  # 防 QP 违反硬件
   ```
2. 输出协议扩展: `actions: (50, 14)` → `actions: (50, 14) + dt: (50,)`
3. 客户端定时器改造: 从均匀 `1/publish_rate` 改为按 `dt[k]` 累积时间驱动 (或用插值)
4. `latency_k` 单位从"步数"改为"时间 (s)", 内部反查丢哪几步 — **本阶段主要工程量**

**与 RTC 兼容性**: 正交 — RTC 在 chunk index 维度引导, QP 在时间维度拉伸。

**风险**: lambda_acc 先取 10 (V2 cloth 默认) 防过激加速; v_max 需查 Piper 规格书。

**验证**: 折叠任务 (cloth) 实测任务耗时下降比例。

### 3.4 阶段 3 — 选项 X 落地

> **v0.11 实施现状**: §3.4.2 推理 serve 已通过 V1 Triton 路径完成 (P50=32.05 ms, 见 §6), 比原计划"自写 PyTorch+Triton"工程量小一个数量级. 剩余: §3.4.1 PyTorch 训练等效性 POC (1-2 周), §3.4.3 sidecar framework 字段 (1 天).

依赖阶段 1 真机测试 1 结果决定优先级:
- 若模型 P50 < 100ms (推理已很快) → 阶段 3 优先级降低, 可推迟
- 若模型 P50 200ms+ → 阶段 3 拿满 5-10× 才能凸显价值

#### 3.4.1 PyTorch 训练等效性 POC (1-2 周)

**目标**: 验证 `train_pytorch.py` 跑 pi05 fine-tune 与 JAX `train.py` 在 deepdive_kai0 数据集 + DDP 集群上等效。

**步骤**:
1. 选一个已完成 JAX 训练的对照 ckpt (例: `task_a_new_pure_1800_mixed1_step49999`), 记下 config + 数据范围 + 最终 inline-eval MAE
2. 用同一 config 在 PyTorch 跑 (gf0 / uc02 / uc03 任一 8× 节点):
   ```bash
   torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py \
       <same_config_name> --exp_name=pytorch_parity_test --save_interval 5000
   ```
3. 关键对比指标 (落表):
   | 指标 | 期望 |
   |---|---|
   | 同 1k step 的 loss 曲线对齐性 | max \|Δloss\| < 5% |
   | inline-eval MAE @ 50k step 差异 | < 10% |
   | 单 step 时间 | PyTorch 应快 10-30% (cuDNN + DDP 优化) |
   | 数值稳定性 | 无 NaN / 梯度爆炸 |
4. 验证 PASS → 进入 3.4.2; FAIL → 退回 fallback Y (§5)

**风险点**:
- pi05 PyTorch 路径在 deepdive_kai0 集群是否 DDP/FSDP 稳定 (advantage 管线用过, 主 fine-tune 没用过)
- norm_stats 加载: PyTorch / JAX 共享 `assets/<asset_id>/norm_stats.json`, dtype 转换可能有 bias
- LoRA 适配 (若用): PyTorch peft vs JAX 自实现的等效性

#### 3.4.2 PyTorch 推理 serve 搭建

> **v0.11 实施状态**: 走 V1 Triton 复用路径 (而非自写), 已完成. **P50 = 32.05 ms**, 详见 §6. 本节保留 5-backend baseline 实测数据 (PyTorch 路径上限) 作为路径背景与决策依据.

**2026-05-19 实测结果 (`optimize/benchmark_pi05_inference.py` on 5090, bf16, 100 iter)**

##### 分位数 (P50 / P95 / P99) 含义

`P50` / `P95` / `P99` 是把 100 次推理耗时**按从小到大排序**后, 取第 50 / 95 / 99 个值。

```
[第1名]  [第2名] ... [第50名] ... [第95名] [第96名] ... [第99名] [第100名]
  最快     ...        ↑中位数         ↑P95               ↑P99      最慢
                       P50          (95%快于它)        (99%快于它)
```

| 指标 | 含义 | 通俗说法 |
|---|---|---|
| **P50** | 50% 的推理 ≤ 这个值 | **中位数** — 典型情况 |
| **P95** | 95% 的推理 ≤ 这个值 | "100 次里 5 次比这个慢" — 较慢情况 |
| **P99** | 99% 的推理 ≤ 这个值 | "100 次里只 1 次比这个慢" — **tail latency** |
| Mean | 算术平均 | 受极端值影响大 |
| Std | 标准差 | 衡量波动程度 |

**为什么不只看 Mean**: 真机控制看 **P99 不是 Mean**。
- 两个 backend Mean = 50ms: X (P50=49ms, P99=52ms) 稳定可预期; Y (P50=30ms, P99=200ms) 偶尔的 200ms 推理会打乱 RTC 节奏
- Y 平均一样但灾难性 — 1 次卡顿 chunk 就错位
- **机器人推理只看 Mean 是错的, P99 才是真实风险指标**

##### Backend 做了什么 (按优化层级)

| Backend | Python 层<br>(dispatcher / autograd / op 查表) | XLA / Inductor 层<br>(kernel fusion + Triton 生成) | CUDA Graph 层<br>(消 launch overhead) | autotune 深度 | 一句话定义 |
|:---:|:---:|:---:|:---:|:---:|---|
| **A** eager | ❌ Python 每个 op 都走 dispatcher (1000+ kernel 每个 ~5-10μs Python 开销) | ❌ 无 fusion, kernel 数 1000+ | ❌ 每个 kernel 都 cuLaunchKernel | — | 纯 PyTorch eager, 把所有 model 代码当 Python 程序逐行翻译成 kernel launch, 是 baseline |
| **B** compile-default | ✅ TorchDynamo 把 Python 字节码提取成 FX graph, 后续 call 跳过 dispatcher | ✅ TorchInductor 把 FX graph 编译成融合 kernel (1000+ → ~200-300), 生成 Triton 代码 | ❌ 每个融合 kernel 仍走单次 cuLaunchKernel | min (Inductor 默认 autotune, 每 GEMM 试 ~5-10 种分块) | `torch.compile(model, mode="default")`. 只做 kernel fusion + Triton 生成, **不**自动叠 CUDA Graph |
| **C** cuda-graph (manual) | ❌ kernel 第一次 capture 时走 eager, 之后 replay 跳过 Python | ❌ 无 fusion, capture 进图的是 eager 的 1000+ kernel | ✅ 显式 `torch.cuda.CUDAGraph()` capture-replay, 整张图一次 cuLaunchGraph | — | 仿 V1 论文 §4.2.1: 预分配 buffer + warm-up + 显式 capture-replay。**FAILED** — `embed_prefix:229` 的 `torch.tensor(att_masks_list)` 是 host list → CUDA tensor copy, 违反 capture 内禁止 host-device sync 约束 |
| **D** compile-reduce-overhead | ✅ 同 B | ✅ 同 B (~200-300 融合 kernel) | ✅ **自动**: Inductor 编译完后 `torch.cuda.make_graphed_callables` 自动包 CUDA Graph | std (与 B 同样的 autotune) | `torch.compile(model, mode="reduce-overhead")`. = default mode + 自动 CUDA Graph。"compile + graph 二合一"标准接口 |
| **E** compile-max-autotune | ✅ 同 B/D | ✅ 同 D | ✅ 同 D (自动) | **max**: Inductor 对每个 GEMM/matmul 跑 20+ Triton 模板变体的完整 sweep, 选最快的 | `torch.compile(model, mode="max-autotune")`. = reduce-overhead + 更深 kernel autotune。**这是 deepdive_kai0 `PI0Pytorch.__init__:113` 当前默认** |

##### 实测数据表

| # | Backend | 优化叠加 | Mean (ms) | Std | P50 | P95 | P99 | Min | Speedup | 首次推理<br>(含 compile) |
|:---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | eager | (baseline) | 240.7 | 3.6 | 239.7 | 246.6 | 254.6 | 234.9 | **1.00×** | 0.6 s |
| B | compile-default | + Inductor fusion | 110.3 | 1.7 | 109.7 | 113.2 | 118.0 | 108.7 | **2.18×** | 40 s |
| C | cuda-graph (manual) | + CUDA Graph (无 fusion) | FAILED | — | — | — | — | — | — | — |
| D | compile-reduce-overhead | + Inductor fusion + auto CUDA Graph | 48.3 | 0.6 | 48.1 | 50.2 | 50.3 | 48.0 | **4.98×** | 37 s |
| E | compile-max-autotune | + Inductor fusion + auto CUDA Graph + 深 autotune | **43.6** | **0.3** | **43.5** | 43.7 | 44.9 | 43.4 | **5.52×** | 39 s |

##### 分量贡献分解

| 边际增量 | Δ (ms) | 倍数 | 解读 |
|:---:|---:|---:|---|
| **B − A** | -130.4 | **2.18×** | 纯 Inductor fusion 单独收益 (kernel 数 1000+ → ~200-300, 减少 launch + matmul fusion) |
| **D − B** | -62.0 | **2.13×** | 在 fusion 之上叠加 CUDA Graph 的边际收益 — **完美印证 V1 论文第一阶段 ~2×** |
| **E − D** | -4.7 | **1.11×** | max-autotune 比 reduce-overhead 多 10% (更深 GEMM 分块搜索) |
| E − A | -197.1 | **5.52×** | 总收益 (deepdive_kai0 当前默认 vs 纯 eager baseline) |

##### 抖动分析 (Std / 稳定性)

| Backend | Std | Std / Mean | P99 − P50 | 抖动级别 |
|:---:|---:|---:|---:|---|
| A eager | 3.6 ms | 1.5% | +14.9 ms | 中等 (Python dispatcher 不稳定) |
| B compile-default | 1.7 ms | 1.5% | +8.3 ms | 中等 |
| D reduce-overhead | 0.6 ms | 1.2% | +2.2 ms | 低 |
| **E max-autotune** | **0.3 ms** | **0.7%** | **+1.4 ms** | **极低** (CUDA Graph 锁定执行路径) |

**关键洞察**: CUDA Graph (D/E) 不只加速, **还显著降低延迟抖动** — P99−P50 从 14.9ms 降到 1.4ms (**抖动降 10×**)。这对真机 RTC 调度极其重要 (P99 接近 P50, 不会出现单次推理超时打乱 chunk 节奏)。

##### 核心结论

- E (deepdive_kai0 当前默认 max-autotune) **P50 = 43.5 ms** — 已远低于 60ms 阈值
- **策略 B (`torch.compile(max-autotune)`) 已饱和**, **不需要走策略 A (V1 手写 Triton 全栈端口)**
- 实际 inference_rate=3Hz timer 节流 333ms, 43.5ms 推理过剩, 可提至 10-20 Hz 让 RTC 更稳

**与 V1 论文对应**: D vs B = 2.13× 印证 V1 第一阶段 CUDA Graph 收益 ~2×; E vs D = 1.11× 反映 max-autotune 多 10% 边际。

##### 首次推理开销 (cold-start vs steady-state)

| Backend | 首次 (ms) | 稳态 P50 (ms) | cold-start 开销 |
|:---:|---:|---:|---|
| A eager | 597 | 239.7 | ~360 ms (cuDNN heuristic 初始化) |
| B compile-default | 40,178 | 109.7 | **40 秒** (TorchDynamo + AOTAutograd + Inductor 编译) |
| D reduce-overhead | 37,175 | 48.1 | 37 秒 (含 CUDA Graph capture) |
| E max-autotune | 39,075 | 43.5 | 39 秒 (autotune 跑 ~20 配置 × 多个 GEMM) |

**生产影响**: 启动 `serve_policy_pytorch` 时需要预热, 加 30-40 秒 cold-start。这是 `start_autonomy_from_ckpt.sh` 启动后需要"等模型 ready"的实际时间。

##### 实测过程踩坑 (前置工程一并落地)

1. **sm_120 不兼容**: kai0/.venv 的 PyTorch 2.7.1+cu126 仅支持到 sm_90, 装独立 `kai0/.venv_5090` (PyTorch nightly 2.12.0.dev + CUDA 12.8 + triton 3.7) 解决
2. **PI0Pytorch 内部 mixed dtype** (sample_noise/dt/time 硬编码 fp32, PaliGemma RMSNorm 强制 fp32 输出, action_out_proj fp32 mismatch action_in_proj bf16): benchmark 中通过 monkey-patch 4 处 (sample_noise / sample_actions / denoise_step / _prepare_attention_masks_4d) 让 bf16 自洽
3. **Dynamo 不能 trace `class SimpleProcessedObservation:` (nested class)**: `torch._dynamo.disable(_preprocessing.preprocess_observation_pytorch)` 让 dynamo 跳过 preprocess (eager 跑)
4. **C 失败 (CUDA Graph capture)**: `embed_prefix:229` 的 `torch.tensor(att_masks_list, ...)` 是 host list→device tensor copy, 违反 capture 约束; 需要把 att_masks 改成预分配 GPU tensor 才能 capture (model code 改动, 阶段 3.2 序后修)

**前置障碍 (已解决)**: `kai0/.venv` 的 PyTorch 2.7.1+cu126 不支持 sm_120 (Blackwell 5090). 解法: 装独立 venv `kai0/.venv_5090` (PyTorch nightly cu128) + `.venv_5090_trt` (稳定 2.7.1+cu128), 与训练 venv 隔离.

**实际实施路径 (v0.11)**: 不自写 Triton, 改走"V1 Triton 复用 + 5090 重 autotune"路径. 见 §6 完整 Step 0-9 记录:
- 工程量: 自写 1-2 周 → V1 复用 3 天
- 性能: PyTorch max-autotune 41 ms → V1 + 5090 tune 32.05 ms
- 代码位置: `optimize/v1_triton/{pi05_infer_tuned.py, convert_kai0_to_v1.py, benchmark_kai0_v1.py}`
- 数值对齐: rel error 1.42%, per-dim MAE < 0.01 rad

**剩余产出**: 把 V1 推理路径包装为 `serve_policy_v1.py` (WebSocket 协议与 JAX serve 一致), 见 §3.1.2 子任务清单 (待做).

#### 3.4.3 sidecar framework 字段 + 启动脚本分支 (1-3 天)

**改动文件**:
- `start_scripts/start_autonomy_from_ckpt.sh`:
  ```bash
  FRAMEWORK=$(python -c "import json; print(json.load(open('$CKPT_DIR/train_config.json')).get('framework', 'jax'))")
  if [ "$FRAMEWORK" = "pytorch" ]; then
      export OPENPI_SERVE_ENTRY="serve_policy_pytorch.py"
      export OPENPI_SERVE_PORT=8001
  else
      export OPENPI_SERVE_ENTRY="serve_policy.py"
      export OPENPI_SERVE_PORT=8000
  fi
  ```
- sidecar `train_config.json` 协议扩展:
  ```jsonc
  // 现有
  {"base_config_name": "...", "override_asset_id": "..."}

  // 扩展
  {"base_config_name": "...", "override_asset_id": "...", "framework": "jax"}
  // 或 "framework": "pytorch", 默认 "jax" 向后兼容
  ```
- `start_scripts/start_autonomy.sh` 接受 `config_name:=` 参数时透传给两个 serve

#### 3.4.4 新 fine-tune 切 PyTorch 策略 (持续, per-ckpt)

落地后, 后续 fine-tune 决策树:
- **简单 ablation / 快迭代实验** → JAX (现有管线稳定, 训练速度无差)
- **计划上 sim01 真机长期部署** → PyTorch (拿推理加速)

训练 + 真机测试都走原生 framework, 不跨框架转换。

#### 3.4.5 TensorRT 路径回顾 (2026-05-20)

> **TL;DR**: TRT 路径尝试过, 全部阻塞失败, 转走 V1 Triton 路径成功 (P50=32 ms, §6). 本节沉淀 5 个阻塞点 + 重启条件, 防止重复趟坑.

**Q**: 为什么 V1 Triton 而非 TensorRT?
**A**: TRT 是 §2 排名 #2 (Y fallback), 价值 3-5×. 实际尝试时遇到 5 个阻塞 (3 个工具链 / 2 个技术), 全部解或绕开后**仍卡在 ONNX export pi05 flow loop**. 同期 V1 Triton 路径直接跑通且更快 (8.0×), 故停 TRT 转 V1.

##### A. 已就绪资产 (留作未来重启用)

| 项 | 状态 | 位置 / 备注 |
|---|---|---|
| `kai0/.venv_5090_trt` | ✅ 完整 | Python 3.10 + **PyTorch 2.7.1+cu128 (stable)** + **TensorRT 10.14.1** + tensorrt_bindings/libs |
| `optimize/pi05_trt_pipeline.py` | ✅ 代码 367 行 | 5-stage 流水线: build → ONNX export → TRT engine → benchmark → numerical compare |
| `optimize/trt_smoke_test.py` | ✅ 工具链验证 254 行 | TinyTransformer → TRT (确认 TRT export 路径对玩具模型可行) |
| `optimize/TRT_30ms_PLAN.md` | 📄 详细 plan 376 行 | 3 个 sub-option A/B/C + ONNX 导出技术难点 |
| `optimize/results/pi05_aoti.pt2` | 🗃 6.3 GB | AOTI 编译产物 (load fail, 留作 PyTorch stable 后重测) |

##### B. 5 个阻塞点 (按尝试顺序)

| # | 阻塞 | 已解 / 未解 |
|:---:|---|---|
| 1 | `kai0/.venv` PyTorch 2.7.1+cu126 **不支持 sm_120** (Blackwell 5090) | ✅ 装 `kai0/.venv_5090` (PyTorch 2.12 nightly + cu128) |
| 2 | `torch_tensorrt` nightly 强制 CUDA 13 (我们 cu128 / 12.8) | ⚠ 绕开: 不用 torch_tensorrt, 直接走 ONNX → trtexec |
| 3 | NVIDIA pypi `pypi.nvidia.com` 子目录 GET 持续 hang (网络层) | ✅ file-copy `tensorrt + tensorrt_bindings + tensorrt_libs` 从 phantom env |
| 4 | phantom env 是 Python 3.10, `.venv_5090` 是 3.12 → C ext ABI 不通 | ✅ 新建 `kai0/.venv_5090_trt` (Python 3.10 + PyTorch 2.7 stable + cu128) |
| 5 | **ONNX export pi05 flow loop 10 步 denoise** — 真正技术难点 | ❌ **未解** — `torch.export` 对 dynamic shape + control flow + flow matching loop 支持不完善; `onnxscript`/`ml_dtypes` 下游依赖也有 sm_120 兼容问题 |

##### C. Backend H (AOTInductor) 同期阻塞

|  | 状态 |
|---|---|
| compile | ✅ 7 分钟出 6.3 GB `optimize/results/pi05_aoti.pt2` (wrapper.so + 60 cubin) |
| runtime load | ❌ `AOTIModelPackageLoader create_func_ API call failed` (C++ runtime line 123) |
| 推测原因 | PyTorch 2.12 nightly 的 AOTI runtime 与 sm_120 cubin loading 不兼容 (nightly bug), 或 nvcc 12.4 编译的 wrapper.so 与 cu128 ABI 不匹配 |

##### D. 最终决策依据 (来自 `optimize/results/FINAL_30ms_attempt_summary.md`)

V1 §4.2.2 列的 **8/8 优化项全部已被 Inductor max-autotune 自动捕获** (Backend K 手动追加 QKV 融合验证 Δ=-0.1ms 噪声内):

| V1 优化 | 我们栈 (E max-autotune) | 手动追加 (K) |
|---|---|---|
| RMSNorm 融合 | ✅ Inductor 自动 | — |
| **QKV 投影融合** | ✅ **Inductor 自动 (horizontal_fuse pass)** | -0.1 ms (噪声) |
| RoPE 融合 | ✅ Inductor 自动 | — |
| 动作时间编码折叠 | ✅ Inductor 自动 | — |
| GEMM 分块调优 | ✅ max-autotune sweep 21 配置 | — (F coord_descent 退化 6×) |
| 门控线性层融合 | ✅ Inductor 自动 | — |
| Split-k | ✅ max-autotune 覆盖 | — (V1 报告本身 <0.1ms) |
| 标量操作融合 | ✅ Inductor epilogue_fusion 自动 | — |

**结论**: PyTorch 内置工具链 5090 sm_120 bf16 极限 = **41.0 ms** (Backend E, §3.4.2 实测). 追加 V1 论文 §4.2.2 全部 fusion 不再有效 — Inductor 已自动. 触达 30ms 需要跳出 PyTorch 工具链.

##### E. 未来重启 TRT 的 4 条路径

| 选项 | 触发条件 | 工程量 | 风险 |
|---|---|---|---|
| **1** | 等 PyTorch 2.13 stable + torch-tensorrt 2.13 stable + cu128 (估 2026 Q2-Q3) | 重测既有 pipeline | 极低 |
| 2 | 5090 跑 sm_90 PTX 模拟 (PyTorch 2.7 stable + cu126) | 1-2 天 (新 venv + ckpt 兼容验证) | 中 (损失 sm_120 新特性) |
| 3 | 完整 ONNX 流水线 (Python 3.10 + PyTorch 2.7 + cu124 + 拆分 flow loop) | 3-5 天 | 高 (ONNX 导出 flow loop 是技术难点) |
| **4** ✅ | **接受 V1 32ms 当前 best, TRT 留 fallback Y** | 0 (当前路径) | 0 |

**当前选 4**. 重启选 1 需要等 PyTorch stable, 预计 2026 Q2-Q3.

##### F. 相关链接

- `optimize/results/FINAL_30ms_attempt_summary.md` — 完整 30ms 攻关时间线 + 全 8 backend 实测表
- `optimize/TRT_30ms_PLAN.md` — TRT plan 原文 (3 个 sub-option A/B/C)
- `optimize/results/phase1_F_G_J_findings.md` — F/G/J backend 失败分析
- `optimize/pi05_trt_pipeline.py` — 完整 5-stage TRT 流水线代码
- §3.4.2 实测踩坑 — 前置障碍解决记录 (sm_120 兼容 / dtype mismatch 等)
- §6 V1 Triton 实施日志 — 取代 TRT 的最终路径

### 3.5 阶段 4 — 任务质量 (3-6 周)

#### #4 V2 速度自适应学习 (V2 §4.4)

**目标**: 折叠衣袖 / 抓叠对位等精细阶段自动降速, 其他阶段加速

**步骤**:
1. 改 teleop 节点 `arm_teleop_node.py` 加油门输入 (脚踏板 / 摇杆 axis)
2. 收集 ~200 个 episode, 每帧记 `(observation, target_speed_factor)`
3. 训轻量回归 head (1 层 MLP), 输入: pi05 image encoder 输出 + 关节状态, 输出: `speed_factor ∈ [0.5, 4.0]`
4. head 训练用哪个框架? **遵从 ckpt framework** (PyTorch ckpt 用 PyTorch 训 head, JAX ckpt 用 JAX)
5. 部署: `speed_factor` → 阶段 2 QP 的 `dt_ref` 乘子 (与 QP 联动)
6. 迭代: 每天训练, 次日部署更高基线速度

**实验迭代瓶颈**: 200 ep 油门 teleop 数采 = ~10-20 hr 操作员时间, AI 不能替代。

**多任务环境**: deepdive_kai0 同时跑 Task A / P / PS, 需 task-conditioned 速度模型, 或 per-task 训练。

### 3.6 阶段 5 — 推理极致 (可选)

#### #3 Flash 推测推理 — **降级为研究项**

**降级理由 (v0.11 数据)**: 当前 baseline V1+autotune **P50=32 ms** (§6). Flash 3× 复合 → ~11 ms 端到端, 但:
- inference_rate=3Hz timer 节流 333ms, 提至 10Hz 也只是 100ms 周期 → 11ms vs 32ms 在 RTC 节奏里无差
- CUDA Graph 已把 P99−P50 抖动控制在 ±1 ms 内
- 工程量 4-8 周 + 每 ckpt 训 draft 3-6 hr GPU, 边际收益与成本不匹配

**仅当以下场景成立才重启**:
- 动态目标任务 (传送带分拣等, 推理 < 30ms 才有意义)
- inference_rate 拉到 30+ Hz (周期 < 33ms)

**实施路径 (备查)**: per-ckpt 110M draft 训练 + flow-matching 端点重建并行验证 + 阶段 fallback 信号 (论文用夹爪过零, deepdive_kai0 折叠/抓叠任务需补关节速度突变 / 视觉特征余弦相似度) + δ 阈值 per-task 调.

#### #7 客户端 MPC + 滞后辨识 (3-5 周)

**前置**: 真机测试 2 测出 Piper t_motion > 50ms; 若 < 30ms 直接 skip

**步骤**:
1. chirp 扫频测单关节响应, 辨识 τ
2. 新增 `ros2_ws/src/piper/scripts/mpc_tracker_node.py` 跑 acados/CasADi MPC
3. 与 Piper 自带 PD 兼容性 — 可能需要降 Piper PD 增益让 MPC 主导

---

## 4. 真机测试方案

### 4.1 测试 1: sim01 模型实际推理延迟 ✅ 完成 (2026-05-20)

**目的**: 区分"模型推理时间" vs "timer 节流时间", 确定推理优化上限.

#### 4.1.1 测量方法

利用 `ros2_ws/src/piper/scripts/policy_inference_node.py:2085-2148` 已内置的 inference timer:

```python
t_start = time.monotonic()              # line 2085
# ... obs 构造 + RTC 注入 + ...
result = self.policy.infer(obs)         # line 2145, WebSocket 同步调用
infer_ms = (time.monotonic() - t_start) * 1000   # line 2147
self.get_logger().info(f'infer {infer_ms:.0f}ms | chunk={actions.shape} | ...')
# line 2229: 每次推理都 log 一条
```

**测量范围 (client 视角)**: `t_start` 含 obs 构造 + WebSocket send → server side JAX `sample_actions` JIT 推理 → WebSocket recv → ROS 落 actions. 这是 **端到端 RTT** , **不是裸 GPU 推理时间**.

**测量工具**: `start_scripts/diag/measure_jax_infer_latency.sh` (87 行) 自动从 `~/.ros/log/` 找最新含 `infer XXXms` 的 log, 提取数值, 算分位数 + 阈值决策.

#### 4.1.2 实验配置

| 项 | 值 |
|---|---|
| **Session log** | `~/.ros/log/python_659068_1779190543244.log` (468 KB) |
| **日期** | 2026-05-19 19:36-19:43 (~7.9 min wall clock) |
| **Config name** | `pi05_flatten_fold_a_new_pure_1200` |
| **Checkpoint** | `/data1/DATA_IMP/checkpoints/task_a_pure200_base_pi05_step49999` |
| **Asset id (norm_stats)** | `a_new_pure_200` |
| **Mode** | RTC enabled (`Pi0Config → Pi0RTCConfig`, `pi05=True`) |
| **Execution** | joint, depth_in=False, ee_pose_in=False, EXECUTE 真机 |
| **Hardware** | RTX 5090 sm_120, sim01/ipc01 |
| **Timer** | `inference_rate = 3.0 Hz` (周期 333 ms) |
| **Chunk** | (50, 14) joint action, 14-DoF dual Piper |

#### 4.1.3 原始数据样本 (前 5 条)

```
[1779190576.541] infer  209ms | chunk=(50, 14) | L[0]=[-0.07,+0.06,-0.52,...] R[0]=[+0.27,+0.30,-0.42,...]
[1779190584.858] infer 8190ms | ...                    ← JIT cold-start outlier (excluded)
[1779190585.135] infer  272ms | chunk=(50, 14) | ...
[1779190585.399] infer  202ms | chunk=(50, 14) | ...
[1779190585.715] infer  184ms | chunk=(50, 14) | ...
```

**清洗规则**: 跳过前 5 个 (JIT compile warmup) + 排除 > 500 ms (1 个 outlier @ 8190 ms 估计为 GC / cache cold).

| | Raw | Cleaned |
|---|---:|---:|
| N samples | 1305 | **1299** |
| Dropped | — | 6 (0.46%) |

#### 4.1.4 实测分位数

| 指标 | 值 (ms) |
|---|---:|
| **P50** | **196.0** |
| P95 | 221.0 |
| P99 | 232.0 |
| Mean ± Std | 198.6 ± 13.2 |
| Min — Max | 174 — 276 |
| P95 − P50 (jitter) | 25.0 |
| P99 − P50 (tail) | 36.0 |
| Session 推理率 | 2.93 Hz (timer 333ms 59% utilization) |

#### 4.1.5 分布直方图 (cleaned, N=1299)

```
[  0, 180) ms |   22 (1.7%)
[180, 190) ms |  380 (29.3%) ██████████████   ← P50 / mode
[190, 200) ms |  356 (27.4%) █████████████
[200, 210) ms |  232 (17.9%) ████████
[210, 220) ms |  222 (17.1%) ████████
[220, 230) ms |   68 (5.2%)  ██
[230, 250) ms |   18 (1.4%)
[250, 300) ms |    1 (0.1%)
[300, 500) ms |    0 (0.0%)
```

**形态**: 单峰窄分布 (CV = Std/Mean = 6.6%), 主体集中 180-220 ms. **无长尾**, P99-P50 仅 36 ms — JAX/XLA 在 5090 上的步进时间非常稳定.

#### 4.1.6 与 V1 路径对比

| 指标 | JAX (sim01 实测, 端到端 RTT) | V1 Triton (offline 5090 benchmark, 裸推理) | 提升 |
|---|---:|---:|---:|
| P50 | **196 ms** | **32 ms** | **6.1×** |
| P95 | 221 ms | ~33 ms | ~6.7× |
| Jitter (P95-P50) | 25 ms | < 1 ms | -96% |
| Std/Mean | 6.6% | < 0.5% | -92% |

> 注: V1 数据来自 §6 offline benchmark, 未含 WebSocket overhead. 真 V1 serve 部署后, RTT 将增 WebSocket + obs 构造开销 (~5-10 ms 量级, B1 profile 待量化). 但即便 +10 ms, V1 serve RTT 估 ~42 ms ≪ JAX 196 ms, 仍 4-5× 加速.

#### 4.1.7 决策映射

P50 = 196 ms → 落在 "100-200ms 标准 5090 baseline" 档:

| 决策项 | 结论 |
|---|---|
| **V1 路径价值** | ✅ **确认 6.1× 加速空间, §6 V1 Triton 实施 + §7 B4 serve 包装方向正确** |
| **AOT compile 优先级** | ❌ 抖动 25ms ≪ 100ms 阈值, JAX 端无 AOT 需求 |
| **inference_rate 提升潜力** | ⏳ JAX 196ms 占满 3Hz × 59%, 提至 5Hz 即超时. **V1 落地后 32ms 可拉 20-30 Hz** (C2 子任务) |
| **#6 JAX 浅层优化 (阶段 1)** | △ 可顺手做 (AOT cache / bf16 局部), 估 1.5-2× → 100-130 ms, 但价值低于直接走 V1 |

阈值表对照:

| 指标 | 后续行动 |
|---|---|
| P50 < 80ms | 模型已很快, V1 收益小, 阶段 3 优先级可降低 |
| **P50 100-200ms** ✅ | **标准 5090 baseline, V1 路径价值明显, 6.1× 加速** |
| P50 > 250ms | 可能有 cache miss / fp32 残留, V1 收益最大 |
| P95-P50 > 100ms | 抖动严重, AOT compile 必做 (sim01 实测 25ms, 不需要) |

#### 4.1.8 测量复跑

```bash
# 自动找最新 log
./start_scripts/diag/measure_jax_infer_latency.sh

# 指定 log
./start_scripts/diag/measure_jax_infer_latency.sh <log_file>

# 调过滤
SKIP_WARMUP=10 MAX_MS=300 ./start_scripts/diag/measure_jax_infer_latency.sh
```

### 4.2 测试 2: Piper 关节 t_motion 滞后

**目的**: 量化 motor 响应滞后, 决定阶段 5 #7 是否值得。

**方法**: 单关节阶跃响应测试
```python
# ros2_ws/src/piper/scripts/test_motor_lag.py (待写)
# 发 0.1 rad 阶跃到 left_arm joint 1, 同时记录 joint_state 回传
# t0 = 发送时刻
# t1 = joint_state 跨越 50% 步进时刻
# t_motion = t1 - t0
# 重复 10 次取均值 / 方差
```

**期望读出 → 后续决策**:
| t_motion | 决策 |
|---|---|
| < 30ms | #7 skip; Piper 自带 PD 已足够 |
| 30-80ms | #7 中等价值, 可后置 |
| 80-150ms | #7 价值升至中段 |
| > 150ms | #7 价值高, 紧急做 |

---

## 5. Fallback 方案 (选项 Y)

### 触发条件
阶段 3.4.1 PyTorch 训练等效性 POC 失败, 例如:
- 同 config 下 PyTorch 训出 ckpt 的 inline-eval MAE > JAX 对照 10%+
- DDP 不稳定 (NaN / 梯度爆炸 / hang)
- 收敛速度显著慢于 JAX (单 step 时间慢 50%+)

### 选项 Y 内容
| 维度 | Y 决策 |
|---|---|
| 训练侧 | JAX only, 完全不变 |
| 推理侧 | JAX 默认 + JAX→ONNX→TRT (高性能版, 按 ckpt 配置) |
| 旧 ckpt | 走 JAX 推理 + #6 浅层 (1.5-2×) |
| 重要 ckpt | 跑 `pack_inference_trt.py` 转 TRT engine, 走 TRT serve (3-5×) |

### 切换步骤
1. **新增** `kai0/scripts/pack_inference_trt.py`:
   - 读 JAX ckpt → JAX→ONNX 导出 (用 `jax2tf` 或 `flax2onnx`)
   - ONNX → TRT engine (用 `trtexec` 或 `tensorrt` Python API)
   - 输出附在 ckpt 目录 `<ckpt>/inference_engine.trt`
2. **新增** `kai0/scripts/serve_policy_trt.py`:
   - 类似 PyTorch serve 但加载 TRT engine
   - WebSocket 协议与 JAX serve 一致
3. **修改** `start_autonomy_from_ckpt.sh`:
   - sidecar `backend: "jax" | "trt"` 字段分发 (与 X 的 `framework` 字段功能类似)
4. **POC 验证点**: pi05 flow matching 10 步去噪循环的 ONNX 导出是否可拆 (主要技术风险)

### Y 期望收益
- 旧 ckpt: 1.5-2× (走 JAX + #6) / 3-5× (走 TRT) 自选
- 新 ckpt: 3-5× (走 TRT)
- 训练侧零变更, 无 X 阶段 3.1 的等效性验证风险

---

## 6. V1 Triton 推理优化实施日志 (2026-05-20)

> **路径决策**: 阶段 3.4.2 原计划"自写 `serve_policy_pytorch.py` + 6 个 Triton kernel" (1-2 周 AI 辅助). 实施时改走**直接复用 V1 (arXiv:2510.26742) `pi05_infer.py` (22+ 个 Triton kernel)** + 写 deepdive_kai0 sentencepiece adapter + JAX→pickle 转换. 工程量从"自写 Triton kernel"降到"集成 + 数值对齐 + 5090 重 tune", 实际 ~3 天.
> **代码位置**: `optimize/v1_triton/` (生产 `pi05_infer_tuned.py`, sweep 脚本 `tune_5090_*.py`, benchmark `benchmark_kai0_v1.py`).
> **最终结果**: **P50 = 32.05 ms** (8.00× vs eager baseline, 比 §3.4.2 max-autotune 43.5ms 再快 26%).

### 6.1 总进度表 (Step 0-9)

真 ckpt: `task_a_mix_b6000_p1200_mixed_1_step49999`, 5090 sm_120 bf16 3-view chunk_size=50, 100-iter benchmark.

| Step | 实施 | Mean (ms) | vs 上一步 | vs eager | 文件 |
|:---:|---|---:|---:|---:|---|
| 0 | PyTorch eager (baseline) | 256.5 | — | 1.00× | `optimize/benchmark_pi05_inference.py` |
| 1 | + torch.compile(default) | 110.3 | -57.0% | 2.33× | (backend B) |
| 2 | + 手动 CUDA Graph | 60.7 | -45.0% | 4.23× | (backend C) |
| 3 | + compile(reduce-overhead) | 48.3 | -20.4% | 5.31× | (backend D) |
| 4 | + compile(max-autotune) (deepdive_kai0 当前默认) | 41.0 | -15.1% | 6.26× | (backend E) |
| 5 | V1 Triton 直接复用 (4090-tuned BLOCK_SIZE) | 35.4 | -13.7% | 7.25× | `pi05_infer.py` (upstream V1) |
| **6** | **+ 5090 BLOCK_SIZE autotune (3 hot decoder kernels)** | **32.3** | **-8.8%** ⭐ | **7.94×** | `pi05_infer_tuned.py` + `tune_5090_all.py` |
| 7 | + Triton pipelining sweep (num_warps × num_stages) | 32.3 | -0.2% 噪声 | 7.95× | `tune_5090_pipelining.py` |
| 10 | + Encoder FFN gate+up BLOCK_SIZE sweep | 32.34 | +0.13% 噪声 | 7.93× | `tune_5090_step10_encoder.py` |
| 8 | + Decoder QKV+RoPE BLOCK_SIZE sweep | 32.25 | -0.4% (-0.13 ms) | 7.95× | `tune_5090_step8_qkv.py` |
| **9** | **+ Decoder Attn QK matmul BLOCK_SIZE sweep** | **32.05** | **-0.6%** (-0.20 ms) | **8.00×** | `tune_5090_step9_attn.py` |

**生产版**: `optimize/v1_triton/pi05_infer_tuned.py` (含 Step 6 + 8 + 9 五个 kernel 全 tune), **P50 = 32.05 ms**.

### 6.2 Sweep 方法论

"Sweep" = **参数空间网格搜索**. 给一个 kernel 的可调参数 (`BLOCK_SIZE_N/M/K` / `num_warps` / `num_stages`) 列 N 个候选, 逐个 benchmark, 比较找最优.

#### 实操步骤

1. **列候选**: 写 Python list 含 10-15 组 BLOCK_SIZE
   ```python
   GATE_FFN_CANDIDATES = [
       (128, 64, 32),    # V1 default (4090)
       (32,  64, 128),   # ← 5090 最优
       (256, 64, 32),    # 退化 -41%
       (256, 128, 64),   # OOM
       # ... 共 10 个
   ]
   ```
2. **每候选独立跑**: monkey-patch `pi05_infer.transformer_decoder` → rebuild `Pi05Inference` (重 capture CUDA Graph) → warm-up 10 + 测 50 iter → 还原 + `empty_cache`
3. **排序选最优**, 应用到生产 `pi05_infer_tuned.py`

#### Greedy multi-kernel sweep

3 个 kernel 同时调时用 greedy (固定其他 default, 一个一个调):
```
[Sweep gate]  fixed (ffn=default, attno=default), 10 候选 → best = (32, 64, 128)
[Sweep ffn]   fixed (gate=best, attno=default),   11 候选 → best = (16, 32, 512)
[Sweep attno] fixed (gate=best, ffn=best),        11 候选 → best = (16, 32, 256)
```
Greedy 32 候选 ≈ 30 min vs 全局 1331 候选 ≈ 22 hr.

#### 我们使用的 sweep 脚本

| 脚本 | 调优对象 | 候选数 |
|---|---|---|
| `tune_5090.py` | matmul_small_gate 单 kernel | 10 |
| `tune_5090_all.py` | 3 个 hot kernels greedy | 32 |
| `tune_5090_pipelining.py` | num_warps × num_stages | 24 |
| `tune_5090_step8_qkv.py` | matmul_rope_qkv | 10 |
| `tune_5090_step9_attn.py` | matmul_abT_scale | 9 |
| `tune_5090_step10_encoder.py` | encoder FFN gate+up | 11 |

#### 与 Triton 内置 `@triton.autotune` 区别

| 维度 | 手动 sweep | `@triton.autotune` |
|---|---|---|
| 候选定义 | Python list, 任意修改 | 写在 kernel 装饰器上 |
| 选择时机 | Build phase, 用户主动跑 | 第一次 call 时自动 |
| 可见性 | 每候选 mean/P50 全可见 | 黑盒 |
| 与 CUDA Graph 兼容 | ✅ 完美 | ⚠️ 可能与 graph capture 冲突 |

### 6.3 PyTorch baseline 路径 (Step 0-4)

> 详细 5-backend 实测见 §3.4.2 (核心结论 + 数据表 + 分量分解 + 抖动分析). 此处仅简述每 Step 边际收益.

| Step | 关键改动 | 为什么有效 |
|:---:|---|---|
| 0 (eager) | PyTorch nn.Module 直跑 | (baseline) Python dispatcher + 1000+ kernel 各自 launch |
| 1 (compile-default) | `torch.compile(mode="default")` | TorchDynamo trace → FX graph → Inductor fusion (1000+ kernel → ~200-300), 跳过 Python dispatcher. **-57%** |
| 2 (CUDA Graph manual) | 手动 `torch.cuda.CUDAGraph()` capture-replay | 1000+ cuLaunchKernel → 1 个 cuLaunchGraph. **-45%** |
| 3 (reduce-overhead) | `mode="reduce-overhead"` (Inductor + 自动 graph) | 二合一: fusion + 自动 CUDA Graph. **-20%** |
| 4 (max-autotune) | `mode="max-autotune"` | Inductor 对每个 GEMM 跑 21 个 Triton 模板, 选最快. **-15%, deepdive_kai0 当前默认** |

**Step 0 → 4 累积 6.26× 加速 (256.5 → 41.0 ms)**. 这是 PyTorch 工具链上限 — Step F/G/H/I/J/K 测试均失败 (详 `optimize/results/FINAL_30ms_attempt_summary.md`).

### 6.4 V1 Triton 路径 (Step 5-9)

#### Step 5: V1 Triton 直接复用 (35.4 ms)

复制 V1 `pi05_infer.py` (22+ 手写 Triton kernel + 手动 CUDA Graph + 预分配 buffer) 进 `optimize/v1_triton/`, 写 `convert_kai0_to_v1.py` (sentencepiece adapter, JAX orbax → 6.7 GB pickle), 跑 `benchmark_kai0_v1.py`.

**为什么有效 (-14%)**: 完全 bypass PyTorch + Inductor, "model = dict of weight tensors + 手写 kernel" 极致路径. Triton kernel `tl.constexpr` BLOCK_SIZE 编译器针对 5090 sm_120 自动生成 PTX, 无需修改即可在 5090 跑.

**数值对齐**: rel error 1.42%, per-dim MAE < 0.01 rad (部署级). 修复要点: 移除 V1 test.py 的 `+ ori_state` (deepdive_kai0 训练绝对 action 不是 delta).

**V1 vs PyTorch E 架构对比**:

| 维度 | V1 Triton | PyTorch E (max-autotune) |
|---|---|---|
| 模型构造 | dict of weight tensors | nn.Module + transformers |
| QKV | 手动 concat 到 1 大矩阵 | 3 个独立 nn.Linear |
| Attention | 手写 Triton softmax+matmul | SDPA (cuDNN) |
| GEMM kernel 数 | 22 个 shape-specific 手写 | Inductor 模板 (21 候选 sweep) |
| Memory | 全预分配, 零 cudaMalloc | PyTorch caching allocator |

#### Step 6: 5090 BLOCK_SIZE autotune (32.3 ms) ⭐ 决定性单步

V1 BLOCK_SIZE 是 4090-tuned. 写 `tune_5090_all.py` 给 3 个 decoder hot kernel (180× per forward) sweep, greedy 选最优:

| Kernel | shape | V1 default (4090) | **5090 tuned** | 单独贡献 |
|---|---|---|---|---:|
| `matmul_small_gate` (FFN gate+up) | 1024→4096 | (128, 64, 32) | **(32, 64, 128)** | **-7.8%** ⭐ |
| `matmul_small_res_gate` (FFN down) | 4096→1024 | (16, 32, 256) | (16, 32, 512) | -0.1% |
| `matmul_small_res_gate` (Attn O) | 2048→1024 | (32, 32, 128) | (16, 32, 256) | -0.9% |

**反直觉发现 — "小 N, 大 K" 在 5090 上更优**:

| Rank | BLOCK (N, M, K) | Mean (ms) | 解读 |
|:---:|---|---:|---|
| **1** | **(32, 64, 128)** | **32.6** | 小 N → 2 grid × 64 = 128 program ≈ 5090 SM (170), 利用率高 |
| 4 | (128, 64, 32) | 35.4 | V1 默认 — 仅 64 program, **5090 浪费 100+ SMs** |
| 9 | (256, 64, 32) | 49.9 | grid=1, **几乎所有 SM 闲置** |
| - | (256, 128, 64) | OOM | shared memory 131KB > 5090 SM 上限 101KB |

**关键洞察**: 5090 SM 170 (vs 4090 128), 真正起作用是 **grid 总数 ≈ SM 数**, 不是 BLOCK 大小本身. 大 K (128) 利用 5090 L2 cache (96MB vs 4090 64MB).

#### Step 7: Triton pipelining sweep (32.3 ms, 噪声内)

固定 Step 6 BLOCK_SIZE, 试 24 个 `num_warps × num_stages` 组合. 最优 (gate: warps=8/stages=3, FFN-down: warps=4/stages=3, Attn-O: warps=4/stages=4) → 32.26 ms, 与 Step 6 (32.33 ms) 差 0.07ms 噪声内.

**为什么无效**: pi05 decoder GEMM **memory-bound** (3.6B × 2B / 1.7 TB/s = 4.2 ms floor), pipelining 主要优化 compute-bound 间隙.

#### Step 10: Encoder FFN gate+up sweep (32.34 ms, 噪声内)

Encoder `rms_matmul_n_2048_16384_gate` (FFN gate+up 2048→16384, 18× per inference, 最大单 GEMM). 11 候选全在 32.34-32.46 ms 窗口 (0.4% 内).

**为什么无效**: Encoder seq_len=775, BLOCK_N=128 时 grid=1792 programs ≫ 5090 SM 170, **已 grid-saturated**.

#### Step 8: Decoder QKV+RoPE sweep (32.25 ms, -0.4%)

`matmul_rope_qkv` (1024→2560, 180×). 10 候选 sweep: V1 default (64,32,64) → 32.38 ms; best (64, 32, 128) → 32.25 ms (-0.13 ms).

**为什么微弱**: QKV 比 FFN gate+up 小 (2560 vs 4096), decoder QKV 用 1D persistent grid `(128,)` 已合理饱和; 大 K (128) 仍有小收益.

#### Step 9: Decoder Attention QK matmul sweep (32.05 ms, -0.6%)

`matmul_abT_scale` (Q × K^T, 180×). 9 候选 sweep: V1 default (32,32,64) → 32.19 ms; best (32, 64, 64) → 32.05 ms (-0.14 ms).

**为什么微弱**: total_queries=400 (50 token × 8 head), total_keys=825. BLOCK_N=64 让 grid 沿 keys 维度更细 (cdiv(825,64)=13), 与 5090 SM 数更匹配.

**Step 8 + 9 累积 -0.28 ms** vs Step 6.

### 6.5 累积分析与硬件下限

#### 最终生产 P50 = 32.05 ms

`pi05_infer_tuned.Pi05InferenceTuned` 应用 5 个 kernel BLOCK_SIZE tune:

| Kernel (decoder, 180×) | V1 default (4090) | 5090 tuned |
|---|---|---|
| matmul_small_gate (FFN g+u) | (128, 64, 32) | **(32, 64, 128)** |
| matmul_small_res_gate (FFN down) | (16, 32, 256) | (16, 32, 512) |
| matmul_small_res_gate (Attn O) | (32, 32, 128) | (16, 32, 256) |
| matmul_rope_qkv (QKV+RoPE) | (64, 32, 64) | (64, 32, 128) |
| matmul_abT_scale (Attn QK) | (32, 32, 64) | (32, 64, 64) |

Mean=32.33ms (Std 1.19), **P50=32.05ms**, P95=32.97ms.

#### 距硬件下限分析

**5090 memory-bound 理论下限**: 3.6B × 2 bytes / 1.7 TB/s = **4.2 ms** (理论 8×)

**实际 32.05 ms 拆分**:
- vision encoder (1×) + transformer encoder (18 layer 1×) + transformer decoder (10 step × 18 layer = 180 attention block)
- 每 decoder block ≈ 32 / 180 ≈ 0.18 ms (含 norm + QKV + SDPA + O + norm + FFN), **已接近硬件极限**
- 剩余 28 ms 主要去向: attention softmax + QK + AV 非纯 matmul (难达 memory peak bandwidth); 22 个 Triton kernel 串行 dispatch; layer 间无法完全 hide 的同步开销

#### 距 30 ms 目标分析

当前 32.05 ms, **距 30 ms 差 -2.05 ms (-6.4%)**.

V1 + 5090 BLOCK_SIZE tune 路径已榨干: 单步收益从 Step 6 的 -7.8% 衰减到 Step 7-10 的 < 1%, 累积已逼近 V1+autotune 路径在 5090 上的极限. **触达 30ms 必须结构性优化** (kernel 重写或 fusion), 不是参数 sweep.

### 6.6 待实施 (Step 11+, 结构性优化)

| Step | 实施 | 预期收益 | 工程量 | 优先级 | 风险 |
|:---:|---|---|---|:---:|---|
| **11** | **Kernel fusion**: 合并 `adarms_norm_style_proj` + 后续 matmul, 减 360+ kernel 边界 sync | **-1-3%** | **3-5 天** | 中 | 中 (需写新 Triton kernel + 数值验证) |
| **12** | **Stream overlap**: vision encoder ‖ decoder denoise 并发 (V1 §4.4 提到 3.7%) | **-1-2%** | 2-3 天 | 低 (decoder 串行依赖, overlap 空间小) | 中 |
| **13** | **wgmma 重写主要 GEMM**: Blackwell 原生 `wgmma.mma_async.m64n*` 替代 `mma.sync` | **-5-10%** | **5-10 天** | 高回报 | 高 (Triton 3.x sm_120 实际生成 wgmma 还是 mma 不确定; OOM shared mem 风险) |
| 14 | **FlashAttention 3**: 替换 attention path | 不确定 (-1 ~ +3%) | 2-3 天 | 低 | 高 (短序列 seq=50 收益不明) |
| 15 | **共享 KV cache cross-step**: 10 步 denoise 部分 encoder K/V 可缓存复用 | 不确定 | 1-2 天 | 中 | 低 (V1 已部分实现) |

#### 推荐路径 (按 ROI / 单位工时排序)

1. **Step 11 (Kernel fusion)** ⭐ — 中等工程量, 收益较确定 (-1-3%), 难度可控
2. Step 13 (wgmma) — 收益最大但风险最高, 适合"豁出去"投入
3. Step 14/15 — 收益不确定, 不优先

#### 结论 (V1 路径上限)

pi05 5090 在不改架构 / 不重训 / 不量化约束下:
- **V1 + 5 kernel autotune = P50 32.05 ms (8.00× vs eager)** ← 当前生产
- **再压到 30 ms 需 3-5 天 kernel fusion 或 5-10 天 wgmma 重写**
- 硬件理论下限 4.2 ms (memory-bound), 当前距下限 8× — 长期目标可参考

---

## 7. Layer B 系统级优化 plan (next phase)

> **路径选择**: kernel 内配置 tune 已尽 (§6.6 Step 11+ 结构性优化是单点突破, ROI 中低). 下一阶段转向**端到端系统级优化** — 量化全链路, 把视野从"推理 32ms"扩大到"相机曝光 → motor 响应"全链.

### 7.1 范围 + 约束

#### 硬件约束 (单 5090)
sim01 真机部署**只用 1 个 5090** (尽管机器物理装 2 张). 这排除了"多 GPU async pipeline"等架构选项:

| 已排除 | 原因 |
|---|---|
| 双 5090 load balance (vision 1 卡, decoder 另 1 卡) | 单 GPU 约束 |
| Multi-GPU batch | 同上 |
| Cross-GPU stream overlap | 同上 |

#### 已选 3 子项 (按依赖顺序)

```
B4 (V1 serve 包装) ──→ 真机可跑 ──→ B1 (profile) ──→ B2 (定向 preprocess)
```

#### 暂不做 (Layer A + Layer C 其他项)

| 项 | 决策 | 原因 |
|---|---|---|
| Layer A1 kernel fusion (-1-3%) | 暂缓 | 收益小, 推理 32ms 已远 < timer 周期 |
| Layer A4 wgmma 重写 (-5-10%) | 暂缓 | 5-10 天高风险, B 层未做完前不投入 |
| Layer C2 inference_rate 调参 | 阶段 1 (并行) | 真机测试 1 完成后即可做 |
| Layer C3 与 QP 联动 | 阶段 2 (依赖 §3.3 #5) | QP 落地后 |
| Layer C4 客户端 MPC | 阶段 5 (条件) | 真机测试 2 测出 t_motion > 50ms 才上 |

### 7.2 B4 V1 serve 包装 ✅ Phase 2 完成 (2026-05-20)

**目标**: 把 §6 的 `Pi05InferenceTuned` 包装成 WebSocket 服务, ROS2 client 无感切换.

#### 实测结果 (kai0/.venv_5090_trt 本机 smoke test, 5 iter Phase 2)

| 段 | iter 2-5 平均 (ms) | 说明 |
|---|---:|---|
| **total** (server side) | **~40.5** | 含全部 5 段 |
| preproc | 5.7-6.7 | PIL resize 224 + bf16 cast (B2 优化候选) |
| state_encode | 0.3-0.4 | sentencepiece + PaliGemma embed lookup |
| infer (V1 forward) | 34.0 | offline P50=32 + 2ms inference_mode/sync overhead |
| postproc | 0.1-0.2 | action denorm + .cpu() |

vs Q2 JAX 196ms: **4.9× 加速**已实现 (server side). 加 WebSocket + ROS2 transit (B1 待量化) 后客户端 RTT 估 ~50-55 ms (vs JAX 196 ms 仍 3.5-4× 加速).

**State conditioning sanity check**: 切换 state ([0.5,-0.3,...] → [0,0,...]), action[0] max diff = 0.286 → state 编码确实流入 action.

#### 实施步骤

#### 步骤

1. **新增** `kai0/scripts/serve_policy_v1.py` (复制 `serve_policy.py` 骨架, WebSocket payload schema 不变)
   - 启动: load V1 pickle (`task_a_mix_b6000_p1200_v1.pkl`, 6.7 GB) → `Pi05InferenceTuned`
   - `predict()`: 调用 `infer.forward(image, noise)`, 输出 action chunk (50, 14)
   - 端口 `:8002` (JAX :8000, PyTorch 备用 :8001 之外)

2. **数据格式 adapter**:
   - V1 期望: `image (num_views, 224, 224, 3) bf16 CUDA` + `noise (50, 32) bf16 CUDA`
   - JAX serve 接收: `(num_views, H_orig, W_orig, 3) uint8 numpy`
   - 写 adapter: decode → resize 224×224 → bf16 → CUDA (B2 优化点)
   - `norm_stats` 复用: 加载 deepdive_kai0 同一份 `assets/<asset_id>/norm_stats.json` 做 denorm

3. **sidecar 协议扩展**:
   ```jsonc
   {"base_config_name": "...", "framework": "jax"}        // 默认, 走 :8000
   {"base_config_name": "...", "framework": "v1_triton"}  // 新增, 走 :8002
   ```

4. **启动脚本分发** (`start_scripts/start_autonomy_from_ckpt.sh`):
   ```bash
   FRAMEWORK=$(python -c "import json; print(json.load(open('$CKPT_DIR/train_config.json')).get('framework', 'jax'))")
   case "$FRAMEWORK" in
     v1_triton) ENTRY=serve_policy_v1.py;  PORT=8002 ;;
     pytorch)   ENTRY=serve_policy_pytorch.py; PORT=8001 ;;
     *)         ENTRY=serve_policy.py;     PORT=8000 ;;
   esac
   ```

5. **WebSocket 协议验证**: ROS2 client 零改动, sim01 跑 autonomy 看推理回包形状一致.

**输出**: `serve_policy_v1.py` + sidecar 协议 + 启动脚本分发 + 1 个真实 ckpt 跑通端到端.

**验收**: ROS2 `policy_inference_node` 调 `:8002` 完成一次 autonomy 周期, action chunk 形状 (50, 14), 推理 RTT 与 §6 benchmark 一致 (~35-40 ms 含 WebSocket).

### 7.3 B1 全链路 latency profiling (2-3 天)

**目标**: 量化 11 段延迟, 找 P50/P95 拐点, 决定 B2 优化方向.

#### 11 段切片

```
t0 相机曝光             ─┐
t1 USB readout          │  → t_camera (~50ms, V2 估)
t2 ROS2 transport      ─┘
t3 client → server WebSocket send
t4 server preprocess     → t_preproc (B2 候选)
t5 server inference      → t_infer (32 ms, 已知)
t6 server postproc / norm denorm
t7 server → client WebSocket recv
t8 ROS2 publish (action chunk)
t9 Piper CAN write
t10 motor 响应           → t_motion (~50-150ms 估, §4.2 测试)
```

#### 步骤

1. **改 `policy_inference_node.py`** (client 侧): `_inference_loop` 每段加 `time.perf_counter()` 串, 落 CSV (列 = t0..t10, 100 cycle 取 P50/P95/P99)
2. **改 `serve_policy_v1.py`** (server 侧): `predict()` 内部 t4/t5/t6 用 GPU event timer (`torch.cuda.Event(enable_timing=True)`)
3. **跑 1-2 个真任务 autonomy session** (sim01, ≥200 cycle), 收集 latency CSV
4. **出表 + 推荐**: 11 段 P50/P95/P99, 标最大头 + 抖动最大段, 落 `docs/deployment/latency_profile_v1.md`
5. **副产物**: 同时拿到 §3.2 #8 `latency_k` 的数据驱动值 (一举两得)

**输出**: `docs/deployment/latency_profile_v1.md` (11 段表 + 优化方向推荐).

**验收**: 找到 ≥1 个 P50 > 5ms 的非推理段, 或确认链路已无 > 5ms 段.

### 7.4 B2 Preprocess 全 GPU 化 (1-3 天, 数据驱动)

**目标**: B1 profile 指出 t4 > 5ms 时才做; 否则确认链路已优化.

#### 候选优化 (按可能瓶颈)

| 候选 | 现状假设 | 改成 | 预期收益 |
|---|---|---|---|
| Image resize | CPU OpenCV / PIL | GPU `torchvision.transforms.v2` 或 V1 vision_encoder 内嵌 | -2-5 ms |
| Normalize (mean/std) | CPU numpy | GPU torch op | -1-2 ms |
| Uint8 → bf16 cast | CPU | GPU `.to(torch.bfloat16, non_blocking=True)` | -0.5-1 ms |
| host→device copy | 阻塞 | Pinned memory + `non_blocking=True` 异步 | -1-2 ms |
| JPEG decode (若 client encode) | CPU PIL | nvJPEG | 视 client 编码方式而定 |

#### 步骤

1. 看 B1 数据定优先级 (只改 > 1ms 段)
2. 改 `serve_policy_v1.py` 内 preprocess pipeline
3. 重测 B1 验证

**验收**: t4 降到 < 3ms, 或确认无优化空间.

### 7.5 时序 + 风险

#### 时序总览

| 周 | 主任务 | 验收 |
|:---:|---|---|
| 1 (Mon-Fri) | B4 V1 serve 包装 + sidecar | sim01 跑通真实 ckpt, ROS2 节点无感 |
| 1 weekend / 2 (Mon-Wed) | B1 profile + autonomy session | latency_profile_v1.md, 11 段延迟表 |
| 2 (Thu-Fri) | B2 定向 preprocess GPU 化 | t4 < 3ms 或确认无空间 |

**总工程**: 1.5-2 周, 主线 B4. 关键里程碑: **B4 完成 = 真机 V1 推理首跑通**.

#### 风险点

| 风险 | 缓解 |
|---|---|
| V1 推理需要的 image preprocessing 与 JAX serve 不一致 (resize 策略 / normalize 参数) | 取 JAX `_preprocess_observation` 作对照, V1 同输入对比输出 |
| `norm_stats.json` 加载 — V1 推理是否完整内嵌? | 检查 `convert_kai0_to_v1.py` 是否把 stats 也存进 pickle; 否则 serve_policy_v1.py 单独加载 |
| Inference cold start (V1 build + CUDA Graph capture, ~30s) 影响 systemd 启动 | 启动脚本里加 readiness probe; `wait_for_serve.sh` 等 graph capture 完成才进 autonomy |
| ROS2 client chunk 协议与 V1 输出不一致 | 端到端用同 schema, V1 serve 内部 padding/格式补齐 |
| 单 5090 资源争抢 (推理 + 其他后台 GPU 进程) | sim01 部署时锁定 `CUDA_VISIBLE_DEVICES=0` 给 serve, 其他进程禁用 GPU |

#### Out of scope (本阶段不做)

- Layer A 任何项 (kernel fusion / wgmma / FA3) — kernel 32ms 已远 < timer 周期 (100ms @ 10Hz), ROI 低
- 双 5090 利用 — 真机约束
- B3 WebSocket payload trim — 用户排除
- 任何破坏 §6 数值对齐 (rel error 1.42%) 的改动

---

## 8. 修订历史

| 版本 | 时间 | 内容 |
|:---:|---|---|
| v0.1 | 2026-05-19 | 初版, P1/P2/P3 排序 + 三论文逐项判断 |
| v0.2 | 2026-05-19 | 成本框架重构: agent 时代 4 维任务复杂度评估, 不再因工时拒绝方案。V1 全栈端口从"不做"调整为 #1。新增 V1 三路径 A/B/C 对比 |
| v0.3 | 2026-05-19 | 整合 Q1-Q5 答案; 发现 PyTorch 训练侧已实装 (advantage 管线), 双栈是既成事实; 新增同架构 4 选项 a/b/c/d, 选项 d 是"无双栈痛点"折中; 关键洞察: 333ms 是 timer 节流; 加真机测试方案 |
| v0.4 | 2026-05-19 | Q4 round 2: 假设双推理架构并存 + 忽略 ckpt 迁移风险。痛点 1-3 消失, 4-7 弱化。新增选项 X/Y/Z, 推荐 X |
| v0.5 | 2026-05-19 | 用户决策选定选项 X。重写 §7 为 5 阶段实施路线图, 每步给出具体改动文件 + 风险点 |
| **v0.6** | **2026-05-19** | **文档全面整理: 加目录; 删冗余 (原 §3 推荐落地路径 ↔ §7 阶段细节重复, 原 §5 Q&A round 1 a/b/c/d 详细对比); §0 收敛为决策摘要 + 5 阶段一览; §1 整合现状/三论文/关键洞察/决策依据; §2 排序与复杂度; §3 实施路线图为主干; §4 真机测试; §5 Fallback Y 详细方案。从 756 行精简到 ~570 行, 主线清晰** |
| v0.7 | 2026-05-19 | **关键前置障碍**: 实测发现 sim01 5090 sm_120 与 `kai0/.venv` PyTorch (2.7.1+cu126) 不兼容; §3.4.2 阶段 3 落地前必须升级 PyTorch nightly / cu128。建议独立 venv `kai0/.venv_5090` 隔离 |
| v0.8 | 2026-05-19 | **pi05 推理 5-backend 实测完成** (`optimize/benchmark_pi05_inference.py`): E max-autotune P50=43.5ms (5.52×), B compile-default 2.18× (纯 fusion), D reduce-overhead 4.98× (+CUDA Graph), V1 论文 CUDA Graph 单独 2× 被印证 (D vs B = 2.13×)。**结论: 策略 B 饱和, 不需要 V1 手写 Triton 路径**。已知 4 个 PI0Pytorch model code 内部 dtype 问题需阶段 3.2 修 (sample_noise/dt/time/RMSNorm output, embed_prefix att_masks list→tensor) |
| v0.9 | 2026-05-19 | §3.4.2 扩展实测结果展示: 加分位数 (P50/P95/P99) 含义说明 + 5-backend 详细描述表 (Python/Inductor/CUDA Graph/autotune 4 层) + 实测数据表 + 分量贡献分解 + 抖动分析 (Std/P99-P50) + cold-start 开销 |
| v0.10 | 2026-05-19 | **根据实测结果更新实验计划**: §3.1 总体阶段图加 "阶段 0 真机推理基线" + 阶段 3 工程量减 50% (3-5 周→1.5-2.5 周, 不需 Triton); 新增 §3.1.1 实测对计划影响对比表; §3.1.2 PI0Pytorch 6 处 model code fix 清单 (P0-P6); §3.1.3 子任务清单 (已完成/待做); §3.6 #3 Flash 降级为研究项 (baseline 43.5ms × 3× 边际效用低) |
| **v0.11** | **2026-05-20** | **V1 Triton 推理优化全程实施 (合并自 `optimize/v1_triton/PROGRESS.md`)**: 新增 §6 完整记录 Step 0-9 的 9 个优化步骤. 路径决策从"自写 PyTorch+Triton"改为"复用 V1 `pi05_infer.py` + 5090 重 autotune", 工程量 1-2 周 → 3 天. **最终 P50 = 32.05 ms (8.00× vs eager, 比 §3.4.2 max-autotune 43.5ms 再快 26%)**. 关键发现: 5090 sm_120 "小 BLOCK_N 大 BLOCK_K" 反直觉最优 (Step 6 单步 -8.8%); decoder GEMM memory-bound, pipelining/encoder sweep 噪声内; 继续突破 30ms 需结构性 kernel fusion (Step 11, 3-5 天) 或 wgmma 重写 (Step 13, 5-10 天). 独立 PROGRESS.md 已删除 |
| **v0.12** | **2026-05-20** | **针对性整理**: §0/§3.1 总体阶段图按 v0.11 实施现状重写 (阶段 0 完成 / 阶段 3 推理 serve 已 ✅); §3.1.1 (原计划影响对比表) 删除, 改为"PI0Pytorch fix 备选路径清单"; §3.1.2 子任务清单状态更新 (V1 路径 4 项完成); §3.4 顶部加 v0.11 现状引导; §3.4.2 末尾自写 Triton 计划折叠 (8 步细节 → 3 行实施路径摘要); §3.6 Flash 降级理由瘦身 (用 32ms 而非 43.5ms 数据). TOC 补 §3.1.1/3.1.2. 总行数 950 → ~900 |
| **v0.13** | **2026-05-20** | **新增 §7 Layer B 系统级优化 plan**: 单 5090 真机约束确认 (排除多 GPU 选项). 子项 B4 (V1 serve 包装, 主线) → B1 (全链路 11 段 latency profile) → B2 (preprocess GPU 化, 数据驱动). 1.5-2 周, 关键里程碑 B4 = 真机 V1 推理首跑通. Layer A (kernel fusion/wgmma) 暂缓 (推理 32ms 已远 < timer 周期, ROI 低). §7 修订历史 → §8. TOC 更新 |
| **v0.14** | **2026-05-20** | **Q2 sim01 JAX 推理延迟实测完成 + B4 Phase 1 serve_policy_v1.py 落地**: §4.1 加 1299-sample 实测表 (P50=196 ms / P95=221 / P99=232 / Std=13.2 / jitter=25 ms), 落在 "100-200ms 标准 5090 baseline" 档, 确认 V1 路径 6.1× 加速空间. JAX 抖动 P95-P50=25ms 不需 AOT compile. 推理 196ms vs timer 333ms = 59% utilization, V1 落地后可拉 inference_rate 到 20-30 Hz. 新增 `kai0/scripts/serve_policy_v1.py` (B4 Phase 1, 343 行) + `start_scripts/diag/measure_jax_infer_latency.sh` (Q2 helper) |
| **v0.15** | **2026-05-20** | **§4.1 扩为正式实验报告 (8 子节)**: 4.1.1 测量方法 (含 policy_inference_node.py:2085-2148 timer 源码片段); 4.1.2 实验配置 (config / ckpt / asset_id / 硬件 / timer 等 9 项 metadata); 4.1.3 原始 log 样本 (前 5 条 + JIT outlier 标注); 4.1.4 分位数表; 4.1.5 **ASCII 分布直方图 9 bucket** (180-220 ms 集中 91%, 单峰窄分布 CV=6.6%, 无长尾); 4.1.6 V1 对比加 jitter / Std/Mean 行 + WebSocket overhead 补偿估算; 4.1.7 决策映射 4 行结论 (V1 路径✅ / AOT❌ / inference_rate 提升潜力⏳ / #6 价值低); 4.1.8 复跑命令. TOC 加 4.1 子节链接 |
| **v0.16** | **2026-05-20** | **新增 §3.4.5 TensorRT 路径回顾**: 沉淀 TRT 攻关失败记录, 防止重复趟坑. 6 子节 A-F: A 已就绪资产 (`.venv_5090_trt` Python 3.10 + PyTorch 2.7.1+cu128 + TRT 10.14, `pi05_trt_pipeline.py` 367 行 5-stage 流水线, AOTI 6.3GB 产物); B **5 个阻塞点** (sm_120 / torch_tensorrt CUDA 13 / pypi hang / Python ABI / **未解 ONNX flow loop**); C AOTInductor Backend H 同期阻塞 (compile OK 但 load fail); D V1 §4.2.2 **8/8 优化已被 Inductor 自动捕获** (PyTorch 工具链 41ms 极限); E **4 条重启路径** (选 1 等 PyTorch 2.13 stable; 选 4 当前接受 V1 32ms); F 相关链接 6 个文件. TOC 加 3.4.5 子节链接 |
| **v0.17** | **2026-05-20** | **B4 Phase 2 + B1 server-side profile 完成**: 实现 `SentencepieceStateEncoder` (kai0 同款 prefix `"Task: {p}, State: {s};\n"` + 256-bin 离散化 + PaliGemma embed lookup + scale√2048), 绕开 V1 prebaked language_embeds via `v1_forward_with_state()` 直写 encoder_x. 新增 `expand_v1_pkl_for_phase2.py` (扩 pkl `language_embeds` 7→200 行, 为 prompt+state 留位). V1Policy.infer() 加 5 段 timing (preproc / state_encode / infer / postproc / total). 本机 smoke test (5 iter): **total ~40.5 ms** (preproc 6 + state 0.3 + infer 34 + post 0.2), state 切换→action max diff 0.286 (验证 state 流入). vs Q2 JAX 196ms = **4.9× server-side speedup**. §7.2 加实测表 |
