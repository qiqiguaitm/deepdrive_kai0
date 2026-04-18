# kai0 & X-VLA 深度分析报告

> 生成时间: 2026-03-27 (最后更新: 2026-03-27)
> 硬件配置:
> - 机器人: 双臂 Piper + Cobot Magic ALOHA 夹爪 + D405 一体式支架 (可调高度) × 2 + D435 (top) × 1
> - 工控机/推理机 **sim01**: tim@sim01, 双卡 RTX 5090 32GB, `/data1/tim/workspace/deepdive_kai0`
> - 训练机 **gf0**: `ssh -p 55555 -R 29290:localhost:29290 tim@14.103.44.161`, 8×A100 80GB
> - 训练机 **gf1**: `ssh -p 11111 -R 29290:localhost:29290 tim@14.103.44.161`, 8×A100 80GB
> - 工作目录同构: `~/workspace/deepdive_kai0` | 密码: tim | gf0/gf1 外网代理登录即生效
> 目标: 复现 kai0 Task A (T恤展平&折叠) 全流程

---

## 目录

- [一、项目概览对比](#一项目概览对比)
- [二、kai0 深度解读](#二kai0-深度解读)
  - [2.1 核心问题与方法论](#21-核心问题与方法论)
  - [2.2 Module 1: Model Arithmetic](#22-module-1-model-arithmetic模型算术)
  - [2.3 Module 2: Stage Advantage](#23-module-2-stage-advantage阶段优势估计)
  - [2.4 Module 3: Train-Deploy Alignment](#24-module-3-train-deploy-alignment训练-部署对齐)
  - [2.5 基座模型架构](#25-基座模型架构)
  - [2.6 实验结果](#26-实验结果)
- [三、X-VLA 深度解读](#三x-vla-深度解读)
  - [3.1 核心创新: Soft Prompt](#31-核心创新-soft-prompt)
  - [3.2 模型架构](#32-模型架构)
  - [3.3 训练流水线](#33-训练流水线)
  - [3.4 实验结果](#34-实验结果)
- [四、硬件匹配度分析](#四硬件匹配度分析)
- [五、相机选型与安装分析](#五相机选型与安装分析)
  - [5.1 D405 vs D435 腕部相机对比](#51-d405-vs-d435-腕部相机对比)
  - [5.2 D405 安装方案 (Cobot Magic ALOHA 一体式)](#52-d405-安装方案)
  - [5.3 D435 Top Camera 位置深度分析](#53-d435-top-camera-位置深度分析)
- [六、数据集开源情况](#六数据集开源情况)
- [七、代码架构分析](#七代码架构分析)
  - [7.1 kai0 代码结构](#71-kai0-代码结构)
  - [7.2 X-VLA 代码结构](#72-x-vla-代码结构)
  - [7.3 源码依赖关系](#73-源码依赖关系)
- [八、两个 repo 推理代码对比](#八两个-repo-推理代码对比)
- [九、kai0 官网独有内容](#九kai0-官网独有内容)
- [十、GPU 算力规划](#十gpu-算力规划)
- [十一、与 kai0 原版差异总结](#十一与-kai0-原版差异总结)
- [十二、推荐方案与上手路径](#十二推荐方案与上手路径)

---

## 一、项目概览对比

| 维度 | kai0 (χ₀) | X-VLA |
|------|-----------|-------|
| **论文** | arXiv:2602.09021 (2026.02) | arXiv:2510.10274 (ICLR 2026) |
| **机构** | OpenDriveLab (港大 MMLab) | 清华 AIR |
| **定位** | 资源高效的鲁棒操作框架 | 跨具身通用 VLA 基座模型 |
| **参数量** | ~3B (π₀.₅ 基座) | 0.9B |
| **基座** | openpi (Physical Intelligence π₀/π₀.₅) | Florence2 + 自研 Transformer |
| **动作生成** | Flow Matching (JAX) | Flow Matching (PyTorch) |
| **控制空间** | 关节空间 (14D) | EE6D 末端执行器 (20D) |
| **叠衣服性能** | 250% 超越 π₀.₅ 基线 | 100% 成功率, ~33件/小时 |
| **硬件匹配** | Piper 原生支持 | Piper 有 SoftFold checkpoint |
| **数据开源** | 完整 (base+dagger+advantage) | SoftFold checkpoint 开源, 数据不确定 |
| **License** | Apache 2.0 (代码), CC BY-NC-SA 4.0 (数据/模型) | Apache 2.0 |

---

## 二、kai0 深度解读

### 2.1 核心问题与方法论

kai0 识别出机器人操作中的三类**分布不一致性 (Distributional Inconsistencies)**:

| 分布 | 含义 | 问题 | 对应模块 |
|------|------|------|---------|
| P_train | 人类示教数据 | 欠采样真实解空间 (Coverage Deficiency) | Model Arithmetic |
| Q_model | 策略归纳偏置 | 缺乏阶段感知信号 | Stage Advantage |
| P_test | 部署执行分布 | 推理延迟、无恢复行为 | Train-Deploy Alignment |

数学框架: 有限时域 MDP (S, A, H), 轨迹在动态 s_{t+1} ~ T(·|s_t, a_t, ξ) 下演化。

### 2.2 Module 1: Model Arithmetic（模型算术）

**核心思想**: 将数据分为 n 个子集分别训练, 然后在权重空间合并:

```
θ_merged = Σ α_i θ_i,  where α_i ≥ 0, Σ α_i = 1
```

**六种合并策略**:

| 方法 | 原理 | 特点 |
|------|------|------|
| Average | α_i = 1/n | 最简单, 无需计算 |
| Inverse Loss | α_i ∝ 1/(L_i + ε)^p | 无需梯度, 最快 |
| Gradient Descent | softmax 参数化, Adam 优化 | 精确但较慢 |
| Adaptive GD | 梯度按当前 loss 缩放 | GD 变体 |
| Greedy | 前向逐步选择减少验证 loss 最多的 ckpt | 实验中表现最好 |
| Manual | 用户手动指定权重 | 灵活 |

**关键发现**:
- 用 DAgger 数据作为 OOD 验证集效果远好于 in-domain 验证集
- 子集训练 + 合并 > 全数据训练 (揭示微调 VLA 的参数冗余)

### 2.3 Module 2: Stage Advantage（阶段优势估计）

**动机**: 长时域任务中标准 BC 对所有帧一视同仁。π₀.₆* 的 value difference A(s,a) = V(s') - V(s) 有数值不稳定问题。

**kai0 的方法**: 直接预测优势 + 阶段条件化

```
A(s, a) = f_θ(s, s')                    # 端到端预测进度差
A_stage(s, a, g) = f_θ(s, s' | g)       # 阶段条件化扩展
```

**五步流水线**:

```
Step 0: 人工标注 stage_progress_gt (每帧 0→1)
  ↓
Step 1: 训练优势估计器 (基于 Pi0 架构, PyTorch)
  ↓
Step 2: 用估计器预测 absolute_advantage + relative_advantage
  ↓
Step 3: 离散化为 binary task_index (top 30% → positive)
  ↓
Step 4: AWBC 训练 (prompt="fold the cloth, Advantage: positive")
```

**AWBC 的巧妙设计**: 将优势信号编码到语言条件中:
- task_index=1 → prompt: "fold the cloth, Advantage: positive"
- task_index=0 → prompt: "fold the cloth, Advantage: negative"
- 推理时只用 "positive" prompt

**任务阶段分解**:

| 任务 | 阶段数 | 分解 |
|------|--------|------|
| Task A (展平折叠) | 2 | 展平 → 折叠 |
| Task B (检索分拣) | 4 | 识别 → 取出 → 展平 → 分类摆放 |
| Task C (挂衣服) | 3 | 抓取衣领 → 穿衣架 → 挂起 |

### 2.4 Module 3: Train-Deploy Alignment（训练-部署对齐）

**时序分块平滑 (Temporal Chunk-wise Smoothing)**:

在重叠区域线性插值新旧 action chunk:
```
ã_i = w_i · a_old_i + (1-w_i) · a_new_i
w_i = 1 - i/max(L-1, 1)
```

**DAgger 数据收集**:
- 标准 DAgger: 在线执行策略 → 检测失败 → 人类纠正
- 启发式 DAgger: 手动预置失败状态 → 直接收集恢复示教

**推理模式 (4种)**:

| 模式 | 说明 |
|------|------|
| Synchronous | 每步一次推理, 最简单 |
| Temporal Smoothing | 滑动窗口平均, 降低抖动 (推荐) |
| Temporal Ensembling | 多次前向传播集成 |
| RTC (Real-Time Chunking) | 最先进, 利用已执行动作前缀引导 |

**数据增强**:
- Time Scaling: 每 N 帧提取一帧 (加速动作)
- Space Mirroring: 水平翻转视频 + 交换左右臂 (数据量翻倍)

### 2.5 基座模型架构

基于 Physical Intelligence 的 openpi 框架, 使用 π₀/π₀.₅ 模型:

```
视觉输入 (3×RealSense) → SigLIP (So400m/14) → 图像嵌入
                            ↓
语言指令 → PaliGemma (Gemma 2B) → 多模态前缀 tokens
                            ↓
             Action Expert (Gemma 300M) → 去噪动作预测
                            ↓ (迭代去噪 10-50 步)
             动作序列 (50步 × 14维)
```

**训练目标 (Flow Matching Loss)**:
```
L = E[||v_t - u_t||²]
其中 v_t = 模型预测速度, u_t = noise - action, x_t = t·noise + (1-t)·action
```

**训练参数**: Action chunk K=50, batch 128, 80K steps, AdamW lr=2.5e-5, 8×A100

### 2.6 实验结果

**总体**: χ₀ 相比 π₀.₅ 基线成功率提升约 250%, 仅用 20h 数据 + 88 A100 GPU-hours

**消融 (Task A)**:
- Base → +MA → 吞吐量和成功率提升
- +SA → 吞吐量主导提升
- +TDA → 成功率提升但 retry cost 增加
- +MA+SA+TDA (完整 χ₀) → 最大组合性能

**失败基线**: X-VLA, GO-1, UniVLA, OpenVLA 成功率可忽略不计 (论文原文)

**24小时连续运行验证**: 从任意初始状态出发, 验证了生产级鲁棒性

---

## 三、X-VLA 深度解读

### 3.1 核心创新: Soft Prompt

为每个数据源分配可学习提示向量:
```
P^H = {p_i}_{i=1}^H,  p_i ∈ R^{k×d}  (k=32, d=1024)
```

**四种异构性处理策略对比**:

| 策略 | 问题 |
|------|------|
| Domain-specific projection | 忽略视觉/任务差异 |
| HPT-style projection | 破坏预训练 VLM 表示 |
| Language prompts | 需手工设计, 不可学习 |
| **Soft Prompts (本文)** | **最稳定, 最低验证误差** |

**参数效率**: 每域仅增加 ~33K 参数 (32×1024), 整体仅 0.04% 非共享参数

### 3.2 模型架构

```
High-Dimensional Stream:
  主视角图像 → Florence2-Large → VLM Features
  辅助视角 → 独立视觉编码 → Aux Visual Features

Low-Dimensional Stream:
  本体感知 + 动作 + 时间嵌入 → 轻量线性投影

SoftPromptedTransformer (24层, 1024D, 16头):
  输入 = [Action_tokens | VLM_proj | Aux_proj | Soft_Prompts(domain_id)]
  → DomainAwareLinear → 动作预测

输出: 30步动作序列 [30 × 20D EE6D]
```

**DomainAwareLinear**: per-sample 自定义层
```python
W = self.fc_embedding(domain_id)  # [B, in×out]
y = torch.matmul(x, W.view(B, in, out))
```

**EE6D 动作空间 (20D)**:
```
左臂: xyz(3) + Rot6D(6) + gripper(1) = 10D
右臂: xyz(3) + Rot6D(6) + gripper(1) = 10D
```

### 3.3 训练流水线

**Phase I: 预训练** — 290K episodes, 7 platforms, 5 arm types (DROID, RoboMind, AgiBot)

**Phase II: 域适配 (两步法)**:
1. Prompt Warm-up: 冻结骨干, 仅训练新 soft prompt + action heads
2. Joint Optimization: 解冻骨干, 联合优化 (soft prompt LR × 0.1)

**学习率策略**:
- VLM backbone: LR=0.0 (始终冻结)
- Transformer: base_lr + cosine decay
- Soft prompts: base_lr × 0.1
- Action heads: base_lr

### 3.4 实验结果

**消融实验 (关键)**:

| 累积改进 | 验证误差 | 适配成功率 |
|---------|---------|-----------|
| Baseline (无预训练) | — | 4.1% |
| +Custom LR | — | 39.6% |
| +异构预训练 | 0.11 | 25.0% |
| +Intention Abstraction | 0.077 | 50.0% |
| +Encoding Pipeline | 0.053 | 64.6% |
| **+Soft Prompt** | **0.041** | **73.8%** |
| +Scale Up (0.9B) | 0.032 | 89.6% |
| +Two-Step Adaptation | 0.032 | **95.8%** |

**基准测试**:

| 基准 | X-VLA | 此前最优 |
|------|-------|---------|
| LIBERO | 98.1% | 94.2% (π₀) |
| Simpler-WidowX | 95.8% | 78.0% |
| Calvin ABC→D | 4.43 | — |
| SoftFold (布料) | 100% | — |

**LoRA 效率**: 9M 参数 (模型的 1%) 达到 π₀ 全参数 3B 的 99% 性能

---

## 四、硬件匹配度分析

### 当前配置: 双臂 Piper + 2×D405 (hand) + 1×D435 (third-person)

| 维度 | kai0 | X-VLA |
|------|------|-------|
| 机械臂 | **Piper 原生 (Task A/B)** | SoftFold 用 Agilex (可能是 Piper) |
| 腕部相机 | D435i (需适配 D405) | 不确定型号 |
| 第三视角 | D435i (与 D435 几乎一致) | astra_camera (奥比中光) |
| 相机数量 | 3 路 | 3 路 |
| 部署代码 | **完整 Piper 推理+DAgger** | 有 Piper 客户端但较简略 |
| 数据可用 | **6,512 ep + advantage 标签** | checkpoint 有, 数据不确定 |
| 上手成本 | **改相机参数即可** | 需写 domain handler + 自行适配 |

**结论: kai0 是明确首选**

---

## 五、相机选型与安装分析

### 5.1 D405 vs D435 腕部相机对比

| 参数 | D405 (当前配置) | D435/D435i (kai0 原版) |
|------|---------|---------|
| 设计用途 | 近距 (7cm~50cm) | 中距 (28cm+) |
| RGB FOV | ~58°×51° | ~69°×42° |
| 尺寸 | 42×42×23mm (很小) | 90×25×25mm |
| 重量 | ~56g | ~72g |
| 近距成像 | 优秀 | 28cm 内模糊 |
| 安装螺丝 | M3 + 1/4-20 三脚架孔 | M3 + 1/4-20 三脚架孔 |

### 5.2 D405 安装方案

**当前配置: Cobot Magic ALOHA 夹爪 + D405 一体式支架, 带可调高度**

与 kai0 原版对比:

| 维度 | kai0 原版 | 当前配置 | 优劣 |
|------|----------|---------|------|
| 相机 | D435i | D405 | D405 近距更优, FOV 不同需微调 |
| 安装件 | 3D 打印 `d435i-centered.STEP` | **ALOHA 原装一体式** | 更稳固更标准 |
| 光心 | centered 居中 | **双目居中** | 对称性更好 |
| 高度 | 固定 | **可调** | 可微调到最优 |
| 夹爪 | 3D 打印 agilex-gripper | **ALOHA 原装夹爪** | 工厂品质 |

kai0 图像处理流水线 (D405 完全兼容):

```
相机采集: 640×480 RGB8 @ 30fps  ← D405 原生支持, 参数不需改
  ↓
resize_with_pad: 等比缩放 + 黑边 → 224×224
  ↓
归一化: (uint8 / 255.0) * 2.0 - 1.0 → [-1, 1] float32
  ↓
训练增强: RandomCrop(95%) + Rotate(±5°) + ColorJitter(0.3/0.4/0.5)
```

模型只用 RGB, 不用深度。D405 和 D435 在 pyrealsense2 API 中完全兼容。

### 5.3 D435 Top Camera 位置深度分析

**数据来源**: setup/README.md + 论文 Figure 5 (实物照片) + 几何推算

#### 文档给出的参数

| 参数 | Task A 值 | 说明 |
|------|----------|------|
| 相机高度 | 76 cm | 桌面到光心 |
| mount angle | 30° | 安装角度 (歧义, 见下) |
| Center base → 左副臂 | 34 cm | 相机支架到 master 臂 |
| Center base → 右副臂 | 34 cm | 相机支架到 master 臂 |

**文档未给出**: 相机支架到 slave 臂的前后距离 (关键缺失)

#### "30° mount angle" 的歧义分析

从论文 Figure 5 照片观察: 相机在高杆顶端, **几乎垂直朝下**, 略向前倾。
这与 "30° from vertical" (大幅前倾) 不一致。

最合理解释: **30° 是支架从竖杆伸出的角度, 不是相机视线偏离垂直的角度**。

```
      竖杆
       │
       │╲ 30° ← 支架角度 (从竖杆伸出)
       │  ╲
       │   [D435] ← 相机装在支架末端, 镜头朝下
       │          实际视线约 15-20° from vertical
```

#### 不同角度解释下的视野计算 (高度 76cm, D435 FOV 69°×42°)

| 实际视线角度 | 光轴前方投影 | 视野纵深 | 视野宽度 | 相机距桌前沿 | 在 slave 臂后方 |
|------------|-----------|---------|---------|------------|---------------|
| 15° (最接近照片) | 20cm | 54cm | 108cm | ~29cm | **~14cm** |
| 20° | 28cm | 65cm | 111cm | ~37cm | **~22cm** |
| 30° (字面解读) | 44cm | 82cm | 121cm | ~53cm | **~38cm** |

(以上 "在 slave 臂后方" 假设视野中心对准操作区中心 ~9cm from 桌前沿)

#### 论文 Figure 5 实物照片分析

```
Figure 5 ("Robot setup of our collaborative dual-arm system") 显示:
- 细金属立杆位于两个 slave 臂之间或略偏后
- 相机在杆顶 (~76cm), 几乎垂直朝下
- 支架从杆顶向前伸出, 相机挂在末端
- 4 个臂 (2 slave + 2 master) 围绕中心支架分布
```

#### 推荐安装位置

```
俯视图:

         桌面 (~120×80cm)
    ┌─────────────────────────────────┐
    │                                 │
    │    ┌──────────────────┐         │
    │    │  衣物操作区        │         │
    │    │  0~18cm from 前沿 │         │
    │    └──────────────────┘         │
    │                                 │
    │  左slave ●────39cm────● 右slave  │ ← 距前沿 15cm (平均)
    │          │            │         │
    │          │  10~20cm   │         │
    │          │     ↕      │         │
    │          │     ★      │         │ ← D435 支架底座
    │          │  (居中)     │         │    距前沿 ~25-35cm
    │          │            │         │
    │  左master             右master  │
    └─────────────────────────────────┘
```

**实操方法 (不需要精确测量)**:

1. 竖杆底座放在两臂连线中点正后方 **10-20cm** (距桌前沿 ~25-35cm)
2. 竖杆升到 **76cm**
3. 支架朝操作区方向伸出, 相机镜头朝下
4. 用 ROS 实时查看画面:
   ```bash
   rosrun image_view image_view image:=/camera_f/color/image_raw
   ```
5. 调整前后/角度, 直到:
   - 两个夹爪完整活动范围在画面内
   - 桌面操作区占画面中心 2/3
   - 两臂基座在画面上方 1/4 处可见

> kai0 训练有 RandomCrop(95%) + Rotate(±5°), 对位置微小偏差有容忍度

---

## 六、数据集开源情况

### 基于实际下载验证 (2026-03-27)

**HuggingFace 仓库结构**:

```
OpenDriveLab-org/Kai0 (dataset, 137.86 GB)
├── Task_A/
│   ├── base/      ← 3,055 ep, ~42h ✅
│   ├── dagger/    ← 3,457 ep, ~13h ✅
│   └── advantage/ ← 3,055 ep (含优势标签) ✅
├── Task_B/
│   ├── base/      ← 5,988 ep, ~31h ✅
│   └── dagger/    ← 769 ep, ~22h ✅
│   (无 advantage/) ❌
├── Task_C/
│   ├── base/      ← 6,954 ep, ~61h ✅
│   └── dagger/    ← 685 ep, ~12h ✅
│   (无 advantage/) ❌
└── README.md

OpenDriveLab-org/Kai0 (model, 52.3 GB)
├── Task_A/ ← best checkpoint ✅
├── Task_B/ ← best checkpoint ✅
└── Task_C/ ← best checkpoint ✅
```

### Parquet 列分析 (实际下载验证)

**Task_A/base** (基础演示):
```
columns: observation.state [14], action [14], timestamp, frame_index,
         episode_index, index, task_index,
         progress_gt ✅,         ← 全局进度 (0→1)
         stage_progress_gt ✅    ← 阶段进度 (带子任务分解)
```

**Task_A/advantage** (优势标签完整版):
```
上述所有列 +
relative_advantage ✅    ← 相对优势 (-1~1, mean=0.03)
absolute_value ✅        ← 绝对值 (0~0.87, mean=0.31)
absolute_advantage ✅    ← 绝对优势 (-0.75~0.68, mean=0.02)
task_index: 0=negative (83%), 1=positive (17%)
```

**Task_A/advantage tasks.jsonl**:
```json
{"task_index": 0, "task": "fold the cloth, Advantage: negative"}
{"task_index": 1, "task": "fold the cloth, Advantage: positive"}
```

**Task_B/base, Task_C/base**: 无 progress_gt, 无 advantage 列

### 开源状态总结

| 数据类型 | 状态 | 说明 |
|---------|------|------|
| 基础演示 (base) | ✅ 全部 | 3 任务 15,997 ep / ~134h |
| DAgger 数据 | ✅ 全部 | 3 任务 4,912 ep / ~47h |
| stage_progress_gt | ⚠️ 仅 Task_A | B/C 无此列 |
| advantage 标签 | ⚠️ 仅 Task_A | relative/absolute/discretized |
| 模型检查点 | ✅ 全部 | 3 任务 best model |

**对叠衣服 (Task_A) 的影响**: 数据链条完整, 可直接走 AWBC Step 4 训练

---

## 七、代码架构分析

### 7.1 kai0 代码结构

```
kai0/
├── src/openpi/                    # 核心源码
│   ├── models/                    # Pi0/Pi05/RTC 模型 (JAX)
│   │   ├── model.py              # 基类, IMAGE_RESOLUTION=224
│   │   ├── pi0.py                # 主扩散模型 (Flow Matching)
│   │   ├── pi0_rtc.py            # RTC 变体
│   │   ├── pi0_config.py         # 配置 dataclass
│   │   ├── gemma.py              # Gemma LLM (含 LoRA)
│   │   ├── siglip.py             # SigLIP 视觉编码器
│   │   └── tokenizer.py          # 语言 tokenizer
│   ├── models_pytorch/            # PyTorch 实现
│   │   ├── pi0_pytorch.py        # PI0Pytorch + AdvantageEstimator
│   │   └── gemma_pytorch.py      # PyTorch Gemma
│   ├── training/                  # 训练管线
│   │   ├── config.py             # 所有预设配置 (~1226行)
│   │   ├── data_loader.py        # LeRobot 数据加载
│   │   ├── advantage_dataset.py  # 优势估计器数据集
│   │   ├── checkpoints.py        # Orbax 检查点 I/O
│   │   └── optimizer.py          # Adam + cosine schedule
│   ├── policies/                  # 策略封装
│   │   ├── policy.py             # 主推理封装
│   │   ├── agilex_policy.py      # Agilex 变换
│   │   └── arx_policy.py         # ARX 变换
│   ├── serving/                   # WebSocket 部署
│   ├── shared/                    # 共享工具
│   │   ├── array_typing.py       # 类型注解
│   │   ├── normalize.py          # 归一化统计
│   │   └── image_tools.py        # resize_with_pad
│   └── transforms.py             # 数据变换管线
├── model_arithmetic/              # 模型合并
│   ├── arithmetic.py             # JAX checkpoint 合并
│   ├── arithmetic_torch.py       # PyTorch 合并
│   ├── common.py                 # mix_params, compute_optimal_weights
│   ├── dump_data.py              # 导出验证数据
│   └── split_data.py             # 数据集切分
├── stage_advantage/               # 阶段优势
│   ├── annotation/
│   │   ├── evaluator.py          # 优势估计器推理
│   │   ├── eval.py               # 数据集标注
│   │   └── discretize_advantage.py # 离散化
│   └── awbc/                     # AWBC 训练
├── train_deploy_alignment/        # 训练-部署对齐
│   ├── data_augment/             # 数据增强
│   │   ├── time_scaling.py
│   │   ├── space_mirroring.py
│   │   └── convert_h5_lerobot.py
│   ├── dagger/agilex/            # Piper DAgger
│   ├── dagger/arx/               # ARX DAgger
│   ├── inference/agilex/         # Piper 推理 (4种模式)
│   └── inference/arx/            # ARX 推理
├── scripts/                       # 入口脚本
│   ├── train.py                  # JAX 训练
│   ├── train_pytorch.py          # PyTorch 训练
│   ├── serve_policy.py           # 策略服务器
│   └── compute_norm_states_fast.py
├── packages/openpi-client/        # 轻量客户端库
├── setup/                         # 3D打印件 + 硬件布局
└── docs/                          # 数据集/推理文档
```

### 7.2 X-VLA 代码结构

```
X-VLA/
├── models/                        # 核心模型
│   ├── modeling_xvla.py          # XVLA 主类 (295行)
│   ├── configuration_xvla.py    # 配置 (24层, 1024D, 30域)
│   ├── processing_xvla.py       # 多模态处理器
│   ├── transformer.py            # SoftPromptedTransformer (403行)
│   ├── action_hub.py             # 动作空间注册 (EE6D/Joint/Auto)
│   └── modeling_florence2.py     # Florence2 编码器
├── datasets/                      # 数据加载
│   ├── dataset.py                # InfiniteDataReader
│   ├── domain_config.py          # 域权重 & ID 映射
│   └── domain_handler/           # 每个数据集的处理器
├── evaluation/                    # 基准评估
│   ├── SoftFold-Agilex/          # Piper 叠衣评估
│   │   ├── deploy/client_eef6d_xvla.py  # 推理客户端
│   │   ├── deploy/utils/rosoperator.py  # ROS 桥接
│   │   ├── deploy/utils/rotation.py     # 旋转工具
│   │   └── Piper_ros_private-ros-noetic/ # Piper ROS 包
│   ├── libero/, simpler/, calvin/, robotwin-2.0/, vlabench/
├── train.py                      # 全参数训练 (282行)
├── peft_train.py                 # LoRA 微调
└── deploy.py                     # FastAPI 推理服务器
```

### 7.3 源码依赖关系

**kai0 分层依赖 (无循环依赖)**:

```
Layer 0: 外部库 (JAX, Flax, PyTorch, Orbax, LeRobot, HF Transformers)
Layer 1: shared/ (array_typing, normalize, image_tools, download)
Layer 2: models/ (model.py → pi0.py → pi0_rtc.py, gemma, siglip)
Layer 3: training/ (config.py, data_loader.py, checkpoints.py, optimizer.py)
Layer 4: policies/ + serving/ (policy.py, websocket_server)
Layer 5: 应用模块 (model_arithmetic, stage_advantage, tda)
Layer 6: scripts/ (train.py, serve_policy.py)
```

**最核心的 5 个文件**:

| 文件 | 被依赖次数 | 角色 |
|------|-----------|------|
| shared/array_typing.py | ~30+ | 类型系统基础 |
| models/model.py | ~20+ | 模型基类+常量 |
| training/config.py | ~15+ | 配置中枢 |
| transforms.py | ~12+ | 数据变换管线 |
| shared/normalize.py | ~10+ | 归一化统计 |

**JAX 与 PyTorch 双路径**:
```
JAX 路径:  model.py → pi0.py → train.py → checkpoints.py (Orbax)
PyTorch:   pi0_pytorch.py → train_pytorch.py → safetensors
共用:      config.py, data_loader.py
```

---

## 八、两个 repo 推理代码对比

### kai0: agilex_inference_openpi_temporal_smoothing.py

```python
# 通信: WebSocket (openpi_client)
from openpi_client import image_tools, websocket_client_policy

# 图像处理: BGR→RGB, resize 224×224, CHW
imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
imgs = image_tools.resize_with_pad(np.array(imgs), 224, 224)

# payload
payload = {
    "state": proprio,          # 14D 关节角度
    "images": {
        "top_head":  imgs[0].transpose(2, 0, 1),
        "hand_right": imgs[1].transpose(2, 0, 1),
        "hand_left":  imgs[2].transpose(2, 0, 1),
    },
    "prompt": "fold the cloth",
}
actions = policy.infer(payload)["actions"]  # [50, 14] 关节空间

# 发布: 直接 JointState
joint_state_msg.position = action[:7]   # 左臂
puppet_arm_left_publisher.publish(...)
```

### X-VLA: client_eef6d_xvla.py

```python
# 通信: HTTP POST (requests + json_numpy)
query = {
    "proprio": json_numpy.dumps(proprio),    # 20D EE6D
    "image0": json_numpy.dumps(main_view),
    "image1": json_numpy.dumps(left_wrist),
    "image2": json_numpy.dumps(right_wrist),
    "language_instruction": "flatten the cloth...",
    "steps": 10,
    "domain_id": 5
}
response = requests.post(url, json=query)
action = response.json()['action']  # [30, 20] EE6D

# 后处理: 6D → Euler → PosCmd
action = abs_6d_2_abs_euler(action)  # 20D → 14D (xyz+euler+grip ×2)
ros_operator.eef_arm_publish(left_action, right_action)  # PosCmd
```

### 关键差异

| 维度 | kai0 | X-VLA |
|------|------|-------|
| 通信协议 | WebSocket (低延迟) | HTTP POST (每次新连接) |
| 控制空间 | 关节角 14D (不需IK) | EE6D 20D → Euler 14D (需IK) |
| chunk_size | 50 步 | 30 步 |
| 执行频率 | ~100 Hz (关节) | 15 Hz (EEF) |
| 推理频率 | ~4 Hz (异步) | ~15 Hz (同步) |
| 时序平滑 | StreamActionBuffer (指数衰减) | 无 (直接执行 chunk) |
| 图像格式 | uint8 CHW via msgpack | uint8 HWC via json_numpy |

---

## 九、kai0 官网独有内容

来源: https://mmlab.hk/research/kai0 (论文中未包含)

### 9.1 "Veni, vidi, vici" 哲学

官网用凯撒名言串联三阶段 (论文中不存在此框架):
- **Veni** (我来了) → 数据采集 (DAgger, 时空增强)
- **Vidi** (我看了) → 模型训练 (Model Arithmetic, Stage Advantage)
- **Vici** (我征服了) → 真机部署 (时序平滑, RTC)

### 9.2 Bottom Line (四条实践洞察, 论文中没有)

1. **"不是所有机器人数据都同等有价值"** — 数据质量 > 数据量
2. **"基座策略的能力很关键, 但知道如何快速评估和选择基座策略更重要"**
3. **"Model Arithmetic 竟然能点石成金"** — 多个平庸模型合并后超越 oracle
4. **"阶段条件化优势估计仍有改进空间"** — 承认 Stage Advantage 不完美

### 9.3 独有可视化资源

| 资源 | 说明 | 论文中有? |
|------|------|----------|
| 3D t-SNE 交互图 | P_train/Q_model/P_test 分布对齐, 可拖拽旋转 | 无 |
| 4 个动画 GIF | DAgger/时序平滑/分布动态/模型合并 过程动画 | 无 (静态图) |
| 开场演示视频 | 3 任务 4K 100x 快放, 关键片段 2-5x | 无 |
| 完整流水线视频 | 4K 30fps 全流程 | 无 |

### 9.4 对复现的关键启示

Model Arithmetic 的**研发迭代效率**: 只训练新数据子集模型 → 与已有 ckpt 合并 → 每次迭代数小时 (vs 全量重训 ~1天)。在双节点 A100 集群上可并行训练多个子集, 大幅加速。

---

## 十、GPU 算力规划

### 10.1 算力分配

```
训练集群 (远端, 14.103.44.161):
  gf0 (ssh -p 55555): 8×A100 80GB → Full 微调 (batch=256) + AWBC + Model Arithmetic
  gf1 (ssh -p 11111): 8×A100 80GB → 数据子集训练 + Advantage 估计器
  外网访问: SSH 反向隧道代理 (:29290), 登录即生效

工控机/推理机 (桌边):
  sim01 (tim@sim01): 双卡 RTX 5090 32GB
  GPU 0: serve_policy.py 推理 (~8GB, :8000)
  GPU 1: 备用 / Model Arithmetic 小规模验证
  兼任 IPC: ROS + 相机 + CAN + Piper SDK
  工作目录: /data1/tim/workspace/deepdive_kai0/kai0
```

### 10.2 关键优势: 训练配置与 kai0 原版完全一致

- batch_size=256 **不需改**
- fsdp_devices=1 **不需改** (单卡 A100 80GB 够)
- gf0 + gf1 可并行训练不同实验, 速度翻倍

### 10.3 工作流

```
[gf0/gf1] 训练 → checkpoint
    │ rsync -e "ssh -p 55555" → [sim01] 推理验证 (host=localhost)
    ↑                                      ↓
    └──── rsync 数据 ←──── [sim01] DAgger 采集
```

---

## 十一、与 kai0 原版差异总结

| 差异点 | kai0 原版 | 当前配置 | 影响 | 处理 |
|--------|----------|---------|------|------|
| 腕部相机 | D435i | D405 (ALOHA 原装) | FOV/色彩差异 | 微调解决 |
| 腕部安装 | 3D 打印 d435i-centered | ALOHA 一体式+可调高度 | 实际更优 | 不需处理 |
| 头部相机 | D435i | D435 | 仅少 IMU | 不需处理 |
| 夹爪 | 3D 打印 agilex-gripper | ALOHA 原装 | 工厂品质 | 直接用 |
| 训练 GPU | 8×A100 | **gf0+gf1 各 8×A100** | **完全一致+更强** | 零改动 |
| 推理 GPU | RTX 4090 | **sim01 双 5090** | 更强 | 不需处理 |
| IPC 架构 | IPC + GPU 主机分离 | **sim01 一体机** | 零网络延迟 | host=localhost |
| batch_size | 256 | **256** | **零差异** | 零改动 |
| config.py | 原版 | 仅改路径 | 零差异 | 只改 repo_id |

**核心结论**: 训练侧与 kai0 原版完全一致 (gf0/gf1)。sim01 一体机方案推理延迟更低。唯一差异是 D405 vs D435i 视觉域, 微调即可解决。

---

## 十二、推荐方案与上手路径

### 上手时间线

```
Day 1-2:  硬件搭建 [sim01]
  ├── 按尺寸图放置双臂 + ALOHA 夹爪/D405 已就位
  ├── D435 top camera (竖杆 76cm, slave 臂后方 10-20cm, 居中)
  └── CAN + USB 接线验证 (全部接 sim01)

Day 3:    软件环境 (三台机器并行)
  ├── [gf0] ssh -p 55555 → uv sync + 下载数据/checkpoint
  ├── [gf1] ssh -p 11111 → uv sync + 下载数据/checkpoint
  └── [sim01] uv sync + conda kai0_inference + ROS + openpi-client

Day 4:    首次推理 [sim01]
  ├── serve_policy.py (GPU 0, :8000)
  ├── 多终端流程跑通 (host=localhost)
  └── 观察动作, 调试链路

Day 5-7:  数据采集 [sim01] + rsync -e "ssh -p 55555" 到 gf0
Day 8-10: [gf0] 训练 (与原版完全一致) → rsync ckpt 回 sim01 验证
Day 11+:  [gf0] Model Arithmetic + [gf1] AWBC + [sim01] DAgger 迭代
```

### 代码改动清单 (仅 3 处)

1. `my_multi_camera.launch` — 填入 D405/D435 serial number
2. `config.py` — `repo_id` 和 `weight_loader` 改为你的路径
3. 推理脚本 — `lang_embeddings = "fold the cloth"` (或 AWBC: `"..., Advantage: positive"`)

---

## 附录 A: kai0 文档索引 (20 个文件)

| # | 路径 | 内容 |
|---|------|------|
| 1 | README.md | 项目总览、三模块说明 |
| 2 | docs/dataset.md | 数据集说明、LeRobot v2.1 格式 |
| 3 | docs/norm_stats_fast.md | 快速归一化统计 |
| 4 | docs/tda_remote_inference.md | server-client 架构 |
| 5 | setup/README.md | 硬件尺寸、3D 打印件 |
| 6 | model_arithmetic/README.md | 6 种合并方法 |
| 7 | stage_advantage/README.md | Step 0-4 流水线 |
| 8 | stage_advantage/annotation/README.md | Step 1-3 |
| 9 | stage_advantage/awbc/README.md | Step 4 AWBC |
| 10 | train_deploy_alignment/README.md | TDA 总览 |
| 11 | train_deploy_alignment/inference/README.md | 推理总览 |
| 12 | train_deploy_alignment/inference/agilex/README.md | Piper 推理 |
| 13 | train_deploy_alignment/inference/arx/README.md | ARX 推理 |
| 14 | train_deploy_alignment/dagger/README.md | DAgger 总览 |
| 15 | train_deploy_alignment/dagger/agilex/README.md | Piper DAgger |
| 16 | train_deploy_alignment/dagger/arx/README.md | ARX DAgger |
| 17 | train_deploy_alignment/data_augment/README.md | 数据增强 |
| 18 | train_deploy_alignment/data_augment/mini_lerobot/README.md | Mini LeRobot |
| 19 | scripts/docker/docker_README.md | Docker (未测试) |
| 20 | Piper_ros.../README(EN).md | Piper SDK + CAN |

## 附录 B: X-VLA 文档索引 (8 个文件)

| # | 路径 | 内容 |
|---|------|------|
| 1 | README.md | 项目总览、安装、EE6D |
| 2 | evaluation/SoftFold-Agilex/readme.md | Piper 叠衣评估 |
| 3 | evaluation/libero/README.md | LIBERO 评估 |
| 4 | evaluation/libero/preprocess.md | 动作预处理 |
| 5 | evaluation/simpler/README.md | WidowX/Google Robot |
| 6 | evaluation/calvin/README.md | CALVIN 评估 |
| 7 | evaluation/robotwin-2.0/README.md | RoboTwin 双臂 |
| 8 | evaluation/vlabench/README.md | VLABench |

## 附录 C: 外部依赖清单

### kai0

| 类别 | 库 |
|------|---|
| JAX 生态 | jax 0.5.3, flax 0.10.2, optax, orbax-checkpoint 0.11.13, jaxtyping, augmax |
| PyTorch | torch 2.7.1, safetensors, transformers 4.53.2 |
| 数据 | lerobot, pandas, pyarrow, h5py, av (PyAV) |
| 机器人 | rospy/rclpy, pyrealsense2, websockets, msgpack, piper_sdk |
| 其他 | wandb, tyro, tqdm, sentencepiece, pydantic, beartype, einops |

### X-VLA

| 类别 | 库 |
|------|---|
| PyTorch | torch 2.1.*, transformers ≤4.51.3, accelerate 1.2.1, peft 0.17.1 |
| 数据 | h5py, pyarrow, av, mmengine |
| 推理 | fastapi, uvicorn, json_numpy |
| 视觉 | timm, einops, opencv, mediapy |
