# deepdive_kai0 vs 官方 kai0 差异、风险分析与修复记录

> 审查日期: 2026-04-02
> 对比对象: 本项目 `/data1/tim/workspace/deepdive_kai0/kai0/` vs 官方 `/home/tim/workspace/kai0/`
> 范围: 推理全链路 (相机采集 → 图像预处理 → 模型推理 → 动作输出 → 机器人执行)

---

## 一、与官方完全一致的组件

以下环节经逐行对比确认与官方 kai0 **代码完全一致**:

| 组件 | 关键文件 |
|------|---------|
| Policy 变换 (obs→model input) | `src/openpi/policies/agilex_policy.py` |
| Policy 配置 (camera rename, action_dim=14) | `src/openpi/policies/policy_config.py` |
| 图像 resize (`resize_with_pad` → 224x224) | `packages/openpi-client/src/openpi_client/image_tools.py` |
| 归一化/反归一化公式 | `src/openpi/shared/normalize.py` — quantile: `(x-q01)/(q99-q01+1e-6)*2-1` |
| 模型配置 (Pi0.5, action_dim=32, horizon=50) | `src/openpi/models/pi0_config.py` |
| Temporal smoothing 算法 | 线性插值权重, m=8, k=8, alpha=0.25 |
| 关节顺序 | left[0:7] + right[0:7], 14 维 |
| Gripper offset | `-0.003`, clamp >= 0 |
| Action 过滤 (超 ±π 置零) | `agilex_policy.py` |
| WebSocket 序列化协议 | `websocket_client_policy.py`, `websocket_policy_server.py` |
| 服务启动脚本 | `scripts/serve_policy.py` |
| 检查点加载 | `src/openpi/training/weight_loaders.py` |
| 归一化统计计算 | `scripts/compute_norm_stats.py` / `compute_norm_states_fast.py` |
| 视频录制 fps | 30fps (DAgger 采集) |

---

## 二、已识别差异与风险修复记录

### 风险 1: 相机采集分辨率宽高比不一致 — **已修复**

**问题**: `scripts/launch_3cam.py` 中 D435 配置为 1920x1080 (16:9), D405 配置为 1280x720 (16:9), 但训练数据为 640x480 (4:3)。`resize_with_pad` 到 224x224 时, 不同宽高比产生不同的 padding 区域, 导致模型看到的图像分布偏移。

**修复**: 三相机统一改为 640x480 @ 30fps, 与训练数据宽高比一致。

```
D435 (top)    → 640x480 RGB + 640x480 Depth @ 30fps
D405-A (left) → 640x480 RGB + 640x480 Depth @ 30fps
D405-B (right)→ 640x480 RGB + 640x480 Depth @ 30fps
```

**文件**: `scripts/launch_3cam.py`

---

### 风险 2: 语言 Prompt 与 checkpoint 不匹配 — **已修复**

**问题**: 官方推理脚本使用 `"fold the sleeve"`, 我们训练配置 `pi05_flatten_fold_normal` 的 `default_prompt` 为 `"Flatten and fold the cloth."`。π0.5 是 language-conditioned 模型, prompt 不同 → action 完全不同。

**分析**: `InjectDefaultPrompt` (transforms.py:109) 仅在 data 中不存在 `"prompt"` 时注入默认值。推理脚本的 `lang_embeddings` 通过客户端 payload 发送, 优先级高于服务端 `default_prompt`。因此推理脚本中的 prompt 必须与训练时的 prompt 严格一致。

**修复**: 4 个 Agilex 推理脚本统一为 `"Flatten and fold the cloth."`:

| 脚本 | 修复前 | 修复后 |
|------|--------|--------|
| `temporal_smoothing.py` | 已正确 | — |
| `temporal_smoothing_ros2.py` | 已正确 | — |
| `temporal_ensembling.py` | `"fold the sleeve"` | `"Flatten and fold the cloth."` |
| `rtc.py` | `"fold the sleeve"` | `"Flatten and fold the cloth."` |

**文件**: `train_deploy_alignment/inference/agilex/inference/` 下 4 个脚本

---

### 风险 3: norm_stats 与 checkpoint 不匹配 — **已验证一致**

**分析**: 推理时 `create_trained_policy()` 从 `checkpoint_dir/assets/<asset_id>/norm_stats.json` 加载归一化统计量。三份 norm_stats.json 的 MD5 完全相同 (`b206072cbfeada1b907c1fa1f649e83e`):

- `data/Task_A/base/norm_stats.json` — 训练时生成的原始文件
- `checkpoints/Task_A/mixed_1/assets/.../norm_stats.json` — checkpoint 内嵌副本 (推理实际加载)
- `checkpoints/Task_A/mixed_1/norm_stats.json` — 顶层便捷副本

**结论**: 无风险, 训练-推理归一化一致。

---

### 风险 4: ROS2 Topic 路径静默失败 — **已修复**

**问题**: `launch_3cam.py` 中 `name='camera_f', namespace='camera_f'` 产生 topic `/camera_f/camera_f/color/image_raw`, 但推理脚本默认订阅 `/camera_f/color/image_raw`。ROS2 订阅未连接的 topic **不报错**, 推理节点会永远收不到图像。

**修复**: 三个相机 `namespace` 改为空字符串:

```python
# 修复前: namespace='camera_f' → /camera_f/camera_f/color/image_raw
# 修复后: namespace=''          → /camera_f/color/image_raw (与推理脚本默认值匹配)
```

**文件**: `scripts/launch_3cam.py`

---

### 风险 5: XLA 编译缓存重启丢失 — **已修复**

**问题**: `JAX_COMPILATION_CACHE_DIR=/tmp/xla_cache`, 重启后 `/tmp` 清空, 每次冷启动需 30-120s JIT 编译。

**修复**: 改为持久路径 `/data1/tim/workspace/deepdive_kai0/.xla_cache`。

**文件**: `scripts/start_server_xla_cache.sh`, `scripts/test_policy_both_mode.sh`, `scripts/test_policy_ros2_mode.sh`

---

### 风险 6: 无关节安全限位 — **已修复 (deepdive_kai0 增强)**

**官方现状**: 官方 kai0 在整条推理链路中不做关节位置/速度 clamp:
- `agilex_policy.py` 仅将超出 ±π 的异常值置零 (过滤 padding 噪声, 非限位)
- 推理脚本仅对 gripper 做 `max(0, gripper - 0.003)`
- `rtc.py` 中 `joint_actions_clip()` 为空实现 (`pass`)
- Piper ROS 节点仅上报限位错误标志, 不做 clamp
- 完全依赖 Piper 固件层硬件限位

**修复**: 新增 `action_safety.py` 模块, 基于官方 URDF (`piper_description.urdf`) 限位:

| Joint | Lower (rad) | Upper (rad) | Velocity (rad/s) |
|-------|------------|------------|-------------------|
| 0 | -2.618 | 2.618 | 5 |
| 1 | 0.000 | 3.140 | 5 |
| 2 | -2.967 | 0.000 | 5 |
| 3 | -1.745 | 1.745 | 5 |
| 4 | -1.220 | 1.220 | 5 |
| 5 | -2.0944 | 2.0944 | 3 |
| gripper | 0.000 | 0.035 | 1 |

**URDF 来源**: `Piper_ros_private-ros-noetic/src/piper_description/urdf/piper_description.urdf`

**功能**:
- 关节位置 clamp — 限制在 URDF 物理极限内
- 关节速度 clamp — 基于 URDF velocity 字段, 按 publish_rate 换算每步最大 delta
- 左右臂独立实例, 各自维护速度历史

**开关配置**:
```bash
# 默认启用 (推荐)
python agilex_inference_openpi_temporal_smoothing_ros2.py ...

# 禁用 (与官方 kai0 行为一致)
python ... --disable_joint_safety

# 仅禁用速度限幅
python ... --no_safety_velocity_clamp

# 更保守的速度限制 (URDF 的 50%)
python ... --safety_velocity_scale 0.5
```

**集成到 5 个推理脚本**, 所有改动处标注 `[deepdive_kai0 增强]` 和 `官方 kai0 无此功能`:

| 脚本 | 动作下发点数量 |
|------|--------------|
| `temporal_smoothing_ros2.py` | 1 |
| `temporal_smoothing.py` | 1 |
| `temporal_ensembling.py` | 1 |
| `rtc.py` | 2 |
| `sync.py` | 1 |

**文件**: `train_deploy_alignment/inference/agilex/inference/action_safety.py` (新增) + 5 个推理脚本

---

### 风险 7: Deque 帧积压竞态 — **已优化 (deepdive_kai0 增强)**

**官方现状**: 所有 sensor deque 使用 `deque()` (无 maxlen), 回调中手动 `if len >= 2000: popleft()`, 存在 check-then-act 竞态。官方 ROS1 版本行为完全一致。

**优化**: ROS2 版改为 `deque(maxlen=2000)`, 回调简化为直接 `append`:
- `maxlen` 溢出由 CPython C 层处理, 原子操作更线程安全
- 去掉 9 个回调中的手动 overflow 检查

**文件**: `train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_temporal_smoothing_ros2.py`

---

### 风险 8: Test 脚本 Topic 路径错误 — **已修复**

**问题**: `test_policy_both_mode.sh` 和 `test_policy_ros2_mode.sh` 中 3 处 topic 错误:

| 位置 | 修复前 | 修复后 |
|------|--------|--------|
| front | `/camera/camera/color/image_raw` (缺 `f`) | `/camera_f/color/image_raw` |
| left | `/camera_l/camera/color/image_rect_raw` | `/camera_l/color/image_raw` |
| right | `/camera_r/camera/color/image_rect_raw` | `/camera_r/color/image_raw` |

**文件**: `scripts/test_policy_both_mode.sh`, `scripts/test_policy_ros2_mode.sh`

---

### 风险 9: 帧同步精度松散 — **未修复 (与官方一致)**

**现状**: `get_frame()` 使用 `min(latest_timestamp)` 跨 3 路相机对齐, 关节状态仅要求 `>= frame_time`, 不参与同步时间计算。左右臂关节可能来自不同时刻, 无最大容差检查。

**官方同款**: 官方 ROS1 版本同一逻辑, 行为完全一致。

**优化方案** (待实施):

| 方案 | 改动量 | 效果 | 推荐 |
|------|--------|------|------|
| **A: 容差检查 + 诊断日志** | 小 | 在现有逻辑上增加: 各传感器实际取帧时间戳差超过阈值 (如 50ms) 时 warn + 丢弃; 关节状态纳入 frame_time 计算 | 推荐先做 |
| **B: ROS2 ApproximateTimeSynchronizer** | 大 | `message_filters` 原生多传感器同步, slop 参数精确控制容差; 需重构回调和 deque 体系 | 标准做法, 但改动大 |
| **C: RealSense 硬件同步** | 硬件 | D435/D405 GPIO 同步触发, 从源头消除时间差 | 效果最好, 需额外接线 |

---

### 风险 10: 单 GPU 使用 — **未修复 (与官方一致)**

**现状**: `CUDA_VISIBLE_DEVICES=0` 仅使用 1 张 RTX 5090 (32GB), 无故障转移到 GPU 1。

**评估**: 32GB 对 Pi0.5 推理足够, 非紧迫问题。可在未来需要并行服务多模型时利用 GPU 1。

---

### 风险 11: `verify_inference_quality.py` 关节限位数据错误 — **已修复**

**问题**: 验证脚本中的 `JOINT_LIMITS` 使用对称近似值, Joint 1/2/4/5 与 URDF 严重不符。

**修复**: 替换为 URDF 权威值:

| Joint | 修复前 | 修复后 (URDF) |
|-------|--------|--------------|
| 0 | [-2.6, 2.6] | [-2.618, 2.618] |
| 1 | [-1.6, 1.6] | [0.0, 3.14] |
| 2 | [-1.6, 1.6] | [-2.967, 0.0] |
| 3 | [-1.8, 1.8] | [-1.745, 1.745] |
| 4 | [-1.6, 1.6] | [-1.22, 1.22] |
| 5 | [-2.6, 2.6] | [-2.0944, 2.0944] |
| gripper | [0, 0.08] | [0, 0.035] |

**文件**: `scripts/verify_inference_quality.py`

---

## 三、非功能性差异 (不影响推理结果)

| 差异 | 我们 | 官方 | 影响 |
|------|------|------|------|
| WebSocket 超时 | `ping/close/open_timeout=300s` | 库默认 (~20s) | 防止首次 JIT 编译时连接断开, 不改变 action 值 |
| `pyproject.toml` 额外依赖 | +`av==13.1.0`, +`mujoco>=3.0.0` | 无 | 仅增加了依赖锁定 |
| `training/config.py` 路径 | 本机实际路径 | 占位符 | 仅影响训练, 不影响推理 |
| `training/config.py` episodes 字段 | 新增 `episodes: list[int] \| None = None` | 无 | 默认 `None` = 全部 episode, 行为一致 |
| ROS2 集成脚本 | `*_ros2.py` (3 个新文件) | 无 | 我们的 ROS2 适配, 官方仅有 ROS1 |
| `.github/` CI | 我们无 (顶层自有) | 有 | 不影响运行 |

---

## 四、文件变更清单

### 修改的官方文件

| 文件 | 改动内容 |
|------|---------|
| `train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_temporal_smoothing.py` | prompt 修正, action_safety 集成, add_safety_args |
| `train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_temporal_smoothing_ros2.py` | action_safety 集成, deque maxlen 优化, 回调简化 |
| `train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_temporal_ensembling.py` | prompt 修正, action_safety 集成 |
| `train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_rtc.py` | prompt 修正, action_safety 集成 (2 处), joint_actions_clip 注释 |
| `train_deploy_alignment/inference/agilex/inference/agilex_inference_openpi_sync.py` | action_safety 集成 |
| `src/openpi/serving/websocket_policy_server.py` | +ping_timeout/close_timeout=300 |
| `packages/openpi-client/src/openpi_client/websocket_client_policy.py` | +ping/close/open_timeout=300 |
| `src/openpi/training/config.py` | 本地路径 + episodes 字段 |
| `pyproject.toml` | +av, +mujoco 依赖 |

### 新增文件

| 文件 | 说明 |
|------|------|
| `train_deploy_alignment/inference/agilex/inference/action_safety.py` | 关节安全限位模块 (URDF 限位 + 速度 clamp + 开关配置) |

### 项目级文件

| 文件 | 改动内容 |
|------|---------|
| `scripts/launch_3cam.py` | namespace 修空, 分辨率统一 640x480 |
| `scripts/start_server_xla_cache.sh` | XLA cache 持久化路径 |
| `scripts/test_policy_both_mode.sh` | topic 路径修正 + XLA cache 路径 |
| `scripts/test_policy_ros2_mode.sh` | topic 路径修正 + XLA cache 路径 |
| `scripts/verify_inference_quality.py` | 关节限位数据修正为 URDF 值 |

---

## 五、标注约定

所有与官方行为不同的代码改动均使用以下标注:

- `[deepdive_kai0 增强]` — 我们新增的功能增强
- `官方 kai0 无此功能` — 明确标注官方不具备此能力
- `[deepdive_kai0 优化]` — 对官方同款问题的改进

禁用增强功能可恢复与官方一致的行为:
```bash
--disable_joint_safety          # 禁用关节安全限位
```
