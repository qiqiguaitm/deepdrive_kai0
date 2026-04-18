# ROS2 图像处理与推理结果校验策略 Review

> 审查日期: 2026-03-29
> 对象: `policy_inference_node.py` (ROS2) vs `agilex_inference_openpi_temporal_smoothing.py` (原版 ROS1)
> 范围: 图像采集 → 预处理 → 模型推理 → 动作输出全链路

---

## 一、图像管线逐步对比

### 1.1 图像采集: `imgmsg_to_cv2`

| 环节 | 原版 (ROS1) | ROS2 版 | 等价? |
|------|------------|---------|-------|
| cv_bridge 调用 | `imgmsg_to_cv2(msg, "passthrough")` | `imgmsg_to_cv2(msg, 'bgr8')` | **有区别, 需分析** |
| RealSense 默认编码 | ROS1 driver 默认 `bgr8` | ROS2 driver 默认 `rgb8` | 不同 |
| 输出格式 | BGR (passthrough 保持原编码) | BGR (cv_bridge 自动将 rgb8→bgr8) | **等价** |

**分析**:
- ROS1: RealSense driver 默认发 `bgr8` → `passthrough` 直接取原始字节 → BGR 数组
- ROS2: RealSense driver 默认发 `rgb8` → cv_bridge 检测源编码, 自动将 `rgb8` 转为 `bgr8` → BGR 数组
- 两者**最终输出均为 BGR**, 因此等价

**潜在风险 (低)**:
- 如果 ROS2 RealSense 被手动配置为 `bgr8` 编码, cv_bridge 的 `'bgr8'` target 不做转换 → 仍为 BGR → 正确
- 如果配置为其他格式 (如 `yuv422`, `rgba8`), cv_bridge 仍会尝试转换 → 通常正确, 但未经测试
- **结论**: ROS2 版的 explicit `'bgr8'` 比原版 `"passthrough"` 实际**更鲁棒**, 因为它保证输出始终为 BGR, 不依赖 driver 默认编码

### 1.2 JPEG Mapping (训练-部署对齐)

```python
# 原版
def jpeg_mapping(img):
    img = cv2.imencode(".jpg", img)[1].tobytes()
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    return img

# ROS2 版
@staticmethod
def _jpeg_mapping(img):
    img = cv2.imencode(".jpg", img)[1].tobytes()
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    return img
```

**完全相同**, 逐字符一致。

作用: 模拟训练数据中 MP4 视频编解码的压缩 artifacts。
- 输入 BGR → `imencode` 按 BGR 编码为 JPEG → `imdecode(IMREAD_COLOR)` 返回 BGR
- 输出始终为 BGR, 与输入通道顺序无关 (JPEG 内部转 YCbCr 再转回)

### 1.3 BGR→RGB 转换

```python
# 原版
imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image_arrs]

# ROS2 版
imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
```

**完全相同**。

### 1.4 Resize with Pad

```python
# 原版
imgs = image_tools.resize_with_pad(np.array(imgs), 224, 224)

# ROS2 版 (增加了不同分辨率的 fallback)
if imgs[0].shape == imgs[1].shape == imgs[2].shape:
    imgs = list(image_tools.resize_with_pad(np.array(imgs), 224, 224))
else:
    imgs = [image_tools.resize_with_pad(im[np.newaxis], 224, 224)[0] for im in imgs]
```

**差异**: ROS2 版增加了对不同分辨率相机的处理 (D435 vs D405 可能分辨率不同)。

`resize_with_pad` 实现分析 (`packages/openpi-client/src/openpi_client/image_tools.py`):
```python
def resize_with_pad(images, height, width, method=Image.BILINEAR):
    if images.shape[-3:-1] == (height, width):
        return images  # 短路: 已经是目标尺寸
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([
        _resize_with_pad_pil(Image.fromarray(im), height, width, method=method)
        for im in images
    ])
```

关键: `Image.fromarray(im)` — PIL 将 3 通道 uint8 数组视为 RGB。
- 输入已经是 RGB (上一步 BGR→RGB) → PIL 正确解释 → 操作后通道顺序不变 → 输出仍为 RGB

**结论**: 当所有相机分辨率一致时 (正常情况), ROS2 版与原版完全等价。fallback 路径也正确。

### 1.5 HWC→CHW 转换

```python
# 原版
"top_head": image_arrs[0].transpose(2, 0, 1),
"hand_right": image_arrs[1].transpose(2, 0, 1),
"hand_left": image_arrs[2].transpose(2, 0, 1),

# ROS2 版
'top_head':   imgs[0].transpose(2, 0, 1),
'hand_right': imgs[1].transpose(2, 0, 1),
'hand_left':  imgs[2].transpose(2, 0, 1),
```

**完全相同**。

### 1.6 相机排列顺序

| 索引 | 原版 | ROS2 版 | payload key |
|------|------|---------|-------------|
| 0 | `config["camera_names"][0]` = `"cam_high"` | `img_front` | `top_head` |
| 1 | `config["camera_names"][1]` = `"cam_right_wrist"` | `img_right` | `hand_right` |
| 2 | `config["camera_names"][2]` = `"cam_left_wrist"` | `img_left` | `hand_left` |

**完全一致**: front→top_head, right→hand_right, left→hand_left。

---

## 二、模型输入验证分析

### 2.1 `AgilexInputs` transform 中的隐式验证

模型侧 (`agilex_policy.py:54-139`) 包含以下校验逻辑:

```python
# 1. 相机名称检查
if set(in_images) - set(self.EXPECTED_CAMERAS) - set(self.EXTRA_CAMERAS):
    raise ValueError(...)

# 2. 图像格式标准化 (CHW→HWC, float→uint8)
if img.shape[0] == 3:
    img = np.transpose(img, (1, 2, 0))
if np.issubdtype(img.dtype, np.floating):
    img = (255 * img).astype(np.uint8)

# 3. State 异常值过滤
state = np.where(state > np.pi, 0, state)
state = np.where(state < -np.pi, 0, state)

# 4. Action 异常值过滤 (训练时)
actions = np.where(actions > np.pi, 0, actions)
actions = np.where(actions < -np.pi, 0, actions)
```

**关键发现**: 模型内部会将 `|state| > pi` 的关节值**置零**而非 clamp。这意味着:
- 如果 Piper 某个关节超出 [-3.14, 3.14] 范围, 该关节的 proprioception 会变为 0
- 这会导致模型"看到"一个错误的关节状态, 可能输出不合理的动作
- Piper 的 joint5 (腕关节) 理论范围是 [-2.79, 2.79] rad, 正常不会超过 pi
- 但如果传感器故障或编码器溢出, 这个"置零"行为会掩盖问题

### 2.2 当前两版均缺失的校验

| 校验项 | 状态 | 影响 |
|--------|------|------|
| 图像尺寸校验 (HxWxC) | **缺失** | 如果相机返回异常分辨率, `resize_with_pad` 会默默处理, 但可能引入 artifact |
| 图像通道数校验 | **缺失** | 灰度或 RGBA 图像会导致 `transpose(2,0,1)` 形状错误 |
| 图像全黑/全白检测 | **缺失** | 相机遮挡或驱动故障时, 模型会基于无效视觉输入做决策 |
| NaN/Inf 检查 (state) | **由模型层 `np.where > pi` 间接覆盖** | NaN 比较返回 False, 会保留原值 (NaN) 进入模型 |
| 关节值连续性检查 | **缺失** | 相邻帧关节值突变 (编码器跳变) 不会被检测 |
| 推理时间监控 | 原版有 `print` 输出, ROS2 版无 | ROS2 版缺少推理延迟的实时监控 |

### 2.3 NaN 传播风险 (中)

```python
state = np.where(state > np.pi, 0, state)
```

如果 `state` 包含 NaN:
- `NaN > np.pi` → `False`
- `NaN < -np.pi` → `False`
- NaN 会通过两次 `np.where` 保留原值, 进入模型

**后果**: JAX 模型中 NaN 会传播到 actions 输出, 最终发送 NaN 给机械臂。

---

## 三、推理结果校验分析

### 3.1 Action 输出处理

```python
# 原版
left_action = act[:7].copy()
right_action = act[7:14].copy()
left_action[6] = max(0.0, left_action[6] - RIGHT_OFFSET)      # gripper >= 0
right_action[6] = max(0.0, right_action[6] - RIGHT_OFFSET)     # gripper >= 0
ros_operator.puppet_arm_publish(left_action, right_action)

# ROS2 版
left = act[:7].copy()
right = act[7:14].copy()
left[6] = max(0.0, left[6] - self.gripper_offset)              # gripper >= 0
right[6] = max(0.0, right[6] - self.gripper_offset)            # gripper >= 0
```

**差异**: 无。两版逻辑完全一致。`RIGHT_OFFSET = gripper_offset = 0.003`。

### 3.2 当前两版均缺失的输出校验

| 校验项 | 状态 | 风险 | 后果 |
|--------|------|------|------|
| **NaN/Inf 检查** | **缺失** | **高** | NaN action 直接发送给机械臂, CAN 驱动行为不确定 |
| **关节位置范围限制** | **缺失** | **高** | 超过物理限位的指令可能损坏机构或触发急停 |
| **关节速度限制** | **缺失** | **高** | 相邻帧 action 差值过大时无减速, 可能导致碰撞 |
| **Gripper 上限 clamp** | **缺失** | 低 | gripper 只有下限 `max(0.0, ...)`, 无上限 clamp (通常 gripper ∈ [0, 0.1]) |
| **Action 维度检查** | **缺失** | 低 | 如果模型返回非 14 维 action, slice `[:7]` 和 `[7:14]` 可能越界 |
| **推理超时处理** | **缺失** | 中 | 如果推理卡死, inference thread 会阻塞, stream_buffer 耗尽后机械臂停止 |

### 3.3 `AgilexOutputs` 的隐式截断

```python
class AgilexOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}
```

模型输出可能维度 > 14 (因为 `pad_to_dim`), `AgilexOutputs` 只取前 14 维。这是正确的, 但没有检查维度是否 >= 14。

---

## 四、帧同步校验

### 4.1 同步策略对比

```python
# 原版 get_frame()
frame_time = min(
    self.img_left_deque[-1].header.stamp.to_sec(),
    self.img_right_deque[-1].header.stamp.to_sec(),
    self.img_front_deque[-1].header.stamp.to_sec(),
)
# Check all sensors' latest timestamp >= frame_time
# Pop stale data, take one frame per sensor

# ROS2 _get_synced_frame()
frame_time = min(
    _stamp_to_sec(self._img_front_deque[-1].header.stamp),
    _stamp_to_sec(self._img_left_deque[-1].header.stamp),
    _stamp_to_sec(self._img_right_deque[-1].header.stamp),
)
# Check all 5 sensors' latest timestamp >= frame_time
# Pop stale data, take one frame per sensor
```

**逻辑等价**, 区别仅为:
- ROS1: `stamp.to_sec()` → float
- ROS2: `stamp.sec + stamp.nanosec * 1e-9` → float

### 4.2 帧同步中缺失的校验

| 校验项 | 状态 | 风险 |
|--------|------|------|
| 帧时间差异上限 (max skew) | **缺失** | 相机之间 > 100ms 偏差时仍视为"同步", 可能导致不一致的视觉输入 |
| 帧率监控 | **缺失** | 某相机掉帧 (降到 < 15fps) 不会被检测 |
| 相机离线检测 | **缺失** | 某相机完全停止发布时, deque 永远非空 (残留旧帧), `_get_synced_frame` 会返回过期数据 |

### 4.3 相机离线导致的"幽灵帧"问题 (中)

如果 cam_l 在运行中断连:
1. `_img_left_deque` 中残留最后若干帧
2. `_get_synced_frame` 计算 `frame_time = min(front_latest, left_STALE, right_latest)`
3. 由于 left 时间戳远小于 front/right, frame_time 被 left 拉低
4. 但随后 check `left_latest >= frame_time` → True (left 的旧帧就是 frame_time)
5. 结果: 返回 left 的旧帧 + front/right 的新帧 → **时间不一致的混合数据**

原版也有同样的问题, 不是 ROS2 移植引入的。

---

## 五、端到端数据流验证矩阵

```
RealSense Camera (rgb8/bgr8)
    ↓ ROS2 Image msg
cv_bridge.imgmsg_to_cv2(msg, 'bgr8')     ← 保证 BGR 输出 ✓
    ↓ BGR numpy [H, W, 3]
_jpeg_mapping()                            ← BGR→JPEG→BGR, 与训练对齐 ✓
    ↓ BGR numpy [H, W, 3]
cv2.cvtColor(BGR, COLOR_BGR2RGB)           ← BGR→RGB ✓
    ↓ RGB numpy [H, W, 3]
image_tools.resize_with_pad(224, 224)      ← RGB→PIL(RGB)→resize→pad→numpy(RGB) ✓
    ↓ RGB numpy [224, 224, 3]
.transpose(2, 0, 1)                        ← HWC→CHW ✓
    ↓ RGB numpy [3, 224, 224]
AgilexInputs.__call__()                    ← CHW→HWC (line 82), float→uint8 (line 79) ✓
    ↓ RGB numpy [224, 224, 3] uint8
Model internal norm                        ← /255.0 → [0,1] float ✓
    ↓ Model inference
AgilexOutputs.__call__()                   ← actions[:, :14] ✓
    ↓ numpy [chunk_size, 14]
StreamActionBuffer.integrate_new_chunk()   ← temporal smoothing ✓
    ↓ numpy [14]
_publish_action()                          ← split left/right, gripper offset ✓
    ↓ JointState msg
/master/joint_left, /master/joint_right
```

**整条链路的通道顺序变换**:
```
Camera → bgr8 → BGR → BGR(jpeg) → RGB → RGB(resize) → CHW_RGB → HWC_RGB(model) → 正确
```

---

## 六、建议的校验策略增强

### P0: 关键安全校验 (必须在部署前实现)

**1. Action NaN/Inf 检查**
```python
def _publish_action(self):
    act = self.stream_buffer.pop_next_action()
    if act is None:
        return
    if not np.all(np.isfinite(act)):
        self.get_logger().error(f'Non-finite action detected: {act}')
        return  # 丢弃该 action, 机械臂保持当前位置
```

**2. 关节位置安全限幅**
```python
# Piper 6-DOF 关节限位 (rad), 参考 Piper SDK 规格
JOINT_LIMITS = np.array([
    [-2.618, 2.618],   # joint0
    [ 0.000, 3.142],   # joint1
    [-3.142, 0.000],   # joint2
    [-1.745, 1.745],   # joint3
    [-1.745, 1.745],   # joint4
    [-2.792, 2.792],   # joint5
    [ 0.000, 0.100],   # joint6 (gripper)
])

def _safe_clamp(self, action_7):
    """Clamp joint values to physical limits."""
    for i in range(7):
        action_7[i] = np.clip(action_7[i], JOINT_LIMITS[i, 0], JOINT_LIMITS[i, 1])
    return action_7
```

**3. 关节速度限制 (jerk protection)**
```python
MAX_JOINT_DELTA = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2])  # rad/step at 30Hz

def _velocity_clamp(self, action_7, prev_action_7):
    if prev_action_7 is None:
        return action_7
    delta = action_7 - prev_action_7
    delta = np.clip(delta, -MAX_JOINT_DELTA, MAX_JOINT_DELTA)
    return prev_action_7 + delta
```

### P1: 诊断与监控

**4. 推理延迟 logging**
```python
# 在 _inference_loop 中
elapsed_ms = (time.monotonic() - t_start) * 1000
self.get_logger().info(f'Inference: {elapsed_ms:.0f}ms')
if elapsed_ms > 500:
    self.get_logger().warn(f'Inference latency spike: {elapsed_ms:.0f}ms')
```

**5. 传感器心跳监控**
```python
def _check_sensor_health(self):
    """Check if all sensors are publishing at expected rate."""
    now = time.monotonic()
    for name, dq in [('front', self._img_front_deque), ...]:
        if len(dq) > 0:
            age = now - _stamp_to_sec(dq[-1].header.stamp)
            if age > 1.0:  # 1 second stale threshold
                self.get_logger().warn(f'Sensor {name} stale: {age:.1f}s')
```

**6. State NaN 检查 (在 `_get_observation` 中)**
```python
qpos = np.concatenate((...), axis=0)
if not np.all(np.isfinite(qpos)):
    self.get_logger().error(f'Non-finite qpos: {qpos}')
    return None  # 跳过本次推理
```

### P2: 可选增强

**7. 帧间时间偏差警告**
```python
# 在 _get_synced_frame 中
max_skew = max(timestamps) - min(timestamps)
if max_skew > 0.1:  # 100ms
    self.get_logger().warn(f'Frame sync skew: {max_skew*1000:.0f}ms')
```

**8. 图像基本检查**
```python
def _validate_image(self, img, name):
    if img is None or img.size == 0:
        self.get_logger().error(f'{name}: empty image')
        return False
    if img.ndim != 3 or img.shape[2] != 3:
        self.get_logger().error(f'{name}: unexpected shape {img.shape}')
        return False
    if img.mean() < 5.0:  # 几乎全黑
        self.get_logger().warn(f'{name}: near-black image (mean={img.mean():.1f})')
    return True
```

---

## 七、总结

### 图像管线一致性: 完全等价

ROS2 版与原版在图像处理的每一步都保持了严格的等价性:
- 通道顺序: BGR → BGR(jpeg) → RGB → CHW_RGB → HWC_RGB(model) ✓
- 空间变换: 保持宽高比, 零填充到 224x224 ✓
- 数值精度: uint8 全程, 无浮点累积误差 ✓
- 相机映射: front→top_head, right→hand_right, left→hand_left ✓

**不存在因图像处理差异导致的复现风险。**

### 推理结果校验: 两版均无安全护栏

原版和 ROS2 版都**没有**对推理输出做任何安全性校验 (NaN检查、关节限位、速度限制)。这不是 ROS2 移植引入的问题, 而是原版就存在的空白。

**建议在 ROS2 版中补充 P0 级安全校验**, 尤其是 NaN 检查和关节限位, 这对于保护硬件至关重要。
