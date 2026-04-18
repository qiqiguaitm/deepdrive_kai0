# kai0 IPC & 推理服务部署 Review: 当前版本 vs 原版差异分析

> 审查日期: 2026-03-29
> 范围: IPC (工控机) 推理服务部署链路, ROS2 移植与原版 ROS1 的一致性

---

## 一、代码修改一览 (git diff)

kai0 子仓库有 5 个文件被修改 (不含 uv.lock):

| 文件 | 修改内容 | 风险等级 |
|------|----------|----------|
| `src/openpi/serving/websocket_policy_server.py` | +`ping_timeout=300, close_timeout=300` | 低 (仅防 XLA 编译超时) |
| `packages/openpi-client/src/openpi_client/websocket_client_policy.py` | +`ping_timeout=300, close_timeout=300, open_timeout=300` | 低 (同上) |
| `src/openpi/training/config.py` | 添加 `episodes` 字段 + 填写本地路径 | 低 (训练配置) |
| `train_deploy_alignment/.../agilex_inference_openpi_temporal_smoothing.py` | prompt 改为 `"Flatten and fold the cloth."` | 中 (见下方分析) |
| `pyproject.toml` | 添加 `av==13.1.0`, `mujoco>=3.0.0` override | 低 (Python 3.12 兼容) |

---

## 二、关键风险点

### 风险 1 (高): Launch 文件中相机 topic 路径不一致

`autonomy_launch.py:146-148` 配置的 topic:

```
/camera_f/camera/color/image_raw   <- 多了 /camera/ 中间层
/camera_l/camera/color/image_raw   <- 多了 /camera/ 中间层
/camera_r/camera/color/image_raw   <- 多了 /camera/ 中间层
```

但 `policy_inference_node.py:226-228` 的默认值是:

```
/camera_f/color/image_raw
/camera_l/color/image_raw
/camera_r/color/image_raw
```

实际 topic 取决于 RealSense ROS2 wrapper 的 namespace + camera_name 组合:

| 相机 | namespace | camera_name | 实际 topic | launch 配置 | 匹配? |
|------|-----------|-------------|------------|-------------|-------|
| cam_f | `camera_f` | `camera` | `/camera_f/camera/color/image_raw` | `/camera_f/camera/color/image_raw` | OK |
| cam_l | `""` | `camera_l` | `/camera_l/color/image_raw` | `/camera_l/camera/color/image_raw` | **不匹配** |
| cam_r | `""` | `camera_r` | `/camera_r/color/image_raw` | `/camera_r/camera/color/image_raw` | **不匹配** |

**后果**: inference node 收不到左/右腕相机数据, `_get_synced_frame()` 永远返回 None, 推理不会执行。

**建议**: 在 sim01 上用 `ros2 topic list` 验证实际相机 topic; 或统一 cam_l/cam_r 也用 namespace 方式配置。

---

### 风险 2 (高): Piper 臂模式差异 -- mode=0 vs mode=1

| 部署方式 | Piper mode | 行为 |
|----------|-----------|------|
| 原版 `start_inference.sh` | `mode:=1 auto_enable:=true` | 控制模式, 订阅 `/master/joint_states` 并驱动从臂 |
| ROS2 `autonomy_launch.py` | `mode: 0, auto_enable: False` | 只读模式, 只发布关节状态, **不接受控制命令** |

`arm_reader_node.py` 中:
- `mode=0`: 同时发布 puppet 和 master joint states, **不订阅控制命令**
- `mode=1`: 订阅 `/master/joint_states` 并控制从臂

在 mode=0 下, inference node 发布到 `/master/joint_left` 和 `/master/joint_right`, 但 Piper 节点不会订阅这些 topic, 机械臂不会执行任何动作。

**后果**: 推理正常运行但机械臂完全不动。

**建议**: launch 文件中改为 `mode: 1, auto_enable: True`。

---

### 风险 3 (中): 缺少初始位姿归位流程

原版 `model_inference()` 在推理循环前执行:

```python
left0 = [0, 0.32, -0.36, 0, 0.24, 0, 0.07]
right0 = [0, 0.32, -0.36, 0, 0.24, 0, 0.07]
ros_operator.puppet_arm_publish_continuous(left0, right0)
input("Press enter to continue")
```

这会将机械臂平滑移动到初始位姿后等待用户确认。ROS2 的 `PolicyInferenceNode` 完全没有这个步骤, 节点启动后直接进入推理循环。

**后果**: 如果机械臂启动时不在初始位姿附近, 第一个 action chunk 可能导致突然的大幅运动, 存在碰撞和损坏风险。

**建议**: 在 inference node 启动时添加归位逻辑或安全检查。

---

### 风险 4 (低): 帧同步策略微小差异

原版 `get_frame()` 基于所有 3 个相机最新帧 timestamp 的 min 做同步, 并检查关节状态 deque 时间也 >= frame_time。

ROS2 版 `_get_synced_frame()` 逻辑相似, 但有细微区别:
- 原版有 depth image 和 robot_base 的同步检查; ROS2 版简化掉了 (对 Task A 无影响)
- 核心 pop 逻辑 `< frame_time` 判断完全一致

**结论**: 无实质风险。

---

## 三、一致性验证: 图像管线

| 步骤 | 原版 | ROS2 版 | 一致? |
|------|------|---------|-------|
| ROS Image -> OpenCV | `bridge.imgmsg_to_cv2` (ROS1 默认 bgr8) | `bridge.imgmsg_to_cv2(msg, 'bgr8')` 显式指定 | 等价 |
| JPEG mapping | `cv2.imencode(".jpg") -> cv2.imdecode` | 相同 | 一致 |
| BGR->RGB | `cv2.cvtColor(im, cv2.COLOR_BGR2RGB)` | 相同 | 一致 |
| Resize | `image_tools.resize_with_pad(np.array(imgs), 224, 224)` | 相同 (含形状检查) | 一致 |
| HWC->CHW | `imgs[i].transpose(2, 0, 1)` | 相同 | 一致 |
| 相机顺序 | `[front, right, left]` | 相同 | 一致 |
| payload keys | `top_head, hand_right, hand_left` | 相同 | 一致 |

**结论**: 图像管线完全一致, 无复现风险。

---

## 四、一致性验证: StreamActionBuffer

ROS2 版的 `StreamActionBuffer` 从原版逐行复制, 关键方法一致性:

- `integrate_new_chunk()`: 完全一致 (latency trimming, linear blending, overlap smoothing)
- `pop_next_action()`: 完全一致 (k 计数, last_action 保存)
- `has_any()`: 完全一致

**结论**: action smoothing 逻辑无差异。

---

## 五、一致性验证: 推理参数默认值

| 参数 | 原版默认 | ROS2版默认 | 一致? |
|------|----------|-----------|-------|
| publish_rate | 30 Hz | 30 Hz | 一致 |
| inference_rate | 3.0 Hz | 3.0 Hz | 一致 |
| chunk_size | 50 | 50 | 一致 |
| latency_k | 8 | 8 | 一致 |
| min_smooth_steps | 8 | 8 | 一致 |
| decay_alpha | 0.25 | 0.25 | 一致 |
| gripper_offset (RIGHT_OFFSET) | 0.003 | 0.003 | 一致 |
| prompt | "Flatten and fold the cloth." | "Flatten and fold the cloth." | 一致 |

---

## 六、一致性验证: WebSocket 协议

添加的 `ping_timeout=300, close_timeout=300, open_timeout=300` 不影响数据传输逻辑, 只防止 XLA 首次编译时的 WebSocket 超时断连。

MessagePack + NumPy 序列化、payload 格式、action 返回格式均未修改。

**结论**: WebSocket 通信层无一致性风险。

---

## 七、总结: 行动项

| 优先级 | 问题 | 建议操作 | 状态 |
|--------|------|----------|------|
| **P0** | Piper mode=0 无法接收控制命令 | launch 文件中改为 `mode: 1, auto_enable: True` | 待修复 |
| **P0** | 相机 topic 路径可能不匹配 (cam_l/cam_r) | 在 sim01 上 `ros2 topic list` 验证; 统一 namespace 配置 | 待验证 |
| **P1** | 缺少初始位姿归位流程 | 在 inference node 启动时添加归位逻辑或安全检查 | 待实现 |
| **P2** | `install/` 下的 node 副本可能未同步 | 修改源码后重新 `colcon build` | 待执行 |

---

## 八、结论

**核心算法层面** (模型推理、图像预处理、action smoothing、WebSocket 协议) 当前版本与原版完全一致, 不存在复现风险。

**部署层面** 有 2 个 P0 问题 (Piper mode 和 camera topic) 需要在实际部署前修复/验证, 否则会导致:
1. 推理节点运行但机械臂不响应 (mode=0 问题)
2. 收不到左/右腕相机图像 (topic 不匹配问题)

这两个问题不影响算法正确性, 属于 ROS2 移植时的配置遗漏, 修复简单且可控。
