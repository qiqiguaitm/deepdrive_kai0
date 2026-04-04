# 在线推理可视化与交互执行控制

> 日期: 2026-04-04
> 修改文件: `policy_inference_node.py`, `inference_full_launch.py`
> 依赖: rerun-sdk, trimesh, scipy, piper_fk

---

## 1. 概述

在现有 `policy_inference_node.py` 基础上增加两个功能模块:

1. **Rerun 3D 可视化** — 实时显示机械臂状态、策略预测轨迹、相机画面、关节时序
2. **交互式执行控制** — 观测/执行模式切换，支持键盘和 ROS2 topic 两种方式

两个模块均通过参数开关控制，默认关闭，对原有推理流程零影响。

---

## 2. 模型输出确认

策略模型输出 **关节角** (joint angles)，不是末端位姿:

```
actions[50, 14]:
  [0:6]   左臂 6 关节角 (rad)
  [6]     左夹爪开合
  [7:13]  右臂 6 关节角 (rad)
  [13]    右夹爪开合
```

发布到 `/master/joint_left`, `/master/joint_right` 时直接作为关节指令，不经 IK。

要在 3D 空间显示末端轨迹，需要对 50 步预测的每一步做 FK (`PiperFK.fk_homogeneous`) 转换为世界坐标系下的 EE 位置。

---

## 3. 交互式执行控制

### 3.1 设计

```
┌─────────────────────────────────────────────────────────┐
│                  PolicyInferenceNode                     │
│                                                         │
│  _execution_enabled: bool                               │
│    │                                                    │
│    ├── 启动参数 execute_mode (默认 false = 观测模式)      │
│    ├── ROS2 topic /policy/execute (Bool)                │
│    └── 键盘: Enter/Space 切换, q/Esc → 观测             │
│                                                         │
│  _inference_loop()  ──→  stream_buffer (始终运行)        │
│                              │                          │
│  _publish_action()           │                          │
│    if not _execution_enabled:│                          │
│      return  ← 门控          │                          │
│    else:                     │                          │
│      pop & publish ──────────┘                          │
└─────────────────────────────────────────────────────────┘
```

关键: 推理始终运行，只在发布端门控。切到观测模式时臂保持最后位姿（电机使能不变），随时可切回执行。

### 3.2 控制方式

| 方式 | 说明 |
|------|------|
| 启动参数 | `execute_mode:=true/false` |
| ROS2 topic | `ros2 topic pub --once /policy/execute std_msgs/msg/Bool '{data: true}'` |
| 键盘 (cbreak) | `Enter`/`Space` 切换, `q`/`Esc` → 观测 |
| 键盘 (line mode) | 输入 `toggle`/`t` 切换, `q`/`quit` → 观测 (ros2 launch 环境) |

### 3.3 停止策略

- Enter/Space: 切�� `_execution_enabled`，不 DisableArm
- q/Esc: 切到观测模式（停发指令），保持电机使能，不 shutdown
- Ctrl+C: 正常 ROS2 shutdown 流程

---

## 4. Rerun 可视化

### 4.1 条件启用

```python
self.declare_parameter('enable_rerun', False)
self.declare_parameter('calibration_config', '')  # 标定 YAML 路径
```

`enable_rerun=true` 时才 import rerun 并初始化。`calibration_config` 提供 FK 所需的 `T_world_baseL/R` 变换矩阵。

### 4.2 相机���面复用

直接在现有的 ROS2 image callback 中追加 Rerun logging，零额外相机开销:

```python
def _cb_img_front(self, msg):
    self._img_front_deque.append(msg)          # 原有逻辑
    if self._use_rerun:
        img = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        rr.log("images/top_head", rr.Image(img))
```

订阅的 topic 与推理管线完全相同:
- `/camera_f/camera/color/image_raw` → `images/top_head`
- `/camera_l/camera/color/image_raw` → `images/hand_left`
- `/camera_r/camera/color/image_raw` → `images/hand_right`

### 4.3 Blueprint 布局

```
┌──────────────────────────────┬────────────────────┐
│                              │ Head │ Left │ Right │
│       3D Scene (60%)         ├────────────────────┤
│                              │    Time Series     │
│  臂mesh + 预测轨迹 + 实际trail │  关节角 + 延迟      │
└──────────────────────────────┴────────────────────┘
```

### 4.4 Rerun 实体层级

```
world/
├── baseL, baseR                    Points3D (static)    基座标记
├── head_cam_frustum                LineStrips3D (static) 头顶相机视锥
├── workspace                       Boxes3D (static)     工作空间包围盒
├── left/
│   ├── base_link                   Mesh3D (static) + Transform3D (~30Hz)
│   ├── link1..link6                Mesh3D (static) + Transform3D (~30Hz)
│   ├── gripper_base, link7, link8  Mesh3D (static) + Transform3D (~30Hz)
│   └── ee                          Points3D (~30Hz)     当前 EE 标记 (蓝)
├── right/
│   └── (同 left, 绿色)
├── predicted/
│   ├── left_traj                   LineStrips3D (3Hz)   50步预测 EE 路径 (蓝)
│   ├── right_traj                  LineStrips3D (3Hz)   50步预测 EE 路径 (绿)
│   ├── left_endpoints              Points3D (3Hz)       起止点标记
│   └── right_endpoints             Points3D (3Hz)       起止点标记
├── actual/
│   ├── left_trail                  Points3D (~30Hz)     rolling 300点实际轨迹 (蓝)
│   └── right_trail                 Points3D (~30Hz)     rolling 300点实际轨迹 (绿)
images/
├── top_head                        Image (~30Hz)
├── hand_left                       Image (~30Hz)
├── hand_right                      Image (~30Hz)
timeseries/
├── left_j0..left_j6                Scalar (~30Hz)       左臂关节角
├── right_j0..right_j6              Scalar (~30Hz)       右臂关节角
├── inference_ms                    Scalar (3Hz)         推理延迟
└── execute_mode                    Scalar (on change)   0=观测, 1=执行
```

### 4.5 Mesh 静态 vs Transform 动态

STL 几何体只 log ���次 (`static=True`)，每帧通过 `Transform3D` 更新位姿驱动运动:

```python
# 初始化 — 几何体 log 一次
rr.log("world/left/link1", rr.Mesh3D(
    vertex_positions=m.vertices, triangle_indices=m.faces, ...), static=True)

# 每帧 — FK 更新位姿
T_world_link = T_world_base @ fk.fk_all_links(q_rad)[0]
rr.log("world/left/link1", rr.Transform3D(
    translation=T_world_link[:3, 3], mat3x3=T_world_link[:3, :3]))
```

### 4.6 预测轨迹数据流

```
policy.infer(obs) → actions[50, 14]
         │
         ▼
  for t in range(50):
    T_left  = T_world_baseL @ PiperFK.fk_homogeneous(actions[t, 0:6])
    T_right = T_world_baseR @ PiperFK.fk_homogeneous(actions[t, 7:13])
    left_path.append(T_left[:3, 3])    # 世界系 EE 位置
    right_path.append(T_right[:3, 3])
         │
         ▼
  rr.log("world/predicted/left_traj",
         rr.LineStrips3D([left_path], colors=蓝, radii=0.003))
  rr.log("world/predicted/right_traj",
         rr.LineStrips3D([right_path], colors=绿, radii=0.003))
```

---

## 5. 依赖与路径

| 依赖 | 用途 | 来源 |
|------|------|------|
| `PiperFK` | 关节角 ��� SE3 变换 | `calib/piper_fk.py` |
| `calibration.yml` | `T_world_baseL/R`, `T_world_camF` | `config/calibration.yml` |
| STL mesh | 机械臂 3D 模型 | `kai0/train_deploy_alignment/inference/agilex/Piper_ros_private-ros-noetic/src/piper_description/meshes/` |
| `rerun-sdk` | 可视化 | pip (v0.31.1) |
| `trimesh` | STL 加载 | pip |
| `scipy` | 夹爪 Euler → 旋转矩阵 | pip |

---

## 6. 用法

### 6.1 观测模式 + Rerun (默认不动臂)

```bash
ros2 launch piper inference_full_launch.py enable_rerun:=true
```

### 6.2 执行模式 + Rerun

```bash
ros2 launch piper inference_full_launch.py enable_rerun:=true execute_mode:=true
```

### 6.3 不带 Rerun (原有行为)

```bash
ros2 launch piper inference_full_launch.py mode:=ros2
```

### 6.4 运行中切换

```bash
# 终端键盘
#   Enter / Space → 切换执行/观测
#   q / Esc       → 切到观测模式

# 或 ROS2 topic
ros2 topic pub --once /policy/execute std_msgs/msg/Bool '{data: true}'   # 开始执行
ros2 topic pub --once /policy/execute std_msgs/msg/Bool '{data: false}'  # 停止执行
```

---

## 7. Launch 参数一览

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mode` | `ros2` | 推理模式: ros2 / websocket / both |
| `config_name` | `pi05_flatten_fold_normal` | 训练配置名 |
| `checkpoint_dir` | `.../checkpoints/Task_A/mixed_1` | 模型权��路径 |
| `execute_mode` | `false` | 启动时是否执行 |
| `enable_rerun` | `false` | 启用 Rerun 可视化 |
| `calibration_config` | `.../config/calibration.yml` | 标定文件 |
| `gpu_id` | `0` | GPU 编号 |
| `host` | `localhost` | WebSocket 主机 (websocket 模式) |
| `port` | `8000` | WebSocket 端口 |
| `prompt` | `Flatten and fold the cloth.` | 语言指令 |

---

## 8. 验证清单

- [ ] 启动观测模式: Rerun 打开，3 路相机画面显示，臂 mesh 跟随实际关节实时运动
- [ ] 预测轨迹: 3D 视图中蓝/绿线条每 ~333ms 更新一次，起止点有标记
- [ ] 键盘 Enter → `[EXECUTE]`，臂开始运动，actual trail (蓝/绿点) 累积
- [ ] 键盘 Enter → `[OBSERVE]`，臂停止发指令，推理继续，预测轨迹仍更新
- [ ] Topic 控制: `ros2 topic pub --once /policy/execute ...` 同样可切换
- [ ] Timeseries 面板: 14 路关节角曲线 + 推理延迟 + 执行状态
- [ ] 不带 Rerun 启动: 原有行为完全不变，无 rerun import
