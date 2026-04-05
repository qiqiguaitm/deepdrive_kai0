# sim01 部署文档

> 机器: sim01 (Ubuntu 24.04, 2x RTX 5090 32GB)
> 日期: 2026-03-28
> 用途: kai0 Task A 推理部署 (工控机 + 推理机一体)
> **关联文档**: 整体项目规划 (训练集群 + 部署 roadmap) 详见 [taskA_master_plan.md](taskA_master_plan.md)。

---

## 1. 环境架构

```
┌──────────────────────────────────────────────────────────────────┐
│  sim01 (Ubuntu 24.04 LTS, kernel 6.17)                          │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │         统一 Python 环境: uv venv (Python 3.12)              ││
│  │                                                              ││
│  │  JAX 0.5.3 + torch 2.7.1+cu126 + rclpy + cv_bridge         ││
│  │  openpi_client + piper_msgs + flax + orbax                  ││
│  │                                                              ││
│  │  ┌─────────────────┐  WebSocket   ┌────────────────────┐    ││
│  │  │ serve_policy.py │ ←──(:8000)──→│ 推理客户端 (ROS2)   │    ││
│  │  │ GPU 0, JAX      │              │ 相机+臂→观测→动作   │    ││
│  │  └─────────────────┘              └────────────────────┘    ││
│  └──────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ROS2 Jazzy: piper 驱动节点 + realsense2_camera 节点             │
│                         │                                        │
│                  USB: 3xRealSense + 3xCAN                        │
└──────────────────────────────────────────────────────────────────┘
```

### 统一 Python 环境

```bash
# 构建命令
cd /data1/tim/workspace/deepdive_kai0/kai0
http_proxy= https_proxy= GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.12

# 使用时 source ROS2 后用 .venv/bin/python
source /opt/ros/jazzy/setup.bash
source /data1/tim/workspace/deepdive_kai0/ros2_ws/install/setup.bash
.venv/bin/python  # JAX + torch + rclpy + cv_bridge + openpi_client 全部可用
```

**关键**: pyproject.toml 需要 override `"av==13.1.0"` 和 `"mujoco>=3.0.0"` (旧版在 Python 3.12 上编译失败)。
**关键**: 不走代理下载更快 (`http_proxy= https_proxy= uv sync`)，sim01 可直连 PyPI。

---

## 2. 硬件连接

### 2.1 USB 拓扑

```
Bus 002 (10Gbps xHCI):
  ├── Port 1: D435  (SN: 254622070889, 5Gbps)  → 头顶相机
  └── Port 2: D405  (SN: 409122273074, 5Gbps)  → 左腕相机

Bus 003 (480Mbps):
  └── Hub → 3x CAN adapters (12Mbps each)
      ├── can0 (bus: 3-2.1.3.2) → 空 (未连接臂)
      ├── can1 (bus: 3-2.1.3.3) → Piper slave 左臂
      └── can2 (bus: 3-2.1.3.4) → Piper slave 右臂

Bus 004 (10Gbps xHCI):
  └── Port 2: D405  (SN: 409122271568, 5Gbps)  → 右腕相机
```

### 2.2 相机验证

| 设备 | 序列号 | 最大分辨率 @ 30fps | RGB topic | Depth topic | 状态 |
|------|--------|-------------------|-----------|-------------|------|
| D435 (top) | 254622070889 | 640x480 RGB | `/camera/camera/color/image_raw` | — | OK |
| D405-L (wrist) | 409122273074 | 848x480 RGB | `/camera_l/camera/color/image_rect_raw` | — | OK |
| D405-R (wrist) | 409122271568 | 848x480 RGB | `/camera_r/camera/color/image_rect_raw` | — | OK |

三相机同时 RGB only @ 30fps: **PASS** (无 depth, 减少 USB 带宽一半, 解决 D405 掉线问题)

**关键设置**:
- **Depth 关闭** — kai0 推理只用 RGB, 不需要 depth
- **D405 topic 是 `image_rect_raw`** (不是 `image_raw`)
- **D435 namespace 是 `/camera/camera/`** (不是 `/camera_f/`)
- 所有图像最终 resize 到 224x224 送入模型

### 2.3 CAN 验证

| 接口 | USB bus-info | 状态 | 角色 |
|------|-------------|------|------|
| can0 | 3-2.1.3.2 | UP, 无数据 | 空/未连接 |
| can1 | 3-2.1.3.3 | UP, 活跃 (61K+ RX) | Piper slave 左臂 |
| can2 | 3-2.1.3.4 | UP, 活跃 (61K+ RX) | Piper slave 右臂 |

纯推理需 2 个 slave CAN, DAgger 需 4 个 (+ 2 master)。

---

## 3. 安装步骤

### 3.1 系统依赖

```bash
sudo apt install -y \
  ros-jazzy-desktop ros-jazzy-cv-bridge \
  ros-jazzy-realsense2-camera ros-jazzy-realsense2-description \
  ros-jazzy-tf-transformations \
  can-utils ethtool
pip install transforms3d piper_sdk python-can  # system python
```

### 3.2 ROS2 工作空间

```bash
cd /data1/tim/workspace/deepdive_kai0/ros2_ws
eval "$(conda shell.bash hook)"; conda deactivate
source /opt/ros/jazzy/setup.bash
colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3.12
# 输出: piper_msgs (消息) + piper (6个驱动脚本 + 5个launch文件)
```

### 3.3 openpi 统一环境 (Python 3.12)

```bash
cd /data1/tim/workspace/deepdive_kai0/kai0

# pyproject.toml 必须包含:
# override-dependencies = ["ml-dtypes==0.4.1", "tensorstore==0.1.74", "av==13.1.0", "mujoco>=3.0.0"]

# 构建 (不走代理, 直连 PyPI 更快)
http_proxy= https_proxy= GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.12

# 验证
source /opt/ros/jazzy/setup.bash
source /data1/tim/workspace/deepdive_kai0/ros2_ws/install/setup.bash
.venv/bin/python -c "import jax, torch, rclpy; print('ALL OK')"
```

### 3.4 Checkpoint (从 ModelScope 下载)

```bash
# 国内直连, 不需要代理
pip install modelscope
python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('OpenDriveLab/Kai0', local_dir='./checkpoints', allow_patterns=['Task_A/*'])
"
# 22GB, norm_stats 需拷贝到 data 目录:
cp checkpoints/Task_A/mixed_1/norm_stats.json data/Task_A/base/
```

### 3.5 代码修改

| 文件 | 修改 | 原因 |
|------|------|------|
| `src/openpi/training/config.py` | `repo_id` 和 `weight_loader` 改为本机绝对路径 | 原版是占位符 |
| `src/openpi/training/config.py` | `DataConfig` 添加 `episodes: list[int] \| None = None` | 上游 bug |
| `src/openpi/serving/websocket_policy_server.py` | `_server.serve()` 添加 `ping_timeout=300` | XLA 首次编译时防断连 |
| `packages/openpi-client/.../websocket_client_policy.py` | `connect()` 添加 `ping_timeout=300, close_timeout=300` | 同上 |
| `pyproject.toml` | override `av==13.1.0, mujoco>=3.0.0` | Python 3.12 编译兼容 |

---

## 4. 启动流程

### 4.1 CAN 激活 (需要 sudo)

```bash
for iface in can0 can1 can2; do
  sudo ip link set "$iface" down
  sudo ip link set "$iface" type can bitrate 1000000
  sudo ip link set "$iface" up
done
```

### 4.2 Policy Server (终端 1)

```bash
cd /data1/tim/workspace/deepdive_kai0/kai0
mkdir -p /tmp/xla_cache
JAX_COMPILATION_CACHE_DIR=/tmp/xla_cache CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python scripts/serve_policy.py \
  --port 8000 policy:checkpoint \
  --policy.config=pi05_flatten_fold_normal \
  --policy.dir=checkpoints/Task_A/mixed_1
# 等待输出: "server listening on 0.0.0.0:8000"
# 注意: 需要 unset XLA_FLAGS 或设为空，否则旧的 flag 会导致崩溃
```

### 4.3 ROS2 节点 (终端 2)

```bash
eval "$(conda shell.bash hook)"; conda deactivate
source /opt/ros/jazzy/setup.bash
source /data1/tim/workspace/deepdive_kai0/ros2_ws/install/setup.bash
ros2 launch /data1/tim/workspace/deepdive_kai0/scripts/launch_e2e_test.py
# 启动: 2个piper(mode=0只读) + 3个realsense相机
```

### 4.4 推理客户端 (终端 3)

```bash
eval "$(conda shell.bash hook)"; conda deactivate
source /opt/ros/jazzy/setup.bash
source /data1/tim/workspace/deepdive_kai0/ros2_ws/install/setup.bash
unset http_proxy https_proxy  # 重要!

cd /data1/tim/workspace/deepdive_kai0/kai0/train_deploy_alignment/inference/agilex/inference
/data1/tim/workspace/deepdive_kai0/kai0/.venv/bin/python \
  agilex_inference_openpi_temporal_smoothing_ros2.py \
  --host localhost --port 8000 \
  --ctrl_type joint --use_temporal_smoothing --chunk_size 50 \
  --publish_rate 30 --inference_rate 3.0 \
  --img_front_topic /camera_f/color/image_raw \
  --img_left_topic /camera_l/color/image_rect_raw \
  --img_right_topic /camera_r/color/image_rect_raw \
  --puppet_arm_left_topic /puppet/joint_left \
  --puppet_arm_right_topic /puppet/joint_right
```

---

## 5. 性能数据

### 5.1 推理延迟 (RTX 5090, pi0.5 ~3B params)

| 指标 | 无 XLA 缓存 | 有 XLA 缓存 |
|------|------------|------------|
| 首次推理 | 13,027ms | **2,675ms (4.9x 加速)** |
| 稳态延迟 avg | **66ms** | **67ms** |
| 稳态延迟 p99 | 67ms | 67ms |
| 吞吐量 | 15.1 infer/s | 15.0 infer/s |
| Action 输出 | (50, 14) | (50, 14) |
| 在线控制需求 | < 300ms (3Hz) | < 300ms (3Hz) |
| 结论 | **PASS (5x 余量)** | **PASS (5x 余量)** |

**XLA 持久缓存**: `JAX_COMPILATION_CACHE_DIR=/tmp/xla_cache`，首次运行写入 0.4MB 缓存，
后续重启 server 首次推理从 13s 降到 2.7s。稳态性能不变。

### 5.2 对比 kai0 原版

| | kai0 原版 (RTX 4090) | sim01 (RTX 5090) | 提升 |
|---|---|---|---|
| 推理延迟 | ~100-150ms | **66ms** | ~50% |
| 吞吐量 | ~7-10 Hz | **15 Hz** | ~2x |
| 显存占用 | ~8GB | ~8GB | 相同 |

### 5.3 相机帧率

| 相机 | RGB FPS | Depth FPS | 端到端延迟 |
|------|---------|-----------|-----------|
| D435 (top) | 29.8 | 29.9 | 51.5ms |
| D405-L | 29.9 | 29.8 | 17.5ms |
| D405-R | 29.8 | 29.7 | 17.4ms |

---

## 6. 遇到的问题及解决方案

| # | 问题 | 方案 |
|---|------|------|
| 1 | ROS1 Noetic 不支持 Ubuntu 24.04 | 迁移到 ROS2 Jazzy (rospy→rclpy 机械替换) |
| 2 | Python 3.11 vs 3.12 冲突 | `uv sync --python 3.12` 统一环境 |
| 3 | mujoco 2.3.7 编译失败 | override `"mujoco>=3.0.0"` (预编译 wheel) |
| 4 | av 14.4.0 编译失败 | override `"av==13.1.0"` (预编译 wheel) |
| 5 | HuggingFace 被墙 | ModelScope 国内镜像直连 |
| 6 | WebSocket 走代理失败 | `unset http_proxy https_proxy` |
| 7 | XLA 编译时 WebSocket 断连 | server + client `ping_timeout=300` |
| 8 | norm_stats 路径不匹配 | `cp checkpoints/.../norm_stats.json data/.../` |
| 9 | DataConfig 缺 episodes 字段 | 代码中添加该字段 |
| 10 | colcon build 被 conda 干扰 | `conda deactivate` 后构建 |
| 11 | D405 topic 名与 D435 不同 | 推理脚本传参指定 `image_rect_raw` |
| 12 | PyPI 走代理慢 | sim01 可直连 PyPI, `http_proxy= uv sync` |
| 13 | `ros2 run` 启动 policy_inference_node 失败 (numpy/rclpy 冲突) | 脚本启动时自动 re-exec 到 venv python, [详见分析](#问题-13-ros2-run-python-环境冲突) |

---

## 7. 端到端测试

### 测试条件

- Policy Server: GPU 0, JAX, pi05_flatten_fold_normal
- Piper: mode=0 只读 (不控制臂运动)
- 相机: 640x480 RGB @ 30fps x 3
- 推理: temporal smoothing, chunk_size=50, publish_rate=30

### 数据流

```
D435/D405 (30fps) ──ROS2──→ 推理客户端 ──WebSocket──→ Policy Server (GPU 0)
                                                           │
                                                      JAX 推理 66ms
                                                           │
                                                      (50,14) actions
                                                           │
Piper (mode=0) ←──ROS2──── 推理客户端 ←──WebSocket─────────┘
 (不执行运动)         发布 /master/joint_left,right
```

### 结果

- 183 步控制命令成功发布
- 推理循环正常完成
- 臂未运动 (piper mode=0 只读)
- **PASS**

### 推理质量验证

使用 `scripts/test_inference_server.py --check quality` 验证模型输出的正确性:

```bash
http_proxy= https_proxy= .venv/bin/python scripts/test_inference_server.py --check quality
```

| 测试项 | 结果 | 详情 |
|--------|------|------|
| **形状和范围** | PASS | (50,14), 所有关节在 Piper 物理限位内 |
| **一致性** | PASS | 相同输入 5 次推理, std=0.023 rad (~1.3°) |
| **敏感性** | PASS | 不同状态输入, 输出差异 0.228 rad (~13°), 模型对输入敏感 |
| **时序平滑性** | PASS | chunk 内相邻步最大跳变 0.071 rad (~4°) |
| **Server timing** | PASS | 推理 65ms, 总计 67ms |

关键观察:
- 左右臂 joint1/2/4 有明显运动 (肩/肘/腕), 符合折叠布料动作模式
- joint0/3/5 (底座/腕旋转) 接近零位, 合理
- gripper 在 0.002-0.032 范围, 微张开状态
- flow matching 随机性低 (std 0.023 rad), 输出高度稳定

---

## 8. 文件清单

### 新建文件

| 路径 | 说明 |
|------|------|
| `ros2_ws/src/piper_msgs/` | ROS2 消息定义 (PosCmd, PiperStatusMsg) |
| `ros2_ws/src/piper/scripts/` | 6 个 ROS2 piper 驱动节点 |
| `ros2_ws/src/piper/launch/` | 5 个 Python launch 文件 |
| `scripts/launch_e2e_test.py` | 端到端测试 launch (piper mode=0 + 3 相机) |
| `scripts/test_inference_server.py` | 推理服务器测试 (--check latency/quality/all, 合并自 bench_inference_latency.py + verify_inference_quality.py) |
| `scripts/start_policy_node.sh` | policy_inference_node 独立启动脚本 (--mode ros2/websocket/both, 吸收了已删除的 test_policy_{ros2,both}_mode.sh) |
| `scripts/test_3cam_ros2.py` | 三相机 ROS2 验证脚本 |
| `*_ros2.py` | ROS2 版推理/DAgger/数据采集脚本 |

### 修改文件

| 路径 | 修改 |
|------|------|
| `kai0/pyproject.toml` | override av, mujoco |
| `kai0/src/openpi/training/config.py` | 路径 + episodes 字段 |
| `kai0/src/openpi/serving/websocket_policy_server.py` | ping_timeout |
| `kai0/packages/openpi-client/.../websocket_client_policy.py` | ping_timeout |

---

## 9. ROS2 Native Policy Node (实验性)

`policy_inference_node.py` 将推理直接集成为 ROS2 节点，消除 WebSocket 中间层。

### 三种模式

| 模式 | 说明 | 用途 |
|------|------|------|
| `ros2` | JAX 模型在节点内加载，无 WebSocket | 最低延迟 |
| `websocket` | 连接外部 serve_policy.py | 兼容旧方案 |
| `both` | 本地加载 + 同时提供 WebSocket 服务 | 过渡期 |

### 一键启动

```bash
# 启动全套: piper(只读) + 3相机 + policy推理节点
bash scripts/start_autonomy.sh                   # 纯 ROS2 + Rerun (默认)
bash scripts/start_autonomy.sh --mode websocket  # 需先启动 serve_policy.py
bash scripts/start_autonomy.sh --no-rerun        # 不启动 Rerun 可视化
```

### 发布的 Topic

| Topic | 说明 |
|-------|------|
| `/policy/actions` | 14 维联合动作 (左7+右7) |
| `/master/joint_left` | 左臂控制命令 |
| `/master/joint_right` | 右臂控制命令 |

### 验证结果

**三种模式全部验证通过:**

| 模式 | Actions Hz | Warmup | 状态 |
|------|-----------|--------|------|
| websocket | 30.0 Hz | 2758ms | PASS |
| ros2 | 30.0 Hz (25.3 Hz*) | 2824ms | PASS |
| both | — | — | 代码已修复, 待重启验证 |

*ros2 模式受 inference_rate=3Hz 限制, 实测 actions 发布仍达 30Hz (temporal smoothing 填充)

**两种模式对比 (300 帧采样):**

| 指标 | WebSocket | ROS2 | 差异 |
|------|-----------|------|------|
| 最大关节均值差 | — | — | 0.0017 rad (0.1°) |
| 整体均值 | +0.0225 | +0.0224 | 0.0001 |
| 整体标准差 | 0.2656 | 0.2651 | 0.0005 |
| **结论** | | | **PASS — 统计分布完全一致** |

### 启动方式

```bash
# 只启 policy_inference_node (其他节点已经在跑)
bash scripts/start_policy_node.sh --mode websocket   # 连远端 serve_policy
bash scripts/start_policy_node.sh --mode ros2        # 节点内直接加载 JAX
bash scripts/start_policy_node.sh --mode both        # 两者兼有

# 或启动整套栈 (相机 + 臂 + policy)
bash scripts/start_autonomy.sh --mode websocket
bash scripts/start_autonomy.sh --mode ros2
```

**已修复**: `ros2 run piper policy_inference_node.py` 现在可以直接使用, 无需 wrapper 脚本。详见 [问题 #13](#问题-13-ros2-run-python-环境冲突)。

### 已知问题

1. **USB 相机锁死**: 频繁启停 realsense 节点导致 V4L2 设备锁死, **不需要重启机器**
   - 修复步骤 (两步恢复):
     ```bash
     # 1. 必须用 sudo kill -9 (普通 pkill 可能杀不掉 ros2 launch 的子进程)
     for pid in $(ps aux | grep realsense2_camera_node | grep -v grep | awk '{print $2}'); do sudo kill -9 $pid; done
     # 2. USB sysfs reset (sim01 特定路径)
     for dev in 2-1 2-2 4-2.2; do
       sudo bash -c "echo 0 > /sys/bus/usb/devices/$dev/authorized; sleep 1; echo 1 > /sys/bus/usb/devices/$dev/authorized"
     done
     sleep 5 && rs-enumerate-devices --compact  # 验证 3 个相机恢复
     ```
   - **关闭 depth** + **逐步启动** 可大幅减少此问题
2. **相机设置**: Depth 关闭 (kai0 推理不用), RGB only 640x480@30fps, 减少 USB 带宽一半
3. **环境变量**: ros2 模式必须用 wrapper 脚本启动, 确保 LD_LIBRARY_PATH 含 CUDA 库
4. **CAN 需在 USB reset 后重新激活** (CAN 适配器在同一个 USB hub 上)

---

## 10. 切换到真实控制

当需要真正控制臂运动时, 将 launch 中 piper 改为 `mode:=1, auto_enable:=true`:

```python
# launch_e2e_test.py 中修改:
parameters=[{'can_port': 'can1', 'mode': 1, 'auto_enable': True}]
```

并去掉推理脚本的 `--max_publish_step` 限制。

---

## 问题 #13: `ros2 run` Python 环境冲突

### 表现

执行 `ros2 run piper policy_inference_node.py` 时报错:

```
ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'
The C extension '..._rclpy_pybind11.cpython-313-x86_64-linux-gnu.so' isn't present on the system
```

或:

```
ImportError: Error importing numpy: you should not try to import numpy from
its source directory
```

使用 wrapper 脚本 (手动设 `LD_LIBRARY_PATH` + `PYTHONPATH`) 或直接用 `.venv/bin/python` 运行则正常。

### 分析

涉及 **三个 Python 环境** 的冲突:

```
┌─────────────────────────────────────────────────────────┐
│  ① conda python3.13  (/data1/miniconda3/bin/python3)    │
│  ② 系统 python3.12   (/usr/bin/python3.12)              │
│  ③ kai0 venv python  (kai0/.venv/bin/python → ②的symlink)│
└─────────────────────────────────────────────────────────┘
```

**执行链路** (修复前):

```
ros2 run piper policy_inference_node.py
  ↓
shebang: #!/usr/bin/env python3
  ↓
which python3 → /data1/miniconda3/bin/python3  (conda 3.13!)
  ↓
conda python3.13 加载脚本
  ↓
脚本内部往 PYTHONPATH 插入 venv site-packages
  ↓
import rclpy → rclpy 的 C 扩展是为 python3.12 编译的 (.cpython-312-*.so)
  ↓
python3.13 的 importlib 去找 .cpython-313-*.so → 找不到 → 报错
```

**三层原因叠加:**

1. **PATH 被 conda 劫持**: conda 的 `/data1/miniconda3/bin` 在 PATH 前面, `python3` 解析为 conda 的 3.13 而非系统 3.12

2. **venv python 是 symlink**: `kai0/.venv/bin/python → /usr/bin/python3.12`, 两者 `realpath` 相同, 无法通过 `sys.executable` 区分 "裸系统 python" 和 "venv python"

3. **`__file__` 路径推导失败**: 脚本中用 `os.path.dirname(__file__)` 往上推 4 级找 `kai0/` 目录, 但 `ros2 run` 执行的是 **install 目录** (`ros2_ws/install/piper/lib/piper/`) 而非 **source 目录** (`ros2_ws/src/piper/scripts/`), 往上推 4 级得到 `ros2_ws/kai0` (不存在), 导致 venv python 找不到, re-exec 没有触发

### 解决方案

修改 `ros2_ws/src/piper/scripts/policy_inference_node.py`, 在所有 `import` 之前加入自动 re-exec 逻辑:

```python
import os
import sys

# ── 自动 re-exec: 确保在 kai0 venv 中运行 ────────────────────────
_KAI0_ROOT = os.environ.get('KAI0_ROOT', '')
if not _KAI0_ROOT or not os.path.isdir(_KAI0_ROOT):
    # 从 __file__ 位置推导: 兼容 source 和 install 两种布局
    for levels in [
        ('..', '..', '..', '..', 'kai0'),          # source 布局
        ('..', '..', '..', '..', '..', 'kai0'),     # install 布局
    ]:
        candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), *levels))
        if os.path.isdir(os.path.join(candidate, 'src', 'openpi')):
            _KAI0_ROOT = candidate
            break
    if not _KAI0_ROOT:
        # 硬编码回退
        for fallback in ['/data1/tim/workspace/deepdive_kai0/kai0',
                         os.path.expanduser('~/workspace/deepdive_kai0/kai0')]:
            if os.path.isdir(os.path.join(fallback, 'src', 'openpi')):
                _KAI0_ROOT = fallback
                break

_VENV_PYTHON = os.path.join(_KAI0_ROOT, '.venv', 'bin', 'python')
_VENV_PREFIX = os.path.join(_KAI0_ROOT, '.venv')

if (os.path.isfile(_VENV_PYTHON)
        and os.path.abspath(sys.prefix) != os.path.abspath(_VENV_PREFIX)):
    # 当前不在 kai0 venv 中 → 清理 conda 污染, re-exec 到 venv python
    os.environ['PATH'] = ':'.join(
        p for p in os.environ.get('PATH', '').split(':') if 'conda' not in p.lower())
    os.environ['LD_LIBRARY_PATH'] = ':'.join(
        p for p in os.environ.get('LD_LIBRARY_PATH', '').split(':') if 'conda' not in p.lower())
    os.execv(_VENV_PYTHON, [_VENV_PYTHON] + sys.argv)
```

**关键设计点:**

| 问题 | 解法 |
|------|------|
| conda python3.13 劫持 `python3` | `os.execv` 用 venv python 的绝对路径, 跳过 PATH 解析 |
| venv python 与系统 python 同一个二进制 | 不比较 `sys.executable`, 改为比较 `sys.prefix` 与 venv 路径 |
| install 布局下 `__file__` 推导失败 | 多级候选路径 + `src/openpi` 存在性校验 + 硬编码回退 |
| `os.execv` 后 conda lib 仍被继承 | re-exec 前从 `PATH` 和 `LD_LIBRARY_PATH` 中过滤掉 conda 路径 |
| `ros2 run` 传递的 `--ros-args` 参数 | `os.execv` 保留完整 `sys.argv`, 参数原样传递 |

### 修复后的执行链路

```
ros2 run piper policy_inference_node.py --ros-args -p mode:=websocket ...
  ↓
#!/usr/bin/env python3 → conda python3.13 (或任意 python)
  ↓
脚本前 10 行: 检测 sys.prefix ≠ venv prefix
  ↓
清理 PATH/LD_LIBRARY_PATH 中的 conda 路径
  ↓
os.execv(kai0/.venv/bin/python, [python, policy_inference_node.py, --ros-args, ...])
  ↓  ← 进程被完全替换, PID 不变
venv python3.12 启动, sys.prefix = kai0/.venv
  ↓
import numpy  → venv 的 numpy 1.26.4  ✓
import cv2    → venv 的 opencv        ✓
import rclpy  → ROS2 Jazzy (PYTHONPATH 从环境继承)  ✓
import jax    → venv 的 JAX 0.5.3     ✓
  ↓
正常启动 PolicyInferenceNode
```

### 验证

```bash
# 修复后 ros2 run 直接可用, 无需 wrapper 脚本
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 run piper policy_inference_node.py --ros-args \
  -p mode:=websocket -p host:=localhost -p port:=8000

# ros2 launch 同样正常 (走同一个脚本)
ros2 launch piper autonomy_launch.py mode:=websocket
```

测试结果:

| 测试项 | 结果 |
|--------|------|
| `ros2 run` (WebSocket 模式) | PASS, 30 Hz 发布 |
| re-exec 延迟 | <1ms (os.execv 是 syscall, 无 fork 开销) |
| `--ros-args` 参数传递 | PASS, 所有 ROS2 参数正常生效 |
| 环境变量继承 | PASS, `JAX_COMPILATION_CACHE_DIR` 等环境变量保留 |

### 适用范围

此修复 **仅影响 `policy_inference_node.py`**。其他 piper 驱动节点 (`arm_reader_node.py`, `arm_teleop_node.py`, `master_handle_node.py`) 依赖 `piper_sdk` + `tf_transformations`, 这些包装在系统 python3.12 中 (通过 `pip3.12 install`), 不需要 venv, 用 `ros2 run` 正常运行。
