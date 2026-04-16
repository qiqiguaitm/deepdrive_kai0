# data_manager — 双臂 VLA 数据采集 UI

按 `../ui_data_collection_plan.md` 实现的前后端骨架，已接真 ROS2 + 真 mp4/parquet 落盘。

## 结构
```
data_manager/
├── backend/        FastAPI + 统计服务 + 录制状态机（PyAV AV1 + pyarrow）+ 模板
├── frontend/       React + TypeScript + Vite
├── config/         collection_templates.yml
├── data_mock/      旧版本地开发占位 (新数据写到 /data1/DATA_IMP/KAI0, 见下)
└── run.sh          一键启动 CAN + 机械臂 + 相机 + 后端 + 前端
```

## 一键启动
```bash
./run.sh start        # 启动全部
./run.sh status       # 查看各服务状态
./run.sh logs backend # 跟踪单个服务日志
./run.sh stop         # 停止全部

# 跳过部分模块（例如已外部启动或仅调前端）
SKIP_CAN=1 SKIP_ARMS=1 SKIP_CAMERAS=1 ./run.sh start
```
启动后：
- 前端 http://HOST:5173/
- 后端 http://HOST:8787/  (REST docs: `/docs`；WS: `/ws/status`)

## 后端 venv 构建

### 为什么不能直接 `python3 -m venv`
后端需要 `import rclpy`。ROS2 Jazzy 只分发了 **Python 3.12** 的 rclpy C 扩展
(`/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/_rclpy_pybind11.cpython-312-*.so`),
其它 Python 版本导入时会报
`No module named 'rclpy._rclpy_pybind11'` → 后端静默回落到 `MockBridge`, UI 上
CAN / cameras / teleop 全红。因此 venv 里的 `python` 必须是 `3.12`。

sim01 的复杂性:
1. `python3` 指向 miniconda 的 3.13 (不匹配 rclpy);
2. `/usr/bin/python3.12` 存在, 但系统没装 `python3.12-venv`, 所以 `python3.12 -m venv`
   会报 `ensurepip is not available`;
3. 我们不总是有 `sudo apt install python3.12-venv` 的权限.

下面给出两条路径: 有 sudo 走 A (干净), 没 sudo 走 B (workaround, 与 gzllll 的现有 venv 一致).

### A. 有 sudo —— 直接用系统 3.12
```bash
sudo apt install -y python3.12-venv

cd web/data_manager/backend
rm -rf .venv
/usr/bin/python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

### B. 无 sudo —— "3.13 骨架 + 3.12 解释器" 组合
用 miniconda 3.13 (有 `venv` 模块) 生成 venv 骨架, 然后把 `bin/python` 换成
`/usr/bin/python3.12`, 再用 `get-pip.py` 给 3.12 装 pip (因为没 `ensurepip`):
```bash
cd web/data_manager/backend
rm -rf .venv

# 1) 用 miniconda 的 3.13 建骨架 (activate / site-packages / include)
/data1/miniconda3/bin/python3 -m venv .venv

# 2) 把 python* 符号链接全部指向系统 3.12, 让实际运行的解释器是 3.12
rm .venv/bin/python .venv/bin/python3 .venv/bin/python3.13
ln -sf /usr/bin/python3.12 .venv/bin/python
ln -sf python .venv/bin/python3
ln -sf python .venv/bin/python3.12
ln -sf python .venv/bin/python3.13

# 3) 没 ensurepip, 用官方 bootstrap 给 3.12 装 pip
curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py

# 4) 装依赖 (会写入 lib/python3.12/site-packages)
.venv/bin/pip install -r requirements.txt
```
产出的 venv: `pyvenv.cfg` 里 version 仍然是 3.13 (骨架来源), 但 `bin/python -V`
返回 `Python 3.12.x`, 且 `lib/python3.12/site-packages` 有全套包。

### 验证
```bash
source /opt/ros/jazzy/setup.bash
source $REPO_ROOT/ros2_ws/install/setup.bash
.venv/bin/python -c "import rclpy; print(rclpy.__file__)"
# 期望: /opt/ros/jazzy/lib/python3.12/site-packages/rclpy/__init__.py

./run.sh start
grep RclpyBridge logs/backend.log
# 期望: [ros_bridge] RclpyBridge online    (不是 "using MockBridge")
```
UI 上 `CAN 左/右`, `teleop`, 三路相机 fps 应全部变绿。

### 常见环境变量
```bash
export KAI0_TEMPLATES=../config/collection_templates.yml
# ROS bridge: auto(默认) / mock(强制假数据, 不依赖 rclpy 可用)
# export KAI0_ROS_BRIDGE=mock
# 数据根: 默认 /data1/DATA_IMP/KAI0 (与 repo 隔离, git clean / 删 venv 不会误清)
# export KAI0_DATA_ROOT=/some/other/path
```
手动启动后端 (`run.sh` 以外):
```bash
cd backend
source /opt/ros/jazzy/setup.bash && source $REPO_ROOT/ros2_ws/install/setup.bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8787 --reload
```

## 前端（手动）
```bash
cd frontend
npm install && npm run dev   # http://localhost:5173
```
Vite 代理 `/api`、`/ws` → `localhost:8787`。

## 角色切换
顶栏按钮切换 collector/admin，前端写入 `localStorage.role`，请求带 `X-Role`；
管理员路由在后端 `require_admin` 校验。

## ROS2 桥 (`backend/app/ros_bridge.py`)
后端常驻 `rclpy` 订阅：
- 关节：`/master/joint_{left,right}` + `/puppet/joint_{left,right}`（见 `config/pipers.yml`）
- 相机：`ros2_topic_color` 三路 RealSense（见 `config/cameras.yml`），CameraInfo 算 fps/latency，Image 同时缓存 JPEG（供 MJPEG 端点）和原始 RGB ndarray（供录制）
- 暴露 `get_health / get_joint_state / get_camera_health / get_latest_jpeg / get_frame_rgb / get_state_action`

当 `rclpy` 或 YAML 不可用时自动退回 `MockBridge`（返回正弦关节值 + 渐变条纹图），端到端链路仍可跑通。强制 mock：`KAI0_ROS_BRIDGE=mock`。

## 录制落盘
`backend/app/recorder.py` — `start() → RECORDING → save()/discard()`：
- `start()` 起 30Hz 采集线程，从 bridge 拉三路 RGB 帧 + 14 维 state/action
- PyAV 编码：默认 `libx264`（双击/浏览器都能播）；`KAI0_VIDEO_CODEC=av1` 切换为 `libsvtav1`/`libaom-av1` 匹配 LeRobot 归档格式；`480×640`、`yuv420p`、30fps
- `save()` flush 容器，`pyarrow` 写 LeRobot v2.1 parquet，更新 `meta/{episodes.jsonl, tasks.jsonl, info.json}`；stats 服务自动 upsert
- `discard()` 关闭容器并删除半成品文件

产出目录遵循 LeRobot v2.1：
```
<KAI0_DATA_ROOT>/Task_A/<base|dagger>/
├── data/chunk-000/episode_NNNNNN.parquet    # obs.state[14], action[14], timestamp, frame_index, ...
├── videos/chunk-000/{top_head,hand_left,hand_right}/episode_NNNNNN.mp4  # AV1
└── meta/{episodes.jsonl, tasks.jsonl, info.json}
```

state = 从臂 (puppet) 关节 = 机器人真实位姿；action = 主臂 (master) 关节 = 遥操指令；
顺序 `[L_j1..L_j6, L_gripper, R_j1..R_j6, R_gripper]`。

### 视频播放
默认录 H.264，本地播放器/浏览器双击直接播。若设 `KAI0_VIDEO_CODEC=av1` 录 AV1（匹配
LeRobot 归档规范），视频端点会在线用 ffmpeg 转成 H.264 碎片 mp4 给浏览器；`?raw=1` 拿原始文件。
已有 AV1 文件想一次性转 H.264：`bash backend/tools/transcode_av1_to_h264.sh [DATA_ROOT]`，
原文件备份到 `*.mp4.av1.bak`，同步把各 `meta/info.json` 里 `video.codec` 改成 `h264`。

## 右下角健康提示 (FloatingHealth)
`frontend/src/components/StatusBar.tsx` 内 `collectFailures` 汇总右下角大牌判据：
- `health.ros2 / can_left / can_right / teleop` 任一为 false
- 任一期望相机 (`top_head / hand_left / hand_right`) 缺失，或实测 `fps < 25`
- 录制状态为 `ERROR`
- 后端下发的 `warnings`（如 `low_disk:<free>GB`）

相机 `dropped` 计数仅用于 `CameraGrid` 展示，**不参与异常判定**，避免瞬时抖动刷红。

## 环境变量速查
| 变量 | 作用 | 默认 |
|------|------|------|
| `KAI0_DATA_ROOT` | 采集落盘根目录 (与 repo 隔离避免误删) | `/data1/DATA_IMP/KAI0` |
| `KAI0_TEMPLATES` | 采集模板 yml | `<repo>/web/data_manager/config/collection_templates.yml` |
| `KAI0_PIPERS_YML` | 机械臂配置 | `<repo>/config/pipers.yml` |
| `KAI0_CAMERAS_YML` | 相机配置 | `<repo>/config/cameras.yml` |
| `KAI0_ROS_BRIDGE` | `auto` / `mock` | `auto` |
| `KAI0_JPEG_QUALITY` | MJPEG JPEG 质量 | `60` |
| `KAI0_JPEG_STRIDE` | MJPEG 下采样 | `2` |
