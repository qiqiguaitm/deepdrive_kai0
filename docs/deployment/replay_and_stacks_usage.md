# Replay 与三栈使用指南

> 创建: 2026-04-30 · 责任人: ruitao jing
> 涵盖: autonomy / replay-only / data_collect 三个互斥栈, 及 collect↔replay 在线切换

## 0. 三栈对照

| 栈 | 启动脚本 | 含 ROS2 节点 | 用途 | 启动时长 | RAM |
|---|---|---|---|---|---|
| **autonomy** | `start_autonomy.sh` | `policy_inference + piper_left/right + multi_camera + rerun_viz` | policy 真机推理 | ~30s (JAX warmup) | ~5 GB |
| **replay-only** ⭐ | `start_replay_stack.sh` | `replay + piper_left/right` | 只跑 replay, 不需 policy/相机 | ~1s | ~50 MB |
| **data_collect** | `start_data_collect.sh` | teleop 主从臂 + cameras + backend + frontend + pedal | 数据采集 | ~5s | — |

**关键约束**: 三栈**互斥**, 任一时刻只能跑一个 arms 层. 因为 `/master/joint_left` `/master/joint_right` 两个 topic 只允许一个 publisher (DDS 多 publisher 会让从臂收到混乱命令).

部署模式由 marker 文件标识: `/tmp/kai0_deployment_mode`, 内容是 `autonomy` / `replay` / `teleop` 之一. 各启动脚本独立维护, **stop 时只删自己写的那个值**, 不会误踩其它栈.

## 1. 命令快速对照

### 1.1 autonomy (跑 policy 推理)
```bash
cd /home/tim/workspace/deepdive_kai0
./start_scripts/start_autonomy.sh                  # 前台
nohup ./start_scripts/start_autonomy.sh > /tmp/autonomy.log 2>&1 &  # 后台
disown
# 停: Ctrl+C 或 pkill -f autonomy_launch
```

### 1.2 replay-only (只 replay, slim, 推荐)
```bash
cd /home/tim/workspace/deepdive_kai0
./start_scripts/start_replay_stack.sh              # 前台
nohup ./start_scripts/start_replay_stack.sh > /tmp/replay.log 2>&1 &  # 后台
disown
# 停 (干净):
./start_scripts/start_replay_stack.sh stop
```

### 1.3 data_collect (采集 + UI)
```bash
cd /home/tim/workspace/deepdive_kai0
./start_scripts/start_data_collect.sh              # 起 5 个服务
./start_scripts/start_data_collect.sh status       # 看状态
./start_scripts/start_data_collect.sh logs backend # 看 backend 日志
./start_scripts/start_data_collect.sh stop         # 全停
```

⚠️ 单独起 backend (不带 teleop):
```bash
cd /data1/tim/workspace/deepdive_kai0/web/data_manager
SKIP_ARMS=1 SKIP_CAMERAS=1 SKIP_PEDAL=1 SKIP_DEPS=1 ./run.sh start backend
```
否则 `./run.sh start backend` 会**整栈拉起 (含 teleop arms)**, 跟 replay 抢 topic.

## 2. Replay 触发方式 (3 种入口, 等价)

### 2.1 Web UI ⭐ 推荐
- 浏览器打开 http://localhost:5173/  (远程: http://192.168.208.51:5173/)
- 顶部切到 **admin** 角色
- 左侧 episode 列表选一条 (注: `task_id` 必须带 `YYYY-MM-DD` 后缀; `kai0_official_base` 这种走 CLI)
- 右侧 ReplayPanel 滚到底, 勾 **真实执行**
- 自动跑 preflight: 显示 `target_node` (`/replay` 或 `/policy_inference`), 起点偏差, 部署模式
- 点红色 **▶ 确认下发到机械臂** → 进度条滚动 ~3 秒 home + 原 episode 时长
- 想停: **⏹ 停止真机回放**

### 2.2 CLI 脚本
```bash
cd /home/tim/workspace/deepdive_kai0
./start_scripts/start_replay_test.sh Task_A/base/2026-04-28/42         # 自录数据
./start_scripts/start_replay_test.sh Task_A/base/kai0_official_base/104 # kai0 官方
./start_scripts/start_replay_test.sh Task_A/base/2026-04-28/42 0.7      # 0.7× 慢速
```
脚本自动: 检 marker → detect 节点 (`/replay` 优先) → S3 parquet shape → 打印 confirm prompt → 真发 → 进度 echo.

### 2.3 Backend curl (脚本化 / 远程)
```bash
cd /data1/tim/workspace/deepdive_kai0/web/data_manager
./test_replay_api.sh preflight      # dry-run, 看姿态偏差
./test_replay_api.sh execute        # 真发
./test_replay_api.sh watch          # 1Hz 轮询进度
./test_replay_api.sh stop           # 中途停

# 换 episode:
EPISODE_ID=25 ./test_replay_api.sh execute
TASK=Task_A SUBSET=base DATE=2026-04-28 EPISODE_ID=42 ./test_replay_api.sh execute
RATE=0.5 ./test_replay_api.sh execute
```

## 3. collect ↔ replay 在线切换

数据采集后想立刻 replay 验证, 不重启 cameras/backend/frontend, 只换 arms 层:

### 3.1 collect → replay
```bash
# 1. 杀 teleop arms (保留 cameras+backend+frontend+pedal)
pkill -9 -f teleop_launch
pkill -9 -f arm_teleop_node
sleep 2

# 2. 切 marker
echo replay > /tmp/kai0_deployment_mode

# 3. 起 replay arms
source /opt/ros/jazzy/setup.bash
source /data1/tim/workspace/deepdive_kai0/ros2_ws/install/setup.bash
nohup ros2 launch piper replay_launch.py > /tmp/replay_arms.log 2>&1 &
disown
sleep 3
ros2 daemon stop && ros2 daemon start && sleep 2
ros2 node list | grep replay   # 期望 /replay 出现
```
此时 web UI 的 "真实执行" toggle 可用, backend 自动 detect `/replay`.

### 3.2 replay → collect
```bash
# 1. 杀 replay arms
pkill -9 -f replay_launch
pkill -9 -f replay_node
pkill -9 -f arm_reader_node
sleep 2

# 2. 切 marker
echo teleop > /tmp/kai0_deployment_mode

# 3. 起回 teleop arms
source /opt/ros/jazzy/setup.bash
source /data1/tim/workspace/deepdive_kai0/ros2_ws/install/setup.bash
nohup ros2 launch piper teleop_launch.py > /tmp/teleop_arms.log 2>&1 &
disown
sleep 3
```

## 4. 安全栏 / 预检字段

backend `/api/replay/preflight` 返回的字段:

| 字段 | 含义 | 失败行为 |
|---|---|---|
| `ok` | 是否所有 gate 都通过 | `false` 时 UI/CLI 拒绝 execute |
| `target_node` | `/replay` 或 `/policy_inference` (active stack) | `null` → 没节点 alive |
| `deployment_mode` | `autonomy` / `replay` / `teleop` / 其它 | 非前两者拒绝 |
| `policy_inference_alive` | (兼容字段) `target_node != null` | — |
| `publisher_conflict` | `/master/joint_left` 上 self 之外的 publisher 列表 | 非空拒绝 |
| `aligned` | 当前从臂姿态与 `action[0]` 偏差 ≤ 5° | `false` + auto_home 关 → 拒绝 |
| `auto_home_will_trigger` | 起点偏 > 5°, 会自动 prepend 慢挪 | `true` 表 OK |
| `home_n_planned` | auto-home 帧数 (默认 90 = 3s @ 30Hz) | — |
| `frames` / `duration_s` / `fps` | parquet 元数据 | — |
| `expected_buffer_total` | 预估 buffer 总帧数 (home + episode @ publish_rate) | — |
| `max_diff_deg` / `per_joint_diff_deg` | 偏差度数 | — |

## 5. 常见错误 → 解决

| 现象 | 根因 | 修 |
|---|---|---|
| `ros2: command not found` | 当前 shell 没 source ROS | `source /opt/ros/jazzy/setup.bash` |
| `target_node: null` | autonomy / replay 都没起 | 起其中一个 |
| `publisher_conflict: ['piper_master_left']` | teleop 和 replay/autonomy 同时跑 | 杀掉 teleop, ros2 daemon stop+start |
| `Setting parameter failed: Node not found` | 节点已死或 daemon 拓扑 stale | `ros2 daemon stop && start`; 还不行就重启 stack |
| 瞬间 done=576/576 | 上次 replay 没 reset 模式 (老 bug, 已修) | 升级到最新代码 |
| arm 跑 2 秒就停 | publisher_conflict / fps 错 (老 bug, 已修) | 升级 |
| 起点偏 68°, replay 拒绝 | aligned=false, auto_home 关了 | `replay_auto_home=true` (默认开), 或手动 home |
| `./run.sh start backend` 拉起整个 stack | run.sh 不接 service 过滤 | 加 `SKIP_ARMS=1 SKIP_CAMERAS=1 SKIP_PEDAL=1 SKIP_DEPS=1` |

## 6. 验证健康

```bash
# 节点列表
source /opt/ros/jazzy/setup.bash
ros2 daemon stop && ros2 daemon start && sleep 2
ros2 node list | sort

# Topic 检查
ros2 topic info /master/joint_left -v   # 看 publisher 谁在
ros2 topic hz /puppet/joint_left        # 应 ~200 Hz (从臂反馈)
ros2 topic hz /master/joint_left        # replay 时 30 Hz, 否则 0

# Backend health
curl -sS http://localhost:8787/api/health    # {"ok": true}

# Backend 选了哪个节点
curl -sS -X POST http://localhost:8787/api/replay/preflight \
  -H 'Content-Type: application/json' \
  -d '{"task":"Task_A","subset":"base","date":"2026-04-28","episode_id":42}' \
  | python3 -c 'import json,sys;d=json.load(sys.stdin);print("target_node:",d.get("target_node"))'
```

## 7. 推荐工作流 (典型场景)

### 场景 A: 纯 replay 测试 (推荐, 最轻)
```bash
# 1. 干净起 slim replay
nohup /home/tim/workspace/deepdive_kai0/start_scripts/start_replay_stack.sh > /tmp/replay.log 2>&1 &
disown
sleep 3

# 2. 单独起 backend (不拉 teleop)
cd /data1/tim/workspace/deepdive_kai0/web/data_manager
SKIP_ARMS=1 SKIP_CAMERAS=1 SKIP_PEDAL=1 SKIP_DEPS=1 ./run.sh start backend

# 3. 浏览器测 / CLI 测均可

# 4. 收
/home/tim/workspace/deepdive_kai0/start_scripts/start_replay_stack.sh stop
pkill -f 'uvicorn.*app.main'
```

### 场景 B: 先采集后 replay 验证
```bash
# 1. 起 data_collect 全栈
/home/tim/workspace/deepdive_kai0/start_scripts/start_data_collect.sh
# 浏览器录数据...

# 2. 录完, 切到 replay (见 §3.1)
pkill -9 -f teleop_launch && sleep 2
echo replay > /tmp/kai0_deployment_mode
source /opt/ros/jazzy/setup.bash
source /data1/tim/workspace/deepdive_kai0/ros2_ws/install/setup.bash
nohup ros2 launch piper replay_launch.py > /tmp/replay_arms.log 2>&1 &
disown && sleep 3 && ros2 daemon stop && ros2 daemon start

# 3. 浏览器选刚录的 episode → 真实执行 → 看回放
```

### 场景 C: autonomy + replay 共用 (policy 测试 + 偶尔回放对比)
- autonomy 跑着时, backend 自动用 `/policy_inference` 路径 replay
- web UI 的 "真实执行" toggle 直接可用
- 但 autonomy 的 inference 跟 replay 互斥 (同一个 node 内部 mode 切换), replay 时 inference 暂停, replay 完自动切回 inference

```bash
# 起 autonomy
nohup /home/tim/workspace/deepdive_kai0/start_scripts/start_autonomy.sh > /tmp/autonomy.log 2>&1 &
disown
# 等 ~30 秒 warmup, 看 log 出现 'Inference loop started'

# 起 backend (单独)
cd /data1/tim/workspace/deepdive_kai0/web/data_manager
SKIP_ARMS=1 SKIP_CAMERAS=1 SKIP_PEDAL=1 SKIP_DEPS=1 ./run.sh start backend

# 浏览器: 选 episode → 真实执行 (用 /policy_inference 跑 replay)
# 退出 replay 自动回 inference 模式
```

## 8. 关键文件 (供后续维护)

| 文件 | 作用 |
|---|---|
| `start_scripts/start_autonomy.sh` | autonomy 启动 + 写 marker=autonomy |
| `start_scripts/start_replay_stack.sh` | slim 启动 + 写 marker=replay + stop 子命令 |
| `start_scripts/start_data_collect.sh` | data_collect 启动 + 写 marker=teleop / stop 删 marker |
| `start_scripts/start_replay_test.sh` | CLI 端到端测试 (auto-detect 节点) |
| `ros2_ws/src/piper/scripts/replay_node.py` | slim replay 节点 (无 JAX) |
| `ros2_ws/src/piper/scripts/policy_inference_node.py` | 完整 policy + replay 节点 |
| `ros2_ws/src/piper/launch/replay_launch.py` | slim 栈 launch |
| `ros2_ws/src/piper/launch/autonomy_launch.py` | 完整 autonomy launch |
| `ros2_ws/src/piper/launch/teleop_launch.py` | data_collect 用的主从臂 launch |
| `web/data_manager/backend/app/replay.py` | backend `/api/replay/*` 实现 + auto-detect 节点 |
| `web/data_manager/backend/app/ros_bridge.py` | `clear_replay_progress`, `publish_execute`, `/replay_progress` 订阅 |
| `web/data_manager/frontend/src/components/ReplayPanel.tsx` | UI toggle / preflight / progress |
| `web/data_manager/test_replay_api.sh` | curl 包装, 测试用 |

## 9. 历史踩坑速查

详见 `docs/deployment/task_a_real_robot_grasp_corner_debug_log.md` 等. 本指南只快速列:

- **"replay 瞬间完成 100%"** → 多个 race / 老 cache bug 全已修 (2026-04-30 完工)
- **"右臂不动"** → 不是数据问题, 是 publisher 抢 (teleop + replay 同跑)
- **"Setting parameter failed: Node not found"** → daemon stale, `ros2 daemon stop && start`
- **"./run.sh start backend 误启 teleop"** → 加 SKIP_*

## 10. 改了什么 vs 没改

不动的:
- 真机硬件 / CAN 配置 / piper SDK
- LeRobot v2.1 数据格式
- gripper_offset 在 autonomy 路径仍是 0.003 (replay 路径忽略)

改了的 (跨 P1 - polish):
- replay 走 mode/buffer/stream_buffer 链, 复用 publish_timer / jump_protection
- auto-home 起点偏 > 5° 时 prepend 90 帧线性插值
- fps 补偿 parquet_fps → publish_rate 上采样
- backend / CLI / Frontend 自动 detect `/replay` 或 `/policy_inference`
- deployment marker 三模式互斥
