#!/bin/bash
###############################################################################
# kai0 Autonomy / Replay 一键启动 (单一入口, 支持 model→arms 与 replay→arms/sim)
#
# 三种模式:
#   1) Autonomy (默认)        模型 → 真实机械臂      autonomy_launch.py
#                              cameras + policy_inference + arms (slave)
#   2) Replay (real arms)     回放数据 → 真实机械臂  replay_launch.py
#                              replay_node + arms (slave); 无 cameras / 无 JAX
#   3) Replay (sim)           回放数据 → 仿真      replay_launch.py + enable_real_arms:=false
#                              仅 replay_node 发 /master/joint_*; 不驱动 CAN/真机
#
# 用法:
#   ./scripts/start_autonomy.sh                         # 1: 默认 autonomy
#   ./scripts/start_autonomy.sh --no-rerun              # autonomy 不要 Rerun
#   ./scripts/start_autonomy.sh --mode websocket        # autonomy WebSocket
#   ./scripts/start_autonomy.sh --execute               # autonomy 直接执行
#   ./scripts/start_autonomy.sh --replay                # 2: 回放真机
#   ./scripts/start_autonomy.sh --replay --sim          # 3: 回放仿真
#
# Marker (/tmp/kai0_deployment_mode):
#   autonomy 模式 → "autonomy"; replay 模式 (real/sim 同) → "replay"
#   两者均能通过 backend `/api/replay/preflight` 的 marker 校验.
#
# 后续触发回放: 用 web 数据管理 UI (data_manager) 或 CLI:
#   ./scripts/start_replay_test.sh <task>/<subset>/<date>/<ep_id>
###############################################################################

set -eo pipefail

# ── 参数解析 ──
MODE="ros2"          # autonomy 模式下的 policy 通信通道 (ros2|websocket|both)
ENABLE_RERUN="true"
EXECUTE_MODE="false"
REPLAY="false"       # true = 走 playback_launch (回放数据 + rerun, 与 autonomy 同架构)
SIM="false"          # true 仅在 REPLAY 下有意义 — 不驱动 CAN/真机
EPISODE=""           # replay 模式必填: <task>/<subset>/<date>/<ep_id> 或绝对 parquet 路径
# ── 扩展模态参数 (depth + EE pose) ──
EXECUTION_MODE="joint"        # joint | ee_pose
ENABLE_DEPTH_INPUT="false"
ENABLE_EE_POSE_INPUT="false"
WS_PORT="8000"                # preflight + autonomy_launch port (JAX :8000 / V1 :8002)
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)       MODE="$2"; shift 2 ;;
        --no-rerun)   ENABLE_RERUN="false"; shift ;;
        --execute)    EXECUTE_MODE="true"; shift ;;
        --replay)     REPLAY="true"; shift ;;
        --sim)        SIM="true"; shift ;;
        --episode)    EPISODE="$2"; shift 2 ;;
        --execution-mode)       EXECUTION_MODE="$2"; shift 2 ;;
        --enable-depth-input)   ENABLE_DEPTH_INPUT="true"; shift ;;
        --enable-ee-pose-input) ENABLE_EE_POSE_INPUT="true"; shift ;;
        --ws-port)    WS_PORT="$2"; shift 2 ;;
        --port)       WS_PORT="$2"; shift 2 ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ "$EXECUTION_MODE" != "joint" && "$EXECUTION_MODE" != "ee_pose" ]]; then
    echo "[FAIL] --execution-mode must be 'joint' or 'ee_pose', got '$EXECUTION_MODE'" >&2
    exit 1
fi

if [[ "$SIM" == "true" && "$REPLAY" != "true" ]]; then
    echo "[FAIL] --sim 必须配合 --replay 使用 (autonomy 默认就是真机, sim 仅适用于 replay)" >&2
    exit 1
fi

if [[ "$REPLAY" == "true" && -z "$EPISODE" ]]; then
    echo "[FAIL] --replay 必须配合 --episode <task/subset/date/ep_id> 或绝对 parquet 路径" >&2
    exit 1
fi

# Resolve --episode → absolute parquet path
EPISODE_PATH=""
if [[ "$REPLAY" == "true" ]]; then
    KAI0_DATA_ROOT="${KAI0_DATA_ROOT:-/data1/DATA_IMP/KAI0}"
    if [[ "$EPISODE" == /* && -f "$EPISODE" ]]; then
        EPISODE_PATH="$EPISODE"
    else
        # parse <task>/<subset>/<date>/<ep_id>
        IFS='/' read -ra _PARTS <<< "$EPISODE"
        if [[ ${#_PARTS[@]} -ne 4 ]]; then
            echo "[FAIL] --episode 期望 'task/subset/date/ep_id' (4 段), 收到 '$EPISODE'" >&2
            exit 1
        fi
        _T="${_PARTS[0]}"; _S="${_PARTS[1]}"; _D="${_PARTS[2]}"; _E="${_PARTS[3]}"
        printf -v _EP6 '%06d' "$_E" 2>/dev/null || { echo "[FAIL] ep_id 非整数: $_E" >&2; exit 1; }
        EPISODE_PATH="$KAI0_DATA_ROOT/$_T/$_S/$_D/data/chunk-000/episode_${_EP6}.parquet"
    fi
    if [[ ! -f "$EPISODE_PATH" ]]; then
        echo "[FAIL] parquet not found: $EPISODE_PATH" >&2
        exit 1
    fi
fi

# ── 路径 ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KAI0_DIR="$PROJECT_ROOT/kai0"
ROS2_WS="$PROJECT_ROOT/ros2_ws"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
info() { echo -e "${CYAN}[INFO]${NC} $1"; }

if [[ "$REPLAY" == "true" ]]; then
    if [[ "$SIM" == "true" ]]; then
        STACK_LABEL="Replay (sim — no real arms)"
    else
        STACK_LABEL="Replay (real arms)"
    fi
else
    STACK_LABEL="Autonomy (model → real arms)"
fi

echo "============================================================"
echo " kai0 Launcher: $STACK_LABEL"
[[ "$REPLAY" != "true" ]] && echo " Mode: $MODE | Rerun: $ENABLE_RERUN | Execute: $EXECUTE_MODE"
echo "============================================================"

# ── 1. 清理残留进程 ──
echo ""
echo "--- Step 1: 清理残留进程 ---"

# Replay stack: 只清自身相关进程; autonomy: 多清 policy/rerun/cameras
if [[ "$REPLAY" == "true" ]]; then
    KILL_PATTERNS="arm_reader_node|replay_node|replay_launch"
else
    KILL_PATTERNS="realsense2_camera_node|arm_reader_node|arm_teleop_node|policy_inference_node|rerun_viz_node|multi_camera_node|autonomy_launch|replay_node|replay_launch"
fi
PIDS=$(ps aux | grep -E "$KILL_PATTERNS" | grep -v grep | grep -v $$ | awk '{print $2}' || true)
if [ -n "$PIDS" ]; then
    COUNT=$(echo "$PIDS" | wc -w)
    info "killing $COUNT leftover processes..."
    echo "$PIDS" | xargs kill -9 2>/dev/null || true
    sleep 3
    ok "cleaned up $COUNT processes"
else
    ok "no leftover processes"
fi

# ROS2 daemon restart (clean DDS state)
eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null; conda deactivate 2>/dev/null || true
source /opt/ros/jazzy/setup.bash
ros2 daemon stop 2>/dev/null || true
ros2 daemon start 2>/dev/null || true

# ── 2. USB 相机 reset (autonomy 才需要; replay/sim 跳过) ──
if [[ "$REPLAY" != "true" ]]; then
    echo ""
    echo "--- Step 2: USB camera reset ---"

    # Reset USB ports to clear stale device state.
    # Uses `sudo tee` (not `sudo bash -c`) so /etc/sudoers.d/kai0-autonomy can
    # grant NOPASSWD on an exact command pattern.
    for dev in 2-1 2-2 4-2.2; do
        auth="/sys/bus/usb/devices/$dev/authorized"
        if [ -e "$auth" ]; then
            echo 0 | sudo -n tee "$auth" >/dev/null 2>&1 || true
            sleep 0.5
            echo 1 | sudo -n tee "$auth" >/dev/null 2>&1 || true
        fi
    done
    sleep 3

    CAM_COUNT=$(lsusb | grep -c "Intel.*RealSense" 2>/dev/null || echo 0)
    if [ "$CAM_COUNT" -ge 3 ]; then
        ok "3 RealSense cameras detected"
    elif [ "$CAM_COUNT" -ge 2 ]; then
        warn "only $CAM_COUNT cameras (need 3)"
    else
        fail "only $CAM_COUNT cameras, check USB"
    fi
else
    info "skip USB camera reset (replay 模式无需相机)"
fi

# ── 3. CAN 激活 (sim 跳过, 否则需要驱动机械臂) ──
if [[ "$SIM" != "true" ]]; then
    echo ""
    echo "--- Step 3: CAN activation ---"

    CAN_UP=0
    if [[ "$REPLAY" == "true" ]]; then
        # replay 只用 slave (mode=1 arm_reader 订阅 /master/joint_* → CAN)
        IFACES="can_left_slave can_right_slave"
    else
        IFACES="can_left_mas can_left_slave can_right_mas can_right_slave"
    fi
    for iface in $IFACES; do
        if ip link show "$iface" &>/dev/null; then
            sudo -n ip link set "$iface" down 2>/dev/null || true
            sudo -n ip link set "$iface" type can bitrate 1000000 2>/dev/null || true
            sudo -n ip link set "$iface" up 2>/dev/null || true
            CAN_UP=$((CAN_UP + 1))
            ok "$iface up"
        fi
    done

    if [[ "$REPLAY" == "true" ]]; then
        if [ "$CAN_UP" -ge 2 ]; then
            ok "$CAN_UP CAN interfaces up (replay slave-only)"
        else
            warn "only $CAN_UP CAN interfaces (replay needs 2: left+right slave)"
        fi
    else
        if [ "$CAN_UP" -ge 4 ]; then
            ok "$CAN_UP CAN interfaces up (dual arm master+slave)"
        elif [ "$CAN_UP" -ge 2 ]; then
            warn "only $CAN_UP CAN interfaces (need 4 for dual arm master+slave)"
        else
            warn "only $CAN_UP CAN interfaces"
        fi
    fi
else
    info "skip CAN activation (--sim: 不驱动真机)"
fi

# ── 4. 依赖检查 (replay 不需要 venv/serve_policy) ──
echo ""
echo "--- Step 4: dependency check ---"

if [[ "$REPLAY" != "true" ]]; then
    # venv (autonomy 需要 JAX)
    if [ -f "$KAI0_DIR/.venv/bin/python" ]; then
        ok "venv: $($KAI0_DIR/.venv/bin/python --version 2>&1)"
    else
        fail "venv not found: $KAI0_DIR/.venv/"
    fi
fi

# ROS2 workspace (两种模式都要)
if [ -f "$ROS2_WS/install/setup.bash" ]; then
    ok "ROS2 workspace: $ROS2_WS"
else
    fail "ROS2 workspace not built: run 'cd $ROS2_WS && colcon build'"
fi

if [[ "$REPLAY" != "true" ]]; then
    # Calibration
    CALIB_FILE="$PROJECT_ROOT/config/calibration.yml"
    if [ -f "$CALIB_FILE" ]; then
        ok "calibration: $CALIB_FILE"
    else
        warn "calibration not found: $CALIB_FILE (FK visualization will be disabled)"
    fi

    # serve_policy (websocket mode) — port follows --ws-port (default 8000=JAX, 8002=V1)
    if [ "$MODE" = "websocket" ] || [ "$MODE" = "both" ]; then
        if ss -tlnp 2>/dev/null | grep -q ":${WS_PORT} "; then
            ok "serve_policy running on :${WS_PORT}"
        else
            fail "serve_policy not running on :${WS_PORT}. Start it first:
    JAX (:8000):  cd $KAI0_DIR && CUDA_VISIBLE_DEVICES=1 JAX_COMPILATION_CACHE_DIR=/tmp/xla_cache \\
      .venv/bin/python scripts/serve_policy.py --port 8000 policy:checkpoint \\
      --policy.config=pi05_flatten_fold_normal --policy.dir=checkpoints/Task_A/mixed_1
    V1 (:8002):   ./start_scripts/start_serve_v1.sh"
        fi
    fi
fi

# ── 5. Rebuild if needed ──
echo ""
echo "--- Step 5: build check ---"

# Check if any source file is newer than the install marker
INSTALL_MARKER="$ROS2_WS/install/piper/.colcon_install_layout"
NEEDS_BUILD=false
if [ ! -f "$INSTALL_MARKER" ]; then
    NEEDS_BUILD=true
else
    # Check if any source is newer than install
    NEWEST_SRC=$(find "$ROS2_WS/src/piper" -name '*.py' -newer "$INSTALL_MARKER" 2>/dev/null | head -1)
    if [ -n "$NEWEST_SRC" ]; then
        NEEDS_BUILD=true
    fi
fi

if [ "$NEEDS_BUILD" = true ]; then
    info "source changed, rebuilding..."
    (cd "$ROS2_WS" && source /opt/ros/jazzy/setup.bash && colcon build --packages-select piper 2>&1 | tail -3)
    ok "rebuild done"
else
    ok "install up to date"
fi

# ── 6. 启动 ──
echo ""
echo "--- Step 6: launching ---"

source /opt/ros/jazzy/setup.bash
source "$ROS2_WS/install/setup.bash"

# Add venv bin to PATH (for rerun CLI). Replay 不依赖 venv 里的 jax/openpi 但保留无害.
[ -d "$KAI0_DIR/.venv/bin" ] && export PATH="$KAI0_DIR/.venv/bin:$PATH"

# Unset proxy vars that can interfere with JAX/gRPC
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

# ── 部署模式 marker ──
# autonomy → "autonomy"; replay (real or sim) → "replay".
# Backend `/api/replay/preflight` 接受任一即可放行 replay.
if [[ "$REPLAY" == "true" ]]; then
    echo replay > /tmp/kai0_deployment_mode
    info "deployment marker = replay"
    info "episode parquet  : $EPISODE_PATH"
    if [[ "$SIM" == "true" ]]; then
        info "mode             : SIM (no real arms; /master→/puppet relay for pose alignment)"
        ENABLE_REAL_ARMS=false
    else
        info "mode             : REAL (arm_reader slave drives CAN)"
        ENABLE_REAL_ARMS=true
    fi
    # rerun_viz_node uses JAX for FK; on Blackwell (RTX 5090, sm_120) jax 0.5.x's
    # XLA autotuner SIGSEGVs during compile. autonomy mode sets these vars
    # later in the script; in replay mode we exec the launch right here so we
    # need to set them BEFORE the exec.
    export JAX_COMPILATION_CACHE_DIR=/tmp/xla_cache
    mkdir -p /tmp/xla_cache
    export XLA_FLAGS="--xla_gpu_autotune_level=0"
    info "JAX env           : XLA_FLAGS=$XLA_FLAGS, cache=$JAX_COMPILATION_CACHE_DIR"
    echo ""
    info "starting ros2 launch piper playback_launch.py (architecture parity with autonomy)"
    echo "  Ctrl+C to stop"
    echo ""
    exec ros2 launch piper playback_launch.py \
        episode_path:="$EPISODE_PATH" \
        enable_real_arms:="$ENABLE_REAL_ARMS" \
        enable_rerun:="$ENABLE_RERUN" \
        "${EXTRA_ARGS[@]}"
fi

# ── Autonomy 专属: GPU + JAX 配置 ──
export JAX_COMPILATION_CACHE_DIR=/tmp/xla_cache
mkdir -p /tmp/xla_cache

# GPU allocation strategy:
#   1. KAI0_GPU_ID=<n> forces the choice (escape hatch).
#   2. Otherwise require ≥MIN_FREE_MB free (π0.5 restore peaks ~2.25 GiB per
#      shard on top of weights + KV cache + cuBLAS context — 6 GB cards OOM).
#      Among qualifying GPUs, prefer fewest running compute apps, then most
#      free memory (empty card > big-but-shared card, per old fragmentation
#      concern).
#   3. If nothing qualifies, fall back to the most-free GPU and warn loudly.
MIN_FREE_MB=${KAI0_MIN_FREE_MB:-12288}
if [ -n "$KAI0_GPU_ID" ]; then
    GPU_ID="$KAI0_GPU_ID"
    info "KAI0_GPU_ID override → using GPU $GPU_ID"
else
    GPU_ID=$(
        MIN_FREE_MB="$MIN_FREE_MB" python3 -c "
import os, subprocess
min_free = int(os.environ['MIN_FREE_MB'])
r = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True)
gpus = {}
for line in r.stdout.strip().split('\n'):
    idx, uuid, free = [x.strip() for x in line.split(',')]
    gpus[uuid] = (int(idx), int(free), 0)
r = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                    '--format=csv,noheader'], capture_output=True, text=True)
for line in r.stdout.strip().split('\n'):
    uuid = line.strip()
    if uuid in gpus:
        idx, free, n = gpus[uuid]
        gpus[uuid] = (idx, free, n + 1)
eligible = [g for g in gpus.values() if g[1] >= min_free]
if eligible:
    best = sorted(eligible, key=lambda g: (g[2], -g[1]))[0]
    print(best[0])
else:
    best = sorted(gpus.values(), key=lambda g: -g[1])[0]
    print(f'{best[0]}:LOW')
" 2>/dev/null
    )
    if [[ "$GPU_ID" == *:LOW ]]; then
        GPU_ID="${GPU_ID%:LOW}"
        warn "no GPU has ≥${MIN_FREE_MB}MB free; falling back to GPU $GPU_ID (inference may OOM). Override with KAI0_GPU_ID=<n> or lower KAI0_MIN_FREE_MB."
    fi
    GPU_ID=${GPU_ID:-0}
fi
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null || echo "?")
info "using GPU $GPU_ID (free: ${FREE_MB}MB, threshold: ${MIN_FREE_MB}MB)"

# Blackwell (RTX 5090 / sm_120) workaround: jax/jaxlib 0.5.3's XLA autotuner
# SIGSEGVs during π₀ backend_compile. Disabling autotune costs ~5-20% infer
# speed but is the only fix short of upgrading jax to ≥0.6.x.
export XLA_FLAGS="--xla_gpu_autotune_level=0"

echo ""
echo "  Mode:    $MODE"
echo "  Rerun:   $ENABLE_RERUN"
echo "  Execute: $EXECUTE_MODE"
echo "  GPU:     $GPU_ID"
echo ""

# ── Deployment mode marker (P1: replay function safety gate) ──
# Lets `replay_mode=replay` pass `_verify_deployment_marker()` check.
echo autonomy > /tmp/kai0_deployment_mode

info "starting ros2 launch piper autonomy_launch.py..."
echo "  Ctrl+C to stop all nodes"
echo ""

exec ros2 launch piper autonomy_launch.py \
    mode:="$MODE" \
    port:="$WS_PORT" \
    enable_rerun:="$ENABLE_RERUN" \
    fg_enable:=false \
    bg_enable:=false \
    execute_mode:="$EXECUTE_MODE" \
    gpu_id:="$GPU_ID" \
    config_name:=pi05_flatten_fold_normal \
    checkpoint_dir:="$KAI0_DIR/checkpoints/Task_A/mixed_1" \
    execution_mode:="$EXECUTION_MODE" \
    enable_depth_input:="$ENABLE_DEPTH_INPUT" \
    enable_ee_pose_input:="$ENABLE_EE_POSE_INPUT" \
    "${EXTRA_ARGS[@]}"
