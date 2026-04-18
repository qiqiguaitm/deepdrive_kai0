#!/bin/bash
###############################################################################
# kai0 自主运行 + Rerun 可视化 一键启动 (autonomy_launch + rerun)
#
# 用法:
#   ./scripts/start_autonomy.sh                    # 默认: ros2 模式 + Rerun
#   ./scripts/start_autonomy.sh --no-rerun         # 不启动 Rerun
#   ./scripts/start_autonomy.sh --mode websocket   # WebSocket 模式
#   ./scripts/start_autonomy.sh --execute          # 直接进入执行模式
#
# 流程:
#   1. 清理残留进程
#   2. USB 相机 reset
#   3. CAN 激活
#   4. 依赖检查
#   5. colcon build (如果源码比 install 新)
#   6. 启动 ros2 launch
###############################################################################

set -eo pipefail

# ── 参数解析 ──
MODE="ros2"
ENABLE_RERUN="true"
EXECUTE_MODE="false"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)       MODE="$2"; shift 2 ;;
        --no-rerun)   ENABLE_RERUN="false"; shift ;;
        --execute)    EXECUTE_MODE="true"; shift ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

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

echo "============================================================"
echo " kai0 Autonomy + Visualization"
echo " Mode: $MODE | Rerun: $ENABLE_RERUN | Execute: $EXECUTE_MODE"
echo "============================================================"

# ── 1. 清理残留进程 ──
echo ""
echo "--- Step 1: 清理残留进程 ---"

KILL_PATTERNS="realsense2_camera_node|arm_reader_node|arm_teleop_node|policy_inference_node|rerun_viz_node|multi_camera_node|autonomy_launch"
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

# ── 2. USB 相机 reset ──
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

# ── 3. CAN 激活 ──
echo ""
echo "--- Step 3: CAN activation ---"

CAN_UP=0
for iface in can_left_mas can_left_slave can_right_mas can_right_slave; do
    if ip link show "$iface" &>/dev/null; then
        sudo -n ip link set "$iface" down 2>/dev/null || true
        sudo -n ip link set "$iface" type can bitrate 1000000 2>/dev/null || true
        sudo -n ip link set "$iface" up 2>/dev/null || true
        CAN_UP=$((CAN_UP + 1))
        ok "$iface up"
    fi
done

if [ "$CAN_UP" -ge 4 ]; then
    ok "$CAN_UP CAN interfaces up (dual arm master+slave)"
elif [ "$CAN_UP" -ge 2 ]; then
    warn "only $CAN_UP CAN interfaces (need 4 for dual arm master+slave)"
else
    warn "only $CAN_UP CAN interfaces"
fi

# ── 4. 依赖检查 ──
echo ""
echo "--- Step 4: dependency check ---"

# venv
if [ -f "$KAI0_DIR/.venv/bin/python" ]; then
    ok "venv: $($KAI0_DIR/.venv/bin/python --version 2>&1)"
else
    fail "venv not found: $KAI0_DIR/.venv/"
fi

# ROS2 workspace
if [ -f "$ROS2_WS/install/setup.bash" ]; then
    ok "ROS2 workspace: $ROS2_WS"
else
    fail "ROS2 workspace not built: run 'cd $ROS2_WS && colcon build'"
fi

# Calibration
CALIB_FILE="$PROJECT_ROOT/config/calibration.yml"
if [ -f "$CALIB_FILE" ]; then
    ok "calibration: $CALIB_FILE"
else
    warn "calibration not found: $CALIB_FILE (FK visualization will be disabled)"
fi

# serve_policy (websocket mode)
if [ "$MODE" = "websocket" ] || [ "$MODE" = "both" ]; then
    if ss -tlnp 2>/dev/null | grep -q ":8000 "; then
        ok "serve_policy running on :8000"
    else
        fail "serve_policy not running on :8000. Start it first:
    cd $KAI0_DIR && CUDA_VISIBLE_DEVICES=1 JAX_COMPILATION_CACHE_DIR=/tmp/xla_cache \\
      .venv/bin/python scripts/serve_policy.py --port 8000 policy:checkpoint \\
      --policy.config=pi05_flatten_fold_normal --policy.dir=checkpoints/Task_A/mixed_1"
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

# Add venv bin to PATH (for rerun CLI)
export PATH="$KAI0_DIR/.venv/bin:$PATH"
export JAX_COMPILATION_CACHE_DIR=/tmp/xla_cache
mkdir -p /tmp/xla_cache

# GPU allocation: pick GPU with fewest active compute processes.
# Free memory alone is misleading because other processes' cuBLAS contexts
# fragment GPU memory and can cause OOM even when nvidia-smi shows free space.
GPU_ID=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
    | sort | uniq -c | sort -n > /tmp/gpu_proc_count.txt || true)
GPU_ID=$(
    python3 -c "
import subprocess
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
# Sort by: fewest processes first, then most free memory
best = sorted(gpus.values(), key=lambda g: (g[2], -g[1]))[0]
print(best[0])
" 2>/dev/null
)
GPU_ID=${GPU_ID:-0}
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null || echo "?")
info "using GPU $GPU_ID (free: ${FREE_MB}MB)"

# Unset proxy vars that can interfere with JAX/gRPC
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true
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
info "starting ros2 launch..."
echo "  Ctrl+C to stop all nodes"
echo ""

exec ros2 launch piper autonomy_launch.py \
    mode:="$MODE" \
    enable_rerun:="$ENABLE_RERUN" \
    fg_enable:=false \
    bg_enable:=false \
    execute_mode:="$EXECUTE_MODE" \
    gpu_id:="$GPU_ID" \
    "${EXTRA_ARGS[@]}"
