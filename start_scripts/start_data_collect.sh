#!/bin/bash
###############################################################################
# kai0 数据采集一键启动
#
# 组合: USB 相机 reset + CAN 激活 + 主从遥操 + 3 相机 + Web 数据管理后端/前端
# 底层复用 web/data_manager/run.sh 的进程管理 (setsid/pidfile/start/stop/status/logs)
#
# 用法:
#   ./scripts/start_data_collect.sh               # 启动全部
#   ./scripts/start_data_collect.sh stop           # 停止全部
#   ./scripts/start_data_collect.sh restart        # 重启
#   ./scripts/start_data_collect.sh status         # 查看各服务状态
#   ./scripts/start_data_collect.sh logs [svc]     # 追踪日志 (arms|cameras|backend|frontend)
#
# 环境变量 (传递给 run.sh):
#   SKIP_ARMS=1        跳过机械臂
#   SKIP_CAMERAS=1     跳过相机
#   SKIP_DEPS=1        跳过后端 pip 依赖同步
#   KAI0_DATA_ROOT=... 采集落盘根目录 (默认 /data1/DATA_IMP/KAI0)
###############################################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROS2_WS="$PROJECT_ROOT/ros2_ws"
RUN_SH="$PROJECT_ROOT/web/data_manager/run.sh"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
info() { echo -e "${CYAN}[INFO]${NC} $1"; }

ACTION="${1:-start}"

# ── Pre-flight: only on start/restart ──
if [[ "$ACTION" == "start" || "$ACTION" == "restart" ]]; then
    echo "============================================================"
    echo " kai0 Data Collection"
    echo " $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo ""

    # 1. ROS2 daemon restart (clean DDS state)
    eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null; conda deactivate 2>/dev/null || true
    source /opt/ros/jazzy/setup.bash 2>/dev/null || true
    ros2 daemon stop  2>/dev/null || true
    ros2 daemon start 2>/dev/null || true

    # 2. USB camera reset (passwordless via /etc/sudoers.d/kai0-autonomy)
    echo "--- USB camera reset ---"
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

    # 3. Rebuild ros2_ws if source changed
    echo ""
    echo "--- Build check ---"
    INSTALL_MARKER="$ROS2_WS/install/piper/.colcon_install_layout"
    NEEDS_BUILD=false
    if [ ! -f "$INSTALL_MARKER" ]; then
        NEEDS_BUILD=true
    else
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
    echo ""
fi

# ── Delegate to run.sh ──
# SETUP_CAN=1: let run.sh activate CAN (start_teleop.sh handles it too,
#              but run.sh's activate_can path is the explicit toggle)
export SETUP_CAN=1
# Data collection needs 30 fps to match training data (launch_3cam.py defaults
# to 15 fps to ease USB bandwidth; override here for full-rate recording).
export CAM_FPS=30

exec bash "$RUN_SH" "$ACTION" "${@:2}"
