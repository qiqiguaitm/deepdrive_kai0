#!/bin/bash
# 一键启动主从遥操 (Master-Slave Teleoperation)
#
# 用法:
#   bash scripts/start_teleop.sh
#
# 流程:
#   1. 激活 CAN 接口并重命名为符号名 (需要 sudo)
#   2. source ROS2 (jazzy) + 工作空间
#   3. 启动 master-slave launch (4 臂)
#
# 停止: Ctrl+C

set -eo pipefail
# 必须加 `|| true`: 无匹配时 pkill 返回 1, 配合 set -e 会让脚本在第一个
# echo 之前就静默退出, 表现为 run.sh 报 "arms failed to start" + 空日志.
pkill -f ros2 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ROS2_WS="$PROJECT_ROOT/ros2_ws"
CAN_ACTIVATE="$PROJECT_ROOT/can_tools/activate_can.sh"
ROS_DISTRO="jazzy"

echo "============================================"
echo "  kai0 主从遥操启动"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""

# ── 1. 激活 CAN ──────────────────────────────────────────────────────────────
echo "[1/3] 激活 CAN 接口..."
# 检查是否已经有 4 个正确符号名接口 UP
EXPECTED_IFACES="can_left_mas can_left_slave can_right_mas can_right_slave"
ALL_UP=true
for iface in $EXPECTED_IFACES; do
    if ! ip link show "$iface" 2>/dev/null | grep -q ",UP"; then
        ALL_UP=false
        break
    fi
done

if $ALL_UP; then
    echo "  CAN 接口已就绪, 跳过激活"
else
    # 需要 root 权限激活 CAN
    if [[ $(id -u) -eq 0 ]]; then
        bash "$CAN_ACTIVATE"
    else
        echo "  需要 root 权限激活 CAN 接口"
        sudo bash "$CAN_ACTIVATE"
    fi
fi
echo ""

# ── 2. Source ROS2 ───────────────────────────────────────────────────────────
echo "[2/3] Source ROS2 环境..."

# 退出 conda 环境, ROS2 Jazzy 需要系统 Python 3.12
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "  退出 conda 环境 ($CONDA_DEFAULT_ENV)..."
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda deactivate 2>/dev/null || true
fi

# ROS2 全局安装
if [[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
    echo "  OK: /opt/ros/${ROS_DISTRO} (python: $(python3 --version 2>&1))"
else
    echo "  [FAIL] /opt/ros/${ROS_DISTRO}/setup.bash 不存在"
    echo "  请先安装 ROS2 ${ROS_DISTRO}: bash install.sh"
    exit 1
fi

# ROS2 工作空间
if [[ -f "$ROS2_WS/install/setup.bash" ]]; then
    source "$ROS2_WS/install/setup.bash"
    echo "  OK: $ROS2_WS"
else
    echo "  [FAIL] $ROS2_WS/install/setup.bash 不存在"
    echo "  请先编译: cd $ROS2_WS && colcon build"
    exit 1
fi
echo ""

# ── 3. 启动 Master-Slave ────────────────────────────────────────────────────
echo "[3/3] 启动 Master-Slave 遥操..."
echo "  master: can_left_mas (左), can_right_mas (右) — 拖拽示教"
echo "  slave:  can_left_slave (左), can_right_slave (右) — 跟随执行"
echo ""
echo "  Ctrl+C 停止"
echo "============================================"
echo ""

ros2 launch piper teleop_launch.py
