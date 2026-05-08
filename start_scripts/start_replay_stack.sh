#!/bin/bash
###############################################################################
# kai0 Replay-only stack — slim alternative to start_autonomy.sh.
#
# Brings up just enough to drive arms from recorded LeRobot v2.1 episodes:
#   - 2× arm_reader_node mode=1 (left + right slave) — drives arms via CAN
#   - 1× replay_node — accepts replay params + publishes /master/joint_*
#
# Skipped vs start_autonomy.sh:
#   ✗ multi_camera (3× RealSense)
#   ✗ depth processing
#   ✗ rerun visualization
#   ✗ policy_inference_node (no JAX, no ckpt load — saves ~30s startup + 5GB RAM)
#
# Usage:
#   ./start_replay_stack.sh                    # foreground
#   nohup ./start_replay_stack.sh > /tmp/replay.log 2>&1 &  # background
#   ./start_replay_stack.sh --no-rerun         # legacy alias (rerun was never enabled here)
#
# After it's up:
#   - Backend (./web/data_manager) auto-detects /replay node and uses it for
#     /api/replay/* endpoints (in addition to /policy_inference if both alive).
#   - CLI: ./start_scripts/start_replay_test.sh <task>/<subset>/<date>/<ep_id>
#
# Marker:
#   Writes /tmp/kai0_deployment_mode = "replay". The backend's preflight gate
#   accepts both 'autonomy' and 'replay' as valid.
###############################################################################
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROS2_WS="$PROJECT_ROOT/ros2_ws"
ACTIVATE_CAN="$PROJECT_ROOT/can_tools/setup_can.sh"
KAI0_DEPLOYMENT_MARKER=/tmp/kai0_deployment_mode

# `stop` action: kill the replay stack + clear marker (only if we own it).
# Mirrors start_data_collect.sh — autonomy's marker is left alone if it's running.
if [[ "${1:-}" == "stop" ]]; then
    pkill -9 -f replay_launch || true
    pkill -9 -f replay_node || true
    pkill -9 -f arm_reader_node || true
    if [[ -f "$KAI0_DEPLOYMENT_MARKER" && "$(cat "$KAI0_DEPLOYMENT_MARKER" 2>/dev/null)" == "replay" ]]; then
        rm -f "$KAI0_DEPLOYMENT_MARKER"
    fi
    echo "[OK] replay stack stopped"
    exit 0
fi

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
info() { echo -e "${CYAN}[INFO]${NC} $1"; }

echo "============================================================"
echo " kai0 Replay-Only Stack (slim, no autonomy/cameras/JAX)"
echo "============================================================"

# ── 1. Cleanup leftovers ──
echo ""
echo "--- Step 1: cleanup leftover ROS processes ---"
KILL_PATTERNS="arm_reader_node|arm_teleop_node|policy_inference_node|replay_node|autonomy_launch|teleop_launch|replay_launch"
PIDS=$(ps aux | grep -E "$KILL_PATTERNS" | grep -v grep | grep -v $$ | awk '{print $2}' || true)
if [ -n "$PIDS" ]; then
    COUNT=$(echo "$PIDS" | wc -w)
    info "killing $COUNT leftover process(es)..."
    echo "$PIDS" | xargs kill -9 2>/dev/null || true
    sleep 2
    ok "cleaned up $COUNT processes"
else
    ok "no leftovers"
fi

# ── 2. ROS2 daemon refresh ──
eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null; conda deactivate 2>/dev/null || true
source /opt/ros/jazzy/setup.bash
ros2 daemon stop 2>/dev/null || true
ros2 daemon start 2>/dev/null || true

# ── 3. CAN ──
echo ""
echo "--- Step 2: CAN activation ---"
if [[ -x "$ACTIVATE_CAN" ]]; then
    bash "$ACTIVATE_CAN" --quick 2>&1 | tail -3 || warn "activate_can returned non-zero"
else
    warn "$ACTIVATE_CAN not found / not executable — skipping. arms may fail to come up."
fi

# ── 4. Build check ──
echo ""
echo "--- Step 3: build check ---"
INSTALL_MARKER="$ROS2_WS/install/piper/.colcon_install_layout"
NEEDS_BUILD=false
if [ ! -f "$INSTALL_MARKER" ]; then
    NEEDS_BUILD=true
else
    NEWEST_SRC=$(find "$ROS2_WS/src/piper" -name '*.py' -newer "$INSTALL_MARKER" 2>/dev/null | head -1)
    [ -n "$NEWEST_SRC" ] && NEEDS_BUILD=true
fi
if [ "$NEEDS_BUILD" = true ]; then
    info "source changed, rebuilding piper..."
    (cd "$ROS2_WS" && source /opt/ros/jazzy/setup.bash && colcon build --packages-select piper 2>&1 | tail -3)
fi
ok "build ready"

# ── 5. Workspace overlay ──
source "$ROS2_WS/install/setup.bash"

# ── 6. Deployment marker (P1: replay safety gate) ──
# Lets `replay_mode=replay` pass the marker check (accepts 'autonomy' or 'replay').
echo replay > /tmp/kai0_deployment_mode
ok "deployment marker = replay"

# ── 7. Launch ──
echo ""
echo "--- Step 4: ros2 launch piper replay_launch.py ---"
echo "  Ctrl+C to stop"
echo ""
exec ros2 launch piper replay_launch.py "${@}"
