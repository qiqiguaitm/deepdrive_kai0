#!/bin/bash
###############################################################################
# Interactive keyboard control for policy execution.
#
# Usage: run in a separate terminal while the inference stack is running.
#   ./scripts/toggle_execute.sh
#
# Keys:
#   Enter/Space  → toggle execute/observe
#   q/Esc        → switch to observe
#   Ctrl+C       → exit
###############################################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROS2_WS="$PROJECT_ROOT/ros2_ws"

eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null; conda deactivate 2>/dev/null || true
source /opt/ros/jazzy/setup.bash
[ -f "$ROS2_WS/install/setup.bash" ] && source "$ROS2_WS/install/setup.bash"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

state="OBSERVE"
echo -e "${CYAN}Policy Execute Control${NC}"
echo "  [Enter/Space] toggle execute"
echo "  [q/Esc]       → observe"
echo "  [Ctrl+C]      exit"
echo ""
echo -e "State: ${YELLOW}$state${NC}"

publish() {
    local data=$1
    ros2 topic pub --once /policy/execute std_msgs/msg/Bool "{data: $data}" >/dev/null 2>&1
}

toggle() {
    if [ "$state" = "OBSERVE" ]; then
        state="EXECUTE"
        publish true
        echo -e "State: ${GREEN}$state${NC}"
    else
        state="OBSERVE"
        publish false
        echo -e "State: ${YELLOW}$state${NC}"
    fi
}

observe() {
    if [ "$state" != "OBSERVE" ]; then
        state="OBSERVE"
        publish false
        echo -e "State: ${YELLOW}$state${NC}"
    fi
}

# Raw mode keyboard read
stty_orig=$(stty -g)
trap 'stty "$stty_orig"; publish false; echo; echo "exited"; exit 0' INT TERM EXIT
stty -echo -icanon min 1 time 0

while true; do
    ch=$(dd bs=1 count=1 2>/dev/null)
    case "$ch" in
        $'\n'|$'\r'|' ')
            toggle
            ;;
        q)
            observe
            ;;
        $'\x1b')
            # Might be Esc alone or an escape sequence; read more with tiny timeout
            stty min 0 time 1
            extra=$(dd bs=1 count=2 2>/dev/null || true)
            stty min 1 time 0
            if [ -z "$extra" ]; then
                observe
            fi
            ;;
    esac
done
