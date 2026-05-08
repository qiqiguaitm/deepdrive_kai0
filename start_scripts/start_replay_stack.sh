#!/bin/bash
###############################################################################
# DEPRECATED — kai0 replay 现已由 start_autonomy.sh 统一管理.
#
# 等价新命令:
#   ./start_autonomy.sh --replay          # 真机回放 (== 旧 start_replay_stack.sh)
#   ./start_autonomy.sh --replay --sim    # 仿真回放 (新增, 不驱动真机)
#
# 本脚本保留为兼容 shim, 转发到 start_autonomy.sh --replay (含 stop 子命令)。
###############################################################################
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ "${1:-}" == "stop" ]]; then
    pkill -9 -f replay_launch || true
    pkill -9 -f replay_node || true
    pkill -9 -f arm_reader_node || true
    if [[ -f /tmp/kai0_deployment_mode && "$(cat /tmp/kai0_deployment_mode 2>/dev/null)" == "replay" ]]; then
        rm -f /tmp/kai0_deployment_mode
    fi
    echo "[OK] replay stack stopped"
    exit 0
fi

echo "[DEPRECATED] start_replay_stack.sh → 转发至 start_autonomy.sh --replay" >&2
exec "$SCRIPT_DIR/start_autonomy.sh" --replay "$@"
