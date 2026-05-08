#!/bin/bash
###############################################################################
# kai0 Replay 命令行测试 (P1.5)
#
# 数据集回放: 把指定 episode 的 action[T,14] 喂给 stream_buffer, publish_timer
# 在 30 Hz 下 pop → /master/joint_left/right, 让机械臂照训练数据走一遍.
#
# 前提:
#   1. ./start_autonomy.sh 已起来 (会写 /tmp/kai0_deployment_mode=autonomy)
#   2. 主臂没运行 (data_collect stop 过, /master/joint_* 无第三方 publisher)
#   3. 机械臂当前姿态与 episode 第 0 帧 action 任意维度 Δ ≤ 5°
#
# 用法:
#   ./start_replay_test.sh <task>/<subset>/<date>/<episode_id> [rate]
#
# 示例:
#   ./start_replay_test.sh Task_A/base/2026-04-28/42
#   ./start_replay_test.sh Task_A/base/2026-04-28/42 0.8       # 慢速 0.8x
#
# 安全栏 (硬性, 不可关):
#   S1 deployment marker = autonomy           → 否则 409 拒绝
#   S2 /master/joint_left 无第三方 publisher  → 否则 409 拒绝
#   S3 parquet 存在 + action shape=[T,14]     → 否则 422
#   S4 起点姿态对齐 ≤ 5°                       → 否则拒绝, 列 per-joint Δ
#   S5 rate ∈ [0.5, 1.5]                       → 自动 clamp
#   S6 单步 jump ≤ 0.5 rad (复用)             → 触发即 flush + abort
#
# 中途停: Ctrl+C 即可 (本脚本会发 replay_mode=inference + execution=false)
###############################################################################
set -eo pipefail

# ── Source ROS2 env if not already sourced ──
# start_autonomy.sh does this internally then exec's, but a fresh terminal
# invoking this script directly may not have ros2 on PATH. Idempotent.
if ! command -v ros2 >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null; conda deactivate 2>/dev/null || true
    source /opt/ros/jazzy/setup.bash 2>/dev/null || true
fi
# Workspace overlay (piper package) — needed only for piper_msgs etc., not core
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -f "$PROJECT_ROOT/ros2_ws/install/setup.bash" ]]; then
    source "$PROJECT_ROOT/ros2_ws/install/setup.bash" 2>/dev/null || true
fi
command -v ros2 >/dev/null 2>&1 || { echo "[FAIL] ros2 still not on PATH after sourcing /opt/ros/jazzy/setup.bash"; exit 5; }

# Auto-detect target node — same logic as backend's detect_replay_node():
# prefer slim /replay (start_replay_stack.sh), fall back to /policy_inference (autonomy).
_detect_node() {
    local nodes
    nodes=$(ros2 node list 2>/dev/null)
    echo "$nodes" | grep -qx '/replay' && { echo "/replay"; return; }
    echo "$nodes" | grep -qx '/policy_inference' && { echo "/policy_inference"; return; }
    echo ""
}
NODE="$(_detect_node)"
if [[ -z "$NODE" ]]; then
    echo "[FAIL] no replay-capable node alive. Start one of:"
    echo "       ./start_scripts/start_replay_stack.sh    (slim, recommended)"
    echo "       ./start_scripts/start_autonomy.sh        (full)"
    exit 6
fi
echo "[OK] target node = $NODE"

EP_ARG="${1:-}"
RATE="${2:-1.0}"

if [[ -z "$EP_ARG" ]]; then
    cat <<EOF
Usage: $0 <task>/<subset>/<date>/<episode_id> [rate]

ep arg 格式: Task_A/base/2026-04-28/42
  → 解析为 \$KAI0_DATA_ROOT/Task_A/base/2026-04-28/data/chunk-000/episode_000042.parquet
EOF
    exit 1
fi

DATA_ROOT="${KAI0_DATA_ROOT:-/data1/DATA_IMP/KAI0}"

# ── Resolve parquet path ──
# Splits the dotted ep_arg into 4 parts (task/subset/date/ep_id) and zero-pads ep_id to 6 digits
ABS_PARQUET="$(python3 -c "
import sys; from pathlib import Path
parts = sys.argv[1].rstrip('/').split('/')
if len(parts) != 4:
    print(f'ERR: ep arg needs 4 parts, got {len(parts)}: {parts}', file=sys.stderr); sys.exit(1)
task, subset, date, ep_id = parts
try: ep_int = int(ep_id)
except ValueError: print(f'ERR: ep_id {ep_id!r} not int', file=sys.stderr); sys.exit(1)
p = Path(sys.argv[2]) / task / subset / date / 'data' / 'chunk-000' / f'episode_{ep_int:06d}.parquet'
print(p.resolve())" "$EP_ARG" "$DATA_ROOT")"
[[ -f "$ABS_PARQUET" ]] || { echo "[FAIL] parquet not found: $ABS_PARQUET"; exit 1; }
echo "[OK] parquet: $ABS_PARQUET"

# ── S1: deployment marker ──
# Accept both 'autonomy' (start_autonomy.sh) and 'replay' (start_replay_stack.sh).
MARKER_VAL="$(cat /tmp/kai0_deployment_mode 2>/dev/null || echo MISSING)"
if [[ "$MARKER_VAL" != "autonomy" && "$MARKER_VAL" != "replay" ]]; then
    echo "[FAIL] /tmp/kai0_deployment_mode=$MARKER_VAL, replay 需 'autonomy' 或 'replay'"
    echo "       先 \`./start_data_collect.sh stop\` (若在跑), 再 \`./start_autonomy.sh\` 或 \`./start_replay_stack.sh\`"
    exit 2
fi
echo "[OK] deployment marker = $MARKER_VAL"

# ── S2: publisher conflict ──
# `ros2 topic info -v` doesn't have `Publishers:`/`Subscriptions:` section headers
# in jazzy — each endpoint block has its own `Endpoint type: PUBLISHER|SUBSCRIPTION`
# line. Use that to distinguish (matches backend's parser fix).
SELF_NAME="${NODE#/}"
OTHERS="$(ros2 topic info /master/joint_left -v 2>/dev/null \
          | awk '
              BEGIN { in_block=0; is_pub=0; node="" }
              /^Node name:/ { if (node!="" && is_pub==1) print node; node=$NF; is_pub=0 }
              /Endpoint type: PUBLISHER/ { is_pub=1 }
              END { if (node!="" && is_pub==1) print node }
            ' \
          | grep -vx "$SELF_NAME" || true)"
if [[ -n "$OTHERS" ]]; then
    echo "[FAIL] /master/joint_left 有其它 publisher (除自身 $SELF_NAME):"
    echo "$OTHERS"
    exit 3
fi
echo "[OK] publisher conflict check passed (self=$SELF_NAME)"

# ── S5: rate clamp (info only; node also clamps) ──
CLAMPED_RATE=$(python3 -c "import sys; print(max(0.5, min(1.5, float(sys.argv[1]))))" "$RATE")
if [[ "$CLAMPED_RATE" != "$RATE" ]]; then
    echo "[WARN] rate $RATE clamped to $CLAMPED_RATE"
fi

# ── Set replay params (path triggers parquet load + shape check S3) ──
echo ""
echo "--- Setting replay params ---"
ros2 param set "$NODE" replay_episode_path "$ABS_PARQUET"
ros2 param set "$NODE" replay_rate "$CLAMPED_RATE"
ros2 param set "$NODE" replay_loop false

# ── Trigger replay mode (this runs S4 pre-flight inside node) ──
echo ""
echo "--- Switching to replay mode (pre-flight: pose alignment) ---"
# CAUTION: `ros2 param set` returns exit 0 even when the node rejects the param
# via SetParametersResult.successful=False. Have to grep stdout for the failure
# string. Otherwise rejection is silently ignored and CLI continues to send
# /policy/execute=true while node stays in 'inference' mode → policy autonomy
# accidentally runs instead of replay.
SWITCH_OUT="$(ros2 param set "$NODE" replay_mode replay 2>&1)"
echo "$SWITCH_OUT"
if echo "$SWITCH_OUT" | grep -qiE 'Setting parameter failed|^Failed'; then
    echo ""
    echo "[FAIL] replay mode 切换失败. 看上面 reason."
    echo "       (起点未对齐时, 把臂手动移动到第 0 帧 action 附近 ≤5°)"
    exit 4
fi
# Defensive double-check: read back replay_mode, must be 'replay'
sleep 0.3
ACTUAL_MODE="$(ros2 param get "$NODE" replay_mode 2>&1 | tail -1 | awk '{print $NF}')"
if [[ "$ACTUAL_MODE" != "replay" ]]; then
    echo "[FAIL] replay_mode readback = $ACTUAL_MODE (expect 'replay'). pre-flight 实际未通过."
    exit 4
fi
echo "[OK] replay_mode = $ACTUAL_MODE (verified)"

# ── Confirm before sending execute=true ──
echo ""
read -p "确认真发到机械臂? [y/N] " yn
if [[ "$yn" != "y" && "$yn" != "Y" ]]; then
    echo "[CANCEL] 切回 inference mode"
    ros2 param set "$NODE" replay_mode inference || true
    exit 0
fi

# ── Cleanup trap: Ctrl+C / exit → abort cleanly ──
cleanup() {
    echo ""
    echo "--- cleanup: aborting replay ---"
    ros2 topic pub --once /policy/execute std_msgs/Bool '{data: false}' 2>/dev/null || true
    ros2 param set "$NODE" replay_mode inference 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# ── Send execute ──
echo ""
echo "--- execute=true, monitoring /replay_progress (Ctrl+C 停) ---"
ros2 topic pub --once /policy/execute std_msgs/Bool '{data: true}' >/dev/null

# ── Monitor progress (Float32MultiArray: [idx, total, done]) ──
# Plain echo: each msg shows 3 numbers (data: [idx, total, done_flag]).
# Watch for 'done_flag = 1.0' line, then Ctrl+C.
ros2 topic echo /replay_progress std_msgs/Float32MultiArray
