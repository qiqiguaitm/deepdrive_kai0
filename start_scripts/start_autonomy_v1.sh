#!/bin/bash
###############################################################################
# 一键启动 V1 Triton 推理真机 autonomy stack (两个后台进程, 概念三个角色):
#
#   后台进程 1: V1 serve_policy_v1.py (:8002, Phase 2 sentencepiece state encoding)
#   后台进程 2: autonomy_launch.py (mode=websocket → :8002)
#               = cameras + arms (slaves) + policy_inference_node (WS client mode)
#
# 顺序: 启 V1 serve → 等 /healthz=OK → 启 autonomy (--mode=websocket)
# 退出: Ctrl-C 同时清理两个进程
#
# 日志:
#   /tmp/v1_serve.log         — V1 serve stdout/stderr
#   /tmp/v1_autonomy.log      — autonomy_launch stdout/stderr (含 policy_inference)
#   /tmp/kai0_latency_<pid>.csv  — B1 client-side 11 段 profile (若 --profile-latency)
#
# Usage:
#   ./scripts/start_autonomy_v1.sh                       # 默认 (无 execute, 无 rerun)
#   ./scripts/start_autonomy_v1.sh --execute             # 真机执行 (危险, 真驱动 Piper)
#   ./scripts/start_autonomy_v1.sh --rerun               # 开 Rerun UI
#   ./scripts/start_autonomy_v1.sh --no-profile          # 关 B1 client latency profile
#   ./scripts/start_autonomy_v1.sh --port 8003           # 自定义 V1 serve 端口
###############################################################################

set -eo pipefail

EXECUTE_FLAG=""
RERUN_FLAG="--no-rerun"
ENABLE_PROFILE=true
PORT=8002
EXTRA_AUTONOMY=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)       EXECUTE_FLAG="--execute"; shift ;;
    --rerun)         RERUN_FLAG=""; shift ;;
    --no-rerun)      RERUN_FLAG="--no-rerun"; shift ;;
    --no-profile)    ENABLE_PROFILE=false; shift ;;
    --port)          PORT="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | head -22
      exit 0
      ;;
    *) EXTRA_AUTONOMY+=("$1"); shift ;;
  esac
done

REPO=/data1/tim/workspace/deepdive_kai0
SERVE_LOG=/tmp/v1_serve.log
AUTO_LOG=/tmp/v1_autonomy.log
PIDS=()

cleanup() {
  echo ""
  echo "=== Cleanup: stopping V1 stack ==="
  # 先杀 autonomy (含 policy_inference + cameras + arms; 它们要 release CAN/USB)
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "  SIGINT → PID $pid"
      kill -INT "$pid" 2>/dev/null || true
    fi
  done
  sleep 3
  # 强杀残余
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "  SIGKILL → PID $pid (still alive)"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
  # autonomy 可能 spawn 子进程 (cameras, arm_reader, policy_inference), 也清理
  pkill -INT -f "policy_inference_node|realsense2_camera_node|arm_reader_node|multi_camera_node|rerun_viz_node" 2>/dev/null || true
  sleep 1
  pkill -KILL -f "policy_inference_node|realsense2_camera_node|arm_reader_node|multi_camera_node|rerun_viz_node" 2>/dev/null || true
  echo "  Done. Logs preserved in /tmp/v1_*.log + /tmp/kai0_latency_*.csv"
}
trap cleanup INT TERM EXIT

# ── Step 1: V1 serve ──────────────────────────────────────────────────
echo ""
echo "[1/2] === Launching V1 Triton serve on :${PORT} (log: $SERVE_LOG) ==="
nohup "$REPO/start_scripts/start_serve_v1.sh" --port "$PORT" > "$SERVE_LOG" 2>&1 &
PID_SERVE=$!
PIDS+=($PID_SERVE)
echo "[1/2] PID=$PID_SERVE, 等待 build + CUDA Graph capture + warmup (~30s) ..."

# 轮询 /healthz 直到响应 OK (最多 90s)
SERVE_READY=false
for i in $(seq 1 90); do
  if curl -s --max-time 1 "http://localhost:${PORT}/healthz" 2>/dev/null | grep -q "OK"; then
    echo "[1/2] ✓ V1 serve healthy after ${i}s"
    SERVE_READY=true
    break
  fi
  if ! kill -0 $PID_SERVE 2>/dev/null; then
    echo ""
    echo "[1/2] ✗ V1 serve process 死掉了 — last 20 lines of log:"
    tail -20 "$SERVE_LOG" 2>&1
    exit 1
  fi
  sleep 1
done
if [ "$SERVE_READY" = "false" ]; then
  echo "[1/2] ✗ V1 serve 90s 内未 ready, 见 $SERVE_LOG"
  exit 1
fi

# ── Step 2: autonomy (websocket mode 接 :PORT) ──────────────────────────
echo ""
echo "[2/2] === Launching autonomy stack (mode=websocket → :${PORT}) (log: $AUTO_LOG) ==="
echo "[2/2] 包含: cameras + arms (slaves) + policy_inference_node (WS client)"

# B1 client-side latency profile env var (autonomy 内的 policy_inference_node 继承)
PROFILE_ENV=""
[ "$ENABLE_PROFILE" = "true" ] && PROFILE_ENV="KAI0_LATENCY_PROFILE=1"

# start_autonomy.sh --ws-port 把 preflight check + autonomy_launch port 一起切到 V1 (:8002)
env $PROFILE_ENV nohup "$REPO/start_scripts/start_autonomy.sh" \
    --mode websocket \
    --ws-port "$PORT" \
    --execution-mode joint \
    $EXECUTE_FLAG \
    $RERUN_FLAG \
    "${EXTRA_AUTONOMY[@]}" \
    > "$AUTO_LOG" 2>&1 &
PID_AUTO=$!
PIDS+=($PID_AUTO)
echo "[2/2] PID=$PID_AUTO, 等待节点初始化 (~10s) ..."
sleep 10
if ! kill -0 $PID_AUTO 2>/dev/null; then
  echo "[2/2] ✗ autonomy_launch 死掉 — last 20 lines:"
  tail -20 "$AUTO_LOG" 2>&1
  exit 1
fi
echo "[2/2] ✓ autonomy_launch started"

# ── 阻塞 / 监控 ─────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  V1 stack RUNNING."
echo "  Logs:"
echo "    serve    : tail -f $SERVE_LOG"
echo "    autonomy : tail -f $AUTO_LOG"
if [ "$ENABLE_PROFILE" = "true" ]; then
  echo "  B1 latency CSV (出现需等 policy_inference_node 进推理循环):"
  echo "    ls -la /tmp/kai0_latency_*.csv | tail -1"
fi
echo ""
echo "  Execute mode: ${EXECUTE_FLAG:-OBSERVE (无执行)}"
echo "  Rerun:        ${RERUN_FLAG:-on}"
echo ""
echo "  Press Ctrl-C 停止全部."
echo "============================================================"

# 持续监控
while true; do
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo ""
      echo "[FAIL] PID $pid 已死, 触发清理"
      exit 1
    fi
  done
  sleep 5
done
