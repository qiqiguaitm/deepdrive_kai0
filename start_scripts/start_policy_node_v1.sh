#!/bin/bash
###############################################################################
# 启动 policy_inference_node, 走 WebSocket client 模式连 V1 Triton serve (:8002)
#
# 前提: start_serve_v1.sh 已在另一终端启动 (:8002 has /healthz OK)
#
# 与 start_policy_node.sh 区别:
#   - 不本地 load JAX (mode=websocket)
#   - 连 :8002 (V1 Triton, P50~32 ms) 而非 :8000 (JAX, P50~196 ms)
#
# Usage:
#   ./scripts/start_policy_node_v1.sh                        # 默认 :8002
#   ./scripts/start_policy_node_v1.sh --port 8003 --host 10.0.0.5
###############################################################################
set -eo pipefail

EXECUTION_MODE="joint"
ENABLE_DEPTH_INPUT="false"
ENABLE_EE_POSE_INPUT="false"
HOST="localhost"
PORT="8002"
ENABLE_LATENCY_PROFILE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execution-mode=*)     EXECUTION_MODE="${1#*=}"; shift ;;
    --execution-mode)       EXECUTION_MODE="$2"; shift 2 ;;
    --enable-depth-input)   ENABLE_DEPTH_INPUT="true"; shift ;;
    --enable-ee-pose-input) ENABLE_EE_POSE_INPUT="true"; shift ;;
    --host)                 HOST="$2"; shift 2 ;;
    --port)                 PORT="$2"; shift 2 ;;
    --profile-latency)      ENABLE_LATENCY_PROFILE="true"; shift ;;
    -h|--help)
      grep '^#' "$0" | head -16
      exit 0
      ;;
    *) shift ;;
  esac
done

if [[ "$EXECUTION_MODE" != "joint" && "$EXECUTION_MODE" != "ee_pose" ]]; then
  echo "[FAIL] --execution-mode must be 'joint' or 'ee_pose', got '$EXECUTION_MODE'" >&2
  exit 1
fi

eval "$(conda shell.bash hook 2>/dev/null)"; conda deactivate 2>/dev/null || true
source /opt/ros/jazzy/setup.bash
source /data1/tim/workspace/deepdive_kai0/ros2_ws/install/setup.bash

VENV=/data1/tim/workspace/deepdive_kai0/kai0/.venv/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$(find $VENV/nvidia -name 'lib' -type d 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH="${VENV}:/data1/tim/workspace/deepdive_kai0/kai0/src:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
unset http_proxy https_proxy

# Opt-in client-side 11-segment latency profile (B1)
[ "$ENABLE_LATENCY_PROFILE" = "true" ] && export KAI0_LATENCY_PROFILE=1

echo "=== Launching policy_inference_node (websocket → ${HOST}:${PORT}) ==="
echo "    execution_mode:  ${EXECUTION_MODE}"
echo "    latency profile: ${ENABLE_LATENCY_PROFILE}"
echo ""

# checkpoint_dir 仍传入但不用 (websocket 模式下 node 不 load 模型, 仍要参数 declared);
# norm_stats 由 V1 serve 端处理; node 端 RTC normalize 仍可用同一 norm_stats.
exec ros2 run piper policy_inference_node.py --ros-args \
  -p mode:=websocket \
  -p host:=${HOST} \
  -p port:=${PORT} \
  -p checkpoint_dir:=/data1/DATA_IMP/checkpoints/task_a_mix_b6000_p1200_mixed_1_step49999 \
  -p config_name:=pi05_flatten_fold_a_new_pure_1200 \
  -p gpu_id:=0 \
  -p img_front_topic:=/camera_f/color/image_raw \
  -p img_left_topic:=/camera_l/color/image_raw \
  -p img_right_topic:=/camera_r/color/image_raw \
  -p execution_mode:=${EXECUTION_MODE} \
  -p enable_depth_input:=${ENABLE_DEPTH_INPUT} \
  -p enable_ee_pose_input:=${ENABLE_EE_POSE_INPUT}
