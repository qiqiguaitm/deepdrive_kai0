#!/bin/bash
# 启动 policy_inference_node (统一入口)
#
# 合并自: test_policy_both_mode.sh + test_policy_ros2_mode.sh
#
# Usage:
#   ./scripts/start_policy_node.sh             # 默认 mode=both
#   ./scripts/start_policy_node.sh --mode ros2
#   ./scripts/start_policy_node.sh --mode both
#   ./scripts/start_policy_node.sh --mode websocket
set -e

MODE="both"
for arg in "$@"; do
  case "$arg" in
    --mode=*) MODE="${arg#*=}" ;;
    --mode)   shift_next=1 ;;
    *)
      if [ "$shift_next" = "1" ]; then
        MODE="$arg"
        shift_next=0
      fi
      ;;
  esac
done

eval "$(conda shell.bash hook 2>/dev/null)"; conda deactivate 2>/dev/null || true
source /opt/ros/jazzy/setup.bash
source /data1/tim/workspace/deepdive_kai0/ros2_ws/install/setup.bash

VENV=/data1/tim/workspace/deepdive_kai0/kai0/.venv/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$(find $VENV/nvidia -name 'lib' -type d 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH="${VENV}:/data1/tim/workspace/deepdive_kai0/kai0/src:${PYTHONPATH}"
export JAX_COMPILATION_CACHE_DIR=/data1/tim/workspace/deepdive_kai0/.xla_cache
export CUDA_VISIBLE_DEVICES=0
unset http_proxy https_proxy
# Blackwell (RTX 5090 / sm_120) workaround: jax/jaxlib 0.5.3's XLA autotuner
# SIGSEGVs during π₀ backend_compile. Disabling autotune costs ~5-20% infer
# speed but is the only fix short of upgrading jax to ≥0.6.x.
export XLA_FLAGS="--xla_gpu_autotune_level=0"

echo "=== Launching policy_inference_node (mode=${MODE}) ==="

exec ros2 run piper policy_inference_node.py --ros-args \
  -p mode:=${MODE} \
  -p checkpoint_dir:=/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/Task_A/mixed_1 \
  -p config_name:=pi05_flatten_fold_normal \
  -p gpu_id:=0 -p ws_port:=8000 \
  -p img_front_topic:=/camera_f/color/image_raw \
  -p img_left_topic:=/camera_l/color/image_raw \
  -p img_right_topic:=/camera_r/color/image_raw
