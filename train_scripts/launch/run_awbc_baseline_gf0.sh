#!/bin/bash
# 单机 gf0 (192.168.0.144) AWBC baseline
# 8x A100 80GB, batch=256 (32/GPU), 100K steps, ~2.3 天
#
# 对应双机 TCP run 的单机版，没有多主机 checkpoint bug、没有跨节点通信开销
# 用法: bash run_single_gf0.sh

CONFIG="pi05_flatten_fold_awbc"
EXP_NAME="gf0_awbc_baseline_v1"
KAI0_ROOT="/vePFS/tim/workspace/deepdive_kai0/kai0"
LOG_DIR="/vePFS/tim/workspace/deepdive_kai0/logs"
PYTHON="$KAI0_ROOT/.venv/bin/python3"
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"

TRAIN_CMD="$PYTHON -u $KAI0_ROOT/scripts/train.py $CONFIG \
  --exp_name=$EXP_NAME \
  --fsdp-devices 8 \
  --batch-size 256 \
  --resume --no-wandb-enabled"

# 单机：不设 JAX_COORDINATOR_ADDRESS，JAX 自动识别 8 本地设备
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
unset JAX_COORDINATOR_ADDRESS JAX_NUM_PROCESSES JAX_PROCESS_INDEX
unset NCCL_IB_DISABLE NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_GID_INDEX \
      NCCL_IB_ROUTABLE_FLID_GID_INDEX NCCL_NET_PLUGIN NCCL_IB_TIMEOUT \
      NCCL_IB_RETRY_CNT NCCL_IB_ADDR_FAMILY NCCL_NET_GDR_LEVEL NCCL_ALGO \
      NCCL_IB_PCI_RELAXED_ORDERING

# 单机 NCCL：仅需 debug 控制
export NCCL_DEBUG=WARN

# JAX / XLA
export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0

cd $KAI0_ROOT
nohup $TRAIN_CMD > $LOG_DIR/gf0_awbc_baseline_${TIMESTAMP}.log 2>&1 &
echo "[gf0] pid=$! log=$LOG_DIR/gf0_awbc_baseline_${TIMESTAMP}.log"
