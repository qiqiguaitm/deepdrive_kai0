#!/bin/bash
# 双节点 TCP 分布式训练启动脚本（稳定生产配置）
#
# 为什么 TCP 不用 RoCE：两节点 mlx5 接口在不同 /27 子网，容器内
# NCCL 数据路径不支持跨子网 QP 建立（verbs vs rdma_cm 路径差异）。
# 详见 docs/multinode_distributed_training_plan.md 问题 3。
#
# 性能：3.51 s/step，batch 256，100K steps ≈ 4 天
#
# 用法: bash run_multinode.sh

CONFIG="pi05_flatten_fold_awbc"
EXP_NAME="multinode_awbc_v1"
KAI0_ROOT="/vePFS/tim/workspace/deepdive_kai0/kai0"
LOG_DIR="/vePFS/tim/workspace/deepdive_kai0/logs"
PYTHON="$KAI0_ROOT/.venv/bin/python3"
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"

TRAIN_CMD="$PYTHON -u $KAI0_ROOT/scripts/train.py $CONFIG \
  --exp_name=$EXP_NAME \
  --fsdp-devices 8 \
  --batch-size 256 \
  --overwrite --no-wandb-enabled"

# Node1
ssh -p 2222 -i /root/.ssh/ssh_worker_rsa_key -o StrictHostKeyChecking=no root@192.168.0.161 "
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export JAX_COORDINATOR_ADDRESS=192.168.0.144:15830
export JAX_NUM_PROCESSES=2
export JAX_PROCESS_INDEX=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth2,eth3,eth4,eth5
export NCCL_DEBUG=WARN
export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p $LOG_DIR
cd $KAI0_ROOT
nohup $TRAIN_CMD > $LOG_DIR/multinode_node1_$TIMESTAMP.log 2>&1 &
echo \"[gf1] pid=\$! log=$LOG_DIR/multinode_node1_$TIMESTAMP.log\"
" &

sleep 2

# Node0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export JAX_COORDINATOR_ADDRESS=192.168.0.144:15830
export JAX_NUM_PROCESSES=2
export JAX_PROCESS_INDEX=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth2,eth3,eth4,eth5
export NCCL_DEBUG=WARN
export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
cd $KAI0_ROOT
nohup $TRAIN_CMD > $LOG_DIR/multinode_node0_$TIMESTAMP.log 2>&1 &
echo "[gf0] pid=$! log=$LOG_DIR/multinode_node0_$TIMESTAMP.log"
