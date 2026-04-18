#!/bin/bash
# AWBC Option A 实验（gf1 单机）
# ================================================================
# 对比 gf0 baseline (pi05_flatten_fold_awbc, binary Advantage prompt):
#   - n_slices=5 + Quality 1-5 prompt
#   - stage-aware (stage-nums=2) rebalance positive samples
#   - [2026-04-18] Dropout 切回 0.0（Option A switch，原 0.15）
#     原因：Step 5000 eval 显示 gf1(dropout=15%) 在多数指标上落后 gf0 baseline。
#     分析表明 demo-only 数据 advantage 方差本就很小（η²≈3%），dropout 进一步稀释
#     弱信号；π0.7 paper 的 dropout 有效前提是多模态 prompt 冗余，我们只有单维度
#     Quality。故禁用 dropout，保留 Quality 1-5 分桶 + stage-aware rebalance。
# ================================================================
#
# 用法: bash run_awbc_q5drop_gf1.sh  (config name 保留以便 resume 已有 checkpoint)
# 运行节点: gf1 (192.168.0.161)
# 前置: bash prepare_advantage_q5.sh (生成 advantage_q5 数据集)

# ==== 参数配置区（按需修改）====
CONFIG="pi05_flatten_fold_awbc_q5drop"
EXP_NAME="gf1_awbc_q5drop_v2"
BATCH_SIZE=256
FSDP_DEVICES=8
NODE_IP="192.168.0.161"         # gf1

# 路径（一般不动）
KAI0_ROOT="/vePFS/tim/workspace/deepdive_kai0/kai0"
LOG_DIR="/vePFS/tim/workspace/deepdive_kai0/logs"
PYTHON="$KAI0_ROOT/.venv/bin/python3"
SSH_KEY="/root/.ssh/ssh_worker_rsa_key"
SSH_PORT=2222
# ==============================

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${EXP_NAME}_${TIMESTAMP}.log"

TRAIN_CMD="$PYTHON -u $KAI0_ROOT/scripts/train.py $CONFIG \
  --exp_name=$EXP_NAME \
  --fsdp-devices $FSDP_DEVICES \
  --batch-size $BATCH_SIZE \
  --no-wandb-enabled"

# gf1 节点单机启动（通过 gf0 跳板 ssh 到 gf1）
ssh -p $SSH_PORT root@192.168.0.144 "
ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no root@$NODE_IP \"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
unset JAX_COORDINATOR_ADDRESS JAX_NUM_PROCESSES JAX_PROCESS_INDEX
unset NCCL_IB_DISABLE NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_GID_INDEX \
      NCCL_IB_ROUTABLE_FLID_GID_INDEX NCCL_NET_PLUGIN NCCL_IB_TIMEOUT \
      NCCL_IB_RETRY_CNT NCCL_IB_ADDR_FAMILY NCCL_NET_GDR_LEVEL NCCL_ALGO \
      NCCL_IB_PCI_RELAXED_ORDERING
export NCCL_DEBUG=WARN
export KAI0_DATA_ROOT=/vePFS/tim/workspace/deepdive_kai0/kai0
export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p $LOG_DIR
cd $KAI0_ROOT
nohup $TRAIN_CMD > $LOG_FILE 2>&1 &
echo '[gf1] pid='\\\$! 'log=$LOG_FILE'
\"
"
