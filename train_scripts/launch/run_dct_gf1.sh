#!/bin/bash
# gf1_v4: DCT frequency-domain aux loss only (RS-CL dropped)
# ================================================================
# History:
#   v1 (q5drop, dropout=0.15): Quality 1-5 prompt + 15% dropout → worse than gf0
#   v2 (Option A, dropout=0.0): same prompt, no dropout → still no improvement
#   v3 (cl_dct): RS-CL + DCT aux losses → MAIN TASK DEGRADED 40-100% on eval MAE
#     Root cause: 0.3·cl_loss=1.67 at step 0 = 2× main loss, gradient dominated
#     by contrastive objective, pushed VLM features toward proprio encoding
#     instead of control-relevant features.
#   v4 (THIS): ONLY DCT loss kept. v3 empirical data showed DCT weighted
#     contribution stayed <1% of main loss throughout → safe; possible small
#     upside from low-freq action smoothing bias (matches slow cloth manip).
# ================================================================
#
# Usage: bash train_scripts/launch/run_dct_gf1.sh
# Node:  gf1 (192.168.0.161 via gf0 jump)

CONFIG="pi05_flatten_fold_dct_only"
EXP_NAME="gf1_dct_v4"
BATCH_SIZE=256
FSDP_DEVICES=8
NODE_IP="192.168.0.161"

KAI0_ROOT="/vePFS/tim/workspace/deepdive_kai0/kai0"
LOG_DIR="/vePFS/tim/workspace/deepdive_kai0/logs"
PYTHON="$KAI0_ROOT/.venv/bin/python3"
SSH_KEY="/root/.ssh/ssh_worker_rsa_key"
SSH_PORT=2222

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${EXP_NAME}_${TIMESTAMP}.log"

TRAIN_CMD="$PYTHON -u $KAI0_ROOT/scripts/train.py $CONFIG \
  --exp_name=$EXP_NAME \
  --fsdp-devices $FSDP_DEVICES \
  --batch-size $BATCH_SIZE \
  --no-wandb-enabled"

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
echo '[gf1-v4] pid='\\\$! 'log=$LOG_FILE'
\"
"
