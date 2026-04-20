#!/bin/bash
# gf0_robust: awbc_v2 with aggressive image augmentation for deploy-robustness
# Parallel to gf1 vanilla run (run_awbc_v2_gf1.sh). Both train 30K steps on awbc_v2_full (6512 ep).
# Differences from gf1 vanilla:
#   - augment_level="aggressive" → larger crop/rotate + stronger color jitter in model.py
#   - Target: D435→D405 sensor swap, top_head pose/height variation, arm-spacing drift
# No state noise augmentation (per user instruction, stay aligned with no-state-noise baseline).

CONFIG="pi05_flatten_fold_awbc_v2_robust"
EXP_NAME="gf0_awbc_v2_robust_v1"
BATCH_SIZE=256
FSDP_DEVICES=8
NODE_IP="192.168.0.144"  # gf0

KAI0_ROOT="/vePFS/tim/workspace/deepdive_kai0/kai0"
LOG_DIR="/vePFS/tim/workspace/deepdive_kai0/logs"
PYTHON="$KAI0_ROOT/.venv/bin/python3"
SSH_PORT=2222

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${EXP_NAME}_${TIMESTAMP}.log"

TRAIN_CMD="$PYTHON -u $KAI0_ROOT/scripts/train.py $CONFIG \
  --exp_name=$EXP_NAME \
  --fsdp-devices $FSDP_DEVICES \
  --batch-size $BATCH_SIZE \
  --no-wandb-enabled"

ssh -p $SSH_PORT root@$NODE_IP "
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
echo '[gf0-robust] pid='\$! 'log=$LOG_FILE'
"
