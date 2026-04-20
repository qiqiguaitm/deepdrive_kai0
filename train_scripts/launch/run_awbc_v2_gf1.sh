#!/bin/bash
# gf1: awbc_v2 vanilla = advantage + dagger_advantage (6512 ep, no mirror)
# ================================================================
# Implements docs/training/stage_classifier_plan.md Phase 7.
#
# Data pipeline (pre-req):
#   1. Stage classifier → pseudo stage_progress_gt on dagger → dagger_with_stage
#   2. AE inference on dagger_with_stage → absolute_advantage column
#   3. discretize_advantage.py --stage-nums 2 --threshold 30 → dagger_advantage
#   4. merge_lerobot.py advantage + dagger_advantage → awbc_v2_full (6512 ep)
#   5. compute_norm_states_fast.py on awbc_v2_full
#
# Mirror skipped per taskE v8 ablation (neutral/slight-negative) + D405 asymmetry risk.
#
# Paired run with gf0 (run_awbc_v2_robust_gf0.sh):
#   - gf1 vanilla:  default "mild" image aug → training-domain baseline
#   - gf0 robust:   "aggressive" image aug → deploy-domain robustness (D435→D405)
# Both: 30K steps, 8×A100 FSDP, batch 256.
# ================================================================

CONFIG="pi05_flatten_fold_awbc_v2"
EXP_NAME="gf1_awbc_v2_vanilla"
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
echo '[gf1-v5] pid='\\\$! 'log=$LOG_FILE'
\"
"
