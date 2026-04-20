#!/bin/bash
# gf1_v3: RS-CL contrastive + DCT frequency-domain aux losses
# ================================================================
# Motivation:
#   v1/v2 (advantage prompting) failed — demo-only data has η²=3% advantage
#   variance; prompt conditioning on weak noisy signal is ignored by the model.
#
# New approach (from 2025-2026 literature):
#   1. RS-CL (arXiv 2510.01711) — InfoNCE between VLM summary token and
#      proprioceptive state. Forces VLM to encode control-relevant signal
#      instead of ignoring visual/prompt inputs. Proven at 60-300 demos scale.
#   2. VLANeXt DCT loss (arXiv 2602.18532) — frequency-domain MSE with low-freq
#      weighting; suppresses high-frequency action jitter for smoother rollout.
#
# Code changes:
#   - pi0_config.py: use_rs_cl, use_dct_loss + hyperparams
#   - pi0.py: projection heads, aux losses in compute_loss (dict return)
#   - train.py: loss_fn handles dict, value_and_grad with has_aux=True
#   - config.py: pi05_flatten_fold_cl_dct TrainConfig (Task_A/base, BC + aux)
# ================================================================
#
# 用法: bash train_scripts/launch/run_cl_dct_gf1.sh
# 运行节点: gf1 (192.168.0.161, ssh via gf0 跳板)

CONFIG="pi05_flatten_fold_cl_dct"
EXP_NAME="gf1_cl_dct_v3"
BATCH_SIZE=256
FSDP_DEVICES=8
NODE_IP="192.168.0.161"         # gf1

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
echo '[gf1-v3] pid='\\\$! 'log=$LOG_FILE'
\"
"
