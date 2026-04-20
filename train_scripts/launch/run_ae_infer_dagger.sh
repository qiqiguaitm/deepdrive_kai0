#!/bin/bash
# Phase 5 Step 1: Run trained AE (ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD) on dagger_with_stage
# Output: dagger_with_stage/data_KAI0_100000/ (new parquet with absolute_advantage column)

set -e

DEEPDIVE=/vePFS/tim/workspace/deepdive_kai0
PYTHON=$DEEPDIVE/kai0/.venv/bin/python3
REPO_ID=$DEEPDIVE/kai0/data/Task_A/dagger_with_stage
CKPT_DIR=$DEEPDIVE/kai0/checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1
CKPT_STEP=100000
LOG=$DEEPDIVE/logs/ae_infer_dagger_$(date -u +%Y%m%d_%H%M%S).log

SSH_KEY=/root/.ssh/ssh_worker_rsa_key
GF0_IP=192.168.0.144
GF1_IP=192.168.0.161
SSH_PORT=2222

# Patch eval.py's MODELS_CONFIG_MAP at runtime via env override (or use a wrapper)
# Simpler: call evaluator directly via a small inline script

echo "[$(date)] AE inference on dagger_with_stage"
echo "  repo_id: $REPO_ID"
echo "  ckpt:    $CKPT_DIR/$CKPT_STEP"
echo "  log:     $LOG"

ssh -p $SSH_PORT root@$GF0_IP "ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no root@$GF1_IP '
    export KAI0_DATA_ROOT=$DEEPDIVE/kai0
    export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
    export PYTORCH_CKPT_BASE=$DEEPDIVE/kai0/checkpoints
    cd $DEEPDIVE/kai0
    # Monkey-patch ckpt_dir via env and call eval directly
    CUDA_VISIBLE_DEVICES=0 nohup $PYTHON -c \"
import sys
sys.path.insert(0, \\\"$DEEPDIVE/kai0/stage_advantage/annotation\\\")
import eval as ev
ev.MODELS_CONFIG_MAP[\\\"Flatten-Fold\\\"][\\\"KAI0\\\"][\\\"ckpt_dir\\\"] = \\\"$CKPT_DIR\\\"
ev.MODELS_CONFIG_MAP[\\\"Flatten-Fold\\\"][\\\"KAI0\\\"][\\\"ckpt_steps\\\"] = $CKPT_STEP
sys.argv = [\\\"eval.py\\\", \\\"Flatten-Fold\\\", \\\"KAI0\\\", \\\"$REPO_ID\\\"]
ev.main()
\" > $LOG 2>&1 &
    echo ae_pid=\$!
'"
echo "Monitor: tail -f $LOG"
