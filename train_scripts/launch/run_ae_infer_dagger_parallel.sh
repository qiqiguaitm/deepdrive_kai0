#!/bin/bash
# Parallel AE inference on dagger_with_stage using 16 GPUs (8 gf1 + 8 gf0)
# Each worker shards by ep_idx % 16.

set -e

DEEPDIVE=/vePFS/tim/workspace/deepdive_kai0
PYTHON=$DEEPDIVE/kai0/.venv/bin/python3
AE_CKPT=$DEEPDIVE/kai0/checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1
DATA_ROOT=$DEEPDIVE/kai0/data/Task_A
LOG_DIR=$DEEPDIVE/logs
NUM_WORKERS=16
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

SSH_KEY=/root/.ssh/ssh_worker_rsa_key
GF0_IP=192.168.0.144
GF1_IP=192.168.0.161
SSH_PORT=2222

mkdir -p $LOG_DIR

launch_worker() {
    local WID=$1
    local HOST=$2     # "gf0" or "gf1"
    local GPU_IDX=$3  # 0..7
    local LOG=$LOG_DIR/ae_infer_${HOST}_w${WID}_${TIMESTAMP}.log

    local PY_BLOCK="
import sys
sys.path.insert(0, \\\"$DEEPDIVE/kai0/stage_advantage/annotation\\\")
import eval as ev
ev.MODELS_CONFIG_MAP[\\\"Flatten-Fold\\\"][\\\"KAI0\\\"][\\\"ckpt_dir\\\"] = \\\"$AE_CKPT\\\"
ev.MODELS_CONFIG_MAP[\\\"Flatten-Fold\\\"][\\\"KAI0\\\"][\\\"ckpt_steps\\\"] = 100000
sys.argv = [\\\"eval.py\\\", \\\"Flatten-Fold\\\", \\\"KAI0\\\", \\\"$DATA_ROOT/dagger_with_stage\\\", \\\"--num-workers\\\", \\\"$NUM_WORKERS\\\", \\\"--worker-id\\\", \\\"$WID\\\"]
ev.main()
"

    if [ "$HOST" = "gf1" ]; then
        ssh -p $SSH_PORT root@$GF0_IP "ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no root@$GF1_IP '
            export KAI0_DATA_ROOT=$DEEPDIVE/kai0
            export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
            export PYTORCH_CKPT_BASE=$DEEPDIVE/kai0/checkpoints
            cd $DEEPDIVE/kai0
            CUDA_VISIBLE_DEVICES=$GPU_IDX nohup $PYTHON -c \"$PY_BLOCK\" > $LOG 2>&1 &
            echo ${HOST}_w${WID}=\$!
        '"
    else
        ssh -p $SSH_PORT root@$GF0_IP "
            export KAI0_DATA_ROOT=$DEEPDIVE/kai0
            export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
            export PYTORCH_CKPT_BASE=$DEEPDIVE/kai0/checkpoints
            cd $DEEPDIVE/kai0
            CUDA_VISIBLE_DEVICES=$GPU_IDX nohup $PYTHON -c \"$PY_BLOCK\" > $LOG 2>&1 &
            echo ${HOST}_w${WID}=\$!
        "
    fi
}

echo "[$(date)] Launching 16 AE workers (8 gf1 + 8 gf0) for dagger_with_stage"

# gf1: worker ids 0-7, GPUs 0-7
for i in 0 1 2 3 4 5 6 7; do
    launch_worker $i "gf1" $i
done

# gf0: worker ids 8-15, GPUs 0-7
for i in 0 1 2 3 4 5 6 7; do
    WID=$((i + 8))
    launch_worker $WID "gf0" $i
done

echo "[$(date)] All 16 workers launched. Logs: $LOG_DIR/ae_infer_*_${TIMESTAMP}.log"
