#!/bin/bash
# gf1: 4 并行 head 探索实验 on V-JEPA 2.1 Large cache
# 每实验 2 GPU (第二卡保留备用，单卡跑已足够；head 仅 4M 参数)
# E1 baseline, E2 capacity, E3 strong-mono, E4 long-train

set -e

DEEPDIVE=/vePFS/tim/workspace/deepdive_kai0
PYTHON=$DEEPDIVE/kai0/.venv/bin/python3
SPLIT=$DEEPDIVE/kai0/data/Task_A/stage_classifier_split.json
CACHE=$DEEPDIVE/kai0/cache/stage_classifier_vjepa2_1_large
CKPT_BASE=$DEEPDIVE/kai0/checkpoints/stage_classifier_vjepa2_1_large
LOG_DIR=$DEEPDIVE/logs

SSH_KEY=/root/.ssh/ssh_worker_rsa_key
GF0_IP=192.168.0.144
GF1_IP=192.168.0.161
SSH_PORT=2222

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
mkdir -p $LOG_DIR $CKPT_BASE

# Launch one experiment via SSH (gf1 = GF0_IP → GF1_IP double-hop)
# Args: exp_name gpu_ids extra_args
launch_exp() {
    local EXP=$1
    local GPUS=$2
    local EXTRA=$3
    local CKPT=$CKPT_BASE/$EXP
    local LOG=$LOG_DIR/stage_classifier_${EXP}_${TIMESTAMP}.log

    mkdir -p $CKPT

    echo "[$(date)] Launching $EXP on GPU $GPUS ..."
    ssh -p $SSH_PORT root@$GF0_IP "ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no root@$GF1_IP '
        CUDA_VISIBLE_DEVICES=$GPUS nohup $PYTHON $DEEPDIVE/train_scripts/stage_classifier/train.py \
            --cache-root $CACHE --split $SPLIT --out-dir $CKPT \
            --num-workers-dl 4 \
            --val-every 1000 --save-every 5000 \
            --log-every 50 \
            $EXTRA \
            > $LOG 2>&1 &
        echo ${EXP}_pid=\$!
    '"
    echo "    Log: $LOG"
    echo "    Ckpt: $CKPT"
}

# E1: baseline — default hyperparameters
launch_exp "E1_baseline" "0,1" "\
    --num-steps 20000 --batch-size 128 --lr 5e-4 \
    --hidden-dim 384 --n-layers 2 --n-heads 8 --dropout 0.1 \
    --class-weight-flat 1.0 --class-weight-fold 3.0 \
    --loss-smooth 0.1 --loss-mono 0.2 \
    --boundary-focus-ratio 0.7"

# E2: capacity — bigger head (hidden 512, 3 layers)
launch_exp "E2_capacity" "2,3" "\
    --num-steps 20000 --batch-size 128 --lr 5e-4 \
    --hidden-dim 512 --n-layers 3 --n-heads 8 --dropout 0.1 \
    --class-weight-flat 1.0 --class-weight-fold 3.0 \
    --loss-smooth 0.1 --loss-mono 0.2 \
    --boundary-focus-ratio 0.7"

# E3: strong monotonicity — enforce cleaner output
launch_exp "E3_strong_mono" "4,5" "\
    --num-steps 20000 --batch-size 128 --lr 5e-4 \
    --hidden-dim 384 --n-layers 2 --n-heads 8 --dropout 0.1 \
    --class-weight-flat 1.0 --class-weight-fold 5.0 \
    --loss-smooth 0.3 --loss-mono 0.5 \
    --boundary-focus-ratio 0.9"

# E4: long training — 40K steps with lower lr + longer warmup
launch_exp "E4_long_train" "6,7" "\
    --num-steps 40000 --batch-size 128 --lr 3e-4 --warmup-steps 2000 \
    --hidden-dim 384 --n-layers 2 --n-heads 8 --dropout 0.1 \
    --class-weight-flat 1.0 --class-weight-fold 3.0 \
    --loss-smooth 0.1 --loss-mono 0.2 \
    --boundary-focus-ratio 0.7"

echo ""
echo "[$(date)] All 4 experiments launched."
echo "Monitor: grep 'Val@' $LOG_DIR/stage_classifier_E*_${TIMESTAMP}.log | tail"
