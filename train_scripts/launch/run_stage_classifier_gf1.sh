#!/bin/bash
# gf1: Stage Classifier training pipeline (2-part)
#   Part A: Pre-compute V-JEPA 2 SSv2 tube features on 8 GPU (parallel)
#   Part B: Train Cross-Attn + MLP on cached features (single GPU)
# See docs/training/stage_classifier_plan.md

set -e

DEEPDIVE=/vePFS/tim/workspace/deepdive_kai0
PYTHON=$DEEPDIVE/kai0/.venv/bin/python3
SOURCE=$DEEPDIVE/kai0/data/Task_A/advantage
SPLIT=$DEEPDIVE/kai0/data/Task_A/stage_classifier_split.json
BACKBONE=${BACKBONE:-vjepa2_1_large}
CACHE=$DEEPDIVE/kai0/cache/stage_classifier_${BACKBONE}
CKPT=$DEEPDIVE/kai0/checkpoints/stage_classifier_${BACKBONE}/run1
LOG_DIR=$DEEPDIVE/logs
DTYPE=${DTYPE:-float32}
export TORCH_HOME=/vePFS/tim/workspace/openpi_cache/torch_hub
export HF_HOME=/vePFS/tim/workspace/openpi_cache/huggingface

SSH_KEY=/root/.ssh/ssh_worker_rsa_key
GF0_IP=192.168.0.144
GF1_IP=192.168.0.161
SSH_PORT=2222

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
mkdir -p $LOG_DIR $CACHE $CKPT

MODE=${1:-"all"}

precompute_parallel() {
    echo "[$(date)] Starting parallel pre-computation on gf1 (8 GPU), backbone=$BACKBONE, dtype=$DTYPE"
    for i in 0 1 2 3 4 5 6 7; do
        LOG=$LOG_DIR/precompute_${BACKBONE}_w${i}_${TIMESTAMP}.log
        ssh -p $SSH_PORT root@$GF0_IP "ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no root@$GF1_IP '
            export HF_HOME=$HF_HOME
            export TORCH_HOME=$TORCH_HOME
            export TRANSFORMERS_OFFLINE=1
            CUDA_VISIBLE_DEVICES=$i nohup $PYTHON $DEEPDIVE/train_scripts/stage_classifier/precompute_features.py \
                --source $SOURCE --split $SPLIT --split-key all \
                --cache-root $CACHE --backbone $BACKBONE \
                --dtype $DTYPE \
                --num-workers 8 --worker-id $i \
                --batch-size 4 \
                > $LOG 2>&1 &
            echo worker$i_pid=\$!
        '"
    done
    echo "[$(date)] All 8 workers launched. Logs: $LOG_DIR/precompute_${BACKBONE}_w*_${TIMESTAMP}.log"
    echo "Monitor: bash $0 status"
}

train_classifier() {
    echo "[$(date)] Starting classifier training on gf1 (1 GPU)..."
    LOG=$LOG_DIR/stage_classifier_train_${TIMESTAMP}.log
    ssh -p $SSH_PORT root@$GF0_IP "ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no root@$GF1_IP '
        CUDA_VISIBLE_DEVICES=0 nohup $PYTHON $DEEPDIVE/train_scripts/stage_classifier/train.py \
            --cache-root $CACHE --split $SPLIT --out-dir $CKPT \
            --num-steps 20000 --batch-size 128 --lr 5e-4 \
            --num-workers-dl 8 --boundary-focus-ratio 0.7 \
            --val-every 1000 --save-every 5000 \
            > $LOG 2>&1 &
        echo train_pid=\$!
    '"
    echo "[$(date)] Training launched. Log: $LOG"
}

check_status() {
    echo "=== gf1 running processes ==="
    ssh -p $SSH_PORT root@$GF0_IP "ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no root@$GF1_IP 'ps aux | grep -E \"precompute|train.py\" | grep -v grep | head'"
    echo "=== cache file count ==="
    ls $CACHE 2>/dev/null | wc -l
    echo "=== latest precompute log tail ==="
    ls -t $LOG_DIR/precompute_w0_*.log 2>/dev/null | head -1 | xargs tail -5 2>/dev/null
    echo "=== latest train log tail ==="
    ls -t $LOG_DIR/stage_classifier_train_*.log 2>/dev/null | head -1 | xargs tail -5 2>/dev/null
}

case $MODE in
    precompute) precompute_parallel ;;
    train)      train_classifier ;;
    status)     check_status ;;
    all)        precompute_parallel ;;
    *) echo "Usage: $0 [precompute|train|status|all]"; exit 1 ;;
esac
