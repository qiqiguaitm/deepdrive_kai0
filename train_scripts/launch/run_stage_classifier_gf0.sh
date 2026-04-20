#!/bin/bash
# gf0: Stage Classifier feature precompute (8 GPU)
# Default backbone: V-JEPA 2 ViT-giant-384 (vitg-384.pt, 15.3 GB)
# Runs in parallel with gf1 precompute/train on V-JEPA 2.1 Large for A/B comparison.
# See docs/training/stage_classifier_plan.md

set -e

DEEPDIVE=/vePFS/tim/workspace/deepdive_kai0
PYTHON=$DEEPDIVE/kai0/.venv/bin/python3
SOURCE=$DEEPDIVE/kai0/data/Task_A/advantage
SPLIT=$DEEPDIVE/kai0/data/Task_A/stage_classifier_split.json
BACKBONE=${BACKBONE:-vjepa2_giant_384}
CACHE=$DEEPDIVE/kai0/cache/stage_classifier_${BACKBONE}
CKPT=$DEEPDIVE/kai0/checkpoints/stage_classifier_${BACKBONE}/run1
LOG_DIR=$DEEPDIVE/logs
DTYPE=${DTYPE:-float32}
export TORCH_HOME=/vePFS/tim/workspace/openpi_cache/torch_hub
export HF_HOME=/vePFS/tim/workspace/openpi_cache/huggingface

GF0_IP=192.168.0.144
SSH_PORT=2222

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
mkdir -p $LOG_DIR $CACHE $CKPT

MODE=${1:-"all"}

precompute_parallel() {
    echo "[$(date)] Starting parallel pre-computation on gf0 (8 GPU), backbone=$BACKBONE, dtype=$DTYPE"
    for i in 0 1 2 3 4 5 6 7; do
        LOG=$LOG_DIR/precompute_${BACKBONE}_w${i}_${TIMESTAMP}.log
        ssh -p $SSH_PORT root@$GF0_IP "
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
            echo worker=\$!
        "
    done
    echo "[$(date)] All 8 workers launched. Logs: $LOG_DIR/precompute_${BACKBONE}_w*_${TIMESTAMP}.log"
    echo "Monitor: bash $0 status"
}

train_classifier() {
    echo "[$(date)] Starting classifier training on gf0 (1 GPU)..."
    LOG=$LOG_DIR/stage_classifier_train_gf0_${BACKBONE}_${TIMESTAMP}.log
    ssh -p $SSH_PORT root@$GF0_IP "
        CUDA_VISIBLE_DEVICES=0 nohup $PYTHON $DEEPDIVE/train_scripts/stage_classifier/train.py \
            --cache-root $CACHE --split $SPLIT --out-dir $CKPT \
            --num-steps 20000 --batch-size 128 --lr 5e-4 \
            --num-workers-dl 8 --boundary-focus-ratio 0.7 \
            --val-every 1000 --save-every 5000 \
            > $LOG 2>&1 &
        echo train_pid=\$!
    "
    echo "[$(date)] Training launched. Log: $LOG"
}

check_status() {
    echo "=== gf0 running processes ==="
    ssh -p $SSH_PORT root@$GF0_IP "ps aux | grep -E 'precompute|train.py' | grep -v grep | head"
    echo "=== gf0 GPU util ==="
    ssh -p $SSH_PORT root@$GF0_IP "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"
    echo "=== cache file count ($BACKBONE) ==="
    ls $CACHE 2>/dev/null | wc -l
}

case $MODE in
    precompute) precompute_parallel ;;
    train)      train_classifier ;;
    status)     check_status ;;
    all)        precompute_parallel ;;
    *) echo "Usage: BACKBONE=<name> $0 [precompute|train|status|all]"; exit 1 ;;
esac
