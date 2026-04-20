#!/bin/bash
# gf1: Dagger pseudo-stage inference (8 parallel workers)
# Uses V-JEPA 2.1 Large backbone + E1 baseline best head
# Writes dagger_with_stage/ dataset + per-episode metrics JSON

set -e

DEEPDIVE=/vePFS/tim/workspace/deepdive_kai0
PYTHON=$DEEPDIVE/kai0/.venv/bin/python3
DAGGER_SOURCE=$DEEPDIVE/kai0/data/Task_A/dagger
DAGGER_OUTPUT=$DEEPDIVE/kai0/data/Task_A/dagger_with_stage
CKPT=$DEEPDIVE/kai0/checkpoints/stage_classifier_vjepa2_1_large/E1_baseline/best.pt
BACKBONE=vjepa2_1_large
METRICS_DIR=$DEEPDIVE/kai0/cache/dagger_infer_metrics
LOG_DIR=$DEEPDIVE/logs
DTYPE=${DTYPE:-float32}
export TORCH_HOME=/vePFS/tim/workspace/openpi_cache/torch_hub
export HF_HOME=/vePFS/tim/workspace/openpi_cache/huggingface

SSH_KEY=/root/.ssh/ssh_worker_rsa_key
GF0_IP=192.168.0.144
GF1_IP=192.168.0.161
SSH_PORT=2222

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
mkdir -p $LOG_DIR $METRICS_DIR

echo "[$(date)] Starting dagger inference on gf1 (8 GPU)"
echo "  source:  $DAGGER_SOURCE"
echo "  output:  $DAGGER_OUTPUT"
echo "  ckpt:    $CKPT"
echo "  metrics: $METRICS_DIR"

for i in 0 1 2 3 4 5 6 7; do
    LOG=$LOG_DIR/dagger_infer_w${i}_${TIMESTAMP}.log
    ssh -p $SSH_PORT root@$GF0_IP "ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no root@$GF1_IP '
        export HF_HOME=$HF_HOME
        export TORCH_HOME=$TORCH_HOME
        export TRANSFORMERS_OFFLINE=1
        CUDA_VISIBLE_DEVICES=$i nohup $PYTHON $DEEPDIVE/train_scripts/stage_classifier/infer_dagger.py \
            --dagger-source $DAGGER_SOURCE --dagger-output $DAGGER_OUTPUT \
            --ckpt $CKPT --backbone $BACKBONE --dtype $DTYPE \
            --num-workers 8 --worker-id $i \
            --metrics-out $METRICS_DIR \
            --stride 8 --batch-size 8 \
            > $LOG 2>&1 &
        echo worker$i=\$!
    '"
done

echo ""
echo "[$(date)] All 8 workers launched. Logs: $LOG_DIR/dagger_infer_w*_${TIMESTAMP}.log"
