#!/bin/bash
# Dagger pseudo-stage inference on gf0 + gf1 (16 GPU total)
# num-workers=16: gf1 w0-7, gf0 w8-15. Resume: skip already-labeled parquets.

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
NUM_WORKERS=16
export TORCH_HOME=/vePFS/tim/workspace/openpi_cache/torch_hub
export HF_HOME=/vePFS/tim/workspace/openpi_cache/huggingface

SSH_KEY=/root/.ssh/ssh_worker_rsa_key
GF0_IP=192.168.0.144
GF1_IP=192.168.0.161
SSH_PORT=2222

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
mkdir -p $LOG_DIR $METRICS_DIR

echo "[$(date)] Launching dagger inference: 16 workers (8 gf1 + 8 gf0)"

# gf1: worker 0-7 (via jumphost gf0)
for i in 0 1 2 3 4 5 6 7; do
    LOG=$LOG_DIR/dagger_infer_gf1_w${i}_${TIMESTAMP}.log
    ssh -p $SSH_PORT root@$GF0_IP "ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no root@$GF1_IP '
        export HF_HOME=$HF_HOME
        export TORCH_HOME=$TORCH_HOME
        export TRANSFORMERS_OFFLINE=1
        CUDA_VISIBLE_DEVICES=$i nohup $PYTHON $DEEPDIVE/train_scripts/stage_classifier/infer_dagger.py \
            --dagger-source $DAGGER_SOURCE --dagger-output $DAGGER_OUTPUT \
            --ckpt $CKPT --backbone $BACKBONE --dtype $DTYPE \
            --num-workers $NUM_WORKERS --worker-id $i \
            --metrics-out $METRICS_DIR \
            --stride 8 --batch-size 8 \
            > $LOG 2>&1 &
        echo gf1_w${i}=\$!
    '"
done

# gf0: worker 8-15 (direct ssh)
for i in 0 1 2 3 4 5 6 7; do
    wid=$((i + 8))
    LOG=$LOG_DIR/dagger_infer_gf0_w${wid}_${TIMESTAMP}.log
    ssh -p $SSH_PORT root@$GF0_IP "
        export HF_HOME=$HF_HOME
        export TORCH_HOME=$TORCH_HOME
        export TRANSFORMERS_OFFLINE=1
        CUDA_VISIBLE_DEVICES=$i nohup $PYTHON $DEEPDIVE/train_scripts/stage_classifier/infer_dagger.py \
            --dagger-source $DAGGER_SOURCE --dagger-output $DAGGER_OUTPUT \
            --ckpt $CKPT --backbone $BACKBONE --dtype $DTYPE \
            --num-workers $NUM_WORKERS --worker-id $wid \
            --metrics-out $METRICS_DIR \
            --stride 8 --batch-size 8 \
            > $LOG 2>&1 &
        echo gf0_w${wid}=\$!
    "
done

echo ""
echo "[$(date)] 16 workers launched."
echo "  gf1 logs: $LOG_DIR/dagger_infer_gf1_w*_${TIMESTAMP}.log"
echo "  gf0 logs: $LOG_DIR/dagger_infer_gf0_w*_${TIMESTAMP}.log"
