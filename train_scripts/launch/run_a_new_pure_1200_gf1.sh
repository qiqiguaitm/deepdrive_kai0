#!/bin/bash
# Task_A A_new_pure_1200 cold-start training on gf1.
# Dataset: 1143 train + 57 val (644 vis_base/*-new originals + 556 hflip mirrors).
# Source: /vePFS/tim/.../vis_base/<date>-new/ (6 dates, all clean).
# 50k steps, peak_lr=1.5e-5 cosine to 1.5e-6, warmup=1k, ema=0.9999.
# inline_eval_every=2 (eval every 4k step = 12 evals total).
# Init from Task_A/mixed_1, fresh norm_stats from current 1143 train.
# exp_name: task_a_new_pure_1200_new_norm (per user spec).
set -euo pipefail

export PATH=/home/tim/miniconda3/bin:/home/tim/.local/bin:$PATH
export PYTHONUNBUFFERED=1
export KAI0_DATA_ROOT=/vePFS/tim/workspace/deepdive_kai0/kai0
export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
export PYTORCH_CKPT_BASE=/vePFS/tim/workspace/openpi_cache/modelscope_cache/lerobot
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_DATASETS_CACHE=/home/tim/.cache/huggingface/datasets
export WANDB_MODE=offline
export LD_LIBRARY_PATH=/home/tim/miniconda3/lib:/home/tim/.cuda_compat:/usr/local/cuda-12.8/targets/x86_64-linux/lib
for d in /home/tim/.kai0_venv/lib/python3.11/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH=$d:$LD_LIBRARY_PATH
done

cd /vePFS/tim/workspace/deepdive_kai0/kai0

echo "[train] === START $(date) ==="
.venv/bin/python scripts/train.py pi05_flatten_fold_a_new_pure_1200 \
  --exp_name=task_a_new_pure_1200_new_norm \
  --resume 2>&1
echo "[train] === END $(date) ==="
