#!/bin/bash
# Task_A mix_apr28_450 cold-start training on gf1.
# Dataset: 405 train + 45 val (150 vis_apr28 + 150 kai0_base + 150 kai0_dagger).
# 30k steps, peak_lr=1.5e-5 cosine to 1.5e-6, warmup=1k, ema=0.9999.
# Save every 2k step (15 ckpts × ~12 GB = 180 GB).
# ETA ~20 hr; deadline Thu Apr 30 12:00 CST.
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
.venv/bin/python scripts/train.py pi05_flatten_fold_mix_apr28_450 \
  --exp_name=mix_apr28_450_v1 \
  --resume 2>&1
echo "[train] === END $(date) ==="
