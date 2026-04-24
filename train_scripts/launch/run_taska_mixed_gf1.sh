#!/bin/bash
# Task_A mixed (visrobot01 + base + dagger, equal N per source)
# Full-param fine-tune from Task_A/mixed_1, EMA=0.999, 5000 steps
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
# NOTE: --resume preserves all existing ckpts; do NOT use --overwrite (it rmtrees entire exp dir).
# First run auto-falls back to weight_loader when no ckpts exist yet.
.venv/bin/python scripts/train.py pi05_flatten_fold_mixed_visrobot01 \
  --exp_name=mixed_visrobot01_v1 \
  --resume 2>&1
echo "[train] === END $(date) ==="
