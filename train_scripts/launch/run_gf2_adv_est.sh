#!/bin/bash
# run_gf2_adv_est.sh — Advantage Estimator only (AWBC already done)
export PATH=/home/tim/miniconda3/bin:/home/tim/.local/bin:$PATH
export PYTHONUNBUFFERED=1
export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_DATASETS_CACHE=/home/tim/.cache/huggingface/datasets
export LD_LIBRARY_PATH=/home/tim/miniconda3/lib:/home/tim/.cuda_compat:/usr/local/cuda-12.8/targets/x86_64-linux/lib
for d in /home/tim/.kai0_venv/lib/python3.11/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH=$d:$LD_LIBRARY_PATH
done
export WANDB_MODE=offline
# Increase NCCL init timeout to 30 min (default 10 min) - rank 0 slow loading pi05_base
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DIST_INIT_BARRIER=1

echo "[cleanup] Killing stale GPU processes..."
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true
sleep 2

cd /home/tim/workspace/deepdive_kai0/kai0
mkdir -p /vePFS/tim/workspace/deepdive_kai0/logs

echo "[gf2] === Advantage Estimator START $(date) ==="
rm -rf experiment/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1
uv run torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py \
  ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD \
  --exp_name=adv_est_v1 \
  --save_interval 10000 \
  --batch-size 144 \
  || echo "[gf2] Advantage Estimator FAILED"
echo "[gf2] === Advantage Estimator END $(date) ==="
