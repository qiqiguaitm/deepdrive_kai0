#!/bin/bash
# run_gf2.sh — gf2: AWBC + Advantage Estimator (基于官方数据和模型)
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

echo "[cleanup] Killing stale GPU processes..."
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true
sleep 2

cd /home/tim/workspace/deepdive_kai0/kai0
mkdir -p /vePFS/tim/workspace/deepdive_kai0/logs

# === Step 1: AWBC (JAX FSDP, 100K steps, 官方预标注数据) ===
echo "[gf2] === AWBC norm_stats $(date) ==="
uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_awbc

echo "[gf2] === AWBC training START $(date) ==="
rm -rf checkpoints/pi05_flatten_fold_awbc/awbc_v1
uv run scripts/train.py pi05_flatten_fold_awbc \
  --exp_name=awbc_v1 \
  --fsdp-devices 8 \
  --batch-size 256 \
  || echo "[gf2] AWBC FAILED"
echo "[gf2] === AWBC training END $(date) ==="

# === Step 2: Advantage Estimator (PyTorch DDP, 100K steps) ===
echo "[gf2] === Advantage Estimator START $(date) ==="
uv run torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py \
  ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD \
  --exp_name=adv_est_v1 \
  --save_interval 10000 \
  --batch-size 144 \
  || echo "[gf2] Advantage Estimator FAILED"
echo "[gf2] === Advantage Estimator END $(date) ==="
