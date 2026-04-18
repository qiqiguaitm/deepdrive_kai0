#!/bin/bash
export PATH=/home/tim/miniconda3/bin:/home/tim/.local/bin:$PATH
export PYTHONUNBUFFERED=1
export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_DATASETS_CACHE=/home/tim/.cache/huggingface/datasets
export WANDB_MODE=offline
export LD_LIBRARY_PATH=/home/tim/miniconda3/lib:/home/tim/.cuda_compat:/usr/local/cuda-12.8/targets/x86_64-linux/lib
for d in /home/tim/.kai0_venv/lib/python3.11/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH=$d:$LD_LIBRARY_PATH
done

echo "[cleanup] Killing stale GPU processes..."
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true
sleep 2

cd /home/tim/workspace/deepdive_kai0/kai0
for i in 0 1 2 3; do
    echo "[train] === START split_$i $(date) ==="
    rm -rf checkpoints/pi05_flatten_fold_split_$i/split_${i}_v1
    EPISODES=$(python3 /vePFS/tim/workspace/deepdive_kai0/get_episodes.py $i)
    uv run scripts/train.py pi05_flatten_fold_split_$i \
      --exp_name=split_${i}_v1 \
      --fsdp-devices 8 \
      --batch-size 256 \
      --data.episodes $EPISODES \
      || echo "[train] split_$i FAILED"
    echo "[train] === END split_$i $(date) ==="
done
