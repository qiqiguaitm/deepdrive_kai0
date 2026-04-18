#!/bin/bash
# 启动 Policy Server，启用 JAX 编译缓存
# RTX 5090 Blackwell 必需：XLA autotuner SIGSEGV 规避。允许外部覆盖。
export XLA_FLAGS="${XLA_FLAGS:-"--xla_gpu_autotune_level=0"}"
export JAX_COMPILATION_CACHE_DIR=/data1/tim/workspace/deepdive_kai0/.xla_cache
export CUDA_VISIBLE_DEVICES=0
cd /data1/tim/workspace/deepdive_kai0/kai0
exec .venv/bin/python scripts/serve_policy.py --port 8000 \
  policy:checkpoint --policy.config=pi05_flatten_fold_normal \
  --policy.dir=checkpoints/Task_A/mixed_1
