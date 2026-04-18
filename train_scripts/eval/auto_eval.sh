#!/bin/bash
# Auto-eval newly-landed v2 ckpts + archive MAE results outside ckpt dir (survives orbax rotation).
set -u
export CUDA_VISIBLE_DEVICES=3
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENPI_DATA_HOME=/data1/tim/workspace/deepdive_kai0/openpi_cache

CKPT_ROOT=/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_stand_box_normal/stand_box_v2_pi05base
VAL=/data1/tim/workspace/deepdive_kai0/kai0/data/Task_E/val
KAI0=/data1/tim/workspace/deepdive_kai0/kai0
ARCHIVE=/data1/tim/workspace/deepdive_kai0/logs/eval_history_v2
mkdir -p "$ARCHIVE"

cd "$KAI0"

report() {
  python3 /data1/tim/workspace/deepdive_kai0/scripts/_print_mae.py "$1" "$2"
}

touch /tmp/evaled_steps_v2.txt

while true; do
  for ckpt in "$CKPT_ROOT"/[0-9]*; do
    [ -d "$ckpt" ] || continue
    step=$(basename "$ckpt")
    archive_json="$ARCHIVE/v2_step_${step}.json"
    # already archived?
    [ -f "$archive_json" ] && continue
    # in progress?
    grep -qxF "$step" /tmp/evaled_steps_v2.txt && continue

    echo "[auto-eval] launching eval for step=$step @ $(date +%T)"
    echo "$step" >> /tmp/evaled_steps_v2.txt

    .venv/bin/python /data1/tim/workspace/deepdive_kai0/scripts/eval_val_action_mse.py \
        --config pi05_stand_box_normal \
        --ckpt "$ckpt" \
        --val "$VAL" \
        --n-sample-frames 50 > "/data1/tim/workspace/deepdive_kai0/logs/eval_v2_${step}.log" 2>&1

    # ckpt/eval_val.json written by script; archive it
    if [ -f "$ckpt/eval_val.json" ]; then
      cp "$ckpt/eval_val.json" "$archive_json"
      report "$step" "$archive_json"
    else
      echo "[auto-eval] WARN: no eval_val.json for step=$step (script failed?)"
      tail -5 "/data1/tim/workspace/deepdive_kai0/logs/eval_v2_${step}.log"
    fi
  done
  sleep 30
done
