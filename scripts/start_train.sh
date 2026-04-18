#!/bin/bash
# Start a kai0 training run with inline val eval.
#
# Usage:
#   ./scripts/start_train.sh <config_name> <exp_name> <gpu_id> [--bs N] [--steps N] [--seed N] [--no-eval]
#
# Examples:
#   ./scripts/start_train.sh pi05_stand_box_normal v2_pi05_baseline 0
#   ./scripts/start_train.sh pi05_stand_box_aug v4_pi05_aug 2 --bs 4 --steps 15000
#   ./scripts/start_train.sh pi05_stand_box_kai0_aug v5_kai0_aug 3 --seed 42 --no-eval
#
# Environment variables (for overriding defaults):
#   VAL_ROOT   (default: Task_E/val)
#   N_FRAMES   (default: 20, per ep)
#   EVAL_EVERY (default: 1, = every save_interval)
set -euo pipefail

# ------ args ------
if [ $# -lt 3 ]; then
  echo "Usage: $0 <config_name> <exp_name> <gpu_id> [--bs N] [--steps N] [--seed N] [--no-eval] [extra-train-args...]"
  exit 1
fi
CONFIG="$1"
EXP_NAME="$2"
GPU="$3"
shift 3

BATCH_SIZE=""
NUM_STEPS=""
SEED="42"
NUM_WORKERS="2"
ENABLE_EVAL="true"
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --bs)        BATCH_SIZE="$2"; shift 2 ;;
    --steps)     NUM_STEPS="$2"; shift 2 ;;
    --seed)      SEED="$2"; shift 2 ;;
    --workers)   NUM_WORKERS="$2"; shift 2 ;;
    --no-eval)   ENABLE_EVAL="false"; shift ;;
    *)           EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# ------ paths ------
PROJECT_ROOT=/data1/tim/workspace/deepdive_kai0
KAI0=$PROJECT_ROOT/kai0
LOGS=$PROJECT_ROOT/logs
VAL_ROOT="${VAL_ROOT:-$KAI0/data/Task_E/val}"
N_FRAMES="${N_FRAMES:-20}"
EVAL_EVERY="${EVAL_EVERY:-1}"

mkdir -p "$LOGS"
LOG="$LOGS/train_${EXP_NAME}.log"

# ------ JAX / XLA env ------
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=$KAI0/.xla_cache
export OPENPI_DATA_HOME=$PROJECT_ROOT/openpi_cache
export WANDB_MODE=offline

# ------ inline eval env ------
if [ "$ENABLE_EVAL" = "true" ]; then
  export INLINE_EVAL_VAL_ROOT="$VAL_ROOT"
  export INLINE_EVAL_N_FRAMES="$N_FRAMES"
  export INLINE_EVAL_EVERY="$EVAL_EVERY"
else
  unset INLINE_EVAL_VAL_ROOT INLINE_EVAL_N_FRAMES INLINE_EVAL_EVERY
fi

# ------ build train.py args ------
ARGS=(--exp_name="$EXP_NAME" --overwrite --seed "$SEED" --num_workers "$NUM_WORKERS")
[ -n "$BATCH_SIZE" ] && ARGS+=(--batch_size "$BATCH_SIZE")
[ -n "$NUM_STEPS" ]  && ARGS+=(--num_train_steps "$NUM_STEPS")
ARGS+=("${EXTRA_ARGS[@]}")

# ------ report ------
cat <<EOF
╭───────────── kai0 train launch ─────────────
│ config:   $CONFIG
│ exp_name: $EXP_NAME
│ GPU:      $GPU
│ inline eval: $ENABLE_EVAL${ENABLE_EVAL:+ (val=$VAL_ROOT, N_FRAMES=$N_FRAMES, EVERY=$EVAL_EVERY)}
│ args:     ${ARGS[@]}
│ log:      $LOG
╰─────────────────────────────────────────────
EOF

cd "$KAI0"
: > "$LOG"
nohup uv run scripts/train.py "$CONFIG" "${ARGS[@]}" > "$LOG" 2>&1 &
PID=$!
disown
echo "PID=$PID  $(date)"
echo "tail -f $LOG    # follow progress"
