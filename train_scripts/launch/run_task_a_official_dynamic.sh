#!/usr/bin/env bash
set -euo pipefail

# One-shot launcher for:
#   1) initial Task_A mixed dataset build
#   2) episodes_stats + norm_stats generation
#   3) first training start
#   4) dynamic watcher start for auto rebuild + resume
#
# Default source layout:
#   /VLA-Data/scripts/lianqing/data/OpenDriveLab-org/Kai0/Task_A/{base,dagger,advantage}
#
# Usage:
#   bash train_scripts/launch/run_task_a_official_dynamic.sh
# Overrides:
#   VIS_ROOT=/path/to/Task_A MIX_ROOT=/path/to/mixed bash train_scripts/launch/run_task_a_official_dynamic.sh

REPO_ROOT="${REPO_ROOT:-/VLA-Data/scripts/xyh/deepdive_kai0}"
KAI0_DIR="${KAI0_DIR:-$REPO_ROOT/kai0}"
VIS_ROOT="${VIS_ROOT:-/VLA-Data/scripts/lianqing/data/OpenDriveLab-org/Kai0/Task_A}"
OLD_ROOT="${OLD_ROOT:-$VIS_ROOT}"
MIX_ROOT="${MIX_ROOT:-$KAI0_DIR/data/Task_A_mixed_gf1}"

CONFIG_NAME="${CONFIG_NAME:-pi05_flatten_fold_mixed_visrobot01}"
EXP_NAME="${EXP_NAME:-mixed_visrobot01_v1}"
PROJECT_NAME="${PROJECT_NAME:-kai0_policy_exp}"
VAL_SIZE="${VAL_SIZE:-21}"
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-}"
LOG_INTERVAL="${LOG_INTERVAL:-}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-}"
ATTACH_LOGS=0
REUSE_MIX=0

TRAIN_LOG="${TRAIN_LOG:-/tmp/train_taska_mixed.log}"
WATCHER_LOG="${WATCHER_LOG:-/tmp/dyn_train.log}"

log() { echo "[$(date +%F' '%T)] $*"; }

validate_existing_mix() {
  local root="$1"
  ./kai0/.venv/bin/python - "$root" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path
import pyarrow.parquet as pq

root = Path(sys.argv[1])
required = [
    root / "manifest.json",
    root / "base" / "meta" / "info.json",
    root / "base" / "meta" / "episodes.jsonl",
    root / "base" / "meta" / "episodes_stats.jsonl",
    root / "base" / "norm_stats.json",
    root / "val" / "meta" / "info.json",
    root / "val" / "meta" / "episodes.jsonl",
    root / "val" / "meta" / "episodes_stats.jsonl",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    print("[ERROR] missing required mixed-dataset files:")
    for p in missing:
        print(f"  {p}")
    sys.exit(2)

required_cols = [
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]
unexpected_cols = {
    "progress_gt",
    "stage_progress_gt",
    "relative_advantage",
    "absolute_value",
    "absolute_advantage",
}

def check_split(split: str) -> tuple[int, list[str]]:
    split_root = root / split
    episodes = [
        json.loads(line)
        for line in (split_root / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    errs: list[str] = []
    for ep in episodes:
        idx = ep["episode_index"]
        pq_path = split_root / "data" / "chunk-000" / f"episode_{idx:06d}.parquet"
        if not pq_path.exists():
            errs.append(f"{split}: missing parquet for episode {idx}")
            if len(errs) >= 20:
                break
            continue
        cols = pq.read_schema(pq_path).names
        missing_cols = [c for c in required_cols if c not in cols]
        extras = [c for c in cols if c in unexpected_cols]
        if missing_cols:
            errs.append(f"{split}: episode {idx} missing parquet columns {missing_cols}")
        if extras:
            errs.append(f"{split}: episode {idx} has unsupported parquet columns {extras}")
        for cam in ("top_head", "hand_left", "hand_right"):
            mp4 = split_root / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{idx:06d}.mp4"
            if not mp4.exists():
                errs.append(f"{split}: missing {cam} video for episode {idx}")
        if len(errs) >= 20:
            break
    return len(episodes), errs

base_n, base_errs = check_split("base")
val_n, val_errs = check_split("val")
errs = base_errs + val_errs
if errs:
    print("[ERROR] existing mixed dataset failed validation:")
    for e in errs[:20]:
        print(f"  {e}")
    if len(errs) > 20:
        print(f"  ... and {len(errs) - 20} more")
    sys.exit(3)

print(f"[OK] validated existing mixed dataset: base={base_n} val={val_n}")
PY
}

usage() {
  cat <<'EOF'
Usage:
  bash train_scripts/launch/run_task_a_official_dynamic.sh [config_name] [options]

Train options:
  --config-name NAME
  --exp-name NAME
  --project-name NAME
  --batch-size N
  --num-train-steps N
  --save-interval N
  --log-interval N
  --num-workers N
  --checkpoint-base-dir DIR
  --attach-logs
      keep launcher in foreground by tailing train/watcher logs
  --foreground
      alias of --attach-logs
  --reuse-mix
      validate existing mixed dataset and skip rebuild/stats/norm recompute

Dataset/build options:
  --vis-root DIR
  --old-root DIR
  --mix-root DIR
  --val-size N

Logs:
  --train-log FILE
  --watcher-log FILE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --config-name)
      CONFIG_NAME="$2"
      shift 2
      ;;
    --exp-name|--exp_name)
      EXP_NAME="$2"
      shift 2
      ;;
    --project-name)
      PROJECT_NAME="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num-train-steps)
      NUM_TRAIN_STEPS="$2"
      shift 2
      ;;
    --save-interval)
      SAVE_INTERVAL="$2"
      shift 2
      ;;
    --log-interval)
      LOG_INTERVAL="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --checkpoint-base-dir)
      CHECKPOINT_BASE_DIR="$2"
      shift 2
      ;;
    --attach-logs|--foreground)
      ATTACH_LOGS=1
      shift
      ;;
    --reuse-mix)
      REUSE_MIX=1
      shift
      ;;
    --vis-root)
      VIS_ROOT="$2"
      shift 2
      ;;
    --old-root)
      OLD_ROOT="$2"
      shift 2
      ;;
    --mix-root)
      MIX_ROOT="$2"
      shift 2
      ;;
    --val-size)
      VAL_SIZE="$2"
      shift 2
      ;;
    --train-log)
      TRAIN_LOG="$2"
      shift 2
      ;;
    --watcher-log)
      WATCHER_LOG="$2"
      shift 2
      ;;
    --*)
      echo "[ERROR] unknown option: $1" >&2
      exit 2
      ;;
    *)
      if [[ "$CONFIG_NAME" == "pi05_flatten_fold_mixed_visrobot01" ]]; then
        CONFIG_NAME="$1"
        shift
      else
        echo "[ERROR] unexpected positional arg: $1" >&2
        exit 2
      fi
      ;;
  esac
done

TRAIN_ARGS=(
  "$CONFIG_NAME"
  "--exp_name" "$EXP_NAME"
  "--project-name" "$PROJECT_NAME"
  "--overwrite"
)
if [[ -n "$BATCH_SIZE" ]]; then
  TRAIN_ARGS+=("--batch-size" "$BATCH_SIZE")
fi
if [[ -n "$NUM_WORKERS" ]]; then
  TRAIN_ARGS+=("--num-workers" "$NUM_WORKERS")
fi
if [[ -n "$NUM_TRAIN_STEPS" ]]; then
  TRAIN_ARGS+=("--num-train-steps" "$NUM_TRAIN_STEPS")
fi
if [[ -n "$SAVE_INTERVAL" ]]; then
  TRAIN_ARGS+=("--save-interval" "$SAVE_INTERVAL")
fi
if [[ -n "$LOG_INTERVAL" ]]; then
  TRAIN_ARGS+=("--log-interval" "$LOG_INTERVAL")
fi
if [[ -n "$CHECKPOINT_BASE_DIR" ]]; then
  TRAIN_ARGS+=("--checkpoint-base-dir" "$CHECKPOINT_BASE_DIR")
fi

if [[ ! -d "$VIS_ROOT" ]]; then
    echo "[ERROR] VIS_ROOT does not exist: $VIS_ROOT" >&2
    exit 1
fi

if pgrep -f "train.py $CONFIG_NAME" >/dev/null; then
    echo "[ERROR] training already running for $CONFIG_NAME" >&2
    exit 1
fi

if pgrep -f "dynamic_dataset_train.sh" >/dev/null; then
    echo "[ERROR] dynamic watcher already running" >&2
    exit 1
fi

export KAI0_DATA_ROOT="${KAI0_DATA_ROOT:-$KAI0_DIR}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$KAI0_DIR/.cache/openpi}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p "$(dirname "$TRAIN_LOG")" "$(dirname "$WATCHER_LOG")"

cd "$REPO_ROOT"
if [[ "$REUSE_MIX" == "1" ]]; then
  log "step 1/6: validate existing mixed dataset at $MIX_ROOT"
  validate_existing_mix "$MIX_ROOT"
  log "step 2/6: reuse existing mixed dataset, skip rebuild"
  log "step 3/6: reuse existing episodes_stats and norm_stats"
else
  log "step 1/6: build mixed dataset into $MIX_ROOT"
  ./kai0/.venv/bin/python train_scripts/data/build_task_a_mixed.py \
    --vis-root "$VIS_ROOT" \
    --old-root "$OLD_ROOT" \
    --out-root "$MIX_ROOT" \
    --val-size "$VAL_SIZE" \
    --force

  log "step 2/6: generate episodes_stats"
  ./kai0/.venv/bin/python train_scripts/data/generate_episodes_stats.py "$MIX_ROOT/base"
  ./kai0/.venv/bin/python train_scripts/data/generate_episodes_stats.py "$MIX_ROOT/val"

  log "step 3/6: compute norm_stats for $CONFIG_NAME"
  cd "$KAI0_DIR"
  ./.venv/bin/python scripts/compute_norm_states_fast.py \
    --config-name "$CONFIG_NAME"
fi

log "step 4/6: start first training run"
cd "$KAI0_DIR"
: > "$TRAIN_LOG"
if [[ "$ATTACH_LOGS" == "1" ]]; then
  ./.venv/bin/python scripts/train.py "${TRAIN_ARGS[@]}" >> "$TRAIN_LOG" 2>&1 &
else
  nohup ./.venv/bin/python scripts/train.py "${TRAIN_ARGS[@]}" >> "$TRAIN_LOG" 2>&1 &
fi
TRAIN_PID=$!
log "training pid=$TRAIN_PID log=$TRAIN_LOG"

log "step 5/6: start dynamic watcher"
cd "$REPO_ROOT"
: > "$WATCHER_LOG"
if [[ "$ATTACH_LOGS" == "1" ]]; then
  env \
    REPO_ROOT="$REPO_ROOT" \
    KAI0_DIR="$KAI0_DIR" \
    VIS_ROOT="$VIS_ROOT" \
    OLD_ROOT="$OLD_ROOT" \
    MIX_ROOT="$MIX_ROOT" \
    CONFIG_NAME="$CONFIG_NAME" \
    EXP_NAME="$EXP_NAME" \
    PROJECT_NAME="$PROJECT_NAME" \
    VAL_SIZE="$VAL_SIZE" \
    BATCH_SIZE="$BATCH_SIZE" \
    NUM_WORKERS="$NUM_WORKERS" \
    NUM_TRAIN_STEPS="$NUM_TRAIN_STEPS" \
    SAVE_INTERVAL="$SAVE_INTERVAL" \
    LOG_INTERVAL="$LOG_INTERVAL" \
    CHECKPOINT_BASE_DIR="$CHECKPOINT_BASE_DIR" \
    KAI0_DATA_ROOT="$KAI0_DATA_ROOT" \
    OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
    TRAIN_LOG="$TRAIN_LOG" \
    WATCHER_LOG="$WATCHER_LOG" \
    DETACH_TRAINING=0 \
    bash train_scripts/launch/dynamic_dataset_train.sh >> "$WATCHER_LOG" 2>&1 &
else
  nohup env \
    REPO_ROOT="$REPO_ROOT" \
    KAI0_DIR="$KAI0_DIR" \
    VIS_ROOT="$VIS_ROOT" \
    OLD_ROOT="$OLD_ROOT" \
    MIX_ROOT="$MIX_ROOT" \
    CONFIG_NAME="$CONFIG_NAME" \
    EXP_NAME="$EXP_NAME" \
    PROJECT_NAME="$PROJECT_NAME" \
    VAL_SIZE="$VAL_SIZE" \
    BATCH_SIZE="$BATCH_SIZE" \
    NUM_WORKERS="$NUM_WORKERS" \
    NUM_TRAIN_STEPS="$NUM_TRAIN_STEPS" \
    SAVE_INTERVAL="$SAVE_INTERVAL" \
    LOG_INTERVAL="$LOG_INTERVAL" \
    CHECKPOINT_BASE_DIR="$CHECKPOINT_BASE_DIR" \
    KAI0_DATA_ROOT="$KAI0_DATA_ROOT" \
    OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
    TRAIN_LOG="$TRAIN_LOG" \
    WATCHER_LOG="$WATCHER_LOG" \
    DETACH_TRAINING=1 \
    bash train_scripts/launch/dynamic_dataset_train.sh >> "$WATCHER_LOG" 2>&1 &
fi
WATCHER_PID=$!
log "watcher pid=$WATCHER_PID log=$WATCHER_LOG"

log "step 6/6: done"
echo "TRAIN_PID=$TRAIN_PID"
echo "WATCHER_PID=$WATCHER_PID"
echo "TRAIN_LOG=$TRAIN_LOG"
echo "WATCHER_LOG=$WATCHER_LOG"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "EXP_NAME=$EXP_NAME"
echo "PROJECT_NAME=$PROJECT_NAME"
echo "BATCH_SIZE=${BATCH_SIZE:-<config-default>}"
echo "NUM_WORKERS=${NUM_WORKERS:-<config-default>}"
echo "NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-<config-default>}"
echo "SAVE_INTERVAL=${SAVE_INTERVAL:-<config-default>}"
echo "LOG_INTERVAL=${LOG_INTERVAL:-<config-default>}"
echo "CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-<config-default>}"
echo "ATTACH_LOGS=$ATTACH_LOGS"

if [[ "$ATTACH_LOGS" == "1" ]]; then
  cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    if kill -0 "$WATCHER_PID" 2>/dev/null; then
      kill "$WATCHER_PID" 2>/dev/null || true
    fi
    pkill -f "train.py $CONFIG_NAME" 2>/dev/null || true
    exit "$exit_code"
  }
  trap cleanup EXIT INT TERM

  echo "ATTACHED_LOGS: tail -F $TRAIN_LOG $WATCHER_LOG"
  tail -n +1 -F "$TRAIN_LOG" "$WATCHER_LOG" &
  TAIL_PID=$!
  wait "$TAIL_PID"
fi
