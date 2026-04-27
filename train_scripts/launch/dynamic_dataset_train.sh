#!/bin/bash
###############################################################################
# Dynamic-dataset training wrapper (Task_A mixed).
#
# Watches a Task_A source root for new complete episodes. Supported layouts:
#   1) dated dynamic layout: <root>/2026-xx-xx/{base,dagger}
#   2) flat official layout: <root>/{base,dagger,advantage}
# When count grows
# beyond what's already in the mixed dataset, it:
#   1) kills the currently-running training process,
#   2) rebuilds the mixed dataset (build_task_a_mixed.py --force),
#   3) regenerates episodes_stats + norm_stats,
#   4) restarts the same experiment with --resume (keeps step/opt/EMA state).
#
# Training is launched OUTSIDE this script. This script only handles the
# resume cycle triggered by data growth. If you need an initial fresh run,
# keep it user-controlled. NEVER use --overwrite casually: it can remove the
# entire experiment dir and wipe step checkpoints irreversibly.
#
# Deploy on gf1 and run in background:
#   nohup /tmp/dynamic_dataset_train.sh > /tmp/dyn_train.log 2>&1 &
#
# Stop: kill the pid printed by 'echo pid=$!' (training is NOT killed by this).
###############################################################################
set -u

# ── paths (env-overridable) ──
REPO_ROOT="${REPO_ROOT:-/vePFS/tim/workspace/deepdive_kai0}"
KAI0_DIR="${KAI0_DIR:-$REPO_ROOT/kai0}"
VIS_ROOT="${VIS_ROOT:-/vePFS/visrobot01/KAI0/Task_A}"
OLD_ROOT="${OLD_ROOT:-$KAI0_DIR/data/Task_A}"
MIX_ROOT="${MIX_ROOT:-$KAI0_DIR/data/Task_A_mixed_gf1}"
BUILD_PY="${BUILD_PY:-$REPO_ROOT/train_scripts/data/build_task_a_mixed.py}"
GEN_STATS="${GEN_STATS:-$REPO_ROOT/train_scripts/data/generate_episodes_stats.py}"
VENV="${VENV:-$KAI0_DIR/.venv/bin/python}"
NORM_SCRIPT="$KAI0_DIR/scripts/compute_norm_states_fast.py"
TRAIN_LOG="${TRAIN_LOG:-/tmp/train_taska_mixed.log}"
WATCHER_LOG="${WATCHER_LOG:-/tmp/dyn_train.log}"
DETACH_TRAINING="${DETACH_TRAINING:-1}"

# ── experiment ──
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

# ── behaviour ──
POLL_SEC=30                  # log poll interval
MIN_REBUILD_INTERVAL=900     # ≥15 min between rebuilds (avoid flapping)
MIN_NEW_EPS=3                # only rebuild if ≥3 new episodes available

LAST_REBUILD_TS=0

log() { echo "[$(date +%H:%M:%S)] $*" ; }

build_train_args() {
    TRAIN_ARGS=("$CONFIG_NAME" "--exp_name=$EXP_NAME" "--project-name=$PROJECT_NAME" "--resume")
    if [ -n "$BATCH_SIZE" ]; then
        TRAIN_ARGS+=("--batch-size=$BATCH_SIZE")
    fi
    if [ -n "$NUM_WORKERS" ]; then
        TRAIN_ARGS+=("--num-workers=$NUM_WORKERS")
    fi
    if [ -n "$NUM_TRAIN_STEPS" ]; then
        TRAIN_ARGS+=("--num-train-steps=$NUM_TRAIN_STEPS")
    fi
    if [ -n "$SAVE_INTERVAL" ]; then
        TRAIN_ARGS+=("--save-interval=$SAVE_INTERVAL")
    fi
    if [ -n "$LOG_INTERVAL" ]; then
        TRAIN_ARGS+=("--log-interval=$LOG_INTERVAL")
    fi
    if [ -n "$CHECKPOINT_BASE_DIR" ]; then
        TRAIN_ARGS+=("--checkpoint-base-dir=$CHECKPOINT_BASE_DIR")
    fi
}

launch_training() {
    build_train_args
    (
        export PATH=/home/tim/miniconda3/bin:/home/tim/.local/bin:$PATH
        export PYTHONUNBUFFERED=1
        export KAI0_DATA_ROOT="${KAI0_DATA_ROOT:-$KAI0_DIR}"
        export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/vePFS/tim/workspace/openpi_cache}"
        export PYTORCH_CKPT_BASE=/vePFS/tim/workspace/openpi_cache/modelscope_cache/lerobot
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
        export HF_DATASETS_CACHE=/home/tim/.cache/huggingface/datasets
        export WANDB_MODE=offline
        export LD_LIBRARY_PATH=/home/tim/miniconda3/lib:/home/tim/.cuda_compat:/usr/local/cuda-12.8/targets/x86_64-linux/lib
        for d in /home/tim/.kai0_venv/lib/python3.11/site-packages/nvidia/*/lib; do
            export LD_LIBRARY_PATH=$d:$LD_LIBRARY_PATH
        done

        cd "$KAI0_DIR"
        echo "[resume] === RESUME $(date) ===" >> "$TRAIN_LOG"
        if [ "$DETACH_TRAINING" = "1" ]; then
            nohup .venv/bin/python scripts/train.py "${TRAIN_ARGS[@]}" >> "$TRAIN_LOG" 2>&1 &
        else
            .venv/bin/python scripts/train.py "${TRAIN_ARGS[@]}" >> "$TRAIN_LOG" 2>&1 &
        fi
        echo "[resume] started PID=$!" >> "$TRAIN_LOG"
    )
}

count_complete_vis() {
    # Returns total source-root complete episodes (parquet + 3 cams)
    local total=0
    if [ -d "$VIS_ROOT/base" ] || [ -d "$VIS_ROOT/dagger" ] || [ -d "$VIS_ROOT/advantage" ]; then
        for kind in base dagger advantage; do
            local kd="$VIS_ROOT/$kind"
            [ -d "$kd/data" ] || continue
            local pq th hl hr n
            pq=$(find "$kd/data" -type f -name 'episode_*.parquet' 2>/dev/null | sed 's#.*/##' | sed "s/\\.parquet$//" | sort -u)
            th=$(find "$kd/videos" -type f -path '*/observation.images.top_head/*.mp4' 2>/dev/null | sed 's#.*/##' | sed "s/\\.mp4$//" | sort -u)
            hl=$(find "$kd/videos" -type f -path '*/observation.images.hand_left/*.mp4' 2>/dev/null | sed 's#.*/##' | sed "s/\\.mp4$//" | sort -u)
            hr=$(find "$kd/videos" -type f -path '*/observation.images.hand_right/*.mp4' 2>/dev/null | sed 's#.*/##' | sed "s/\\.mp4$//" | sort -u)
            [ -z "$pq" ] && continue
            n=$(comm -12 <(echo "$pq") <(echo "$th") | comm -12 - <(echo "$hl") | comm -12 - <(echo "$hr") | grep -c episode_)
            total=$((total + n))
        done
    else
        for d in "$VIS_ROOT"/2026-*/; do
            [ -d "$d" ] || continue
            for kind in base dagger; do
                local kd="${d}${kind}"
                [ -d "$kd/data/chunk-000" ] || continue
                local pq th hl hr n
                pq=$(ls "$kd/data/chunk-000/" 2>/dev/null | sed "s/\\.parquet$//" | sort -u)
                th=$(ls "$kd/videos/chunk-000/top_head/" 2>/dev/null | sed "s/\\.mp4$//" | sort -u)
                hl=$(ls "$kd/videos/chunk-000/hand_left/" 2>/dev/null | sed "s/\\.mp4$//" | sort -u)
                hr=$(ls "$kd/videos/chunk-000/hand_right/" 2>/dev/null | sed "s/\\.mp4$//" | sort -u)
                [ -z "$pq" ] && continue
                n=$(comm -12 <(echo "$pq") <(echo "$th") | comm -12 - <(echo "$hl") | comm -12 - <(echo "$hr") | grep -c episode_)
                total=$((total + n))
            done
        done
    fi
    echo "$total"
}

used_vis_count() {
    # Read manifest and sum buckets that originate from the watched source root.
    [ -f "$MIX_ROOT/manifest.json" ] || { echo 0; return; }
    python3 -c "
import json
m = json.load(open('$MIX_ROOT/manifest.json'))
counts = m.get('source_train_val_counts')
if isinstance(counts, dict):
    total = 0
    for name, c in counts.items():
        if str(name).startswith(('visroot/', 'task_a_root/')):
            total += int(c.get('train', 0)) + int(c.get('val', 0))
    print(total)
else:
    print(m.get('N_train_per_source', 0) + m.get('N_val_per_source', 0))
"
}

do_rebuild_and_resume() {
    log ">>> rebuild+resume cycle start"

    # 1. Kill current training
    if pgrep -f "train.py $CONFIG_NAME" > /dev/null; then
        log "step 1/5: killing training process..."
        pkill -f "train.py $CONFIG_NAME"
        sleep 8
        if pgrep -f "train.py $CONFIG_NAME" > /dev/null; then
            log "[warn] training still alive after 8s, sending SIGKILL"
            pkill -9 -f "train.py $CONFIG_NAME"
            sleep 3
        fi
        log "training killed"
    else
        log "step 1/5: no training running (nothing to kill)"
    fi

    # 2. Rebuild mixed dataset
    log "step 2/5: rebuilding mixed dataset..."
    "$VENV" "$BUILD_PY" --vis-root "$VIS_ROOT" --old-root "$OLD_ROOT" --out-root "$MIX_ROOT" --val-size "$VAL_SIZE" --force >> "$WATCHER_LOG" 2>&1
    if [ $? -ne 0 ]; then
        log "[ERROR] build failed, aborting resume (leaving training stopped)"
        return 1
    fi

    # 3. Regenerate episodes_stats for both splits
    log "step 3/5: regenerating episodes_stats..."
    "$VENV" "$GEN_STATS" "$MIX_ROOT/base" >> "$WATCHER_LOG" 2>&1 || { log "[ERROR] base stats failed"; return 1; }
    "$VENV" "$GEN_STATS" "$MIX_ROOT/val"  >> "$WATCHER_LOG" 2>&1 || { log "[ERROR] val stats failed"; return 1; }

    # 4. Recompute norm_stats
    log "step 4/5: recomputing norm_stats..."
    ( cd "$REPO_ROOT" \
      && source ./setup_env.sh >/dev/null 2>&1 \
      && export KAI0_DATA_ROOT OPENPI_DATA_HOME PYTORCH_CKPT_BASE \
      && cd kai0 \
      && "$VENV" "$NORM_SCRIPT" --config-name "$CONFIG_NAME" ) >> "$WATCHER_LOG" 2>&1
    if [ $? -ne 0 ]; then
        log "[ERROR] norm_stats failed"
        return 1
    fi

    # 5. Launch --resume
    log "step 5/5: launching --resume..."
    launch_training
    LAST_REBUILD_TS=$(date +%s)
    log "<<< rebuild+resume cycle complete"
}

# ── main loop ──
log "=========================================="
log "dynamic dataset watcher started"
log "poll=${POLL_SEC}s  min-rebuild-interval=${MIN_REBUILD_INTERVAL}s"
log "min-new-eps=${MIN_NEW_EPS}"
log "config=${CONFIG_NAME} exp=${EXP_NAME} project=${PROJECT_NAME}"
log "batch_size=${BATCH_SIZE:-<config-default>} num_workers=${NUM_WORKERS:-<config-default>} num_train_steps=${NUM_TRAIN_STEPS:-<config-default>} save_interval=${SAVE_INTERVAL:-<config-default>} log_interval=${LOG_INTERVAL:-<config-default>} checkpoint_base_dir=${CHECKPOINT_BASE_DIR:-<config-default>}"
log "=========================================="

last_eval_step=-1

while true; do
    # Is training still running?
    if ! pgrep -f "train.py $CONFIG_NAME" > /dev/null; then
        log "training not running, watcher will exit"
        break
    fi

    # Look for new inline-eval line
    latest_eval_step=$(grep 'inline-eval' "$TRAIN_LOG" 2>/dev/null | tail -1 | grep -oE 'step=[0-9]+' | head -1 | cut -d= -f2)
    [ -z "$latest_eval_step" ] && latest_eval_step=0

    if [ "$latest_eval_step" -gt "$last_eval_step" ]; then
        last_eval_step="$latest_eval_step"
        cur=$(count_complete_vis)
        used=$(used_vis_count)
        diff=$((cur - used))
        log "eval@step=${latest_eval_step}  source_complete=${cur}  in-dataset=${used}  new=${diff}"

        now=$(date +%s)
        since=$((now - LAST_REBUILD_TS))

        if [ "$diff" -ge "$MIN_NEW_EPS" ] && [ "$since" -ge "$MIN_REBUILD_INTERVAL" ]; then
            log "*** trigger: +${diff} new eps available, last rebuild ${since}s ago ***"
            do_rebuild_and_resume
        elif [ "$diff" -ge "$MIN_NEW_EPS" ]; then
            log "new data ready but within rebuild cooldown (${since}s < ${MIN_REBUILD_INTERVAL}s)"
        fi
    fi

    sleep "$POLL_SEC"
done

log "watcher exit"
