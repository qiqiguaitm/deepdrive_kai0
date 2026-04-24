#!/bin/bash
###############################################################################
# Dynamic-dataset training wrapper (Task_A mixed).
#
# Watches visrobot01/KAI0/Task_A for new complete episodes. When count grows
# beyond what's already in the mixed dataset, it:
#   1) kills the currently-running training process,
#   2) rebuilds the mixed dataset (build_task_a_mixed.py --force),
#   3) regenerates episodes_stats + norm_stats,
#   4) restarts the same experiment with --resume (keeps step/opt/EMA state).
#
# Training is launched OUTSIDE this script (initial run uses --resume, which auto
# falls back to weight_loader if no ckpts exist). This script only handles the
# resume cycle triggered by data grow. NEVER use --overwrite — it rmtrees the
# entire exp dir and wipes ALL step ckpts irreversibly.
#
# Deploy on gf1 and run in background:
#   nohup /tmp/dynamic_dataset_train.sh > /tmp/dyn_train.log 2>&1 &
#
# Stop: kill the pid printed by 'echo pid=$!' (training is NOT killed by this).
###############################################################################
set -u

# ── paths ──
VIS_ROOT="/vePFS/visrobot01/KAI0/Task_A"
MIX_ROOT="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A_mixed_gf1"
BUILD_PY="/vePFS/tim/workspace/deepdive_kai0/train_scripts/data/build_task_a_mixed.py"
GEN_STATS="/vePFS/tim/workspace/deepdive_kai0/train_scripts/data/generate_episodes_stats.py"
VENV="/home/tim/workspace/deepdive_kai0/kai0/.venv/bin/python"
KAI0_DIR="/vePFS/tim/workspace/deepdive_kai0/kai0"
NORM_SCRIPT="$KAI0_DIR/scripts/compute_norm_states_fast.py"
TRAIN_LOG="/tmp/train_taska_mixed.log"
WATCHER_LOG="/tmp/dyn_train.log"

# ── experiment ──
CONFIG_NAME="pi05_flatten_fold_mixed_visrobot01"
EXP_NAME="mixed_visrobot01_v1"
VAL_SIZE=21

# ── behaviour ──
POLL_SEC=30                  # log poll interval
MIN_REBUILD_INTERVAL=900     # ≥15 min between rebuilds (avoid flapping)
MIN_NEW_EPS=3                # only rebuild if ≥3 new episodes available

LAST_REBUILD_TS=0

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$WATCHER_LOG" ; }

count_complete_vis() {
    # Returns total visrobot01 complete episodes (parquet + 3 cams)
    local total=0
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
    echo "$total"
}

used_vis_count() {
    # Read manifest: N_train_per_source + N_val_per_source = vis used
    [ -f "$MIX_ROOT/manifest.json" ] || { echo 0; return; }
    python3 -c "
import json
m = json.load(open('$MIX_ROOT/manifest.json'))
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
    "$VENV" "$BUILD_PY" --val-size "$VAL_SIZE" --force >> "$WATCHER_LOG" 2>&1
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
    ( cd /vePFS/tim/workspace/deepdive_kai0 \
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
    (
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

        cd "$KAI0_DIR"
        echo "[resume] === RESUME $(date) ===" >> "$TRAIN_LOG"
        nohup .venv/bin/python scripts/train.py "$CONFIG_NAME" \
            --exp_name="$EXP_NAME" \
            --resume >> "$TRAIN_LOG" 2>&1 &
        echo "[resume] started PID=$!" >> "$TRAIN_LOG"
    )
    LAST_REBUILD_TS=$(date +%s)
    log "<<< rebuild+resume cycle complete"
}

# ── main loop ──
log "=========================================="
log "dynamic dataset watcher started"
log "poll=${POLL_SEC}s  min-rebuild-interval=${MIN_REBUILD_INTERVAL}s"
log "min-new-eps=${MIN_NEW_EPS}"
log "config=${CONFIG_NAME} exp=${EXP_NAME}"
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
        log "eval@step=${latest_eval_step}  visrobot01 complete=${cur}  in-dataset=${used}  new=${diff}"

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
