#!/bin/bash
# Launch the Task E 4-GPU differentiated experiment matrix with staggered ckpt restore.
#
# Why staggered: v3+v5 share the 25 GB Task_A/mixed_1/params ckpt; v4+v8 share the 12 GB
# pi05_base/params. Two procs concurrently cold-reading a 25 GB file triggered silent
# OOM-kill (v5 died at restore twice). Fix: pre-warm into page cache, then launch in two
# waves so no pair reads the same source cold.
#
#   GPU0 = v8 (pi05_base + mirror-only)
#   GPU1 = v3 (kai0_mixed_1 + base)
#   GPU2 = v4 (pi05_base + full-aug)
#   GPU3 = v5 (kai0_mixed_1 + full-aug)
#
# Usage:  ./scripts/start_task_e_4gpu.sh [--dry-run] [--no-prewarm]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DRY=""
PREWARM=1
for arg in "$@"; do
  case "$arg" in
    --dry-run)      DRY="echo [dry-run]" ;;
    --no-prewarm)   PREWARM=0 ;;
  esac
done

# --- guard against overlapping runs ---
if pgrep -f "scripts/train.py pi05_stand_box_" >/dev/null; then
  echo "[WARN] existing kai0 train.py processes detected:"
  pgrep -fa "scripts/train.py pi05_stand_box_"
  echo "Kill them first (pkill -f 'scripts/train.py pi05_stand_box_')"
  exit 2
fi

# Each launch wave = (config, exp_name, gpu). Wave 1 and 2 use the same 2 ckpt sources
# but one proc per source at a time, avoiding 2× concurrent cold reads.
WAVE1=(
  "pi05_stand_box_kai0init  v3_kai0_base   1"   # kai0_mixed_1 (25 GB)
  "pi05_stand_box_aug       v4_pi05_aug    2"   # pi05_base    (12 GB)
)
WAVE2=(
  "pi05_stand_box_kai0_aug  v5_kai0_aug    3"   # kai0_mixed_1 (hot cache)
  "pi05_stand_box_mirror    v8_pi05_mirror 0"   # pi05_base    (hot cache)
)

KAI0_CKPT="$PROJECT_ROOT/kai0/checkpoints/Task_A/mixed_1/params"
PI05_CKPT="$PROJECT_ROOT/openpi_cache/openpi-assets/checkpoints/pi05_base/params"

cd "$PROJECT_ROOT"

echo "╭──── Task E 4-GPU parallel launch (staggered) ────"
echo "│ project_root: $PROJECT_ROOT"
echo "│ prewarm:      $PREWARM"
echo "│ Wave 1 (t=0):"
for e in "${WAVE1[@]}"; do read -r c n g <<<"$e"; printf "│   GPU%s  %-28s %s\n" "$g" "$c" "$n"; done
echo "│ Wave 2 (after wave-1 restore done):"
for e in "${WAVE2[@]}"; do read -r c n g <<<"$e"; printf "│   GPU%s  %-28s %s\n" "$g" "$c" "$n"; done
echo "╰──────────────────────────────────────────────────"

# --- Pre-warm ckpt sources into OS page cache (concurrent reads later hit cache) ---
if [ "$PREWARM" = "1" ]; then
  echo "[prewarm] reading $KAI0_CKPT into page cache..."
  $DRY bash -c "find '$KAI0_CKPT' -type f -print0 | xargs -0 -P 4 -I{} dd if={} of=/dev/null bs=8M status=none"
  echo "[prewarm] reading $PI05_CKPT into page cache..."
  $DRY bash -c "find '$PI05_CKPT' -type f -print0 | xargs -0 -P 4 -I{} dd if={} of=/dev/null bs=8M status=none"
  $DRY free -g | head -2
fi

# --- Wave 1 ---
echo ""
echo "[wave 1] launching..."
for entry in "${WAVE1[@]}"; do
  read -r cfg exp gpu <<<"$entry"
  $DRY ./scripts/start_train.sh "$cfg" "$exp" "$gpu"
done

# --- Wait for wave 1 ckpt restore to complete ---
if [ -z "$DRY" ]; then
  echo ""
  echo "[wave 1] waiting for ckpt restore (up to 5 min)..."
  deadline=$(( $(date +%s) + 300 ))
  for entry in "${WAVE1[@]}"; do
    read -r _ exp _ <<<"$entry"
    log="$PROJECT_ROOT/logs/train_${exp}.log"
    until grep -q "Finished restoring checkpoint" "$log" 2>/dev/null; do
      [ $(date +%s) -ge "$deadline" ] && { echo "[ERROR] $exp did not finish restore in 5 min"; exit 3; }
      sleep 3
    done
    echo "[wave 1] $exp restored ✅"
  done
fi

# --- Wave 2 ---
echo ""
echo "[wave 2] launching..."
for entry in "${WAVE2[@]}"; do
  read -r cfg exp gpu <<<"$entry"
  $DRY ./scripts/start_train.sh "$cfg" "$exp" "$gpu"
done

echo ""
echo "all 4 launched. follow logs:"
echo "  tail -f logs/train_{v3_kai0_base,v4_pi05_aug,v5_kai0_aug,v8_pi05_mirror}.log"
echo "stop all:  pkill -f 'scripts/train.py pi05_stand_box_'"
