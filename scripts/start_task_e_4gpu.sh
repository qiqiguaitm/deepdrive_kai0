#!/bin/bash
# Launch the Task E 4-GPU differentiated experiment matrix.
#   GPU0 = v8 (pi05_base + mirror-only)
#   GPU1 = v3 (kai0_mixed_1 + base)
#   GPU2 = v4 (pi05_base + full-aug)
#   GPU3 = v5 (kai0_mixed_1 + full-aug)
#
# All 4 use: abs action + freeze PaliGemma (action expert only) + batch=4 + 15k steps + inline eval.
# Usage:  ./scripts/start_task_e_4gpu.sh [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="echo [dry-run]"

# --- guard against overlapping runs ---
if pgrep -f "scripts/train.py pi05_stand_box_" >/dev/null; then
  echo "[WARN] existing kai0 train.py processes detected:"
  pgrep -fa "scripts/train.py pi05_stand_box_"
  echo "Kill them first (pkill -f 'scripts/train.py pi05_stand_box_') or pass --force (unimpl.)"
  exit 2
fi

# --- experiments: config_name | exp_name | gpu ---
EXPERIMENTS=(
  "pi05_stand_box_mirror    v8_pi05_mirror 0"
  "pi05_stand_box_kai0init  v3_kai0_base   1"
  "pi05_stand_box_aug       v4_pi05_aug    2"
  "pi05_stand_box_kai0_aug  v5_kai0_aug    3"
)

cd "$PROJECT_ROOT"

echo "╭──── Task E 4-GPU parallel launch ────"
echo "│ project_root: $PROJECT_ROOT"
echo "│ script:       start_train.sh (inline eval on)"
echo "│ experiments:"
for entry in "${EXPERIMENTS[@]}"; do
  read -r cfg exp gpu <<<"$entry"
  printf "│   GPU%s  %-28s %s\n" "$gpu" "$cfg" "$exp"
done
echo "╰───────────────────────────────────────"

for entry in "${EXPERIMENTS[@]}"; do
  read -r cfg exp gpu <<<"$entry"
  $DRY ./scripts/start_train.sh "$cfg" "$exp" "$gpu"
done

echo ""
echo "all 4 launched. follow logs:"
echo "  tail -f logs/train_{v3_kai0_base,v4_pi05_aug,v5_kai0_aug,v8_pi05_mirror}.log"
echo "stop all:  pkill -f 'scripts/train.py pi05_stand_box_'"
