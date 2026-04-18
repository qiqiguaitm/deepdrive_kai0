#!/bin/bash
# π0.7-style advantage 数据集预处理
# 从 data/Task_A/advantage/ 生成 data/Task_A/advantage_q5/
#   - n_slices=5 离散化
#   - stage-nums=2（Task_A flat/fold 独立计算 percentile）
#   - tasks.jsonl 格式改为 π0.7 风格 "Quality: 1/5" ~ "Quality: 5/5"
#   - 复用原 norm_stats.json（state/action 分布不变）
#
# 用法: bash prepare_advantage_q5.sh
# 只需跑一次（幂等：已存在 $DST 会先删再建）

set -euo pipefail

SRC=/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/advantage
DST=/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/advantage_q5
KAI0_ROOT=/vePFS/tim/workspace/deepdive_kai0/kai0
DISCRETIZE=$KAI0_ROOT/stage_advantage/annotation/discretize_advantage.py
PY=$KAI0_ROOT/.venv/bin/python3

if [ ! -d "$SRC" ]; then
  echo "ERROR: source dataset missing: $SRC"
  exit 1
fi

if [ -d "$DST" ] || [ -L "$DST/videos" ]; then
  echo "⚠️  existing $DST will be removed (parquet/meta). videos symlink kept if possible."
  rm -rf "$DST"
fi

echo "[1/5] Copying parquets + meta (videos symlinked to save ~60GB)"
mkdir -p "$DST"
cp -r "$SRC/meta" "$DST/"
cp -r "$SRC/data" "$DST/"     # ~500MB parquets (task_index will be rewritten)
ln -s "$SRC/videos" "$DST/videos"

echo "[2/5] Copy norm_stats (state/action distributions unchanged)"
if [ -f "$SRC/norm_stats.json" ]; then
  cp "$SRC/norm_stats.json" "$DST/norm_stats.json"
  echo "    ✓ norm_stats.json copied"
else
  echo "    ⚠️  $SRC/norm_stats.json missing — you MUST run compute_norm_states_fast.py later"
fi

echo "[3/5] Mark as timestamp-validated (skip the 20-min check at first launch)"
touch "$DST/.kai0_ts_validated"

echo "[4/5] Run discretize_advantage.py (n_slices=5, stage-nums=2)"
cd "$KAI0_ROOT/stage_advantage/annotation"
$PY discretize_advantage.py "$DST" \
    --discretion-type n_slices \
    --n-slices 5 \
    --advantage-source absolute_advantage \
    --stage-nums 2

echo "[5/5] Override tasks.jsonl to π0.7 Quality format"
cat > "$DST/meta/tasks.jsonl" <<'EOF'
{"task_index": 0, "task": "Flatten and fold the cloth. Quality: 1/5"}
{"task_index": 1, "task": "Flatten and fold the cloth. Quality: 2/5"}
{"task_index": 2, "task": "Flatten and fold the cloth. Quality: 3/5"}
{"task_index": 3, "task": "Flatten and fold the cloth. Quality: 4/5"}
{"task_index": 4, "task": "Flatten and fold the cloth. Quality: 5/5"}
EOF

echo ""
echo "✅ advantage_q5 ready at $DST"
echo ""
echo "Final task_index distribution (sampled from first parquet):"
$PY -c "
import pandas as pd, glob
fs = sorted(glob.glob('$DST/data/chunk-000/episode_*.parquet'))
df = pd.concat([pd.read_parquet(f) for f in fs[:5]])
print('  first 5 episodes task_index distribution:', df['task_index'].value_counts().sort_index().to_dict())
"
echo ""
echo "tasks.jsonl:"
cat "$DST/meta/tasks.jsonl"
