#!/bin/bash
# Phase 5-6 pipeline: AE inference → sanity check → discretize → merge → norm_stats
# Prerequisites: dagger_with_stage/ (3457 ep) produced by stage classifier (Phase 4)
# Runs on gf1 (single GPU for AE + CPU for merge/norm)

set -e

DEEPDIVE=/vePFS/tim/workspace/deepdive_kai0
PYTHON=$DEEPDIVE/kai0/.venv/bin/python3
DATA_ROOT=$DEEPDIVE/kai0/data/Task_A
LOG_DIR=$DEEPDIVE/logs

SSH_KEY=/root/.ssh/ssh_worker_rsa_key
GF0_IP=192.168.0.144
GF1_IP=192.168.0.161
SSH_PORT=2222

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

# ─── Step 5a: AE inference on dagger_with_stage ────────────────
echo "[$(date)] Step 5a: AE inference on dagger_with_stage (single GPU on gf1)"
AE_LOG=$LOG_DIR/ae_infer_dagger_${TIMESTAMP}.log
AE_CKPT=$DEEPDIVE/kai0/checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1
AE_STEP=100000

ssh -p $SSH_PORT root@$GF0_IP "ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no root@$GF1_IP '
    export KAI0_DATA_ROOT=$DEEPDIVE/kai0
    export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
    export PYTORCH_CKPT_BASE=$DEEPDIVE/kai0/checkpoints
    cd $DEEPDIVE/kai0
    CUDA_VISIBLE_DEVICES=0 nohup $PYTHON -c \"
import sys
sys.path.insert(0, \\\"$DEEPDIVE/kai0/stage_advantage/annotation\\\")
import eval as ev
ev.MODELS_CONFIG_MAP[\\\"Flatten-Fold\\\"][\\\"KAI0\\\"][\\\"ckpt_dir\\\"] = \\\"$AE_CKPT\\\"
ev.MODELS_CONFIG_MAP[\\\"Flatten-Fold\\\"][\\\"KAI0\\\"][\\\"ckpt_steps\\\"] = $AE_STEP
sys.argv = [\\\"eval.py\\\", \\\"Flatten-Fold\\\", \\\"KAI0\\\", \\\"$DATA_ROOT/dagger_with_stage\\\"]
ev.main()
\" > $AE_LOG 2>&1 &
    echo ae_pid=\$!
'"
echo "AE inference launched. Log: $AE_LOG"
echo "Wait for completion... (poll every 30s)"

# Wait for AE to finish (check if process alive on gf1)
while ssh -p $SSH_PORT root@$GF0_IP "ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no root@$GF1_IP 'pgrep -f \"eval.py\" >/dev/null 2>&1'"; do
    sleep 30
    EP_COUNT=$(ls $DATA_ROOT/dagger_with_stage/data_KAI0_${AE_STEP}/*/episode_*.parquet 2>/dev/null | wc -l)
    echo "  [$(date)] AE progress: $EP_COUNT parquets written"
done
echo "[$(date)] AE inference done."

# ─── Step 5b: Sanity check abs_adv distribution ────────────────
echo "[$(date)] Step 5b: sanity check abs_adv distribution"
$PYTHON -c "
import pyarrow.parquet as pq, numpy as np
from pathlib import Path
for name, path in [
    ('advantage', '$DATA_ROOT/advantage/data'),
    ('dagger',    '$DATA_ROOT/dagger_with_stage/data_KAI0_${AE_STEP}'),
]:
    vals = []
    for p in list(Path(path).rglob('*.parquet'))[:100]:
        try:
            t = pq.read_table(p)
            if 'absolute_advantage' in t.column_names:
                vals.extend(t['absolute_advantage'].to_numpy())
        except Exception: pass
    v = np.array(vals)
    print(f'{name:10s}: n={len(v):>7d}  std={v.std():.4f}  range=[{v.min():.3f}, {v.max():.3f}]  mean={v.mean():+.4f}')
"

# ─── Step 5c: Create dagger_advantage/ dataset and discretize ──
echo "[$(date)] Step 5c: prepare dagger_advantage/ + discretize"
DAGGER_ADV=$DATA_ROOT/dagger_advantage
mkdir -p $DAGGER_ADV
ln -sfn $DATA_ROOT/dagger_with_stage/videos $DAGGER_ADV/videos
cp -rf $DATA_ROOT/dagger_with_stage/meta $DAGGER_ADV/meta
[ -f $DATA_ROOT/dagger_with_stage/norm_stats.json ] && cp -f $DATA_ROOT/dagger_with_stage/norm_stats.json $DAGGER_ADV/norm_stats.json || true
rm -rf $DAGGER_ADV/data
cp -r $DATA_ROOT/dagger_with_stage/data_KAI0_${AE_STEP} $DAGGER_ADV/data

$PYTHON $DEEPDIVE/kai0/stage_advantage/annotation/discretize_advantage.py $DAGGER_ADV \
    --threshold 30 \
    --chunk-size 50 \
    --discretion-type binary \
    --advantage-source absolute_advantage \
    --stage-nums 2

# ─── Step 6: Merge advantage + dagger_advantage → awbc_v2_full ─
echo "[$(date)] Step 6: merge → awbc_v2_full"
AWBC_V2=$DATA_ROOT/awbc_v2_full

$PYTHON $DEEPDIVE/kai0/scripts/merge_lerobot.py \
    --src_paths $DATA_ROOT/advantage $DAGGER_ADV \
    --tgt_path $AWBC_V2 \
    --repo_id awbc_v2_full

# compute_norm_stats
echo "[$(date)] Step 6b: compute_norm_stats_fast"
ssh -p $SSH_PORT root@$GF0_IP "ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no root@$GF1_IP '
    export KAI0_DATA_ROOT=$DEEPDIVE/kai0
    export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
    cd $DEEPDIVE/kai0
    $PYTHON scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_awbc_v2
'"

echo "[$(date)] ✅ Phase 5-6 done. Ready for Phase 7 dual-node training."
