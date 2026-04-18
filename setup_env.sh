#!/usr/bin/env bash
# Source this file before running any train_scripts/ or start_scripts/.
#
#   source setup_env.sh
#
# Overrides:
#   - Set any variable BEFORE sourcing to override the auto-detected default.
#     e.g.  KAI0_DATA_ROOT=/custom/path source setup_env.sh
#   - Or edit the per-host block below.
#
# Env vars exported:
#   KAI0_DATA_ROOT      → base dir of deepdive_kai0/kai0 (holds data/ and local checkpoints/)
#   OPENPI_DATA_HOME    → cache root for `gs://openpi-assets/...` downloads (openpi convention)
#   PYTORCH_CKPT_BASE   → root for ADVANTAGE_TORCH PyTorch pretrained weights (modelscope cache)

_host="$(hostname 2>/dev/null)"

# Profile selection: hostname match first, then filesystem probe, then HOME fallback.
if [[ "$_host" == gf* ]] || [[ -d /vePFS/tim/workspace ]]; then
    _profile=gf
elif [[ "$_host" == sim01 ]] || [[ -d /data1/tim/workspace ]]; then
    _profile=sim01
else
    _profile=default
fi

case "$_profile" in
    gf)
        : "${KAI0_DATA_ROOT:=/vePFS/tim/workspace/deepdive_kai0/kai0}"
        : "${OPENPI_DATA_HOME:=/vePFS/tim/workspace/openpi_cache}"
        : "${PYTORCH_CKPT_BASE:=/vePFS/tim/workspace/openpi_cache/modelscope_cache/lerobot}"
        ;;
    sim01)
        : "${KAI0_DATA_ROOT:=/data1/tim/workspace/deepdive_kai0/kai0}"
        : "${OPENPI_DATA_HOME:=$HOME/.cache/openpi}"
        : "${PYTORCH_CKPT_BASE:=/data1/tim/workspace/openpi_cache/modelscope_cache/lerobot}"
        ;;
    default)
        : "${KAI0_DATA_ROOT:=$HOME/workspace/deepdive_kai0/kai0}"
        : "${OPENPI_DATA_HOME:=$HOME/.cache/openpi}"
        : "${PYTORCH_CKPT_BASE:=$HOME/.cache/openpi/modelscope_cache/lerobot}"
        ;;
esac

export KAI0_DATA_ROOT OPENPI_DATA_HOME PYTORCH_CKPT_BASE

echo "[setup_env] host=$_host profile=$_profile"
unset _host _profile

echo "[setup_env] KAI0_DATA_ROOT=$KAI0_DATA_ROOT"
echo "[setup_env] OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
echo "[setup_env] PYTORCH_CKPT_BASE=$PYTORCH_CKPT_BASE"
