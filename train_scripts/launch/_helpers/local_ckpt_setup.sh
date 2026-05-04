#!/bin/bash
# Helper: ensure ckpt for a training run goes to /home/tim/local_ckpts/<config>/<exp>/
# instead of the default <KAI0_DATA_ROOT>/checkpoints/<config>/<exp>/
# Sets up a per-exp symlink at the workspace path, idempotent.
#
# Usage (in launcher):
#   source /home/tim/workspace/deepdive_kai0/train_scripts/launch/_helpers/local_ckpt_setup.sh \
#     pi05_flatten_fold_mix_b6000_p1200_init_mixed_1 \
#     task_a_mix_base6000_pure1200_new_norm_base_mixed_1
#
# Or explicitly call: setup_local_ckpt <config> <exp> [kai0_root]

setup_local_ckpt() {
    local config="$1"
    local exp="$2"
    local kai0_root="${3:-${KAI0_DATA_ROOT:-/home/tim/workspace/deepdive_kai0/kai0}}"
    local local_root="${LOCAL_CKPT_ROOT:-/home/tim/local_ckpts}"

    local local_dir="${local_root}/${config}/${exp}"
    local workspace_dir="${kai0_root}/checkpoints/${config}/${exp}"

    mkdir -p "${local_dir}"
    mkdir -p "$(dirname "${workspace_dir}")"

    # If workspace path is already a real dir, we won't overwrite (caller must move first)
    if [ -d "${workspace_dir}" ] && [ ! -L "${workspace_dir}" ]; then
        echo "[local_ckpt_setup] WARN: ${workspace_dir} is a real dir, not a symlink. Refusing to overwrite."
        echo "[local_ckpt_setup] To migrate: mv ${workspace_dir} ${local_dir} && ln -s ${local_dir} ${workspace_dir}"
        return 1
    fi

    # Idempotently set the symlink
    ln -sfn "${local_dir}" "${workspace_dir}"

    echo "[local_ckpt_setup] OK ${workspace_dir} -> ${local_dir}"
}

# If sourced with args, auto-call
if [ $# -ge 2 ]; then
    setup_local_ckpt "$@"
fi
