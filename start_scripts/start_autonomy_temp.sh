#!/bin/bash
# DEPRECATED — 不再维护. 用 start_autonomy_from_ckpt.sh 替代.
#
# 历史: 本文件曾是手工 ckpt 实验日志, 注释了几十个 launch 命令 (A/B/C/...).
# 每换一个 ckpt 都得改 src/openpi/training/config.py 加 asset_id, 或换这里的注释.
#
# 现状 (2026-05-08 之后): ckpt 自带 train_config.json sidecar
# ({"base_config_name": ..., "override_asset_id": ...}), 启动只需一行:
#
#   ./start_scripts/start_autonomy_from_ckpt.sh <ckpt_dir> [extra_ros_args...]
#
# 该 launcher 自己读 sidecar + 注入 OPENPI_EXTRA_CONFIG, 不再编辑 config.py.
# Sidecar 由 train_scripts/data/pack_inference_ckpt.py 在打包推理 bundle 时
# 自动生成. 历史不带 sidecar 的 ckpt 可参考 docs/deployment/checkpoints_layout.md
# §"Type A flat bundle" 手工补齐 (~5 KB 额外文件, 不动 12 GB params).
#
# 实验记录请写到 docs/training/<exp>_results.md, 不要再往这里堆.

echo "[DEPRECATED] start_autonomy_temp.sh 已废弃, 用 start_autonomy_from_ckpt.sh:" >&2
echo "  ./start_scripts/start_autonomy_from_ckpt.sh <ckpt_dir>" >&2
exit 1
