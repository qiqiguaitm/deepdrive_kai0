#!/bin/bash
###############################################################################
# 启动 V1 Triton WebSocket policy serve (:8002, B4 Phase 2)
#
# 与 start_server_xla_cache.sh (JAX :8000) 并列, 互不冲突.
# policy_inference_node 用 --mode=websocket -p port:=8002 连接.
#
# Usage:
#   ./scripts/start_serve_v1.sh                     # 默认 task_a_mix ckpt
#   ./scripts/start_serve_v1.sh --phase 1           # 跳过 state encoding (固定 prompt)
#   ./scripts/start_serve_v1.sh --port 8003 --pkl <path>
###############################################################################
set -eo pipefail

REPO=/data1/tim/workspace/deepdive_kai0
PKL_DEFAULT=$REPO/optimize/results/task_a_mix_b6000_p1200_v1_p200.pkl
NORM_DEFAULT=/data1/DATA_IMP/checkpoints/task_a_mix_b6000_p1200_mixed_1_step49999/assets/mix_b6000_p1200/norm_stats.json
TOK_DEFAULT=$REPO/openpi_cache/big_vision/paligemma_tokenizer.model
PORT_DEFAULT=8002
PROMPT_DEFAULT="Flatten and fold the cloth"

PKL=$PKL_DEFAULT
NORM=$NORM_DEFAULT
TOK=$TOK_DEFAULT
PORT=$PORT_DEFAULT
PROMPT=$PROMPT_DEFAULT
PHASE=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pkl)     PKL="$2"; shift 2 ;;
    --norm)    NORM="$2"; shift 2 ;;
    --norm-stats) NORM="$2"; shift 2 ;;
    --tokenizer) TOK="$2"; shift 2 ;;
    --port)    PORT="$2"; shift 2 ;;
    --prompt|--default-prompt) PROMPT="$2"; shift 2 ;;
    --phase)   PHASE="$2"; shift 2 ;;
    --phase=*) PHASE="${1#*=}"; shift ;;
    -h|--help)
      grep '^#' "$0" | head -16
      exit 0
      ;;
    *) echo "[WARN] unknown arg: $1" >&2; shift ;;
  esac
done

# 路径检查
for f in "$PKL" "$NORM"; do
  if [ ! -f "$f" ]; then
    echo "[FAIL] not found: $f" >&2
    exit 1
  fi
done
if [ "$PHASE" = "2" ] && [ ! -f "$TOK" ]; then
  echo "[FAIL] tokenizer not found (Phase 2 needs it): $TOK" >&2
  exit 1
fi

# 关闭 conda; .venv_5090_trt 需要的环境
eval "$(conda shell.bash hook 2>/dev/null)"; conda deactivate 2>/dev/null || true

VENV=$REPO/kai0/.venv_5090_trt
export CUDA_VISIBLE_DEVICES=0
unset http_proxy https_proxy

echo "=== Launching V1 Triton serve on :${PORT} (Phase ${PHASE}) ==="
echo "    pkl:        $PKL"
echo "    norm-stats: $NORM"
[ "$PHASE" = "2" ] && echo "    tokenizer:  $TOK"
echo "    prompt:     '$PROMPT'"
echo ""

TOK_ARG=""
[ "$PHASE" = "2" ] && TOK_ARG="--tokenizer $TOK"

exec $VENV/bin/python $REPO/kai0/scripts/serve_policy_v1.py \
  --pkl "$PKL" \
  --norm-stats "$NORM" \
  $TOK_ARG \
  --default-prompt "$PROMPT" \
  --port "$PORT" \
  --num-views 3 \
  --chunk-size 50 \
  --action-dim 14 \
  --state-dim 14
