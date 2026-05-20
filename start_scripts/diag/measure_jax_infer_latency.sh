#!/bin/bash
###############################################################################
# Q2: sim01 JAX 模型实际推理延迟测量 (server-side end-to-end RTT, 含 WebSocket)
#
# 利用 policy_inference_node.py:2147 内置的 infer_ms log (`infer XXXms | chunk=...`),
# 从最新 ros2 log 提取 N 次推理时间, 算 P50/P95/P99.
#
# 用法 (在 sim01 上跑):
#   ./start_scripts/diag/measure_jax_infer_latency.sh                 # 自动找最新 ros2 log
#   ./start_scripts/diag/measure_jax_infer_latency.sh <log_file>      # 指定 log 文件
#
# 先决条件: 已跑过 start_autonomy.sh 至少一次 (产生过 infer XXXms log).
# 若无, 先跑 start_autonomy.sh, 等约 30s (允许 ~30-60 次推理), Ctrl-C, 再跑本脚本.
###############################################################################

set -eo pipefail

LOG_DIR="${HOME}/.ros/log"

if [ -n "$1" ]; then
    LOGFILE="$1"
else
    # 自动找最新含 "infer XXXms" 的 ros2 log
    LOGFILE=$(find "${LOG_DIR}" -name "*stdout*" -type f 2>/dev/null \
              | xargs ls -t 2>/dev/null \
              | xargs grep -l "infer [0-9]*ms" 2>/dev/null \
              | head -1)
    if [ -z "$LOGFILE" ]; then
        echo "[FAIL] No ros2 log with 'infer XXXms' found under ${LOG_DIR}"
        echo "       Run start_autonomy.sh first to generate inference traces."
        exit 1
    fi
fi

echo "=== Log file: $LOGFILE ==="
echo "=== Size: $(du -h "$LOGFILE" | cut -f1) ==="
echo ""

# 提取所有 'infer XXXms' 数值, 喂给 python 算分位数
grep -oP 'infer \K\d+(?=ms)' "$LOGFILE" | python3 - <<'PYEOF'
import sys
import numpy as np

vals = []
for line in sys.stdin:
    line = line.strip()
    if line:
        vals.append(int(line))

if not vals:
    print("[FAIL] No 'infer XXXms' entries found in log.")
    sys.exit(1)

vals = np.array(vals)
n = len(vals)
print(f"=== JAX inference latency (server-side end-to-end RTT, n={n}) ===")
print(f"  Mean: {vals.mean():.1f} ms")
print(f"  Std:  {vals.std():.1f} ms")
print(f"  Min:  {vals.min()} ms")
print(f"  P50:  {np.percentile(vals, 50):.1f} ms")
print(f"  P95:  {np.percentile(vals, 95):.1f} ms")
print(f"  P99:  {np.percentile(vals, 99):.1f} ms")
print(f"  Max:  {vals.max()} ms")
print()
print(f"  P95 - P50: {np.percentile(vals,95) - np.percentile(vals,50):.1f} ms (jitter)")
print(f"  P99 - P50: {np.percentile(vals,99) - np.percentile(vals,50):.1f} ms (tail)")
print()
print(f"  V1 Triton baseline (offline 5090 benchmark, raw model): 32.05 ms")
print(f"  PyTorch E max-autotune (offline 5090): 43.5 ms")
print(f"  Expected JAX RTT range: 100-300 ms (含 WebSocket + JAX overhead)")

# 跨阈值判断 (用于决策推理优化路线)
p50 = np.percentile(vals, 50)
print()
print("=== Decision per docs/deployment/realtime_vla_optimization_analysis.md §4.1 ===")
if p50 < 80:
    print(f"  P50 = {p50:.0f} ms → 模型已很快, #6 浅层收益小, 阶段 3 优先级可降低")
elif p50 < 200:
    print(f"  P50 = {p50:.0f} ms → 标准 5090 baseline, #6 (1.5-2×) 拿 50-100ms 收益")
elif p50 < 250:
    print(f"  P50 = {p50:.0f} ms → 接近标准, 但偏慢")
else:
    print(f"  P50 = {p50:.0f} ms → 可能有 cache miss / fp32 残留, #6 收益最大")

if np.percentile(vals,95) - p50 > 100:
    print(f"  P95-P50 = {np.percentile(vals,95) - p50:.0f} ms → 抖动严重, AOT compile 必做")
PYEOF
