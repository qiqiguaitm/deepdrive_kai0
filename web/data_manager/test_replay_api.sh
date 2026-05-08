#!/bin/bash
# P2 Replay API 快速测试
# 用法:
#   ./test_replay_api.sh preflight                        # dry-run, 看姿态偏差等
#   ./test_replay_api.sh execute                          # 真发 (autonomy 必须起着 + admin)
#   ./test_replay_api.sh progress                         # 当前进度
#   ./test_replay_api.sh watch                            # 1Hz 轮询进度
#   ./test_replay_api.sh stop                             # 停止
#
# 默认拉 Task_A/base/kai0_official_base/episode_104. 改 EPISODE_ID 选别条.
set -e
BASE=http://localhost:8787
TASK="${TASK:-Task_A}"
SUBSET="${SUBSET:-base}"
DATE="${DATE:-kai0_official_base}"
EPISODE_ID="${EPISODE_ID:-104}"
RATE="${RATE:-1.0}"
ROLE="${ROLE:-admin}"

BODY=$(printf '{"task":"%s","subset":"%s","date":"%s","episode_id":%d,"rate":%s}' \
    "$TASK" "$SUBSET" "$DATE" "$EPISODE_ID" "$RATE")

case "${1:-preflight}" in
    preflight) curl -sS -X POST "$BASE/api/replay/preflight" -H 'Content-Type: application/json' -d "$BODY" | python3 -m json.tool ;;
    execute)   curl -sS -X POST "$BASE/api/replay/execute"   -H 'Content-Type: application/json' -H "X-Role: $ROLE" -d "$BODY" | python3 -m json.tool ;;
    stop)      curl -sS -X POST "$BASE/api/replay/stop"      -H "X-Role: $ROLE" | python3 -m json.tool ;;
    progress)  curl -sS "$BASE/api/replay/progress" | python3 -m json.tool ;;
    watch)     while true; do clear; curl -sS "$BASE/api/replay/progress" | python3 -m json.tool 2>/dev/null; sleep 1; done ;;
    *)         echo "Usage: $0 [preflight|execute|stop|progress|watch]"; exit 1 ;;
esac
