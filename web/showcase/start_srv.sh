#!/bin/bash
###############################################################################
# deepdive_kai0 showcase — start/stop/status/restart
#
# Usage:
#   ./start_srv.sh start      # start in background
#   ./start_srv.sh stop
#   ./start_srv.sh restart
#   ./start_srv.sh status
#   ./start_srv.sh logs       # tail server log
#   ./start_srv.sh fg         # run in foreground (dev/debug)
#
# Env:
#   SHOWCASE_PORT (default 8765)
#   SHOWCASE_HOST (default 0.0.0.0)
#   SHOWCASE_VENV (default: $HOME/.venvs/showcase or system python3)
###############################################################################
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$LOG_DIR/server.pid"
LOG_FILE="$LOG_DIR/server.log"

PORT="${SHOWCASE_PORT:-8765}"
HOST="${SHOWCASE_HOST:-0.0.0.0}"

# Prefer dedicated venv if present; otherwise fall back to system python3.
if [ -n "$SHOWCASE_VENV" ] && [ -x "$SHOWCASE_VENV/bin/python" ]; then
    PY="$SHOWCASE_VENV/bin/python"
elif [ -x "$HOME/.venvs/showcase/bin/python" ]; then
    PY="$HOME/.venvs/showcase/bin/python"
else
    PY="$(command -v python3)"
fi

mkdir -p "$LOG_DIR"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1" >&2; }
info() { echo -e "${CYAN}[INFO]${NC} $1"; }

is_running() {
    [ -f "$PID_FILE" ] || return 1
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

cmd_status() {
    if is_running; then
        ok "showcase server running (pid=$(cat "$PID_FILE"), port=$PORT)"
        return 0
    fi
    info "showcase server NOT running"
    return 1
}

cmd_start() {
    if is_running; then
        ok "already running (pid=$(cat "$PID_FILE"))"
        return 0
    fi
    if ! command -v "$PY" >/dev/null; then
        fail "python3 not found ($PY)"
        return 1
    fi
    info "checking deps (fastapi + uvicorn)..."
    if ! "$PY" -c "import fastapi, uvicorn" 2>/dev/null; then
        warn "fastapi/uvicorn missing — run: $PY -m pip install fastapi uvicorn"
        return 1
    fi
    if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
        fail "port $PORT already bound by another process"
        return 1
    fi
    info "starting showcase server on $HOST:$PORT ..."
    nohup setsid "$PY" "$SCRIPT_DIR/server.py" --host "$HOST" --port "$PORT" \
        >>"$LOG_FILE" 2>&1 < /dev/null &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    sleep 1
    if ! kill -0 "$pid" 2>/dev/null; then
        fail "server died immediately. Check $LOG_FILE"
        tail -20 "$LOG_FILE" || true
        return 1
    fi
    # Wait for /api/health up to 8s
    local i
    for i in 1 2 3 4 5 6 7 8; do
        if curl -fsS "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
            ok "showcase up at http://$HOST:$PORT/ (pid=$pid)"
            return 0
        fi
        sleep 1
    done
    warn "server pid=$pid running but /api/health not responding yet — check $LOG_FILE"
}

cmd_stop() {
    if ! is_running; then
        info "not running"
        rm -f "$PID_FILE"
        return 0
    fi
    local pid
    pid="$(cat "$PID_FILE")"
    info "stopping pid=$pid ..."
    kill "$pid" 2>/dev/null || true
    for i in 1 2 3 4 5; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    ok "stopped"
}

cmd_restart() {
    cmd_stop || true
    sleep 0.5
    cmd_start
}

cmd_logs() {
    [ -f "$LOG_FILE" ] || { fail "no log at $LOG_FILE"; return 1; }
    tail -f "$LOG_FILE"
}

cmd_fg() {
    if is_running; then
        warn "background instance running (pid=$(cat "$PID_FILE")). Stop it first or use a different port."
        return 1
    fi
    info "foreground mode (Ctrl+C to stop), $HOST:$PORT"
    exec "$PY" "$SCRIPT_DIR/server.py" --host "$HOST" --port "$PORT"
}

ACTION="${1:-start}"
case "$ACTION" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_restart ;;
    status)  cmd_status ;;
    logs)    cmd_logs ;;
    fg)      cmd_fg ;;
    *) echo "usage: $0 {start|stop|restart|status|logs|fg}"; exit 2 ;;
esac
