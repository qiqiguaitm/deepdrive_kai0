#!/usr/bin/env bash
# data_manager 一键启动：CAN + 机械臂 + 相机 + 后端 + 前端
#
# 用法：
#   ./run.sh start     # 启动全部
#   ./run.sh stop      # 停止全部
#   ./run.sh status    # 查看状态
#   ./run.sh logs [svc]  # 追踪日志（svc: arms|cameras|backend|frontend）
#
# 环境变量（可选）：
#   SETUP_CAN=1        启动时激活 CAN（默认跳过；假设已经手动激活过）
#   SKIP_ARMS=1        跳过机械臂节点
#   SKIP_CAMERAS=1     跳过相机节点
#   SKIP_DEPS=1        跳过后端 pip 依赖同步（默认每次 start 会同步）
#   KAI0_DATA_ROOT=... 采集落盘根目录（默认 /data1/DATA_IMP/KAI0, 与项目目录隔离避免误删）
#   KAI0_ROS_BRIDGE=mock  强制 mock（无 ROS2 环境时调试用）

set -u
# ros2 孤儿清理: pkill -f ros2 只能匹配到 `ros2 launch` 封装进程,
# 匹配不到它拉起的 realsense2_camera_node / arm_teleop_node 子进程
# (命令行里已经没有 "ros2" 字样)。这些孤儿会独占 CAN 和 RealSense 设备,
# 导致下次启动看不见相机/机械臂, 所以这里补一遍, 先 TERM 再 KILL。
# 注意: 只在 start/stop 时调用, status/logs 绝不能碰 —— 否则 `./run.sh status`
# 会把活着的 arms/cameras 连根拔起 (历史 bug: pkill 曾在顶层, status 都杀).
kill_ros2_orphans() {
    pkill -f ros2 2>/dev/null || true
    pkill -f realsense2_camera_node 2>/dev/null || true
    pkill -f arm_teleop_node 2>/dev/null || true
    sleep 1
    pkill -9 -f realsense2_camera_node 2>/dev/null || true
    pkill -9 -f arm_teleop_node 2>/dev/null || true
}
# ------------------------------------------------------------ paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WEB_DIR="$REPO_ROOT/web/data_manager"
BACKEND_DIR="$WEB_DIR/backend"
FRONTEND_DIR="$WEB_DIR/frontend"
LOG_DIR="$WEB_DIR/logs"
PID_DIR="$WEB_DIR/.pids"

mkdir -p "$LOG_DIR" "$PID_DIR"

ROS_SETUP="/opt/ros/jazzy/setup.bash"
ROS2_WS_SETUP="$REPO_ROOT/ros2_ws/install/setup.bash"
ACTIVATE_CAN="$REPO_ROOT/piper_tools/activate_can.sh"
START_TELEOP="$REPO_ROOT/scripts/start_teleop.sh"
LAUNCH_3CAM="$REPO_ROOT/scripts/launch_3cam.py"

SERVICES=(arms cameras backend frontend)

# 服务 → 监听端口 (用于孤儿清理)。没监听端口的服务 (arms/cameras) 留空。
# 历史背景: 之前 start_svc 用 $! 写 pidfile, 抓到的是瞬死的 setsid 父 PID,
# stop_svc kill 已死 PID = no-op, 真正的 uvicorn 一直占着 8787 没人管。
# 接下来 ./run.sh start 想再起 uvicorn, 端口被占 → 新 uvicorn 立刻退出 →
# pidfile 又变 dead PID。循环往复, 用户一直看老版本数据。
# 现在 pidfile 已经修对了, 但仍然 belt-and-suspenders 加端口级清理兜底。
declare -A SVC_PORT=( [backend]=8787 [frontend]=5173 )

# ------------------------------------------------------------ helpers
log()  { echo -e "\033[36m[$(date +%H:%M:%S)]\033[0m $*"; }
warn() { echo -e "\033[33m[warn]\033[0m $*" >&2; }
err()  { echo -e "\033[31m[err]\033[0m $*" >&2; }

pid_file() { echo "$PID_DIR/$1.pid"; }
log_file() { echo "$LOG_DIR/$1.log"; }

is_running() {
    local svc="$1" pf; pf="$(pid_file "$svc")"
    [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null
}

# 杀掉占用 TCP $1 端口的所有进程 (孤儿清理)。安全无操作如果端口空闲。
kill_port() {
    local port="$1"
    local pids
    pids=$(ss -lntp 2>/dev/null | awk -v p=":$port\$" '$4 ~ p' | grep -oP 'pid=\K[0-9]+' | sort -u)
    [[ -z "$pids" ]] && return 0
    log "kill orphan(s) on :$port → $(echo $pids | tr '\n' ' ')"
    kill -TERM $pids 2>/dev/null || true
    sleep 0.5
    kill -KILL $pids 2>/dev/null || true
}

start_svc() {
    local svc="$1"; shift
    local cmd="$*"
    if is_running "$svc"; then
        warn "$svc already running (pid $(cat "$(pid_file "$svc")"))"
        return 0
    fi
    log "start $svc: $cmd"
    # 为什么绕这么一圈: 简单的 `setsid bash -c "$cmd" &; echo $!` 不可靠,
    # setsid 在不同组合下可能 fork 并让父进程立即退出, $! 拿到的是一个瞬死的
    # 父 PID —— 即使 `--fork --wait` 也观察到不稳定。结果: is_running 一秒后
    # 判 DEAD, 打印 "arms failed to start", 而真正的服务在另一个 PID 下跑得好好的,
    # 这就是 "err 但 teleop ok" 的根因。
    #
    # 解决: 让子 bash 自己把 $$ 写进 pidfile, 我们读它而不是读 $!。
    # setsid 仍然提供 session 隔离 (pgid == $$), 保证 stop 时 kill -- -pid 能
    # 整组杀掉整个进程树 (source + start_teleop.sh + ros2 launch + 4 个 node).
    #   bash -c 'SCRIPT' arg0 arg1  —— arg0 填到 $0, arg1 填到 $1
    #   第一条命令 echo $$ > $0  写本 bash 的 PID 到 pidfile
    #   exec bash -c "$1"       原地替换, PID 不变, 继续跑真实命令
    local pidf; pidf="$(pid_file "$svc")"
    rm -f "$pidf"
    # Wrapper bash 写自己的 PID 到 pidfile (不是用 $!), 然后 eval 跑真实命令,
    # 保证 pidfile 始终追踪活着的 bash。上个版本发现空 log + dead PID 的根因其实
    # 不在这里 —— 是 start_teleop.sh 第 15 行的 `pkill -f ros2` 在无匹配时返回 1,
    # 配合它自己的 `set -e` 在第一个 echo 之前静默退出。那个已经修了。
    setsid bash -c 'echo $$ > "$0"; eval "$1"' "$pidf" "$cmd" \
        >"$(log_file "$svc")" 2>&1 &
    # 等 pidfile 写入 (通常 <100 ms)
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        [[ -s "$pidf" ]] && break
        sleep 0.1
    done
    sleep 1
    if is_running "$svc"; then
        log "  -> $svc ok (pid $(cat "$(pid_file "$svc")"))  log: $(log_file "$svc")"
    else
        err "$svc failed to start; see $(log_file "$svc")"
    fi
}

stop_svc() {
    local svc="$1" pf pid
    pf="$(pid_file "$svc")"
    if [[ -f "$pf" ]]; then
        pid="$(cat "$pf")"
        if kill -0 "$pid" 2>/dev/null; then
            log "stop $svc (pid $pid, group)"
            # 杀整个进程组（setsid 启动时 pgid = pid）
            kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
            for _ in 1 2 3 4 5; do
                kill -0 "$pid" 2>/dev/null || break
                sleep 0.5
            done
            kill -KILL -- -"$pid" 2>/dev/null
        fi
        rm -f "$pf"
    fi
    # 兜底: 如果该服务有已知监听端口, 把仍在占用端口的进程也清掉
    # (旧 pidfile 是死 PID 但真服务还活着的情况, 见 SVC_PORT 注释)
    local port="${SVC_PORT[$svc]:-}"
    [[ -n "$port" ]] && kill_port "$port"
}

# ------------------------------------------------------------ actions
do_start() {
    # 0a) ROS2 孤儿清理 (只在 start 时, 避免误伤 status/logs 看到的活服务).
    kill_ros2_orphans
    # 0b) 端口级孤儿清理: 如果 backend/frontend 端口已被占用 (上次没清干净的孤儿,
    #     或别的进程占了), 提前杀掉, 避免新 uvicorn/vite 因 "address in use" 静默退出
    #     然后 pidfile 又指向死 PID 的循环。
    for svc in "${!SVC_PORT[@]}"; do
        kill_port "${SVC_PORT[$svc]}"
    done

    # 1) CAN（默认跳过；显式 SETUP_CAN=1 时才激活）
    if [[ "${SETUP_CAN:-0}" == "1" ]]; then
        if [[ -x "$ACTIVATE_CAN" ]]; then
            log "activate CAN ..."
            bash "$ACTIVATE_CAN" || warn "activate_can.sh returned non-zero (continuing)"
        else
            warn "activate_can.sh not executable: $ACTIVATE_CAN"
        fi
    else
        log "skip CAN activation (set SETUP_CAN=1 to enable)"
    fi

    # 2) 机械臂读数节点
    # start_teleop.sh 自己会 source ROS + ros2_ws (见脚本头部), 这里不再重复 source
    # 并用 exec 让 wrapper bash 原地替换成 bash-start_teleop, 避免 `&&` 链导致
    # wrapper bash fork 后不等待, 被误判为失败.
    if [[ "${SKIP_ARMS:-0}" != "1" ]]; then
        start_svc arms "exec bash '$START_TELEOP'"
    fi

    # 3) 相机
    if [[ "${SKIP_CAMERAS:-0}" != "1" ]]; then
        start_svc cameras "source '$ROS_SETUP' && source '$ROS2_WS_SETUP' && ros2 launch '$LAUNCH_3CAM'"
    fi

    # 4) 后端（必须 source ROS 才能用 rclpy）
    if [[ "${SKIP_DEPS:-0}" != "1" ]]; then
        if [[ -x "$BACKEND_DIR/.venv/bin/pip" ]]; then
            log "sync backend deps (av / pyarrow / ...)"
            "$BACKEND_DIR/.venv/bin/pip" install -q -r "$BACKEND_DIR/requirements.txt" \
                || warn "pip install returned non-zero (continuing)"
        else
            warn "backend venv missing at $BACKEND_DIR/.venv — create it with: python -m venv $BACKEND_DIR/.venv && $BACKEND_DIR/.venv/bin/pip install -r $BACKEND_DIR/requirements.txt"
        fi
    fi
    start_svc backend "source '$ROS_SETUP' && source '$ROS2_WS_SETUP' && cd '$BACKEND_DIR' && .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8787"

    # 5) 前端
    start_svc frontend "cd '$FRONTEND_DIR' && npm run dev -- --host"

    echo
    log "all services launched."
    log "  前端:  http://$(hostname -I | awk '{print $1}'):5173/"
    log "  后端:  http://$(hostname -I | awk '{print $1}'):8787/  (docs: /docs)"
    log "查看日志: ./run.sh logs <arms|cameras|backend|frontend>"
}

do_stop() {
    # 按启动反序停
    for svc in frontend backend cameras arms; do
        stop_svc "$svc"
    done
    # stop_svc 走 pidfile, 可能漏掉 pidfile 失同步的 ros2 子进程, 兜底清一次.
    kill_ros2_orphans
}

do_status() {
    printf "%-10s %-8s %-8s %s\n" "SERVICE" "STATE" "PID" "LOG"
    for svc in "${SERVICES[@]}"; do
        if is_running "$svc"; then
            printf "%-10s \033[32m%-8s\033[0m %-8s %s\n" "$svc" "running" "$(cat "$(pid_file "$svc")")" "$(log_file "$svc")"
        else
            printf "%-10s \033[31m%-8s\033[0m %-8s %s\n" "$svc" "stopped" "-" "$(log_file "$svc")"
        fi
    done
}

do_logs() {
    local svc="${1:-}"
    if [[ -z "$svc" ]]; then
        tail -n 20 -F "$LOG_DIR"/*.log
    else
        tail -n 50 -F "$(log_file "$svc")"
    fi
}

case "${1:-start}" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop; sleep 1; do_start ;;
    status)  do_status ;;
    logs)    shift; do_logs "${1:-}" ;;
    *) echo "usage: $0 {start|stop|restart|status|logs [svc]}"; exit 1 ;;
esac
