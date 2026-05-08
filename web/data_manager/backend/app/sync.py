"""Post-save 数据同步: rsync 录完的一条 episode 到 gf0/gf1。

设计点:
  * 异步: recorder.save() 返回前 spawn 后台线程, 不阻塞 UI
  * 幂等: rsync -a 只传递差异; 失败不影响下一次触发
  * 可观测: 所有 rsync 结果写 web/data_manager/logs/sync.log, 带 remote / 用时 / rc
  * 可关闭: KAI0_SYNC_ENABLED=0 整体禁用
  * 可重定向: KAI0_SYNC_REMOTES (JSON list) 覆盖默认 gf0/gf1
  * 可补偿: sync_remaining() 手动一次性同步已存在的 task/subset/date (见 bottom)

rsync 策略 (v2 layout, subset 在 date 之上):
  - src:  {DATA_ROOT}/{task}/{subset}/{date}    (无 trailing slash, src 目录本身被推)
  - dst:  {user}@{host}:{dest_root}/{task}/{subset}/
  - --mkpath 自动建远端父目录 (rsync 3.2.3+, sim01 是 3.2.7 ✓)
  - 不加 --delete: 本地误删时不传播到云端
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .config import DATA_ROOT


# ----------------------------- config -------------------------------------
@dataclass(frozen=True)
class Remote:
    name: str
    user: str
    host: str
    port: int
    dest_root: str
    # transport options:
    #  - "rsync"        : 直推 ssh+rsync 到 dest_root (老链路, 小文件 OK 但 gf0 WAN 慢)
    #  - "tos_via_gf1"  : tar → TOS → ssh gf1 extract → /vePFS (legacy, 写 vePFS)
    #  - "tos_only"     : 直接 per-file upload 到 TOS, 不 tar 不 extract; gf 端通过
    #                     fuse mount /transfer-shanghai/KAI0 直接读, 不写 vePFS.
    #                     这是 2026-05+ 推荐路径 (gf2 cron pull TOS, gf0/1 fuse, vePFS 只留 ckpt).
    transport: str = "rsync"


# Default push targets. Adding a remote only needs (name, user, host, port, dest_root).
#
# gf0-vepfs: gf0 和 gf1 的 /vePFS/visrobot01 是同一块 gpfs 卷 (fs_vepfs-cnsh075262e1f815),
#   推 gf0 = 两台同时到。rsync 3.2.7. 一次性: sudo chown -R tim:tim /vePFS/visrobot01/KAI0.
# bja2-vla:  单独的开发机, root SSH, rsync 3.1.3 (不支持 --mkpath, 见 _rsync_cmd 里用
#   --rsync-path 兼容老版本).
DEFAULT_REMOTES: list[Remote] = [
    # 2026-05+: 直 upload 到 TOS bucket. gf0/gf1 通过 /transfer-shanghai/KAI0 fuse
    # mount 直读, 不再写 vePFS. gf2 cron */5 拉 TOS, gf3 lsyncd 镜像 gf2.
    # dest_root 在 tos_only transport 下用作 TOS prefix 标记 (= "KAI0/").
    Remote(name="gf0-tos", user="tim", host="14.103.44.161", port=11111,
           dest_root="KAI0", transport="tos_only"),
    # bja2 直推 rsync 速率 (~13 MB/s) 足以跟上录制, 不需要走 TOS.
    Remote(name="bja2-vla", user="root", host="115.190.97.39", port=37686,
           dest_root="/VLA-Data/scripts/lianqing/data/bipiper_dataset",
           transport="rsync"),
]


def _load_remotes() -> list[Remote]:
    raw = os.environ.get("KAI0_SYNC_REMOTES", "")
    if not raw:
        return list(DEFAULT_REMOTES)
    try:
        items = json.loads(raw)
        return [Remote(**it) for it in items]
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logging.getLogger(__name__).error(
            "invalid KAI0_SYNC_REMOTES=%r, falling back to defaults: %s", raw, e
        )
        return list(DEFAULT_REMOTES)


ENABLED: bool = os.environ.get("KAI0_SYNC_ENABLED", "1") == "1"
REMOTES: list[Remote] = _load_remotes()
RETRIES: int = int(os.environ.get("KAI0_SYNC_RETRIES", "3"))
BACKOFF_BASE_S: float = float(os.environ.get("KAI0_SYNC_BACKOFF_S", "2"))
TIMEOUT_S: int = int(os.environ.get("KAI0_SYNC_TIMEOUT_S", "1200"))  # 单次 rsync 上限 (20 min)
# 为啥 1200s: gf0 WAN 带宽 2-3 MB/s, 每条 episode 900+ zarr chunks + 3 mp4 (~200MB
# 总传输量) 加上 remote stat 开销, 600s 容易 timeout. 1200s 给足缓冲.
# 带宽上限 (KB/s). 白天 BWLIMIT_KBPS_DAY, 夜里 BWLIMIT_KBPS_NIGHT.
# 夜间默认 0 = 不限, 白天默认 2000 (~2 MB/s). 0 = 不限流。
# 夜间窗口默认 [00:00, 08:00) local time; 由 NIGHT_START/END_HOUR 控制。
BWLIMIT_KBPS_DAY: int = int(os.environ.get("KAI0_SYNC_BWLIMIT_KBPS_DAY", "2000"))
BWLIMIT_KBPS_NIGHT: int = int(os.environ.get("KAI0_SYNC_BWLIMIT_KBPS_NIGHT", "0"))
NIGHT_START_HOUR: int = int(os.environ.get("KAI0_SYNC_NIGHT_START_HOUR", "0"))   # 00:00
NIGHT_END_HOUR: int = int(os.environ.get("KAI0_SYNC_NIGHT_END_HOUR", "8"))       # 08:00


def _current_bwlimit_kbps() -> int:
    """按 local 时钟决定此刻 bwlimit。夜间窗口 [start, end) 放开, 其它时段限流。"""
    h = datetime.now().hour
    if NIGHT_START_HOUR <= NIGHT_END_HOUR:
        is_night = NIGHT_START_HOUR <= h < NIGHT_END_HOUR
    else:  # 跨零点窗口 e.g. [22, 6)
        is_night = h >= NIGHT_START_HOUR or h < NIGHT_END_HOUR
    return BWLIMIT_KBPS_NIGHT if is_night else BWLIMIT_KBPS_DAY


# 旧全局 (兼容其它可能引用). 首次 import 时的快照, 实际每次 rsync 调用会动态取。
BWLIMIT_KBPS: int = _current_bwlimit_kbps()
# CAMERAS 和 recorder.py 对齐; 单 episode 同步用它构造文件列表。
_CAMERAS = ("top_head", "hand_left", "hand_right")


def _load_depth_cameras() -> tuple[str, ...]:
    """Read DEPTH_CAMERAS from config/camera_depth_flags.py — duplicated
    inline (instead of `from .recorder import DEPTH_CAMERAS`) to avoid a
    circular import: recorder imports sync.sync_episode_files, so sync
    must not depend on recorder at import time.

    rsync 同步白名单和 recorder 的写入白名单必须一致, 否则会列出根本不存在
    的腕部 zarr 路径触发 "vanished" 警告 (rsync 退出码 24).
    """
    import importlib.util
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config" / "camera_depth_flags.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "kai0_camera_depth_flags_sync", candidate)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return tuple(mod.DEPTH_CAMERAS)
    return ()


_DEPTH_CAMERAS = _load_depth_cameras()

# 每个 remote 一把 lock, 串行处理 post-save 推送, 防止 gf0 那种慢链路背压积压:
# 录制 ~15s/ep 但 gf0 同步 ~120s/ep → 并发会指数堆积. 串行让后进 episode 排队,
# rsync 用 --partial 保证断续可续, 没有丢数据.
_remote_locks: dict[str, threading.Lock] = {}
_remote_locks_guard = threading.Lock()


def _get_remote_lock(name: str) -> threading.Lock:
    with _remote_locks_guard:
        lock = _remote_locks.get(name)
        if lock is None:
            lock = threading.Lock()
            _remote_locks[name] = lock
        return lock


# 独立文件 logger, 避免 uvicorn access log 里塞满 rsync 统计
_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "sync.log"
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_sync_log = logging.getLogger("kai0.sync")
if not _sync_log.handlers:
    h = logging.FileHandler(_LOG_PATH)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _sync_log.addHandler(h)
    _sync_log.setLevel(logging.INFO)
    _sync_log.propagate = False


# ----------------------------- workers ------------------------------------
@dataclass
class _Job:
    src: Path
    task: str
    date: str
    subset: str
    remotes: list[Remote] = field(default_factory=list)


def _rsync_cmd(src: Path, remote: Remote, task: str, subset: str) -> list[str]:
    """目录级推送: src 是 date 目录 (不带 trailing slash), 会落到 remote 的
    dest_root/task/subset/ 下作为同名 (date) 子目录。

    **注意**: 这个函数会 rsync *整个 date 树*, 触发 remote+local 全树 stat,
    对大 subset (万级文件) 可能超时。post-save 钩子应当使用 _rsync_cmd_episode
    走单 episode 文件列表, 不要走这个。本函数保留给 sync_all / 手动 full rsync。

    用 --rsync-path='mkdir -p X && rsync' 而非 --mkpath, 兼容 rsync 3.1.x
    (bja2 是 3.1.3). gf0 是 3.2.7 用哪种都行。"""
    ssh = (
        f"ssh -p {remote.port} -o BatchMode=yes "
        f"-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
    )
    dst_parent_path = f"{remote.dest_root}/{task}/{subset}"
    dst_parent = f"{remote.user}@{remote.host}:{dst_parent_path}/"
    remote_wrap = f"mkdir -p {shlex.quote(dst_parent_path)} && rsync"
    cmd: list[str] = []
    # 低优先级 wrapper: nice + ionice, 不抢 uvicorn / recorder 的 CPU/IO
    if shutil_which("nice") and shutil_which("ionice"):
        cmd += ["nice", "-n", "19", "ionice", "-c", "3"]
    cmd += [
        "rsync", "-a", "--partial",
        f"--rsync-path={remote_wrap}",
        f"--timeout={TIMEOUT_S}",
        "-e", ssh,
    ]
    bw = _current_bwlimit_kbps()
    if bw > 0:
        cmd.append(f"--bwlimit={bw}")
    cmd += [str(src), dst_parent]
    return cmd


def _episode_rel_paths(episode_id: int) -> list[str]:
    """列出单条 episode 相对 subset_root 的所有文件 / 目录路径。
    目录路径末尾带 /, rsync 会递归进去 (zarr chunks)。"""
    eid = f"episode_{episode_id:06d}"
    paths = [
        f"data/chunk-000/{eid}.parquet",
        # meta 整组小文件, 直接列整条
        "meta/episodes.jsonl",
        "meta/info.json",
        "meta/tasks.jsonl",
    ]
    for cam in _CAMERAS:
        paths.append(f"videos/chunk-000/{cam}/{eid}.mp4")
    for cam in _DEPTH_CAMERAS:
        paths.append(f"videos/chunk-000/{cam}_depth/{eid}.zarr/")
    return paths


def _rsync_cmd_episode(src_date: Path, remote: Remote, task: str, subset: str,
                       date: str, rel_paths: list[str]) -> list[str]:
    """单 episode 级推送 —— 不扫整棵树, 只 stat 列表里的 ~10 条。
    这是 post-save 钩子应当走的路径。

    构造: rsync -a --files-from=- --bwlimit=N src_date/ remote:dst/<task>/<subset>/<date>/
    files-from 通过 stdin 传; caller 负责把 rel_paths 用换行连接喂进去。
    """
    ssh = (
        f"ssh -p {remote.port} -o BatchMode=yes "
        f"-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
    )
    dst_date_path = f"{remote.dest_root}/{task}/{subset}/{date}"
    dst_date = f"{remote.user}@{remote.host}:{dst_date_path}/"
    remote_wrap = f"mkdir -p {shlex.quote(dst_date_path)} && rsync"
    cmd: list[str] = []
    if shutil_which("nice") and shutil_which("ionice"):
        cmd += ["nice", "-n", "19", "ionice", "-c", "3"]
    cmd += [
        "rsync", "-a", "--partial",
        "--files-from=-",  # rel_paths 从 stdin 喂
        f"--rsync-path={remote_wrap}",
        f"--timeout={TIMEOUT_S}",
        "-e", ssh,
    ]
    bw = _current_bwlimit_kbps()
    if bw > 0:
        cmd.append(f"--bwlimit={bw}")
    cmd += [f"{src_date}/", dst_date]  # src 带 trailing slash: 内容 → dst (不再套一层 date 目录)
    return cmd


def shutil_which(name: str) -> str | None:  # small wrapper to avoid import cost in module scope
    import shutil
    return shutil.which(name)


def _run_one(cmd: list[str], stdin_text: str | None = None) -> tuple[int, str]:
    """跑一条 rsync, 返回 (rc, short_summary)。stderr 合进 stdout, 截 200 字。
    stdin_text: 给 --files-from=- 用, rel_paths 通过 stdin 喂进 rsync。"""
    try:
        p = subprocess.run(
            cmd, input=stdin_text, capture_output=True, text=True,
            timeout=TIMEOUT_S + 30,
        )
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    tail = (p.stdout + p.stderr).strip().splitlines()[-1:]
    return p.returncode, (tail[0] if tail else "")


def _push_one_remote(src: Path, task: str, date: str, subset: str, remote: Remote) -> None:
    cmd = _rsync_cmd(src, remote, task, subset)
    for attempt in range(1, RETRIES + 1):
        t0 = time.time()
        rc, summary = _run_one(cmd)
        dt_ms = int((time.time() - t0) * 1000)
        tag = f"[{remote.name}] {task}/{subset}/{date}"
        if rc == 0:
            _sync_log.info("%s ok in %d ms (attempt %d)", tag, dt_ms, attempt)
            return
        _sync_log.warning(
            "%s rc=%d attempt=%d/%d dt=%d ms: %s | cmd=%s",
            tag, rc, attempt, RETRIES, dt_ms, summary, shlex.join(cmd),
        )
        if attempt < RETRIES:
            time.sleep(BACKOFF_BASE_S * (2 ** (attempt - 1)))
    _sync_log.error("%s FAILED after %d attempts", tag, RETRIES)


def _worker(job: _Job) -> None:
    """给每个 remote 并行发一条 rsync, 彼此独立。"""
    threads = []
    for r in job.remotes:
        t = threading.Thread(
            target=_push_one_remote,
            args=(job.src, job.task, job.date, job.subset, r),
            name=f"sync-{r.name}-{job.task}-{job.date}",
            daemon=True,
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


# ─── TOS circuit breaker — 连续失败时自动降级到 rsync, 同时通知用户 ───
# 触发: 同一 remote 上 _push_episode_via_tos 连续 N 次失败 (默认 3)
# 行为: 标记电路开路, 之后该 remote 的 ep 推送改走 rsync (不走 TOS)
#      经过 RETRY_AFTER 秒后下一次推送会再尝试 TOS, 成功则关闭电路
# 通知: log ERROR + 写 marker file + notify-send (有桌面会话则桌面气泡)
TOS_FAILURE_THRESHOLD: int = int(os.environ.get("KAI0_TOS_FAILURE_THRESHOLD", "3"))
TOS_CIRCUIT_RETRY_AFTER_S: int = int(os.environ.get("KAI0_TOS_CIRCUIT_RETRY_AFTER_S", "600"))  # 10 min
_TOS_FAILURE_STREAK: dict[str, int] = {}
_TOS_CIRCUIT_OPEN_AT: dict[str, float] = {}
_TOS_CIRCUIT_LOCK = threading.Lock()
_TOS_CIRCUIT_MARKER = Path("/tmp/kai0_tos_circuit.txt")


def _tos_circuit_is_open(remote_name: str) -> bool:
    """查电路是否处于打开 (= 走 rsync 旁路) 状态; 超过 RETRY_AFTER 自动允许重试."""
    with _TOS_CIRCUIT_LOCK:
        opened_at = _TOS_CIRCUIT_OPEN_AT.get(remote_name, 0.0)
        if opened_at == 0.0:
            return False
        if time.time() - opened_at > TOS_CIRCUIT_RETRY_AFTER_S:
            return False  # 允许探测一次, 但不立刻清状态; 探测成功后才清
        return True


def _notify_user(title: str, msg: str) -> None:
    """尽量多通道通知: marker file + notify-send + log ERROR (sync_log 调用方做)."""
    try:
        _TOS_CIRCUIT_MARKER.write_text(f"{datetime.now().isoformat()} {title}: {msg}\n")
    except OSError:
        pass
    try:
        # notify-send 仅在有桌面 DBUS session 时有效, 失败静默
        subprocess.run(
            ["notify-send", "-u", "critical", "-i", "dialog-warning", title, msg],
            timeout=2, capture_output=True,
        )
    except Exception:
        pass


def _on_tos_failure(remote_name: str, exc: Exception) -> bool:
    """记一次 TOS 失败. 返回 True 表示应触发电路打开 (调用方降级 rsync 并通知)."""
    with _TOS_CIRCUIT_LOCK:
        streak = _TOS_FAILURE_STREAK.get(remote_name, 0) + 1
        _TOS_FAILURE_STREAK[remote_name] = streak
        already_open = remote_name in _TOS_CIRCUIT_OPEN_AT
        if streak >= TOS_FAILURE_THRESHOLD and not already_open:
            _TOS_CIRCUIT_OPEN_AT[remote_name] = time.time()
            return True  # 此次触发 — caller 通知 + 降级
    return False


def _on_tos_success(remote_name: str) -> bool:
    """记一次 TOS 成功. 返回 True 表示电路从打开状态恢复 (调用方通知 RECOVERED)."""
    with _TOS_CIRCUIT_LOCK:
        was_open = remote_name in _TOS_CIRCUIT_OPEN_AT
        _TOS_FAILURE_STREAK[remote_name] = 0
        if was_open:
            _TOS_CIRCUIT_OPEN_AT.pop(remote_name, None)
            try:
                _TOS_CIRCUIT_MARKER.unlink(missing_ok=True)
            except OSError:
                pass
            return True
    return False


def _push_one_remote_episode(src_date: Path, task: str, date: str, subset: str,
                             episode_id: int, remote: Remote) -> None:
    """单 episode push, 根据 remote.transport 分派到 rsync 或 TOS 路径.

    src_date 是 v2 layout 的 `<DATA_ROOT>/<task>/<subset>/<date>` 目录 (date 叶节点).

    **串行锁**: 同一 remote 同时只有一个 job 执行, 超出能力的慢链路自然 backlog
    但不堆并发, 不产生 orphan 进程.

    **TOS 电路 breaker**: 若该 remote 配置 transport=tos_via_gf1 但电路打开中,
    自动降级走 _push_episode_via_rsync, 不再尝试 TOS. RETRY_AFTER 后才放行 1 次
    探测, 成功就关电路."""
    lock = _get_remote_lock(remote.name)
    tag = f"[{remote.name}] {task}/{subset}/{date}/ep{episode_id:06d}"
    t_wait_start = time.time()
    with lock:
        wait_ms = int((time.time() - t_wait_start) * 1000)
        if wait_ms > 1000:
            _sync_log.info("%s waited %d ms for remote lock (backlog)", tag, wait_ms)
        if remote.transport == "tos_via_gf1":
            if _tos_circuit_is_open(remote.name):
                _sync_log.warning("%s TOS circuit OPEN — falling back to direct rsync", tag)
                _push_episode_via_rsync(src_date, task, date, subset, episode_id, remote, tag)
            else:
                _push_episode_via_tos(src_date, task, date, subset, episode_id, remote, tag)
        elif remote.transport == "tos_only":
            if _tos_circuit_is_open(remote.name):
                _sync_log.warning("%s TOS circuit OPEN — episode delayed (no rsync fallback in tos_only)", tag)
                return
            _push_episode_via_tos_only(src_date, task, date, subset, episode_id, remote, tag)
        else:  # rsync (default)
            _push_episode_via_rsync(src_date, task, date, subset, episode_id, remote, tag)


def _push_episode_via_rsync(src_date: Path, task: str, date: str, subset: str,
                            episode_id: int, remote: Remote, tag: str) -> None:
    rel_paths = _episode_rel_paths(episode_id)
    cmd = _rsync_cmd_episode(src_date, remote, task, subset, date, rel_paths)
    stdin_text = "\n".join(rel_paths) + "\n"
    for attempt in range(1, RETRIES + 1):
        t0 = time.time()
        rc, summary = _run_one(cmd, stdin_text=stdin_text)
        dt_ms = int((time.time() - t0) * 1000)
        if rc == 0:
            _sync_log.info("%s ok in %d ms (attempt %d)", tag, dt_ms, attempt)
            return
        _sync_log.warning("%s rc=%d attempt=%d/%d dt=%d ms: %s",
                          tag, rc, attempt, RETRIES, dt_ms, summary)
        if attempt < RETRIES:
            time.sleep(BACKOFF_BASE_S * (2 ** (attempt - 1)))
    _sync_log.error("%s FAILED after %d attempts", tag, RETRIES)


# ---------------- TOS transport (gf0 real-time via gf1 fuse extract) ------
# Credentials are loaded from env (KAI0_TOS_AK / KAI0_TOS_SK) — NEVER hard-code here.
# Set them in web/data_manager/.env (gitignored) and run.sh auto-source the file.
# When env is missing, _get_tos_client() raises; caller falls back to non-TOS path.
_TOS_CREDS = {
    "ak": os.environ.get("KAI0_TOS_AK", ""),
    "sk": os.environ.get("KAI0_TOS_SK", ""),
    "endpoint": os.environ.get("KAI0_TOS_ENDPOINT", "tos-cn-shanghai.volces.com"),
    "region": os.environ.get("KAI0_TOS_REGION", "cn-shanghai"),
    "bucket": os.environ.get("KAI0_TOS_BUCKET", "transfer-shanghai"),
}
_TOS_CLIENT_LAZY: "object | None" = None
_TOS_CLIENT_LOCK = threading.Lock()


def _get_tos_client():
    global _TOS_CLIENT_LAZY
    with _TOS_CLIENT_LOCK:
        if _TOS_CLIENT_LAZY is None:
            # strip proxies before tos import (intra-region path)
            for k in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY",
                     "ftp_proxy", "FTP_PROXY", "all_proxy", "ALL_PROXY",
                     "no_proxy", "NO_PROXY"):
                os.environ.pop(k, None)
            import tos  # lazy; avoids requiring tos in envs without it
            _TOS_CLIENT_LAZY = tos.TosClientV2(
                _TOS_CREDS["ak"], _TOS_CREDS["sk"],
                _TOS_CREDS["endpoint"], _TOS_CREDS["region"],
            )
        return _TOS_CLIENT_LAZY


def _push_episode_via_tos_only(src_date: Path, task: str, date: str, subset: str,
                               episode_id: int, remote: Remote, tag: str) -> None:
    """直接 per-file upload 到 TOS, 不 tar 不 ssh extract 不写 vePFS.

    TOS object key = `<dest_root>/<task>/<subset>/<date>/<rel_path>` (= sim01 同名).
    gf0/gf1 通过 /transfer-shanghai/KAI0 fuse 直读, 不需要落到 gpfs.

    zarr 是目录, walk 内部 chunk 文件逐个 upload. 全量小于一个 ep tar 大小,
    并行多线程 upload 与 tar+upload 总时长接近, 但去掉了 ssh extract 步骤.
    """
    cli = _get_tos_client()
    bucket = _TOS_CREDS["bucket"]
    prefix = (remote.dest_root or "KAI0").strip("/")
    rel_subset = src_date.relative_to(DATA_ROOT)  # task/subset/date

    # 列出本 ep 所有要上传的 (local_path, tos_key) 对
    rel_paths = _episode_rel_paths(episode_id)
    upload_pairs: list[tuple[Path, str]] = []
    for rp in rel_paths:
        local = src_date / rp.rstrip("/")
        if local.is_dir():
            # zarr / 目录: 递归找文件
            for f in local.rglob("*"):
                if f.is_file():
                    rel_in_subset = f.relative_to(src_date)
                    key = f"{prefix}/{rel_subset}/{rel_in_subset}"
                    upload_pairs.append((f, key))
        elif local.is_file():
            key = f"{prefix}/{rel_subset}/{rp}"
            upload_pairs.append((local, key))
        # else: 文件不存在, skip (e.g. 部分相机无 depth)

    if not upload_pairs:
        _sync_log.warning("%s no files to upload (skip)", tag)
        return

    # 上传
    t0 = time.time()
    total_bytes = 0
    n_done = 0
    n_err = 0
    last_err = None
    for local, key in upload_pairs:
        try:
            cli.put_object_from_file(bucket, key, str(local))
            n_done += 1
            try:
                total_bytes += local.stat().st_size
            except OSError:
                pass
        except Exception as e:  # noqa: BLE001
            n_err += 1
            last_err = e
            if n_err <= 3:
                _sync_log.error("%s tos_only upload %s FAILED: %s", tag, key, e)

    up_ms = int((time.time() - t0) * 1000)

    if n_err > 0 and n_done == 0:
        # 全失败 → TOS 电路 breaker
        if _on_tos_failure(remote.name, last_err):
            _sync_log.error(
                "%s TOS CIRCUIT OPENED after %d consecutive failures",
                tag, TOS_FAILURE_THRESHOLD,
            )
            _notify_user(
                "kai0 sync: TOS DOWN",
                f"{remote.name} tos_only upload 连续失败 {TOS_FAILURE_THRESHOLD} 次"
            )
        return

    # 部分/全成功 — reset streak
    if _on_tos_success(remote.name):
        _sync_log.warning("%s TOS CIRCUIT CLOSED — TOS path restored", tag)

    rate = total_bytes / max(up_ms / 1000, 0.001) / 1e6
    _sync_log.info(
        "%s tos_only ok files=%d/%d size=%dKB elapsed=%dms rate=%.1fMB/s err=%d",
        tag, n_done, len(upload_pairs), total_bytes // 1024, up_ms, rate, n_err,
    )


def _push_episode_via_tos(src_date: Path, task: str, date: str, subset: str,
                          episode_id: int, remote: Remote, tag: str) -> None:
    """tar ep specific files → TOS → ssh gf1 extract from /transfer-shanghai fuse.

    src_date 是 v2 layout 的 `<DATA_ROOT>/<task>/<subset>/<date>`; 相对 DATA_ROOT 的
    rel 直接是 `<task>/<subset>/<date>`, tar 解到 dest_root 自动落到对位 v2 路径。

    每步计时记日志。失败多次后 fall back 不做 (数据仍在本地, nightly batch 会兜底)。
    """
    tmp_dir = Path("/data1")  # same fs as src, tar is cheap
    tar_name = f"rt_{remote.name}_{task}_{subset}_{date}_ep{episode_id:06d}.tar"
    tar_path = tmp_dir / tar_name
    tos_key = f"KAI0/realtime/{tar_name}"
    rel_paths = _episode_rel_paths(episode_id)

    # 1. tar (只打本 ep 的 ~10 条 path, ~200 MB)
    t0 = time.time()
    rel = src_date.relative_to(DATA_ROOT)
    tar_cmd = ["tar", "cf", str(tar_path), "--ignore-failed-read",
               "-C", str(DATA_ROOT)] + [f"{rel}/{p}" for p in rel_paths]
    p = subprocess.run(tar_cmd, capture_output=True, text=True, timeout=300)
    if p.returncode not in (0, 1):  # rc=1 = "file changed", ok
        _sync_log.error("%s tar FAILED rc=%d: %s", tag, p.returncode, p.stderr[-200:])
        tar_path.unlink(missing_ok=True)
        return
    tar_ms = int((time.time() - t0) * 1000)
    tar_size = tar_path.stat().st_size

    # 2. TOS multi-part upload (intra-region 85 MB/s)
    t0 = time.time()
    try:
        cli = _get_tos_client()
        cli.upload_file(_TOS_CREDS["bucket"], tos_key, str(tar_path),
                        task_num=8, part_size=64 * 1024 * 1024)
    except Exception as e:
        _sync_log.error("%s TOS upload FAILED: %s", tag, e)
        tar_path.unlink(missing_ok=True)
        # 记入电路 breaker; 连续失败到阈值时通知用户并切 rsync 旁路
        if _on_tos_failure(remote.name, e):
            _sync_log.error(
                "%s TOS CIRCUIT OPENED after %d consecutive failures — "
                "subsequent eps fall back to direct rsync until %ds re-probe",
                tag, TOS_FAILURE_THRESHOLD, TOS_CIRCUIT_RETRY_AFTER_S,
            )
            _notify_user(
                "kai0 sync: TOS DOWN",
                f"{remote.name} TOS upload 连续失败 {TOS_FAILURE_THRESHOLD} 次, "
                f"自动降级走 rsync. 详情: {_sync_log.handlers[0].baseFilename if _sync_log.handlers else 'sync.log'}"
            )
        return
    up_ms = int((time.time() - t0) * 1000)
    # 这次 TOS 成功 — 重置 streak; 若之前电路是开的, 通知恢复
    if _on_tos_success(remote.name):
        _sync_log.warning("%s TOS CIRCUIT CLOSED — TOS path restored", tag)
        _notify_user(
            "kai0 sync: TOS RECOVERED",
            f"{remote.name} TOS 路径恢复, 自动切回 TOS 中转 (100 MB/s)."
        )

    # 3. ssh gf1 extract tar from fuse → /vePFS (shared gpfs, gf0 immediately sees)
    t0 = time.time()
    remote_tar = f"/transfer-shanghai/{tos_key}"
    dst_root = remote.dest_root  # /vePFS/visrobot01/KAI0
    ssh_cmd = (f"ssh -p {remote.port} -o BatchMode=yes "
               f"-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "
               f"{remote.user}@{remote.host}").split()
    remote_sh = (
        f"set -e; mkdir -p {shlex.quote(dst_root)}; "
        f"tar xf {shlex.quote(remote_tar)} -C {shlex.quote(dst_root)} --overwrite; "
        f"echo OK"
    )
    p = subprocess.run(ssh_cmd + [remote_sh], capture_output=True, text=True, timeout=600)
    ext_ms = int((time.time() - t0) * 1000)
    if p.returncode != 0 or "OK" not in p.stdout:
        _sync_log.error("%s gf1 extract FAILED rc=%d: %s", tag, p.returncode, p.stderr[-200:])
        # leave TOS object + local tar for nightly retry
        return

    # 4. cleanup
    try:
        cli.delete_object(_TOS_CREDS["bucket"], tos_key)
    except Exception as e:
        _sync_log.warning("%s TOS delete failed (non-fatal): %s", tag, e)
    tar_path.unlink(missing_ok=True)

    _sync_log.info("%s ok via tos in %d ms (tar=%dms up=%dms ext=%dms size=%.1fMB)",
                   tag, tar_ms + up_ms + ext_ms, tar_ms, up_ms, ext_ms, tar_size / 1e6)


def _worker_episode(src_date: Path, task: str, date: str, subset: str,
                    episode_id: int, remotes: list[Remote]) -> None:
    threads = []
    for r in remotes:
        t = threading.Thread(
            target=_push_one_remote_episode,
            args=(src_date, task, date, subset, episode_id, r),
            name=f"sync-{r.name}-{task}-{subset}-{date}-ep{episode_id}",
            daemon=True,
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


# ----------------------------- public API ---------------------------------
def sync_episode_files(task: str, date: str, subset: str, episode_id: int) -> None:
    """recorder.save() 应当调这个 (单 episode, O(1) 开销).

    只推这一条 episode 相关的 ~10 个文件/目录 (parquet + 3 mp4 + 3 zarr + meta),
    用 rsync --files-from= 跳过全树 stat。对 10k+ 文件的 subset 关键: 每次 save 仅
    几秒就完成, 不再因为扫描慢触发 timeout.
    """
    if not ENABLED:
        return
    if not REMOTES:
        _sync_log.warning("no remotes configured; skipping sync of %s/%s/%s/ep%06d",
                          task, subset, date, episode_id)
        return
    src = _resolve_src(task, subset, date)
    if src is None:
        _sync_log.error("sync source missing: task=%s subset=%s date=%s", task, subset, date)
        return
    threading.Thread(
        target=_worker_episode,
        args=(src, task, date, subset, episode_id, REMOTES),
        name=f"sync-ep-{task}-{subset}-{date}-{episode_id}",
        daemon=True,
    ).start()


def sync_episode_subset(task: str, date: str, subset: str) -> None:
    """旧 API (全 subset 推): 不再被 recorder.save() 调用, 只留给 sync_all / 管理员
    手动全量对齐用。对大 subset 会很慢, 见 _rsync_cmd 注释。
    """
    if not ENABLED:
        return
    if not REMOTES:
        _sync_log.warning("no remotes configured; skipping sync of %s/%s/%s",
                          task, subset, date)
        return
    src = _resolve_src(task, subset, date)
    if src is None:
        _sync_log.error("sync source missing: task=%s subset=%s date=%s", task, subset, date)
        return
    job = _Job(src=src, task=task, date=date, subset=subset, remotes=REMOTES)
    threading.Thread(target=_worker, args=(job,), name=f"sync-main-{task}-{subset}-{date}",
                     daemon=True).start()


def _resolve_src(task: str, subset: str, date: str) -> Path | None:
    """v2 → v1 → v0 三段式回退 (与 layout.compound_to_subset_root 同优先级)."""
    v2 = DATA_ROOT / task / subset / date
    if v2.is_dir():
        return v2
    v1 = DATA_ROOT / task / date / subset
    if v1.is_dir():
        return v1
    v0 = DATA_ROOT / f"{task}_{date}" / subset
    if v0.is_dir():
        return v0
    return None


def sync_all(only_task: str | None = None) -> int:
    """一次性把 DATA_ROOT 下所有 task/subset/date 推到远端 (同步性, 阻塞)。
    用于首次搭建 / 迁移完成后全量对齐。返回任务数。"""
    if not REMOTES:
        return 0
    from .layout import path_to_compound, split_compound, glob_all_episodes
    seen: set[tuple[str, str, str]] = set()  # (task, date, subset)
    for pq in glob_all_episodes():
        parsed = path_to_compound(pq)
        if parsed is None:
            continue
        compound, subset = parsed
        sp = split_compound(compound)
        if sp is None:
            continue
        task, date = sp
        if only_task and task != only_task:
            continue
        seen.add((task, date, subset))
    for task, date, subset in sorted(seen):
        sync_episode_subset(task, date, subset)
    return len(seen)


def recent_log_tail(n: int = 50) -> list[str]:
    """返回最近 n 行 sync.log, 供 UI / 调试用。"""
    if not _LOG_PATH.exists():
        return []
    try:
        with _LOG_PATH.open("r", encoding="utf-8", errors="replace") as f:
            return f.readlines()[-n:]
    except OSError:
        return []


def status() -> dict:
    """给 /api/sync/status 用的摘要。"""
    return {
        "enabled": ENABLED,
        "remotes": [
            {"name": r.name, "host": r.host, "port": r.port, "dest_root": r.dest_root}
            for r in REMOTES
        ],
        "log_path": str(_LOG_PATH),
        "log_tail": [ln.rstrip() for ln in recent_log_tail(20)],
        "ts": datetime.utcnow().isoformat() + "Z",
    }
