"""Replay (P2) — backend wrapper around policy_inference_node's replay-mode params.

Frontend ReplayPanel toggles "真实执行" → calls /api/replay/preflight (read-only) for
pose-alignment check, shows result, then on user confirm calls /api/replay/execute
which actually flips replay_mode + execute=true.

ROS2 calls go via subprocess (matches existing CLI patterns rtc_apply.sh /
start_replay_test.sh). Latency ~200ms per call, fine for UI button cadence.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np

from .config import DATA_ROOT
from .ros_bridge import bridge

# Two stacks expose replay params: full autonomy (/policy_inference) or slim
# replay-only (/replay, see start_replay_stack.sh + replay_node.py).
# We auto-detect which is alive at preflight time and target whichever responds.
NODE_AUTONOMY = "/policy_inference"
NODE_REPLAY_ONLY = "/replay"
DEPLOYMENT_MARKER = "/tmp/kai0_deployment_mode"
ALIGN_THRESHOLD_RAD = float(np.deg2rad(5.0))
ROS2_BIN = os.environ.get("KAI0_ROS2_BIN", "/opt/ros/jazzy/bin/ros2")


def _ros2(args: list[str], timeout: float = 8.0) -> subprocess.CompletedProcess:
    """Run ros2 CLI via absolute path. Backend's parent process has LD_LIBRARY_PATH
    etc. set (rclpy works), so the CLI inherits a usable env."""
    return subprocess.run(
        [ROS2_BIN] + args, capture_output=True, text=True, timeout=timeout)


# ── Path resolution (CLI-style 4-tuple) ─────────────────────────────────────

def parquet_path(task: str, subset: str, date: str, episode_id: int) -> Path:
    """`Task_A/base/2026-04-28/42` → absolute parquet path under DATA_ROOT.

    `date` is treated as a free-form path component — supports both
    YYYY-MM-DD and arbitrary strings like 'kai0_official_base'."""
    return DATA_ROOT / task / subset / date / "data" / "chunk-000" / f"episode_{int(episode_id):06d}.parquet"


# ── Parquet metadata ────────────────────────────────────────────────────────

def load_first_action(p: Path) -> dict:
    """Read action[0] + recording fps from a LeRobot v2.1 parquet.
    Returns {ok, frames, fps, duration_s, action0}; ok=False on parse error."""
    if not p.is_file():
        return {"ok": False, "error": f"parquet not found: {p}"}
    try:
        import pyarrow.parquet as pq
        tbl = pq.read_table(str(p), columns=["action", "timestamp"])
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"parquet read failed: {e}"}
    try:
        actions = np.asarray(tbl.column("action").to_pylist(), dtype=np.float32)
        ts = np.asarray(tbl.column("timestamp").to_pylist(), dtype=np.float64)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"column decode failed: {e}"}
    if actions.ndim != 2 or actions.shape[1] != 14:
        return {"ok": False, "error": f"bad action shape {actions.shape}"}
    duration = float(ts[-1] - ts[0]) if len(ts) >= 2 else 0.0
    fps = (len(ts) - 1) / duration if duration > 1e-6 else 30.0
    return {
        "ok": True,
        "frames": int(actions.shape[0]),
        "duration_s": duration,
        "fps": fps,
        "action0": [float(x) for x in actions[0]],
        "action_last": [float(x) for x in actions[-1]],
    }


# ── Safety gates (mirror node-side checks; node is authoritative) ───────────

def check_deployment_marker() -> tuple[bool, str]:
    """Marker is written by start_autonomy.sh ('autonomy') or
    start_replay_stack.sh ('replay'). Either is valid for replay."""
    if not os.path.isfile(DEPLOYMENT_MARKER):
        return False, f"{DEPLOYMENT_MARKER} missing — start_autonomy.sh or start_replay_stack.sh not active"
    try:
        v = open(DEPLOYMENT_MARKER).read().strip()
    except Exception as e:  # noqa: BLE001
        return False, f"{DEPLOYMENT_MARKER} read failed: {e}"
    return (v in ("autonomy", "replay")), v


def detect_replay_node(diag: dict | None = None) -> str | None:
    """Return name of the replay-capable node alive in graph, or None.
    Tries /replay (slim replay-only stack) first, then /policy_inference (full autonomy).
    Both nodes accept the same `replay_*` ros2 params, so we just target the one
    that exists. fills diag with debug info if provided."""
    try:
        out = _ros2(["node", "list"], timeout=5)
        if diag is not None:
            diag["rc"] = out.returncode
            diag["stdout_first200"] = out.stdout[:200]
            diag["stderr_first200"] = out.stderr[:200]
        if out.returncode != 0:
            return None
        nodes = set(out.stdout.split())
        if NODE_REPLAY_ONLY in nodes:
            return NODE_REPLAY_ONLY
        if NODE_AUTONOMY in nodes:
            return NODE_AUTONOMY
        return None
    except Exception as e:  # noqa: BLE001
        if diag is not None:
            diag["exception"] = repr(e)
        return None


def policy_inference_alive(diag: dict | None = None) -> bool:
    """Backwards-compat: True iff a replay-capable node is alive.
    Field name kept for existing UI; underlying check covers both stacks."""
    return detect_replay_node(diag) is not None


def publisher_conflicts(self_node: str | None = None) -> tuple[bool, list[str]]:
    """List nodes publishing to /master/joint_left, excluding the active replay
    target (self_node, leading slash optional). Default self = /policy_inference
    for backwards-compat with callers that don't auto-detect."""
    try:
        out = _ros2(["topic", "info", "/master/joint_left", "-v"], timeout=5)
    except subprocess.TimeoutExpired:
        return False, ["ros2_topic_info timeout"]
    if out.returncode != 0:
        return False, [f"rc={out.returncode}: {out.stderr.strip()[:200]}"]
    blocks = re.split(r"\n(?=Node name:)", out.stdout)
    pubs = []
    for blk in blocks:
        if "Endpoint type: PUBLISHER" not in blk:
            continue
        m = re.search(r"Node name:\s*(\S+)", blk)
        if m:
            pubs.append(m.group(1))
    self_bare = (self_node or "/policy_inference").lstrip("/")
    others = [p for p in pubs if p != self_bare]
    return len(others) == 0, others


def current_joint_state() -> list[float] | None:
    """Read latest puppet joint state from ros_bridge → 14-dim action-aligned vector."""
    try:
        state, _action = bridge.get_state_action()
    except Exception:  # noqa: BLE001
        return None
    if not state or len(state) != 14:
        return None
    return [float(x) for x in state]


# ── Preflight: read-only diagnostic ─────────────────────────────────────────

def preflight(task: str, subset: str, date: str, episode_id: int) -> dict:
    """No side effects. Returns everything the UI needs to decide whether to proceed."""
    p = parquet_path(task, subset, date, episode_id)
    meta = load_first_action(p)
    if not meta["ok"]:
        return {"ok": False, "step": "load_parquet", "reason": meta["error"]}

    marker_ok, marker_val = check_deployment_marker()
    node_diag: dict = {}
    target_node = detect_replay_node(node_diag)
    node_alive = target_node is not None
    pub_ok, pub_others = publisher_conflicts(target_node)
    current = current_joint_state()

    target = meta["action0"]
    if current is not None:
        diff = [abs(target[i] - current[i]) for i in range(14)]
        max_diff = max(diff)
        per_joint_deg = [float(np.rad2deg(d)) for d in diff]
        max_diff_deg = float(np.rad2deg(max_diff))
        aligned = max_diff <= ALIGN_THRESHOLD_RAD
    else:
        max_diff_deg = float("inf")
        per_joint_deg = []
        aligned = False

    can_run = bool(marker_ok and node_alive and pub_ok and meta["ok"])
    # Default node-side ros2 params (matches policy_inference_node declare_parameter).
    # Frontend uses these to compute video↔action sync.
    PUBLISH_RATE_DEFAULT = 30
    HOME_DUR_DEFAULT = 3.0
    auto_home = (current is not None) and (not aligned)
    home_n_planned = int(round(HOME_DUR_DEFAULT * PUBLISH_RATE_DEFAULT)) if auto_home else 0
    return {
        "ok": can_run,
        "parquet_path": str(p),
        "frames": meta["frames"],
        "fps": meta["fps"],
        "duration_s": meta["duration_s"],
        "action0": target,
        "current_state": current,
        "max_diff_deg": max_diff_deg,
        "per_joint_diff_deg": per_joint_deg,
        "aligned": aligned,
        "deployment_mode": marker_val,
        "policy_inference_alive": node_alive,  # backwards-compat field name
        "target_node": target_node,            # which stack: /replay or /policy_inference
        "_node_alive_diag": node_diag,
        "publisher_conflict": pub_others,
        # auto_home info (for UI to decide messaging + video sync):
        "auto_home_will_trigger": auto_home,
        "home_n_planned": home_n_planned,    # # of pre-episode interp frames in buffer
        "publish_rate": PUBLISH_RATE_DEFAULT,
        # Total estimated buffer length (home + episode resampled to publish_rate)
        "expected_buffer_total": home_n_planned + int(round(meta["duration_s"] * PUBLISH_RATE_DEFAULT)),
    }


# ── Execute: actually fire the replay ───────────────────────────────────────

def _set_param(node: str, name: str, value: str) -> tuple[bool, str]:
    """`ros2 param set` exits 0 even on rejection; have to grep stdout."""
    out = _ros2(["param", "set", node, name, value], timeout=10)
    text = (out.stdout + out.stderr)
    if "Setting parameter failed" in text or out.returncode != 0:
        return False, text.strip()[:500]
    return True, text.strip()[:200]


def execute(task: str, subset: str, date: str, episode_id: int,
            rate: float = 1.0, loop: bool = False) -> dict:
    """Set params, switch to replay mode, fire execute=true. Returns flow trace."""
    p = parquet_path(task, subset, date, episode_id)
    if not p.is_file():
        return {"ok": False, "step": "resolve_path", "reason": f"parquet not found: {p}"}

    rate = max(0.5, min(1.5, float(rate)))
    trace = []

    # Detect target node (slim /replay or full /policy_inference, whichever alive).
    node = detect_replay_node()
    if node is None:
        return {"ok": False, "step": "detect_node",
                "reason": "no replay-capable node alive (start_replay_stack.sh or start_autonomy.sh)"}
    trace.append({"step": "detect_node", "ok": True, "msg": f"target={node}"})

    # Clear any stale /replay_progress cache (esp. previous run's done=true) so
    # frontend's polling doesn't immediately see "done" before the new session
    # has produced its first progress msg.
    bridge.clear_replay_progress()

    for name, value in [
        ("replay_episode_path", str(p)),
        ("replay_rate", str(rate)),
        ("replay_loop", "true" if loop else "false"),
        ("replay_mode", "replay"),  # last → triggers node-side pre-flight
    ]:
        ok, msg = _set_param(node, name, value)
        trace.append({"step": f"set:{name}", "ok": ok, "msg": msg})
        if not ok:
            # Reset replay_mode to inference so we don't leave the node stuck.
            _set_param(node, "replay_mode", "inference")
            return {"ok": False, "step": f"set:{name}", "reason": msg, "trace": trace}

    # Fire /policy/execute=true. Prefer native pub via ros_bridge (instant);
    # fall back to subprocess if bridge is mock.
    fired = bridge.publish_execute(True)
    if not fired:
        out = _ros2(["topic", "pub", "--once", "/policy/execute",
                     "std_msgs/Bool", "{data: true}"], timeout=8)
        fired = (out.returncode == 0)
        trace.append({"step": "pub_execute_subprocess", "ok": fired,
                      "msg": (out.stdout + out.stderr).strip()[:200]})
    else:
        trace.append({"step": "pub_execute_native", "ok": True, "msg": ""})

    return {"ok": fired, "started": fired, "trace": trace}


# ── Stop ────────────────────────────────────────────────────────────────────

def stop() -> dict:
    """Cleanup: execute=false + replay_mode=inference. Idempotent."""
    trace = []
    fired = bridge.publish_execute(False)
    if not fired:
        out = _ros2(["topic", "pub", "--once", "/policy/execute",
                     "std_msgs/Bool", "{data: false}"], timeout=8)
        fired = (out.returncode == 0)
        trace.append({"step": "pub_execute_subprocess",
                      "ok": fired, "msg": (out.stdout + out.stderr).strip()[:200]})
    else:
        trace.append({"step": "pub_execute_native", "ok": True, "msg": ""})

    node = detect_replay_node()
    if node is None:
        trace.append({"step": "set:replay_mode=inference", "ok": False, "msg": "no replay node alive"})
        return {"ok": fired, "stopped": True, "trace": trace}
    ok, msg = _set_param(node, "replay_mode", "inference")
    trace.append({"step": f"set:replay_mode=inference (on {node})", "ok": ok, "msg": msg})
    return {"ok": fired and ok, "stopped": True, "trace": trace}


# ── Progress ────────────────────────────────────────────────────────────────

def progress() -> dict:
    """Latest /replay_progress + computed fraction. None if never received."""
    p = bridge.get_replay_progress()
    if p is None:
        return {"ok": False, "reason": "no /replay_progress message yet"}
    total = p["total"]
    idx = p["idx"]
    fraction = (idx / total) if total > 0 else 0.0
    return {
        "ok": True,
        "idx": idx,
        "total": total,
        "done": p["done"],
        "fraction": fraction,
        "age_s": p["age_s"],
    }
