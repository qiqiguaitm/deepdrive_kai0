#!/usr/bin/python3
"""ROS2 replay-only node — drives slave arms from a recorded LeRobot v2.1 episode.

This is a slim alternative to running the full `policy_inference_node` for replay.
NO JAX, NO policy load, NO cameras, NO rerun → starts in <1s and ~50MB RSS instead
of 30s + 5GB. Suitable for replay-only deployments (no inference/autonomy needed).

Topology (mirror of autonomy-side):
  Subscriptions:
    /puppet/joint_left   sensor_msgs/JointState   — current slave-left state
    /puppet/joint_right  sensor_msgs/JointState   — current slave-right state
    /policy/execute      std_msgs/Bool            — toggle execute on/off
  Publishers:
    /master/joint_left   sensor_msgs/JointState   — drives slave-left (mode=1 reader subs)
    /master/joint_right  sensor_msgs/JointState   — drives slave-right
    /replay_progress     std_msgs/Float32MultiArray  [frame_idx, total, done_flag]

ros2 params (all hot-reloadable):
  replay_mode             string  inference|replay|idle    default=replay  (this node only does replay)
  replay_episode_path     string  abs path to .parquet     default=""
  replay_rate             float   [0.5, 1.5]               default=1.0
  replay_loop             bool                              default=false
  replay_auto_home        bool                              default=true   (slow-walk to action[0] if >5°)
  replay_home_duration    float   seconds [1, 10]          default=3.0
  publish_rate            int     Hz                       default=30
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray


def _to_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


class StreamActionBuffer:
    """Drained one frame per pop_next_action() at publish_rate. cur_chunk replaced
    on integrate_new_chunk; for replay we inject the entire episode in one go."""

    def __init__(self, state_dim: int = 14):
        self.lock = threading.Lock()
        self.cur_chunk: deque = deque()
        self.k = 0
        self.last_action = None
        self.state_dim = state_dim

    def pop_next_action(self):
        with self.lock:
            if not self.cur_chunk:
                return None
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(self.cur_chunk[0], dtype=float).copy()
            act = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self.k += 1
            return act

    def flush(self, seed_action=None):
        with self.lock:
            self.cur_chunk.clear()
            self.k = 0
            self.last_action = seed_action


class ReplayNode(Node):
    """Slim replay node — see module docstring."""

    def __init__(self):
        super().__init__("replay")

        # ── ros2 params ──
        self.declare_parameter("replay_mode", "replay")
        self.declare_parameter("replay_episode_path", "")
        self.declare_parameter("replay_rate", 1.0)
        self.declare_parameter("replay_loop", False)
        self.declare_parameter("replay_auto_home", True)
        self.declare_parameter("replay_home_duration", 3.0)
        self.declare_parameter("publish_rate", 30)

        self._replay_mode = str(self.get_parameter("replay_mode").value)
        self._replay_rate = max(0.5, min(1.5, float(self.get_parameter("replay_rate").value)))
        self._replay_loop = _to_bool(self.get_parameter("replay_loop").value)
        self._replay_auto_home = _to_bool(self.get_parameter("replay_auto_home").value)
        self._replay_home_duration = max(1.0, min(10.0, float(self.get_parameter("replay_home_duration").value)))
        self.publish_rate = int(self.get_parameter("publish_rate").value)

        # ── State ──
        self._sensor_lock = threading.Lock()
        self._joint_left_deque: deque = deque(maxlen=200)
        self._joint_right_deque: deque = deque(maxlen=200)

        self._exec_lock = threading.Lock()
        self._execution_enabled = False
        self._last_published_action = None
        self._MAX_JOINT_JUMP_RAD = 0.5  # ~28.6°

        self.stream_buffer = StreamActionBuffer(state_dim=14)

        self._replay_lock = threading.Lock()
        self._replay_actions = None
        self._replay_path = ""
        self._replay_parquet_fps = float(self.publish_rate)
        self._replay_buffer_total = 0
        self._replay_aligned_threshold_rad = float(np.deg2rad(5.0))
        self._replay_progress_last_ts = 0.0
        self._replay_info_last_ts = 0.0

        # ── Subscribers ──
        self.create_subscription(JointState, "/puppet/joint_left",
                                 self._cb_joint_left, 100)
        self.create_subscription(JointState, "/puppet/joint_right",
                                 self._cb_joint_right, 100)
        self.create_subscription(Bool, "/policy/execute", self._cb_execute, 5)

        # ── Publishers ──
        self.pub_left = self.create_publisher(JointState, "/master/joint_left", 10)
        self.pub_right = self.create_publisher(JointState, "/master/joint_right", 10)
        self.pub_replay_progress = self.create_publisher(
            Float32MultiArray, "/replay_progress", 5)

        # ── Timers ──
        period = 1.0 / max(1, self.publish_rate)
        self.create_timer(period, self._publish_action)
        self.create_timer(0.1, self._tick_replay_timer)  # 10 Hz

        # ── Param hot-reload ──
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info(
            f"[REPLAY] node ready. publish_rate={self.publish_rate} Hz, "
            f"mode={self._replay_mode}, auto_home={self._replay_auto_home}")

    # ── Sensor callbacks ──
    def _cb_joint_left(self, msg):
        with self._sensor_lock:
            self._joint_left_deque.append(msg)

    def _cb_joint_right(self, msg):
        with self._sensor_lock:
            self._joint_right_deque.append(msg)

    def _cb_execute(self, msg):
        with self._exec_lock:
            was_enabled = self._execution_enabled
            if msg.data and not was_enabled:
                # In replay mode the buffer is preloaded; skip flush which would wipe it.
                if self._replay_mode != "replay":
                    self._flush_stale_buffer()
            self._execution_enabled = bool(msg.data)
        self.get_logger().info(
            f"[REPLAY] execute → {'ON' if self._execution_enabled else 'OFF'}")

    def _flush_stale_buffer(self):
        with self._sensor_lock:
            jl = self._joint_left_deque[-1] if self._joint_left_deque else None
            jr = self._joint_right_deque[-1] if self._joint_right_deque else None
        seed = None
        if jl is not None and jr is not None:
            ql = np.asarray(jl.position[:7], dtype=np.float32)
            qr = np.asarray(jr.position[:7], dtype=np.float32)
            if len(ql) >= 7 and len(qr) >= 7:
                seed = np.concatenate([ql, qr])
        self.stream_buffer.flush(seed_action=seed)
        self._last_published_action = None

    # ── Param hot-reload ──
    def _on_set_parameters(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            try:
                if p.name == "replay_episode_path":
                    new_path = str(p.value)
                    if not new_path:
                        with self._replay_lock:
                            self._replay_actions = None
                            self._replay_path = ""
                        self.get_logger().info("[REPLAY] cleared episode path")
                    else:
                        ok, msg = self._load_replay_episode(new_path)
                        if not ok:
                            return SetParametersResult(successful=False, reason=msg)
                elif p.name == "replay_rate":
                    self._replay_rate = max(0.5, min(1.5, float(p.value)))
                    self.get_logger().info(f"replay_rate → {self._replay_rate}")
                elif p.name == "replay_loop":
                    self._replay_loop = _to_bool(p.value)
                elif p.name == "replay_auto_home":
                    self._replay_auto_home = _to_bool(p.value)
                elif p.name == "replay_home_duration":
                    self._replay_home_duration = max(1.0, min(10.0, float(p.value)))
                elif p.name == "replay_mode":
                    new_rm = str(p.value)
                    if new_rm not in ("inference", "replay", "idle"):
                        return SetParametersResult(
                            successful=False,
                            reason=f"invalid replay_mode {new_rm!r}")
                    if new_rm == "replay":
                        # Always re-enter (re-fill buffer + re-check pose)
                        ok, reason = self._enter_replay_mode()
                        if not ok:
                            return SetParametersResult(
                                successful=False,
                                reason=f"replay pre-flight failed: {reason}")
                    elif new_rm != self._replay_mode and self._replay_mode == "replay":
                        self._exit_replay_mode()
                    self._replay_mode = new_rm
                    self.get_logger().info(f"replay_mode → {new_rm}")
            except Exception as e:  # noqa: BLE001
                return SetParametersResult(successful=False, reason=f"{p.name}: {e}")
        return SetParametersResult(successful=True)

    # ── Parquet load ──
    def _load_replay_episode(self, parquet_path: str):
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            return False, f"pyarrow not installed: {e}"
        if not os.path.isfile(parquet_path):
            return False, f"parquet not found: {parquet_path}"
        try:
            tbl = pq.read_table(parquet_path, columns=["action", "timestamp"])
        except Exception as e:  # noqa: BLE001
            return False, f"parquet read failed: {e}"
        try:
            actions = np.asarray(tbl.column("action").to_pylist(), dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            return False, f"action column decode failed: {e}"
        if actions.ndim != 2 or actions.shape[1] != 14:
            return False, f"bad action shape {actions.shape}"
        try:
            ts = np.asarray(tbl.column("timestamp").to_pylist(), dtype=np.float64)
            if len(ts) >= 2:
                duration_s = float(ts[-1] - ts[0])
                parquet_fps = (len(ts) - 1) / duration_s if duration_s > 1e-6 else float(self.publish_rate)
            else:
                duration_s = 0.0
                parquet_fps = float(self.publish_rate)
        except Exception:  # noqa: BLE001
            duration_s = actions.shape[0] / float(self.publish_rate)
            parquet_fps = float(self.publish_rate)
        with self._replay_lock:
            self._replay_actions = actions
            self._replay_path = parquet_path
            self._replay_parquet_fps = parquet_fps
        self.get_logger().info(
            f"[REPLAY] loaded {parquet_path}: {actions.shape[0]} frames, "
            f"duration={duration_s:.2f}s, parquet_fps={parquet_fps:.2f} Hz")
        return True, f"loaded {actions.shape[0]} frames @ {parquet_fps:.1f} Hz"

    # ── Pose alignment ──
    def _check_start_pose_aligned(self):
        with self._replay_lock:
            if self._replay_actions is None:
                return False, float("inf"), []
            target = self._replay_actions[0].copy()
        with self._sensor_lock:
            jl = self._joint_left_deque[-1] if self._joint_left_deque else None
            jr = self._joint_right_deque[-1] if self._joint_right_deque else None
        if jl is None or jr is None:
            return False, float("inf"), []
        ql = np.asarray(jl.position[:7], dtype=np.float32)
        qr = np.asarray(jr.position[:7], dtype=np.float32)
        if len(ql) < 7 or len(qr) < 7:
            return False, float("inf"), []
        current = np.concatenate([ql, qr])
        diff = np.abs(current - target)
        max_diff = float(diff.max())
        return bool(max_diff <= self._replay_aligned_threshold_rad), max_diff, [float(x) for x in diff]

    def _verify_no_publisher_conflict(self):
        try:
            out = subprocess.run(
                ["ros2", "topic", "info", "/master/joint_left", "-v"],
                capture_output=True, text=True, timeout=5)
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
            my_name = self.get_name()
            others = [p for p in pubs if p != my_name]
            return len(others) == 0, others
        except Exception as e:  # noqa: BLE001
            return False, [f"check_failed: {e}"]

    def _verify_deployment_marker(self):
        marker = "/tmp/kai0_deployment_mode"
        if not os.path.isfile(marker):
            return False, f"{marker} missing"
        try:
            with open(marker) as f:
                val = f.read().strip()
        except Exception as e:  # noqa: BLE001
            return False, f"{marker} read failed: {e}"
        # Accept both 'autonomy' and 'replay' deployments.
        if val not in ("autonomy", "replay"):
            return False, f"{marker}={val!r}, replay requires autonomy or replay"
        return True, val

    def _resample_actions(self, actions: np.ndarray, rate: float) -> np.ndarray:
        rate = float(rate)
        if abs(rate - 1.0) < 1e-3:
            return actions
        T = actions.shape[0]
        new_T = max(1, int(round(T / rate)))
        if new_T == T:
            return actions
        old_idx = np.linspace(0.0, T - 1.0, T)
        new_idx = np.linspace(0.0, T - 1.0, new_T)
        return np.array(
            [np.interp(new_idx, old_idx, actions[:, d]) for d in range(actions.shape[1])],
            dtype=np.float32).T

    def _enter_replay_mode(self):
        with self._replay_lock:
            if self._replay_actions is None:
                return False, "no episode loaded (set replay_episode_path first)"
        ok, reason = self._verify_deployment_marker()
        if not ok:
            return False, f"deployment marker: {reason}"
        ok, others = self._verify_no_publisher_conflict()
        if not ok:
            return False, f"topic /master/joint_left has other publishers: {others}"
        aligned, max_d, per_joint = self._check_start_pose_aligned()
        home_n = 0
        home_frames = None
        if not aligned:
            if not self._replay_auto_home:
                deg_per = ", ".join(f"{np.rad2deg(x):.1f}" for x in per_joint)
                return False, (f"start pose not aligned: max_Δ={np.rad2deg(max_d):.2f}°, "
                               f"threshold=5°. per-joint(°)=[{deg_per}]")
            with self._sensor_lock:
                jl = self._joint_left_deque[-1] if self._joint_left_deque else None
                jr = self._joint_right_deque[-1] if self._joint_right_deque else None
            if jl is None or jr is None:
                return False, "auto_home: missing joint_state"
            ql = np.asarray(jl.position[:7], dtype=np.float32)
            qr = np.asarray(jr.position[:7], dtype=np.float32)
            current = np.concatenate([ql, qr])
            target = self._replay_actions[0]
            home_n = max(1, int(round(self._replay_home_duration * float(self.publish_rate))))
            per_step_max = float(np.max(np.abs(target - current))) / home_n
            if per_step_max > self._MAX_JOINT_JUMP_RAD:
                return False, (f"auto_home: per-step Δ={np.rad2deg(per_step_max):.2f}° > "
                               f"{np.rad2deg(self._MAX_JOINT_JUMP_RAD):.0f}° threshold")
            alphas = np.linspace(1.0/home_n, 1.0, home_n, dtype=np.float32).reshape(-1, 1)
            home_frames = ((1.0 - alphas) * current[None, :] +
                           alphas * target[None, :]).astype(np.float32)
            self.get_logger().info(
                f"[REPLAY] auto-home: {home_n} interp frames, "
                f"max_Δ={np.rad2deg(max_d):.2f}°, per-step={np.rad2deg(per_step_max):.3f}°")

        publish_rate = float(self.publish_rate)
        parquet_fps = float(self._replay_parquet_fps)
        episode_eff_rate = self._replay_rate * (parquet_fps / publish_rate)
        if episode_eff_rate < 0.05 or episode_eff_rate > 3.0:
            episode_eff_rate = max(0.05, min(3.0, episode_eff_rate))
        if abs(episode_eff_rate - 1.0) > 1e-3:
            episode_actions = self._resample_actions(self._replay_actions, episode_eff_rate)
        else:
            episode_actions = self._replay_actions

        if home_frames is not None:
            actions = np.vstack([home_frames, episode_actions]).astype(np.float32)
        else:
            actions = np.asarray(episode_actions, dtype=np.float32)

        with self.stream_buffer.lock:
            self.stream_buffer.cur_chunk = deque([a.copy() for a in actions], maxlen=None)
            self.stream_buffer.k = 0
            self.stream_buffer.last_action = None
        with self._replay_lock:
            self._replay_buffer_total = int(actions.shape[0])
        self.get_logger().info(
            f"[REPLAY] entered replay: buffer={actions.shape[0]} frames "
            f"(home={home_n} + episode={actions.shape[0]-home_n}), "
            f"wall_play={actions.shape[0]/publish_rate:.2f}s")
        return True, ""

    def _exit_replay_mode(self):
        self.stream_buffer.flush(seed_action=self._last_published_action)
        with self._exec_lock:
            self._execution_enabled = False
        self.get_logger().info("[REPLAY] exited replay mode")

    # ── Replay tick: publish progress + handle done ──
    def _tick_replay_timer(self):
        if self._replay_mode != "replay":
            return
        try:
            self._tick_replay()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"replay tick error: {e}")

    def _tick_replay(self):
        with self._replay_lock:
            actions = self._replay_actions
            total = self._replay_buffer_total
        if actions is None or total <= 0:
            return
        with self.stream_buffer.lock:
            remaining = len(self.stream_buffer.cur_chunk)
        idx = max(0, total - remaining)
        done = (remaining == 0)
        now = time.monotonic()
        if now - self._replay_progress_last_ts >= 0.1 or done:
            msg = Float32MultiArray()
            msg.data = [float(idx), float(total), 1.0 if done else 0.0]
            try:
                self.pub_replay_progress.publish(msg)
            except Exception:  # noqa: BLE001
                pass
            self._replay_progress_last_ts = now
            if now - self._replay_info_last_ts >= 1.0 or done:
                with self._exec_lock:
                    enabled = self._execution_enabled
                self.get_logger().info(
                    f"[REPLAY] frame {idx}/{total} ({100*idx/max(1,total):.1f}%) "
                    f"remaining={remaining} execute={enabled}")
                self._replay_info_last_ts = now
        if done:
            if self._replay_loop:
                ok, reason = self._enter_replay_mode()
                if not ok:
                    self.get_logger().warn(f"[REPLAY] loop end: {reason}")
                    with self._exec_lock:
                        self._execution_enabled = False
                    self._replay_mode = "inference"
            else:
                self.get_logger().info("[REPLAY] episode finished, execution OFF, mode→inference")
                with self._exec_lock:
                    self._execution_enabled = False
                self._replay_mode = "inference"

    # ── Publisher (timer @ publish_rate) ──
    def _publish_action(self):
        with self._exec_lock:
            enabled = self._execution_enabled
        if not enabled:
            self._last_published_action = None
            return
        act = self.stream_buffer.pop_next_action()
        if act is None:
            return
        # Jump protection
        reference = self._last_published_action
        if reference is None:
            with self._sensor_lock:
                jl = self._joint_left_deque[-1] if self._joint_left_deque else None
                jr = self._joint_right_deque[-1] if self._joint_right_deque else None
            if jl is not None and jr is not None:
                ql = np.array(jl.position[:7])
                qr = np.array(jr.position[:7])
                if len(ql) >= 7 and len(qr) >= 7:
                    reference = np.concatenate([ql, qr])
        if reference is not None:
            max_jump = float(np.max(np.abs(act[:14] - reference[:14])))
            if max_jump > self._MAX_JOINT_JUMP_RAD:
                self.get_logger().warn(
                    f"Jump protection: rejected action max Δ={np.degrees(max_jump):.1f}°, "
                    f"flushing buffer")
                self._flush_stale_buffer()
                return
        self._last_published_action = act.copy()

        left = act[:7].copy()
        right = act[7:14].copy()
        # No gripper_offset for replay (recorded data is raw master joint state).
        left[6] = max(0.0, float(left[6]))
        right[6] = max(0.0, float(right[6]))

        now = self.get_clock().now().to_msg()
        for pub, values in [(self.pub_left, left), (self.pub_right, right)]:
            msg = JointState()
            msg.header.stamp = now
            msg.name = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
            msg.position = values.tolist()
            pub.publish(msg)


def main():
    rclpy.init()
    node = ReplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
