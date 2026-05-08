#!/usr/bin/python3
"""ROS2 node — replays a recorded LeRobot v2.1 episode's videos as if they were
live RealSense streams. Drop-in replacement for `multi_camera_node` during
playback / replay tests.

For each of {top_head → /camera_f, hand_left → /camera_l, hand_right → /camera_r}:
  * Color: opens `videos/chunk-000/<cam>/episode_NNNNNN.mp4` and publishes to
    `/camera_<f|l|r>/camera/color/image_raw` (sensor_msgs/Image, rgb8) at
    `publish_rate` Hz.
  * Depth: if `videos/chunk-000/<cam>_depth/episode_NNNNNN.zarr` exists (uint16
    HxW, mm), publishes 16UC1 to
    `/camera_<f|l|r>/camera/aligned_depth_to_color/image_raw` at the same rate.
    Missing zarr → silently skip depth (mirror dataset has none).

Sync with replay_node:
  * Both subscribe to /policy/execute (start/stop together).
  * We additionally subscribe to /replay_progress = [idx, total, done]; idx
    accounts for `auto_home` interpolation frames prepended before the episode,
    so we hold frame-0 during home and only advance once idx ≥ home_n
    (= total - own video frame count).
  * On done=1 we stop emitting and reset to frame 0.

Topic names + QoS match multi_camera_node so rerun_viz_node picks us up
unchanged.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray

# dataset cam name → ROS camera id (matches multi_camera_node + cameras.yml)
CAM_MAP = {
    "top_head":   "f",   # /camera_f/...
    "hand_left":  "l",   # /camera_l/...
    "hand_right": "r",   # /camera_r/...
}


class _VideoSrc:
    """Lazy-decoded color (mp4) + depth (zarr) for one camera. Frame indexing is
    bounded — request beyond last frame returns the last decoded frame."""

    def __init__(self, mp4_path: Path, zarr_path: Path | None, logger):
        self.mp4_path = mp4_path
        self.zarr_path = zarr_path if zarr_path and zarr_path.is_dir() else None
        self._logger = logger
        self._cap = None
        self._cur_idx = -1
        self._last_color = None
        self._color_w = 0
        self._color_h = 0
        self._color_n = 0
        self._zarr = None
        self._depth_n = 0

    def open(self) -> tuple[bool, str]:
        if not self.mp4_path.is_file():
            return False, f"mp4 missing: {self.mp4_path}"
        cap = cv2.VideoCapture(str(self.mp4_path))
        if not cap.isOpened():
            return False, f"cannot open {self.mp4_path}"
        self._cap = cap
        self._color_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._color_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._color_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.zarr_path is not None:
            try:
                import zarr
                self._zarr = zarr.open(str(self.zarr_path), mode="r")
                self._depth_n = int(self._zarr.shape[0])
            except Exception as e:  # noqa: BLE001
                self._logger.warn(f"zarr open failed for {self.zarr_path}: {e}")
                self._zarr = None
        return True, "ok"

    @property
    def color_frames(self) -> int:
        return self._color_n

    @property
    def has_depth(self) -> bool:
        return self._zarr is not None

    def get_color(self, idx: int):
        """Return BGR ndarray for color frame `idx`, clamped to [0, n-1]. Reuses
        the decoder when idx == cur_idx + 1 (sequential — fast). Else seeks."""
        if self._cap is None or self._color_n <= 0:
            return None
        idx = max(0, min(self._color_n - 1, int(idx)))
        if idx == self._cur_idx and self._last_color is not None:
            return self._last_color
        if idx == self._cur_idx + 1:
            ok, frame = self._cap.read()
            if not ok:
                return self._last_color
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self._cap.read()
            if not ok:
                return self._last_color
        self._cur_idx = idx
        self._last_color = frame
        return frame

    def get_depth(self, idx: int):
        """Return uint16 (H, W) ndarray for depth frame `idx`, or None if no zarr."""
        if self._zarr is None or self._depth_n <= 0:
            return None
        idx = max(0, min(self._depth_n - 1, int(idx)))
        try:
            return np.asarray(self._zarr[idx], dtype=np.uint16)
        except Exception:  # noqa: BLE001
            return None

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoPublisherNode(Node):

    def __init__(self):
        super().__init__("video_publisher")

        self.declare_parameter("episode_path", "")  # parquet path; videos derived
        self.declare_parameter("publish_rate", 30)

        self._publish_rate = max(1, int(self.get_parameter("publish_rate").value))
        episode_path = str(self.get_parameter("episode_path").value)

        # ── State ──
        self._lock = threading.Lock()
        self._sources: dict[str, _VideoSrc] = {}
        self._video_frames = 0   # max color frame count across cams (canonical episode T)
        self._exec = False
        self._idx = 0            # /replay_progress idx
        self._total = 0          # /replay_progress total (= home_n + episode_T_resampled)
        self._video_idx_from_replay = -1  # data[3] of /replay_progress, -1 = legacy/not yet

        if episode_path:
            self._open_episode(episode_path)
        else:
            self.get_logger().warn("[VID] empty episode_path — set via param later")

        # ── QoS (match multi_camera_node) ──
        img_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST)
        depth_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                               history=HistoryPolicy.KEEP_LAST)

        # ── Publishers (one pair per camera) ──
        self._pubs: dict[str, dict[str, object]] = {}
        for cam_name, cam_id in CAM_MAP.items():
            self._pubs[cam_name] = {
                "color": self.create_publisher(
                    Image, f"/camera_{cam_id}/camera/color/image_raw", img_qos),
                "depth": self.create_publisher(
                    Image, f"/camera_{cam_id}/camera/aligned_depth_to_color/image_raw", depth_qos),
            }

        # ── Subscribers ──
        self.create_subscription(Bool, "/policy/execute", self._cb_execute, 5)
        self.create_subscription(
            Float32MultiArray, "/replay_progress", self._cb_replay_progress, 5)

        # ── Timer ──
        self.create_timer(1.0 / self._publish_rate, self._tick)

        # ── Param hot-reload (mostly for episode_path) ──
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info(
            f"[VID] node ready. publish_rate={self._publish_rate} Hz, "
            f"episode_T={self._video_frames}, "
            f"sources={list(self._sources.keys())}")

    # ── Episode load ──
    def _open_episode(self, parquet_path: str):
        p = Path(parquet_path)
        if not p.is_file():
            self.get_logger().error(f"[VID] parquet not found: {p}")
            return
        # parquet: <root>/data/chunk-000/episode_NNNNNN.parquet
        try:
            ep_str = p.stem.replace("episode_", "")
            ep_id = int(ep_str)
        except Exception:
            self.get_logger().error(f"[VID] cannot parse episode id from {p.name}")
            return
        videos_dir = p.parent.parent.parent / "videos" / "chunk-000"
        if not videos_dir.is_dir():
            self.get_logger().error(f"[VID] videos dir missing: {videos_dir}")
            return

        with self._lock:
            for src in self._sources.values():
                src.close()
            self._sources = {}
            video_frames = 0
            for cam_name in CAM_MAP:
                mp4 = videos_dir / cam_name / f"episode_{ep_id:06d}.mp4"
                zarr_dir = videos_dir / f"{cam_name}_depth" / f"episode_{ep_id:06d}.zarr"
                src = _VideoSrc(mp4, zarr_dir if zarr_dir.is_dir() else None,
                                self.get_logger())
                ok, msg = src.open()
                if not ok:
                    self.get_logger().warn(f"[VID] {cam_name}: {msg}")
                    continue
                self._sources[cam_name] = src
                video_frames = max(video_frames, src.color_frames)
                self.get_logger().info(
                    f"[VID] {cam_name}: color={src.color_frames}f"
                    f"{' + depth-zarr' if src.has_depth else ' (no depth)'}")
            self._video_frames = video_frames
            self._idx = 0
            self._total = 0

    def _on_set_parameters(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == "episode_path":
                if str(p.value):
                    self._open_episode(str(p.value))
        return SetParametersResult(successful=True)

    # ── Subscribers ──
    def _cb_execute(self, msg):
        with self._lock:
            self._exec = bool(msg.data)
            if not self._exec:
                # reset so next run starts from frame 0 of color/depth
                for src in self._sources.values():
                    src._cur_idx = -1
                self._video_idx_from_replay = -1
                self._idx = 0
                self._total = 0
        self.get_logger().info(f"[VID] execute → {'ON' if msg.data else 'OFF'}")

    def _cb_replay_progress(self, msg):
        try:
            data = list(msg.data)
            if len(data) >= 3:
                with self._lock:
                    self._idx = int(data[0])
                    self._total = int(data[1])
                    # data[3] (extended): source-mp4 frame index — the canonical
                    # answer to "which frame should I show now?". Used in _tick
                    # to bypass the (lossy) idx-vs-video_T heuristic that breaks
                    # under resampling (e.g. parquet 15.7Hz → buffer 30Hz).
                    if len(data) >= 4:
                        self._video_idx_from_replay = int(data[3])
                    if data[2] > 0.5:
                        self._exec = False  # done flag → stop emitting
        except Exception:  # noqa: BLE001
            pass

    # ── Tick ──
    def _tick(self):
        with self._lock:
            if not self._exec or not self._sources or self._video_frames <= 0:
                return
            idx = self._idx
            total = self._total
            video_T = self._video_frames
            explicit_video_idx = self._video_idx_from_replay

        if explicit_video_idx >= 0:
            # Authoritative: replay_node already mapped buffer→source frame,
            # accounting for home pad + any resampling. Just clamp + use.
            video_idx = max(0, min(video_T - 1, explicit_video_idx))
        else:
            # Legacy /replay_progress (3-field): assume buffer is 1:1 with mp4
            # except for an optional auto-home prepend. Wrong under resampling
            # but matches behavior pre-fix.
            home_n = max(0, total - video_T)
            video_idx = max(0, min(video_T - 1, idx - home_n))

        now_stamp = self.get_clock().now().to_msg()
        for cam_name, src in self._sources.items():
            color = src.get_color(video_idx)
            if color is not None:
                self._publish_color(cam_name, color, now_stamp)
            depth = src.get_depth(video_idx) if src.has_depth else None
            if depth is not None:
                self._publish_depth(cam_name, depth, now_stamp)

    # ── Image publishing helpers ──
    def _publish_color(self, cam_name: str, bgr: np.ndarray, stamp):
        # mp4 decode is BGR; encode as rgb8 to match RealSense convention.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = f"camera_{CAM_MAP[cam_name]}_color_optical_frame"
        msg.height, msg.width = h, w
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = rgb.tobytes()
        self._pubs[cam_name]["color"].publish(msg)

    def _publish_depth(self, cam_name: str, depth_u16: np.ndarray, stamp):
        # 16-bit single-channel mm depth — same encoding as RealSense aligned depth.
        if depth_u16.dtype != np.uint16:
            depth_u16 = depth_u16.astype(np.uint16)
        h, w = depth_u16.shape[:2]
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = f"camera_{CAM_MAP[cam_name]}_depth_optical_frame"
        msg.height, msg.width = h, w
        msg.encoding = "16UC1"
        msg.is_bigendian = 0
        msg.step = w * 2
        msg.data = depth_u16.tobytes()
        self._pubs[cam_name]["depth"].publish(msg)


def main():
    rclpy.init()
    node = VideoPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for src in node._sources.values():
            src.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
