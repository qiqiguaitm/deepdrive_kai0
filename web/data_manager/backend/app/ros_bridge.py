"""ROS2 bridge.

真实实现：后台线程 spin 一个 rclpy Node，订阅 Piper 双臂 JointState + 三路 RealSense
camera_info，聚合为 StatusHub 需要的 get_health / get_joint_state / get_camera_health
三个同步接口。

当环境缺少 rclpy 或 YAML 配置时，自动退回到 MockBridge（保留仿真行为，便于在
无机器的机器上调试前端）。

可通过 KAI0_ROS_BRIDGE=mock 强制使用 mock；默认 auto。
"""
from __future__ import annotations

import math
import os
import random
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque

import yaml

# 默认配置路径：工程根目录/config/{pipers,cameras}.yml
_REPO_ROOT = Path(__file__).resolve().parents[4]  # .../deepdive_kai0
_DEFAULT_PIPERS_YML = _REPO_ROOT / "config" / "pipers.yml"
_DEFAULT_CAMERAS_YML = _REPO_ROOT / "config" / "cameras.yml"

PIPERS_YML = Path(os.environ.get("KAI0_PIPERS_YML", _DEFAULT_PIPERS_YML))
CAMERAS_YML = Path(os.environ.get("KAI0_CAMERAS_YML", _DEFAULT_CAMERAS_YML))

# 数据新鲜度阈值（秒）—— 超过则视为该数据源断开
FRESH_JOINT_S = 2.0
FRESH_CAM_S = 2.0


def _load_depth_enabled_map() -> dict[str, bool]:
    """Inline-load CAMERA_DEPTH_ENABLED from config/camera_depth_flags.py.

    Mirrored in recorder.py / sync.py — same path-probing pattern, since
    config/ isn't a Python package and `from config...` won't resolve at
    runtime.
    """
    import importlib.util
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config" / "camera_depth_flags.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "kai0_camera_depth_flags_bridge", candidate)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return dict(mod.CAMERA_DEPTH_ENABLED)
    # 找不到宏文件就当全部禁用 — 比误开订阅一通 depth topic 安全.
    return {}


# Per-camera depth on/off, populated once at import. Consumers (RclpyBridge)
# check this map before subscribing or accepting incoming depth frames.
_DEPTH_ENABLED_MAP = _load_depth_enabled_map()


# ---------------------------- Mock ------------------------------------------
class MockBridge:
    kind = "mock"

    def __init__(self) -> None:
        self._t0 = time.time()

    def get_health(self) -> dict[str, bool]:
        return {"ros2": False, "can_left": False, "can_right": False, "teleop": False}

    def get_camera_health(self) -> dict[str, dict]:
        return {
            cam: {"fps": round(29.0 + random.random(), 1), "target_fps": 30, "dropped": 0, "latency_ms": 35}
            for cam in ("top_head", "hand_left", "hand_right")
        }

    def get_joint_state(self) -> dict:
        t = time.time() - self._t0
        left = [0.5 * math.sin(t + i * 0.3) for i in range(7)]
        right = [0.5 * math.cos(t + i * 0.3) for i in range(7)]
        return {
            "left_joints": left,
            "right_joints": right,
            "left_gripper": 0.5 * (1 + math.sin(t)),
            "right_gripper": 0.5 * (1 + math.cos(t)),
            "left_temp": [round(35 + random.random() * 2, 1) for _ in range(7)],
            "right_temp": [round(35 + random.random() * 2, 1) for _ in range(7)],
            "left_torque": [round(random.random() * 0.5, 2) for _ in range(7)],
            "right_torque": [round(random.random() * 0.5, 2) for _ in range(7)],
        }

    def emergency_stop(self) -> bool:
        return True

    def get_frame_rgb(self, cam_name: str):
        import numpy as _np
        # 渐变条纹的假图，方便端到端 smoke test
        t = int((time.time() - self._t0) * 30) % 256
        arr = _np.zeros((480, 640, 3), dtype=_np.uint8)
        arr[:, :, 0] = (_np.arange(640) + t) % 256
        arr[:, :, 1] = (_np.arange(480)[:, None] + t) % 256
        return arr

    def get_frame_depth(self, cam_name: str):
        import numpy as _np
        # mock depth: 简单距离梯度 (mm)
        return ((_np.arange(640) + _np.arange(480)[:, None]).astype(_np.uint16) * 4)

    def get_replay_progress(self) -> dict | None:
        return None

    def clear_replay_progress(self) -> None:
        return

    def publish_execute(self, enabled: bool) -> bool:
        return False

    def get_state_action(self):
        js = self.get_joint_state()
        state = list(js["left_joints"]) + [js["left_gripper"]] + list(js["right_joints"]) + [js["right_gripper"]]
        # mock 没有独立 master，action=state
        return state[:14] + [0.0] * max(0, 14 - len(state)), state[:14] + [0.0] * max(0, 14 - len(state))


# ---------------------------- Real rclpy bridge -----------------------------
def _normalize_topic(topic: str) -> str:
    """去掉相邻重复段：/camera_f/camera_f/color/image_raw -> /camera_f/color/image_raw"""
    parts = [p for p in topic.split("/") if p]
    out: list[str] = []
    for p in parts:
        if out and out[-1] == p:
            continue
        out.append(p)
    return "/" + "/".join(out)


def _derive_camera_info_topic(color_topic: str) -> str:
    """/camera_f/color/image_raw -> /camera_f/color/camera_info（先规范化）"""
    base = _normalize_topic(color_topic).rsplit("/", 1)[0]
    return f"{base}/camera_info"


class RclpyBridge:
    """在后台线程中运行 rclpy，聚合最新话题数据。"""

    kind = "rclpy"

    def __init__(self, rclpy_mod, msg_mods, pipers_cfg: dict, cameras_cfg: dict) -> None:
        self._rclpy = rclpy_mod
        self._JointState = msg_mods["JointState"]
        self._CameraInfo = msg_mods["CameraInfo"]
        self._qos_sensor = msg_mods["qos_sensor"]
        # Replay (P2): subscribe /replay_progress, cache latest [idx, total, done].
        self._Float32MultiArray = msg_mods.get("Float32MultiArray")
        self._Bool = msg_mods.get("Bool")
        self._replay_progress: dict | None = None  # {idx, total, done, ts}
        self._replay_pub_execute = None  # publisher to /policy/execute (lazy)

        self._pipers = pipers_cfg["arms"]
        self._cameras = cameras_cfg["cameras"]

        # 最新数据缓存（线程安全：dict 单次赋值是原子的，读写简单字段够用；
        # 复杂结构用锁保护）
        self._lock = threading.Lock()
        self._latest_joint: dict[str, dict] = {}  # key -> {pos, eff, ts}
        self._cam_stamps: dict[str, Deque[float]] = {
            name: deque(maxlen=120) for name in self._cameras
        }
        self._cam_latency_ms: dict[str, float] = {name: 0.0 for name in self._cameras}
        self._cam_last_seq: dict[str, int] = {name: -1 for name in self._cameras}
        # 深度帧缓存: 由 recorder 每 tick 抓最新一张写入 zarr。无订阅则保持 None。
        self._cam_depth: dict[str, object] = {name: None for name in self._cameras}
        self._cam_depth_ts: dict[str, float] = {name: 0.0 for name in self._cameras}
        # 最新 JPEG 字节（供 MJPEG 端点读取）；事件用于阻塞式等待新帧
        self._cam_jpeg: dict[str, bytes] = {name: b"" for name in self._cameras}
        self._cam_frame_event: dict[str, threading.Event] = {
            name: threading.Event() for name in self._cameras
        }
        # 最新原始 RGB ndarray(H,W,3) uint8，供 recorder 抓帧编码
        self._cam_rgb: dict[str, object] = {name: None for name in self._cameras}
        self._cam_rgb_ts: dict[str, float] = {name: 0.0 for name in self._cameras}
        # JPEG 压缩/下采样参数（用环境变量可调）
        self._jpeg_quality = int(os.environ.get("KAI0_JPEG_QUALITY", "60"))
        self._jpeg_stride = max(1, int(os.environ.get("KAI0_JPEG_STRIDE", "2")))
        self._Image = msg_mods["Image"]
        self._PIL = msg_mods["PIL"]
        self._np = msg_mods["np"]

        self._node = None
        self._executor = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="rclpy-bridge")
        self._started = threading.Event()
        self._stop = threading.Event()
        self._thread.start()
        # 等 ROS 起好再返回
        self._started.wait(timeout=5.0)

    # ---- 后台 spin ----
    def _run(self) -> None:
        try:
            if not self._rclpy.ok():
                self._rclpy.init(args=None)
            self._node = self._rclpy.create_node("datacollect_bridge")

            # 关节订阅：/puppet/joint_{left,right} 和 /master/joint_{left,right}
            for arm_key, arm in self._pipers.items():
                topic = arm.get("ros2_topic_joint")
                if not topic:
                    continue
                self._node.create_subscription(
                    self._JointState, topic,
                    lambda msg, k=arm_key: self._on_joint(k, msg), 10,
                )

            # 相机订阅：camera_info (fps/latency) + Image (MJPEG 流)
            for cam_name, cam in self._cameras.items():
                color = cam.get("ros2_topic_color")
                if not color:
                    continue
                color = _normalize_topic(color)
                info_topic = _derive_camera_info_topic(color)
                self._node.create_subscription(
                    self._CameraInfo, info_topic,
                    lambda msg, k=cam_name: self._on_cam_info(k, msg),
                    self._qos_sensor,
                )
                self._node.create_subscription(
                    self._Image, color,
                    lambda msg, k=cam_name: self._on_cam_image(k, msg),
                    self._qos_sensor,
                )
                # 深度: 16-bit mono (encoding=16UC1), 单独缓存供 recorder 拉取.
                # 仅当宏 (config/camera_depth_flags.py) 把该相机标记为 True 时
                # 才订阅, 否则就算 cameras.yml 列了 topic 也不消费.
                depth = cam.get("ros2_topic_depth")
                if depth and _DEPTH_ENABLED_MAP.get(cam_name, False):
                    depth = _normalize_topic(depth)
                    self._node.create_subscription(
                        self._Image, depth,
                        lambda msg, k=cam_name: self._on_cam_depth(k, msg),
                        self._qos_sensor,
                    )

            # Replay progress subscriber (P2). Float32MultiArray data = [idx, total, done].
            if self._Float32MultiArray is not None:
                self._node.create_subscription(
                    self._Float32MultiArray, '/replay_progress',
                    self._on_replay_progress, 5,
                )
            # Lazy publisher for /policy/execute (Bool) — created on first use to avoid
            # consuming a publisher slot when replay is never invoked.
            if self._Bool is not None:
                self._replay_pub_execute = self._node.create_publisher(
                    self._Bool, '/policy/execute', 5)

            self._executor = self._rclpy.executors.SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._started.set()

            while not self._stop.is_set() and self._rclpy.ok():
                self._executor.spin_once(timeout_sec=0.1)
        except Exception as e:  # noqa: BLE001
            print(f"[ros_bridge] rclpy thread crashed: {e}")
            self._started.set()
        finally:
            try:
                if self._executor is not None:
                    self._executor.shutdown()
                if self._node is not None:
                    self._node.destroy_node()
            except Exception:  # noqa: BLE001
                pass

    # ---- Replay (P2) ----
    def _on_replay_progress(self, msg) -> None:
        """Cache latest /replay_progress: data = [idx, total, done_flag]."""
        try:
            data = list(msg.data)
            if len(data) >= 3:
                with self._lock:
                    self._replay_progress = {
                        "idx": int(data[0]),
                        "total": int(data[1]),
                        "done": bool(data[2] > 0.5),
                        "ts": time.time(),
                    }
        except Exception:  # noqa: BLE001
            pass

    def get_replay_progress(self) -> dict | None:
        """Return latest /replay_progress payload or None if never received.
        Stale (>2s old) reports get age annotation but are still returned."""
        with self._lock:
            p = dict(self._replay_progress) if self._replay_progress else None
        if p is not None:
            p["age_s"] = max(0.0, time.time() - p.pop("ts"))
        return p

    def clear_replay_progress(self) -> None:
        """Reset cached progress so a stale done=True from the previous run
        doesn't bleed into the next session. Call right before firing execute."""
        with self._lock:
            self._replay_progress = None

    def publish_execute(self, enabled: bool) -> bool:
        """Publish std_msgs/Bool to /policy/execute. Returns True on success."""
        if self._replay_pub_execute is None or self._Bool is None:
            return False
        try:
            msg = self._Bool()
            msg.data = bool(enabled)
            self._replay_pub_execute.publish(msg)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[ros_bridge] publish_execute failed: {e}")
            return False

    # ---- 回调 ----
    def _on_joint(self, arm_key: str, msg) -> None:
        with self._lock:
            self._latest_joint[arm_key] = {
                "pos": list(msg.position),
                "eff": list(msg.effort) if len(msg.effort) else [],
                "ts": time.time(),
            }

    def _on_cam_image(self, cam_name: str, msg) -> None:
        """把 sensor_msgs/Image 编码成 JPEG，存入最新帧槽。"""
        try:
            w, h, enc = msg.width, msg.height, msg.encoding
            data = bytes(msg.data)
            np = self._np
            if enc in ("rgb8", "bgr8"):
                arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
                if enc == "bgr8":
                    arr = arr[:, :, ::-1]
            elif enc in ("mono8", "8UC1"):
                arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w)
            else:
                return  # 不支持的编码
            # 原图（RGB）缓存给录制模块（独立于 MJPEG 下采样）
            if arr.ndim == 3:
                rgb_full = np.ascontiguousarray(arr)
                with self._lock:
                    self._cam_rgb[cam_name] = rgb_full
                    self._cam_rgb_ts[cam_name] = time.time()
            # 下采样降低 MJPEG 带宽
            s = self._jpeg_stride
            if s > 1:
                arr = arr[::s, ::s]
            img = self._PIL.fromarray(arr)
            import io
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self._jpeg_quality)
            jpeg = buf.getvalue()
            with self._lock:
                self._cam_jpeg[cam_name] = jpeg
            self._cam_frame_event[cam_name].set()
        except Exception as e:  # noqa: BLE001
            print(f"[ros_bridge] encode {cam_name} failed: {e}")

    def _on_cam_depth(self, cam_name: str, msg) -> None:
        """缓存最新一张 depth 帧 (uint16, mm). 不做编码, recorder 直接读 ndarray 写 zarr。"""
        try:
            w, h, enc = msg.width, msg.height, msg.encoding
            if enc not in ("16UC1", "mono16"):
                return  # RealSense depth 是 16UC1; 其它编码先忽略
            np = self._np
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(h, w)
            arr = np.ascontiguousarray(arr)
            with self._lock:
                self._cam_depth[cam_name] = arr
                self._cam_depth_ts[cam_name] = time.time()
        except Exception as e:  # noqa: BLE001
            print(f"[ros_bridge] decode depth {cam_name} failed: {e}")

    def get_frame_depth(self, cam_name: str):
        """返回最新一帧 (H,W) uint16 深度图 (mm); 无订阅或无数据返回 None."""
        with self._lock:
            return self._cam_depth.get(cam_name)

    def get_latest_jpeg(self, cam_name: str, wait_timeout: float = 1.0) -> bytes | None:
        """阻塞等待一帧新 JPEG；返回 None 表示超时。"""
        ev = self._cam_frame_event.get(cam_name)
        if ev is None:
            return None
        got = ev.wait(timeout=wait_timeout)
        ev.clear()
        if not got:
            return None
        with self._lock:
            return self._cam_jpeg.get(cam_name) or None

    def get_frame_rgb(self, cam_name: str):
        """返回最新一帧 (H,W,3) uint8 RGB；无帧时返回 None。"""
        with self._lock:
            return self._cam_rgb.get(cam_name)

    def get_state_action(self):
        """返回 (state14, action14) float32 list。
        state = puppet(从臂) 关节 = 真实机器人当前位姿；
        默认 action = state（官方 KAI0 约定，模型当 state predictor 训）；
        顺序: [L_j1..L_j6, L_gripper, R_j1..R_j6, R_gripper] (7+7=14)。
        Set KAI0_ACTION_EQ_STATE=0 to fall back to legacy bilateral capture
        (action = master, fallback slave when master topic missing).
        """
        def _pick(key: str) -> list[float]:
            with self._lock:
                j = self._latest_joint.get(key)
            if not j:
                return [0.0] * 7
            pos = list(j["pos"])[:7]
            pos += [0.0] * (7 - len(pos))
            return pos

        state = _pick("left_slave") + _pick("right_slave")
        if os.environ.get("KAI0_ACTION_EQ_STATE", "1") == "1":
            return state, list(state)

        l_master = _pick("left_master")
        r_master = _pick("right_master")
        with self._lock:
            has_lm = "left_master" in self._latest_joint
            has_rm = "right_master" in self._latest_joint
        action_l = l_master if has_lm else state[:7]
        action_r = r_master if has_rm else state[7:]
        return state, action_l + action_r

    def _on_cam_info(self, cam_name: str, msg) -> None:
        now = time.time()
        stamp = msg.header.stamp
        msg_t = stamp.sec + stamp.nanosec * 1e-9
        lat_ms = max(0.0, (now - msg_t) * 1000.0) if msg_t > 0 else 0.0
        with self._lock:
            self._cam_stamps[cam_name].append(now)
            self._cam_latency_ms[cam_name] = lat_ms

    # ---- 对外接口 ----
    def get_health(self) -> dict[str, bool]:
        now = time.time()
        with self._lock:
            ls = self._latest_joint.get("left_slave") or self._latest_joint.get("left_master")
            rs = self._latest_joint.get("right_slave") or self._latest_joint.get("right_master")
            lm = self._latest_joint.get("left_master")
            rm = self._latest_joint.get("right_master")
        return {
            "ros2": True,
            "can_left": bool(ls and now - ls["ts"] < FRESH_JOINT_S),
            "can_right": bool(rs and now - rs["ts"] < FRESH_JOINT_S),
            "teleop": bool(
                lm and rm
                and now - lm["ts"] < FRESH_JOINT_S
                and now - rm["ts"] < FRESH_JOINT_S
            ),
        }

    def get_joint_state(self) -> dict:
        """优先取从臂 (puppet) 作为当前真实状态；取不到就退回主臂或零值。"""
        def _pick(side: str) -> dict:
            with self._lock:
                j = self._latest_joint.get(f"{side}_slave") or self._latest_joint.get(f"{side}_master")
            if not j:
                return {"pos": [0.0] * 7, "eff": [0.0] * 7}
            pos = list(j["pos"]) + [0.0] * (7 - len(j["pos"]))
            eff = list(j["eff"]) + [0.0] * (7 - len(j["eff"]))
            return {"pos": pos[:7], "eff": eff[:7]}

        left, right = _pick("left"), _pick("right")
        return {
            "left_joints": left["pos"][:6],
            "right_joints": right["pos"][:6],
            "left_gripper": left["pos"][6] if len(left["pos"]) > 6 else 0.0,
            "right_gripper": right["pos"][6] if len(right["pos"]) > 6 else 0.0,
            # 温度通道暂无，占位为 None
            "left_temp": [None] * 7,
            "right_temp": [None] * 7,
            "left_torque": [round(abs(v), 3) for v in left["eff"]],
            "right_torque": [round(abs(v), 3) for v in right["eff"]],
        }

    def get_camera_health(self) -> dict[str, dict]:
        now = time.time()
        out: dict[str, dict] = {}
        with self._lock:
            for cam in self._cameras:
                stamps = self._cam_stamps[cam]
                recent = [s for s in stamps if now - s < 1.0]
                fps = round(float(len(recent)), 1)
                target = float(self._cameras[cam].get("fps", 30))
                # `dropped` 是"过去 1 秒内相对 target_fps 缺的帧数", 瞬时值,
                # 不再累加 (旧版本 += 后, StatusHub 以 2Hz 调用 → 同一秒的缺帧
                # 被算两次, 而且数字只增不减; 哪怕相机已恢复也一直显示历史峰值).
                # 现在: 健康时显示 0; 卡顿时显示当前缺多少帧/秒, 自动随状态变化.
                dropped = max(0, int(round(target - fps)))
                last_seen = stamps[-1] if stamps else 0.0
                stale = (now - last_seen) > FRESH_CAM_S if last_seen else True
                out[cam] = {
                    "fps": 0.0 if stale else fps,
                    "target_fps": int(target),
                    "dropped": dropped,
                    "latency_ms": 0 if stale else int(round(self._cam_latency_ms[cam])),
                }
        return out

    def emergency_stop(self) -> bool:
        # TODO: 若需要真实急停，在此发布 /enable_flag=False 或自定义 topic
        return True

    def shutdown(self) -> None:
        self._stop.set()


# ---------------------------- Factory ---------------------------------------
def _make_bridge():
    mode = os.environ.get("KAI0_ROS_BRIDGE", "auto").lower()
    if mode == "mock":
        print("[ros_bridge] KAI0_ROS_BRIDGE=mock -> MockBridge")
        return MockBridge()

    try:
        import rclpy  # noqa: F401
        import rclpy.executors  # noqa: F401
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from sensor_msgs.msg import CameraInfo, Image, JointState
        import numpy as np
        from PIL import Image as PILImage
    except Exception as e:  # noqa: BLE001
        print(f"[ros_bridge] rclpy unavailable ({e}); using MockBridge")
        return MockBridge()

    if not PIPERS_YML.exists() or not CAMERAS_YML.exists():
        print(f"[ros_bridge] yml missing ({PIPERS_YML} / {CAMERAS_YML}); using MockBridge")
        return MockBridge()

    with open(PIPERS_YML) as f:
        pipers_cfg = yaml.safe_load(f)
    with open(CAMERAS_YML) as f:
        cameras_cfg = yaml.safe_load(f)

    qos_sensor = QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
    )

    try:
        import rclpy as _rclpy
        from std_msgs.msg import Float32MultiArray, Bool as BoolMsg
        b = RclpyBridge(
            _rclpy,
            {
                "JointState": JointState, "CameraInfo": CameraInfo, "Image": Image,
                "PIL": PILImage, "np": np, "qos_sensor": qos_sensor,
                "Float32MultiArray": Float32MultiArray, "Bool": BoolMsg,
            },
            pipers_cfg,
            cameras_cfg,
        )
        print("[ros_bridge] RclpyBridge online")
        return b
    except Exception as e:  # noqa: BLE001
        print(f"[ros_bridge] failed to start RclpyBridge ({e}); falling back to MockBridge")
        return MockBridge()


bridge = _make_bridge()
