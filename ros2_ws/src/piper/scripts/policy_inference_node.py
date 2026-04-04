#!/usr/bin/python3
"""
ROS2 Policy Inference Node — 将 pi0.5 推理直接集成为 ROS2 节点

三种推理模式 (通过 --mode 参数选择):
  1. "ros2"      — 纯 ROS2 模式: JAX 推理在本节点内完成, 无 WebSocket
  2. "websocket" — 原版模式: 通过 WebSocket 连接外部 serve_policy.py
  3. "both"      — 同时启动: 本节点加载模型, 同时兼容 WebSocket 客户端

用法:
  # 模式 1: 纯 ROS2 (推荐, 最低延迟)
  ros2 run piper policy_inference_node.py --ros-args \
    -p mode:=ros2 \
    -p config_name:=pi05_flatten_fold_normal \
    -p checkpoint_dir:=gs://openpi-assets/checkpoints/pi05_base/params

  # 模式 2: WebSocket 客户端 (兼容旧的 serve_policy.py)
  ros2 run piper policy_inference_node.py --ros-args \
    -p mode:=websocket -p host:=localhost -p port:=8000

  # 模式 3: 两者兼有
  ros2 run piper policy_inference_node.py --ros-args \
    -p mode:=both \
    -p config_name:=pi05_flatten_fold_normal \
    -p checkpoint_dir:=gs://openpi-assets/checkpoints/pi05_base/params \
    -p ws_port:=8000

订阅:
  /camera_f/camera/color/image_raw  (sensor_msgs/Image)     — 头顶相机
  /camera_l/camera/color/image_raw  (sensor_msgs/Image)     — 左腕相机
  /camera_r/camera/color/image_raw  (sensor_msgs/Image)     — 右腕相机
  /puppet/joint_left            (sensor_msgs/JointState) — 左臂关节状态
  /puppet/joint_right           (sensor_msgs/JointState) — 右臂关节状态

发布:
  /policy/actions               (sensor_msgs/JointState) — 推理输出动作 (14 维)
  /master/joint_left            (sensor_msgs/JointState) — 左臂控制命令
  /master/joint_right           (sensor_msgs/JointState) — 右臂控制命令
"""

import os
import sys

# ── 自动 re-exec: 确保在 kai0 venv 中运行 ────────────────────────
# ros2 run 通过 shebang (#!/usr/bin/env python3) 启动, 可能命中 conda 的
# python3.13 或系统 python3.12, 而本节点依赖 venv 中的 JAX/numpy/cv2 等包.
# 检测当前是否在 kai0 venv 中, 如果不是则 re-exec.
# KAI0_ROOT 查找顺序: 环境变量 > 相对路径推导 (source 和 install 两种布局)
_KAI0_ROOT = os.environ.get('KAI0_ROOT', '')
if not _KAI0_ROOT or not os.path.isdir(_KAI0_ROOT):
    # 从 __file__ 位置推导: source 布局 (ros2_ws/src/piper/scripts/ → ../../.. → kai0)
    for levels in [
        ('..', '..', '..', '..', 'kai0'),          # source: ros2_ws/src/piper/scripts/
        ('..', '..', '..', '..', '..', 'kai0'),     # install: ros2_ws/install/piper/lib/piper/
    ]:
        candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), *levels))
        if os.path.isdir(os.path.join(candidate, 'src', 'openpi')):
            _KAI0_ROOT = candidate
            break
    if not _KAI0_ROOT:
        # 最终回退: home-relative paths (avoid machine-specific hardcoded paths)
        for fallback in [os.path.expanduser('~/workspace/deepdive_kai0/kai0'),
                         os.path.join(os.path.expanduser('~'), 'kai0')]:
            if os.path.isdir(os.path.join(fallback, 'src', 'openpi')):
                _KAI0_ROOT = fallback
                break
_VENV_PYTHON = os.path.join(_KAI0_ROOT, '.venv', 'bin', 'python')
_VENV_PREFIX = os.path.join(_KAI0_ROOT, '.venv')

if (os.access(_VENV_PYTHON, os.X_OK)
        and os.path.abspath(sys.prefix) != os.path.abspath(_VENV_PREFIX)):
    # 当前不在 kai0 venv 中 (可能是 conda python3.13 或裸系统 python3.12)
    # 清理 PATH 中的 conda 路径, 防止 conda 的 libpython/importlib 污染 re-exec 后的进程
    _clean_path = ':'.join(p for p in os.environ.get('PATH', '').split(':')
                           if 'conda' not in p.lower())
    os.environ['PATH'] = _clean_path
    # 确保 LD_LIBRARY_PATH 也不含 conda
    _clean_ld = ':'.join(p for p in os.environ.get('LD_LIBRARY_PATH', '').split(':')
                         if 'conda' not in p.lower())
    os.environ['LD_LIBRARY_PATH'] = _clean_ld
    # 用 venv python 重新启动自己, 保留所有命令行参数
    os.execv(_VENV_PYTHON, [_VENV_PYTHON] + sys.argv)

import time
import threading
from collections import deque

# 确保 openpi src 可被 import
_KAI0_SRC = os.path.join(_KAI0_ROOT, 'src')
if os.path.isdir(_KAI0_SRC) and _KAI0_SRC not in sys.path:
    sys.path.insert(0, _KAI0_SRC)

# 确保 CUDA 库路径在 JAX import 前设好
import glob as _glob
_venv_lib = os.path.join(_KAI0_ROOT, '.venv', 'lib')
_venv_pydirs = sorted(_glob.glob(os.path.join(_venv_lib, 'python3.*')))
_venv_sp = os.path.join(_venv_pydirs[-1], 'site-packages') if _venv_pydirs else os.path.join(_venv_lib, 'python3.12', 'site-packages')
_venv_nvidia = os.path.join(_venv_sp, 'nvidia')
_nvidia_libs = ':'.join(sorted(_glob.glob(os.path.join(_venv_nvidia, '*', 'lib'))))
if _nvidia_libs:
    os.environ['LD_LIBRARY_PATH'] = _nvidia_libs + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    import ctypes
    try:
        for lib_dir in _nvidia_libs.split(':'):
            for so in sorted(_glob.glob(os.path.join(lib_dir, '*.so*'))):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
    except Exception:
        pass

import select
import cv2
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Header
from scipy.spatial.transform import Rotation as R_


def _stamp_to_sec(stamp):
    """Convert a ROS2 stamp (sec + nanosec) to float seconds."""
    return stamp.sec + stamp.nanosec * 1e-9


def _to_bool(value) -> bool:
    """Convert a ROS2 parameter value to bool.

    LaunchConfiguration always passes strings, so 'false'/'true' must be
    handled explicitly — Python's bool('false') is True!
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('true', '1', 'yes')
    return bool(value)


# ────────────────────────────────────────────────────────────────────────────
# StreamActionBuffer — 从原版 agilex_inference_openpi_temporal_smoothing_ros2.py
#                      逐行复制, 不做任何修改
# ────────────────────────────────────────────────────────────────────────────
class StreamActionBuffer:
    """
    Maintains a queue of action chunks; each chunk is a deque([action0, action1, ...]).
    - New inferred chunks are appended from the right;
    - For each published step, popleft() the leftmost action from each chunk;
    - Empty chunks are dropped.
    """
    def __init__(self, max_chunks=10, decay_alpha=0.25, state_dim=14, smooth_method="temporal"):
        self.chunks = deque()                 # Kept for backward compatibility
        self.max_chunks = max_chunks
        self.lock = threading.Lock()
        self.decay_alpha = float(decay_alpha)  # Smoothing strength (exponential weight)
        self.state_dim = state_dim
        self.smooth_method = smooth_method
        self.cur_chunk = deque()              # Current sequence to publish (after smoothing)
        self.k = 0                            # Published step count (for latency trimming)
        self.last_action = None               # Last successfully popped action

    def push_chunk(self, actions_chunk: np.ndarray):
        """Legacy interface (no longer used)."""
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            dq = deque([a.copy() for a in actions_chunk], maxlen=None)
            self.chunks.append(dq)
            while len(self.chunks) > self.max_chunks:
                self.chunks.popleft()

    def integrate_new_chunk(self, actions_chunk: np.ndarray, max_k: int, min_m: int = 8):
        """
        Integrate a new inference chunk:
        1) Trim the front of the new chunk by current k and max_k (latency compensation).
        2) If there is an existing chunk (cur_chunk), apply temporal smoothing on the overlap:
           - Overlap: first element 100% old / 0% new, last element 0% old / 100% new.
           - Extra tail from the new chunk is appended.
        3) Reset k=0 as the new current execution sequence.
        """
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            max_k = max(0, int(max_k))
            min_m = max(1, int(min_m))
            drop_n = min(self.k, max_k)
            if drop_n >= len(actions_chunk):
                # Entire chunk trimmed; skip this update
                return
            new_chunk = [a.copy() for a in actions_chunk[drop_n:]]
            # Build old sequence: if empty but last_action exists, extend with last_action to min_m steps;
            # if non-empty and len < m, pad tail to min_m; if both empty, take new sequence as-is
            if len(self.cur_chunk) == 0 and self.last_action is not None:
                old_list = [np.asarray(self.last_action, dtype=float).copy() for _ in range(min_m)]
                self.last_action = None
            else:
                old_list = list(self.cur_chunk)
                if len(old_list) > 0 and len(old_list) < min_m:
                    tail = np.asarray(old_list[-1], dtype=float).copy()
                    old_list.extend([tail.copy() for _ in range(min_m - len(old_list))])
                elif len(old_list) == 0:
                    self.cur_chunk = deque(new_chunk, maxlen=None)
                    self.k = 0
                    return
            new_list = list(new_chunk)

            # Overlap length = min of remaining old length and new length
            overlap_len = min(len(old_list), len(new_list))
            if overlap_len <= 0:
                # No overlap; use new sequence as-is
                self.cur_chunk = deque(new_list, maxlen=None)
                self.k = 0
                return

            # If old sequence is longer than new, trim old tail
            if len(old_list) > len(new_list):
                old_list = old_list[:len(new_list)]
                overlap_len = len(new_list)

            # Linear weights: first element 100% old, last element 0% old
            if overlap_len == 1:
                w_old = np.array([1.0], dtype=float)
            else:
                w_old = np.linspace(1.0, 0.0, overlap_len, dtype=float)
            w_new = 1.0 - w_old

            # Smooth the overlap region
            smoothed = [
                (w_old[i] * np.asarray(old_list[i], dtype=float) +
                 w_new[i] * np.asarray(new_list[i], dtype=float))
                for i in range(overlap_len)
            ]
            # Append the extra tail from the new sequence
            combined = smoothed + new_list[overlap_len:]
            self.cur_chunk = deque([a.copy() for a in combined], maxlen=None)
            self.k = 0

    def has_any(self):
        with self.lock:
            return len(self.cur_chunk) > 0

    def pop_next_action(self) -> np.ndarray | None:
        """Pop and return the next action to publish; k += 1."""
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            # If about to pop the last element, save it as last_action
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(self.cur_chunk[0], dtype=float).copy()
            act = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self.k += 1
            return act

    def flush(self, seed_action: np.ndarray | None = None):
        """Discard all buffered actions and optionally seed for smooth blending."""
        with self.lock:
            self.cur_chunk.clear()
            self.k = 0
            self.last_action = seed_action


# ────────────────────────────────────────────────────────────────────────────
# PolicyInferenceNode
# ────────────────────────────────────────────────────────────────────────────
class PolicyInferenceNode(Node):
    """ROS2 node that integrates policy inference directly."""

    def __init__(self):
        super().__init__('policy_inference_node')

        # ── Parameters ──
        self.declare_parameter('mode', 'ros2')  # ros2 | websocket | both
        self.declare_parameter('config_name', 'pi05_flatten_fold_normal')
        self.declare_parameter('checkpoint_dir', '')
        self.declare_parameter('host', 'localhost')
        self.declare_parameter('port', 8000)
        self.declare_parameter('ws_port', 8000)
        self.declare_parameter('prompt', 'Flatten and fold the cloth.')
        self.declare_parameter('publish_rate', 30)
        self.declare_parameter('inference_rate', 3.0)
        self.declare_parameter('chunk_size', 50)
        self.declare_parameter('latency_k', 8)
        self.declare_parameter('min_smooth_steps', 8)
        self.declare_parameter('decay_alpha', 0.25)
        self.declare_parameter('gripper_offset', 0.003)
        self.declare_parameter('img_front_topic', '/camera_f/camera/color/image_raw')
        self.declare_parameter('img_left_topic', '/camera_l/camera/color/image_raw')
        self.declare_parameter('img_right_topic', '/camera_r/camera/color/image_raw')
        self.declare_parameter('puppet_left_topic', '/puppet/joint_left')
        self.declare_parameter('puppet_right_topic', '/puppet/joint_right')
        self.declare_parameter('depth_front_topic', '/camera_f/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('gpu_id', 0)
        self.declare_parameter('execute_mode', False)
        self.declare_parameter('enable_rerun', False)
        self.declare_parameter('calibration_config', '')

        self.mode = self.get_parameter('mode').value
        self.prompt = self.get_parameter('prompt').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.inference_rate = self.get_parameter('inference_rate').value
        self.chunk_size = self.get_parameter('chunk_size').value
        self.latency_k = self.get_parameter('latency_k').value
        self.min_smooth_steps = self.get_parameter('min_smooth_steps').value
        self.decay_alpha = self.get_parameter('decay_alpha').value
        self.gripper_offset = self.get_parameter('gripper_offset').value

        self.get_logger().info(f'Mode: {self.mode}')

        # ── State ──
        self.bridge = CvBridge()
        self.policy = None
        self.stream_buffer = StreamActionBuffer(
            decay_alpha=self.decay_alpha, state_dim=14)

        # ── Sensor deques (原版帧同步模式: 回调 append, get_synced_frame 消费) ──
        self._sensor_lock = threading.Lock()
        self._img_front_deque = deque(maxlen=200)
        self._img_left_deque = deque(maxlen=200)
        self._img_right_deque = deque(maxlen=200)
        self._joint_left_deque = deque(maxlen=200)
        self._joint_right_deque = deque(maxlen=200)

        # ── Execution control ──
        self._exec_lock = threading.Lock()
        self._execution_enabled = _to_bool(self.get_parameter('execute_mode').value)
        self._last_published_action = None  # for jump detection
        self._MAX_JOINT_JUMP_RAD = 0.5  # ~28.6°, reject actions with larger per-joint jump
        mode_str = 'EXECUTE' if self._execution_enabled else 'OBSERVE'
        self.get_logger().info(f'Execution mode: [{mode_str}]')

        # ── Subscribers ──
        # 执行控制 topic
        self.create_subscription(Bool, '/policy/execute', self._cb_execute, 1)
        # 图像: RELIABLE 匹配 RealSense v4.56+ 默认 QoS (RELIABLE + TRANSIENT_LOCAL)
        # BEST_EFFORT subscriber 无法接收 RELIABLE publisher 的消息
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(Image,
            self.get_parameter('img_front_topic').value,
            self._cb_img_front, img_qos)
        self.create_subscription(Image,
            self.get_parameter('img_left_topic').value,
            self._cb_img_left, img_qos)
        self.create_subscription(Image,
            self.get_parameter('img_right_topic').value,
            self._cb_img_right, img_qos)
        # Depth image for Rerun point cloud (head camera only, aligned to color)
        self.create_subscription(Image,
            self.get_parameter('depth_front_topic').value,
            self._cb_depth_front, img_qos)
        self.create_subscription(JointState,
            self.get_parameter('puppet_left_topic').value,
            self._cb_joint_left, 1000)
        self.create_subscription(JointState,
            self.get_parameter('puppet_right_topic').value,
            self._cb_joint_right, 1000)

        # ── Publishers ──
        self.pub_action = self.create_publisher(JointState, '/policy/actions', 10)
        self.pub_left = self.create_publisher(JointState, '/master/joint_left', 10)
        self.pub_right = self.create_publisher(JointState, '/master/joint_right', 10)
        # Full predicted action chunk for visualization (50 steps × 14 dims)
        from std_msgs.msg import Float32MultiArray
        self.pub_action_chunk = self.create_publisher(
            Float32MultiArray, '/policy/action_chunk', 5)

        # ── Rerun visualization (conditional) ──
        # Rerun set_time+log is non-atomic. All ROS2 callbacks/timers MUST share
        # a single thread. _rr_require_single_thread is checked in on_configure
        # and main() to enforce this at runtime, not just via comments.
        self._rr_require_single_thread = False
        self._use_rerun = _to_bool(self.get_parameter('enable_rerun').value)
        self._rr = None  # rerun module, lazy import
        self._rr_rec = None  # rerun RecordingStream
        self._rr_bridge = None  # separate CvBridge for Rerun callbacks (thread safety)
        self._fk = None
        self._T_world_baseL = None
        self._T_world_baseR = None
        self._T_world_camF = None
        self._cam_f_fx = self._cam_f_fy = self._cam_f_cx = self._cam_f_cy = None
        self._actual_trail_left = deque(maxlen=300)
        self._actual_trail_right = deque(maxlen=300)
        self._rr_frame_idx_left = 0
        self._rr_frame_idx_right = 0
        self._rr_fk_skip = 3  # only run FK visualization every N-th joint callback
        self._rr_fk_queue = deque(maxlen=60)  # (arm_label, q, T_base, stamp) for off-callback FK
        self._rr_fk_drops = 0  # count of dropped FK frames due to queue overflow
        self._rr_pred_queue = deque(maxlen=2)  # (actions, stamp) for off-thread predicted traj FK
        self._rr_pred_drops = 0  # count of dropped predicted trajectory frames
        # Queue for Rerun events from non-executor threads (keyboard, etc.)
        # Entries: (entity: str, data_factory: callable, stamp_sec: float)
        self._rr_event_queue = deque(maxlen=20)
        if self._use_rerun:
            self._init_rerun()
            # Rerun timer — processes FK queue, predicted traj, and cross-thread events
            if self._use_rerun:
                self._rr_require_single_thread = True
                self.create_timer(1.0 / 10.0, self._rr_process_fk_queue)  # ~10Hz

        # ── Load policy ──
        self._load_policy()

        # ── Keyboard listener thread ──
        self._kb_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self._kb_thread.start()

        # ── Inference thread ──
        self._infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._infer_thread.start()

        # ── Publish timer ──
        period = 1.0 / self.publish_rate
        self.create_timer(period, self._publish_action)

        self.get_logger().info('Policy inference node ready')

    # ── Policy loading ──────────────────────────────────────────────

    def _load_policy(self):
        """Load policy based on mode."""
        if self.mode in ('ros2', 'both'):
            self._load_jax_policy()
            if self.mode == 'both':
                self._start_ws_server()
        elif self.mode == 'websocket':
            self._load_ws_policy()

    def _load_jax_policy(self):
        """Load JAX model directly (no WebSocket)."""
        config_name = self.get_parameter('config_name').value
        checkpoint_dir = self.get_parameter('checkpoint_dir').value
        gpu_id = str(self.get_parameter('gpu_id').value)

        if not checkpoint_dir:
            raise ValueError(
                'checkpoint_dir is required for ros2/both mode. '
                'Set via: -p checkpoint_dir:=gs://openpi-assets/checkpoints/pi05_base/params')

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR', '/tmp/xla_cache')
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['JAX_COMPILATION_CACHE_DIR'] = cache_dir

        # 确保 CUDA 库路径 (reuse auto-detected site-packages path)
        venv_nvidia = os.path.join(_venv_sp, 'nvidia')
        if os.path.exists(venv_nvidia):
            cuda_libs = ':'.join(
                os.path.join(venv_nvidia, d, 'lib')
                for d in os.listdir(venv_nvidia)
                if os.path.isdir(os.path.join(venv_nvidia, d, 'lib'))
            )
            os.environ['LD_LIBRARY_PATH'] = cuda_libs + ':' + os.environ.get('LD_LIBRARY_PATH', '')

        self.get_logger().info(f'Loading JAX policy: config={config_name}, ckpt={checkpoint_dir}, GPU={gpu_id}')
        t0 = time.monotonic()

        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        train_config = _config.get_config(config_name)
        self.policy = _policy_config.create_trained_policy(
            train_config, checkpoint_dir)

        self.get_logger().info(f'JAX policy loaded in {time.monotonic()-t0:.1f}s')

    def _load_ws_policy(self):
        """Connect to external serve_policy.py via WebSocket."""
        host = self.get_parameter('host').value
        port = self.get_parameter('port').value
        self.get_logger().info(f'Connecting to WebSocket policy at {host}:{port}')

        from openpi_client import websocket_client_policy
        self.policy = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.get_logger().info('WebSocket policy connected')

    def _start_ws_server(self):
        """In 'both' mode, also serve the loaded policy via WebSocket."""
        ws_port = self.get_parameter('ws_port').value
        self.get_logger().info(f'Starting WebSocket server on :{ws_port}')

        from openpi.serving import websocket_policy_server
        server = websocket_policy_server.WebsocketPolicyServer(
            policy=self.policy, host='0.0.0.0', port=ws_port,
            metadata=getattr(self.policy, 'metadata', {}))
        ws_thread = threading.Thread(target=server.serve_forever, daemon=True)
        ws_thread.start()
        self.get_logger().info(f'WebSocket server running on :{ws_port}')

    # ── Sensor callbacks (原版 deque 模式, maxlen=200) ──────────────

    def _rr_log_with_time(self, entity, data, stamp_sec):
        """Log to Rerun with explicit set_time + log, minimizing window for interleaving.

        NOTE: set_time + log is NOT atomic. This is safe because all callers
        (image/joint callbacks, FK timer) run on the ROS2 single-threaded executor.
        Do NOT switch to MultiThreadedExecutor without adding a Rerun lock.
        """
        rr = self._rr
        rr.set_time("ros_time", timestamp=stamp_sec)
        rr.log(entity, data)

    def _cb_img_front(self, msg):
        with self._sensor_lock:
            self._img_front_deque.append(msg)
        if self._use_rerun and self._rr_bridge is not None:
            try:
                img = self._rr_bridge.imgmsg_to_cv2(msg, 'rgb8')
                self._rr_log_with_time("images/top_head", self._rr.Image(img),
                                       _stamp_to_sec(msg.header.stamp))
            except Exception as e:
                self.get_logger().debug(f'Rerun img_front error: {e}')

    def _cb_img_left(self, msg):
        with self._sensor_lock:
            self._img_left_deque.append(msg)
        if self._use_rerun and self._rr_bridge is not None:
            try:
                img = self._rr_bridge.imgmsg_to_cv2(msg, 'rgb8')
                self._rr_log_with_time("images/hand_left", self._rr.Image(img),
                                       _stamp_to_sec(msg.header.stamp))
            except Exception as e:
                self.get_logger().debug(f'Rerun img_left error: {e}')

    def _cb_img_right(self, msg):
        with self._sensor_lock:
            self._img_right_deque.append(msg)
        if self._use_rerun and self._rr_bridge is not None:
            try:
                img = self._rr_bridge.imgmsg_to_cv2(msg, 'rgb8')
                self._rr_log_with_time("images/hand_right", self._rr.Image(img),
                                       _stamp_to_sec(msg.header.stamp))
            except Exception as e:
                self.get_logger().debug(f'Rerun img_right error: {e}')

    def _cb_depth_front(self, msg):
        """Project aligned depth from head camera into a 3D point cloud for Rerun."""
        if not self._use_rerun or self._rr is None:
            return
        if self._T_world_camF is None:
            return
        try:
            rr = self._rr
            bridge = self._rr_bridge or CvBridge()
            depth = bridge.imgmsg_to_cv2(msg, 'passthrough')  # uint16, mm
            if depth.dtype == np.uint16:
                depth_m = depth.astype(np.float32) * 0.001
            else:
                depth_m = depth.astype(np.float32)

            # Downsample for performance (every 4th pixel)
            step = 4
            h, w = depth_m.shape
            v, u = np.mgrid[0:h:step, 0:w:step]
            z = depth_m[0:h:step, 0:w:step]
            valid = (z > 0.05) & (z < 1.5)
            u = u[valid].astype(np.float32)
            v = v[valid].astype(np.float32)
            z = z[valid]

            if len(z) == 0:
                return

            # Unproject using head camera intrinsics
            fx = self._cam_f_fx
            fy = self._cam_f_fy
            cx = self._cam_f_cx
            cy = self._cam_f_cy
            x_cam = (u - cx) * z / fx
            y_cam = (v - cy) * z / fy
            pts_cam = np.stack([x_cam, y_cam, z], axis=-1)  # (N, 3)

            # Transform to world frame
            R_wc = self._T_world_camF[:3, :3]
            t_wc = self._T_world_camF[:3, 3]
            pts_world = (R_wc @ pts_cam.T).T + t_wc

            stamp_sec = _stamp_to_sec(msg.header.stamp)
            rr.set_time("ros_time", timestamp=stamp_sec)
            rr.log("world/point_cloud", rr.Points3D(
                pts_world, radii=[0.002], colors=[[180, 180, 180]]))
        except Exception as e:
            self.get_logger().debug(f'Rerun depth_front error: {e}')

    def _cb_joint_left(self, msg):
        with self._sensor_lock:
            self._joint_left_deque.append(msg)
        if self._use_rerun and self._rr is not None:
            try:
                q = np.array(msg.position)
                self._rr_frame_idx_left += 1
                frame_idx = self._rr_frame_idx_left
                stamp = _stamp_to_sec(msg.header.stamp)
                rr = self._rr
                rr.set_time("ros_time", timestamp=stamp)
                rr.set_time("frame_left", sequence=frame_idx)
                for i in range(min(7, len(q))):
                    rr.log(f"timeseries/left_j{i}", rr.Scalar(float(q[i])))
                # Queue FK work for the visualization timer (avoid blocking callback)
                if self._fk is not None and len(q) >= 7 and frame_idx % self._rr_fk_skip == 0:
                    self._rr_fk_queue.append(('left', q[:7].copy(), self._T_world_baseL, stamp))
            except Exception as e:
                self.get_logger().debug(f'Rerun left joint error: {e}')

    def _cb_joint_right(self, msg):
        with self._sensor_lock:
            self._joint_right_deque.append(msg)
        if self._use_rerun and self._rr is not None:
            try:
                q = np.array(msg.position)
                self._rr_frame_idx_right += 1
                frame_idx = self._rr_frame_idx_right
                stamp = _stamp_to_sec(msg.header.stamp)
                rr = self._rr
                rr.set_time("ros_time", timestamp=stamp)
                rr.set_time("frame_right", sequence=frame_idx)
                for i in range(min(7, len(q))):
                    rr.log(f"timeseries/right_j{i}", rr.Scalar(float(q[i])))
                if self._fk is not None and len(q) >= 7 and frame_idx % self._rr_fk_skip == 0:
                    self._rr_fk_queue.append(('right', q[:7].copy(), self._T_world_baseR, stamp))
            except Exception as e:
                self.get_logger().debug(f'Rerun right joint error: {e}')

    # ── Execution control ────────────────────────────────────────

    def _cb_execute(self, msg):
        """ROS2 topic callback for /policy/execute (Bool)."""
        with self._exec_lock:
            was_enabled = self._execution_enabled
            if msg.data and not was_enabled:
                # Flush BEFORE enabling to prevent publishing stale actions
                self._flush_stale_buffer()
            self._execution_enabled = msg.data
        mode_str = 'EXECUTE' if msg.data else 'OBSERVE'
        self.get_logger().info(f'[{mode_str}] 已切换到{"执行" if msg.data else "观测"}模式')
        if self._use_rerun:
            self._rr_queue_execute_mode(msg.data)

    def _toggle_execute(self):
        """Toggle execution mode."""
        with self._exec_lock:
            enabled = not self._execution_enabled
            if enabled:
                # Flush BEFORE enabling to prevent publishing stale actions
                self._flush_stale_buffer()
            self._execution_enabled = enabled
        mode_str = 'EXECUTE' if enabled else 'OBSERVE'
        self.get_logger().info(f'[{mode_str}] 已切换到{"执行" if enabled else "观测"}模式')
        if self._use_rerun:
            self._rr_queue_execute_mode(enabled)

    def _switch_to_observe(self):
        """Switch to observe mode if currently executing."""
        with self._exec_lock:
            if self._execution_enabled:
                self._execution_enabled = False
                self.get_logger().info('[OBSERVE] 已切换到观测模式')
                if self._use_rerun:
                    self._rr_queue_execute_mode(False)

    def _rr_queue_execute_mode(self, enabled):
        """Thread-safe: queue a Rerun execute_mode scalar for the executor timer to log.

        Safe to call from any thread (keyboard listener, ROS2 executor, etc.).
        """
        stamp_sec = self.get_clock().now().nanoseconds * 1e-9
        val = 1.0 if enabled else 0.0
        # Capture val in closure to avoid late-binding issues
        self._rr_event_queue.append((
            "timeseries/execute_mode",
            lambda v=val: self._rr.Scalar(v),
            stamp_sec,
        ))

    def _flush_stale_buffer(self):
        """Discard stale actions, seed last_action with current joint pos for smooth transition.

        May be called with or without _exec_lock held (from _cb_execute/_toggle_execute
        under _exec_lock, or from _publish_action without it).

        Lock ordering: _exec_lock → _sensor_lock. Never acquire _exec_lock while
        holding _sensor_lock to avoid deadlock."""
        with self._sensor_lock:
            jl = self._joint_left_deque[-1] if self._joint_left_deque else None
            jr = self._joint_right_deque[-1] if self._joint_right_deque else None

        seed = None
        if jl is not None and jr is not None:
            ql = np.array(jl.position[:7])
            qr = np.array(jr.position[:7])
            if len(ql) < 7:
                ql = np.pad(ql, (0, 7 - len(ql)))
            if len(qr) < 7:
                qr = np.pad(qr, (0, 7 - len(qr)))
            seed = np.concatenate([ql, qr])  # always 14-dim
        self.stream_buffer.flush(seed_action=seed)

    def _keyboard_listener(self):
        """Non-blocking keyboard listener for interactive control.

        Enter/Space → toggle execute/observe
        q/Esc       → switch to observe mode
        """
        import termios
        import tty

        self.get_logger().info(
            'Keyboard control: [Enter/Space] toggle execute | [q/Esc] → observe')
        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
        except (termios.error, ValueError, OSError):
            # Not a real terminal (e.g., launched via ros2 launch)
            # Fall back to line-buffered stdin
            self._keyboard_listener_line_mode()
            return

        try:
            tty.setcbreak(fd)
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0.2)[0]:
                    ch = sys.stdin.read(1)
                    if ch in ('\n', '\r', ' '):
                        self._toggle_execute()
                    elif ch == '\x1b':
                        # Drain trailing escape sequence bytes (e.g. arrow keys: \x1b[A)
                        # to avoid interpreting them as separate keypresses
                        has_seq = False
                        while select.select([sys.stdin], [], [], 0.05)[0]:
                            sys.stdin.read(1)
                            has_seq = True
                        if has_seq:
                            # Was a multi-byte escape sequence (arrow key etc.), ignore
                            continue
                        # Bare Esc key — fall through to observe mode
                        self._switch_to_observe()
                    elif ch == 'q':
                        self._switch_to_observe()
        except Exception as e:
            self.get_logger().warn(f'Keyboard listener exited: {e}')
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _keyboard_listener_line_mode(self):
        """Fallback: line-buffered stdin (for ros2 launch environments)."""
        try:
            # Check stdin is readable (may be closed/redirected under ros2 launch)
            if sys.stdin is None or sys.stdin.closed:
                self.get_logger().info('stdin not available, keyboard control disabled')
                return
            sys.stdin.fileno()  # will raise if stdin is not a real fd
        except (ValueError, OSError):
            self.get_logger().info('stdin not available, keyboard control disabled')
            return

        while rclpy.ok():
            try:
                if select.select([sys.stdin], [], [], 0.5)[0]:
                    line = sys.stdin.readline()
                    if not line:
                        # EOF — stdin closed
                        self.get_logger().info('stdin EOF, keyboard control stopped')
                        return
                    line = line.strip().lower()
                    if line in ('', 'toggle', 't'):
                        self._toggle_execute()
                    elif line in ('q', 'quit', 'stop'):
                        self._switch_to_observe()
            except (OSError, ValueError):
                # stdin closed/broken
                self.get_logger().info('stdin error, keyboard control stopped')
                return
            except Exception:
                pass

    # ── Rerun visualization ────────────────────────────────────────

    def _rr_process_fk_queue(self):
        """Timer callback: drain FK queue, event queue, and predicted traj.

        Time-budgeted to avoid blocking the executor for too long (target <30ms).
        """
        rr = self._rr
        if rr is None:
            return

        t_budget_start = time.monotonic()
        BUDGET_SEC = 0.030  # 30ms max per timer tick

        # 1. Drain cross-thread event queue (keyboard toggle, etc.) — always process first (cheap)
        while self._rr_event_queue:
            try:
                entity, data_factory, stamp_sec = self._rr_event_queue.popleft()
            except IndexError:
                break
            try:
                rr.set_time("ros_time", timestamp=stamp_sec)
                rr.log(entity, data_factory())
            except Exception as e:
                self.get_logger().debug(f'Rerun event queue error: {e}')

        if self._fk is None:
            return

        # 2. Adaptive skip for arm FK queue
        queue_len = len(self._rr_fk_queue)
        if queue_len >= self._rr_fk_queue.maxlen:
            self._rr_fk_drops += 1
            if self._rr_fk_skip < 15:
                self._rr_fk_skip = min(self._rr_fk_skip + 1, 15)
            if self._rr_fk_drops % 50 == 1:
                self.get_logger().warn(
                    f'Rerun FK queue full ({queue_len}), increased skip to {self._rr_fk_skip} '
                    f'(total drops: {self._rr_fk_drops})')
        elif queue_len <= self._rr_fk_queue.maxlen // 4 and self._rr_fk_skip > 3:
            # Queue has headroom, gradually restore skip
            self._rr_fk_skip = max(self._rr_fk_skip - 1, 3)

        # 3. Drain arm FK requests (with time budget)
        processed = 0
        while self._rr_fk_queue and processed < 10:
            if time.monotonic() - t_budget_start > BUDGET_SEC:
                break
            try:
                arm_label, q6, T_base, stamp = self._rr_fk_queue.popleft()
            except IndexError:
                break
            processed += 1
            try:
                rr.set_time("ros_time", timestamp=stamp)
                ee_pos = self._rr_log_arm_fk(arm_label, q6, T_base)
                if ee_pos is not None:
                    if arm_label == 'left':
                        self._actual_trail_left.append(ee_pos.copy())
                        trail = np.array(list(self._actual_trail_left))
                        rr.log("world/actual/left_trail", rr.Points3D(
                            trail, colors=[[50, 100, 255]], radii=[0.002]))
                    else:
                        self._actual_trail_right.append(ee_pos.copy())
                        trail = np.array(list(self._actual_trail_right))
                        rr.log("world/actual/right_trail", rr.Points3D(
                            trail, colors=[[50, 255, 100]], radii=[0.002]))
            except Exception as e:
                self.get_logger().debug(f'Rerun FK viz error ({arm_label}): {e}')

        # 4. Process predicted trajectory FK (time-budgeted: only one per tick)
        if self._rr_pred_queue and (time.monotonic() - t_budget_start) < BUDGET_SEC:
            try:
                actions, stamp = self._rr_pred_queue.popleft()
            except IndexError:
                pass
            else:
                try:
                    rr.set_time("ros_time", timestamp=stamp)
                    self._rr_log_predicted_trajectory(actions)
                except Exception as e:
                    self.get_logger().debug(f'Rerun predicted traj error: {e}')
            # Log if predictions are being dropped
            if self._rr_pred_queue:
                self._rr_pred_drops += len(self._rr_pred_queue)
                self._rr_pred_queue.clear()
                if self._rr_pred_drops % 10 == 1:
                    self.get_logger().info(
                        f'Rerun predicted traj queue overflow (total drops: {self._rr_pred_drops})')

    def _init_rerun(self):
        """Initialize Rerun viewer with blueprint and static elements."""
        try:
            import rerun as rr
            import rerun.blueprint as rrb
        except ImportError:
            self.get_logger().warn('rerun not installed, disabling visualization')
            self._use_rerun = False
            return

        self._rr = rr
        self._rr_bridge = CvBridge()

        # Rerun >=0.21 uses new_recording(); older versions use init()
        if hasattr(rr, 'new_recording'):
            try:
                self._rr_rec = rr.new_recording("inference_viz", make_default=True, make_thread_default=True)
            except TypeError:
                # Older new_recording() may not support make_thread_default
                self._rr_rec = rr.new_recording("inference_viz", make_default=True)
        else:
            self._rr_rec = rr.init("inference_viz")
            # Old Rerun: ensure recording is set as default for all threads
            if self._rr_rec is not None and hasattr(self._rr_rec, 'set_global'):
                self._rr_rec.set_global()

        # Spawn viewer or save to file
        if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
            try:
                rr.spawn()
            except Exception as e:
                from datetime import datetime
                rrd_path = f'/tmp/inference_viz_{datetime.now():%Y%m%d_%H%M%S}.rrd'
                self.get_logger().warn(
                    f'rr.spawn() failed ({e}), saving to {rrd_path}. '
                    f'View with: rerun {rrd_path}')
                rr.save(rrd_path)
        else:
            from datetime import datetime
            rrd_path = f'/tmp/inference_viz_{datetime.now():%Y%m%d_%H%M%S}.rrd'
            self.get_logger().warn(
                f'No display detected, Rerun saving to {rrd_path}. '
                f'View with: rerun {rrd_path}')
            rr.save(rrd_path)
        # Pre-log empty images so Rerun blueprint can resolve entity paths
        import numpy as np
        _placeholder = rr.Image(np.zeros((1, 1, 3), dtype=np.uint8))
        for _entity in ("images/top_head", "images/hand_left", "images/hand_right"):
            rr.log(_entity, _placeholder, static=True)

        rr.send_blueprint(rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world", contents="world/**", name="3D Scene"),
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial2DView(origin="images/top_head", name="Head"),
                        rrb.Spatial2DView(origin="images/hand_left", name="Left"),
                        rrb.Spatial2DView(origin="images/hand_right", name="Right"),
                    ),
                    rrb.TimeSeriesView(origin="timeseries", contents="timeseries/**",
                                       name="Telemetry"),
                    row_shares=[1, 1],
                ),
                column_shares=[3, 2],
            ),
        ))

        # Load calibration for FK
        calib_path = self.get_parameter('calibration_config').value
        if not calib_path or not os.path.isfile(calib_path):
            self.get_logger().warn(
                f'calibration_config not found: {calib_path!r}, '
                'FK visualization disabled')
            self._fk = None
        else:
            self._load_calibration_and_fk(calib_path)

        # Log initial execute mode
        rr.log("timeseries/execute_mode",
               rr.Scalar(1.0 if self._execution_enabled else 0.0))

        self.get_logger().info('Rerun viewer initialized')

    def _load_calibration_and_fk(self, calib_path):
        """Load calibration YAML and FK, log static mesh elements."""
        rr = self._rr

        # Load calibration
        with open(calib_path) as f:
            calib = yaml.safe_load(f)
        transforms = calib['transforms']
        for key in transforms:
            transforms[key] = np.array(transforms[key])

        self._T_world_baseL = transforms['T_world_baseL']
        self._T_world_baseR = transforms['T_world_baseR']
        self._T_world_camF = transforms.get('T_world_camF')

        # Head camera intrinsics for depth → point cloud projection
        intrinsics = calib.get('intrinsics', {}).get('cam_f', {})
        self._cam_f_fx = intrinsics.get('fx', 606.5)
        self._cam_f_fy = intrinsics.get('fy', 605.7)
        self._cam_f_cx = intrinsics.get('cx', 326.5)
        self._cam_f_cy = intrinsics.get('cy', 256.9)

        # FK — find calib/ dir relative to project root or calibration_config location
        calib_dir = None
        for candidate in [
            os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '..', '..', '..', '..', 'calib')),
            os.path.normpath(os.path.join(_KAI0_ROOT, '..', 'calib')),
            os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(calib_path)),
                                          '..', 'calib')),
        ]:
            if os.path.isfile(os.path.join(candidate, 'piper_fk.py')):
                calib_dir = candidate
                break
        if calib_dir is None:
            self.get_logger().warn('Cannot find calib/piper_fk.py, FK disabled')
            self._fk = None
            return
        if calib_dir not in sys.path:
            sys.path.insert(0, calib_dir)
        from piper_fk import PiperFK
        self._fk = PiperFK()

        # Base markers
        for label, T_key in [('baseL', 'T_world_baseL'), ('baseR', 'T_world_baseR')]:
            pos = transforms[T_key][:3, 3]
            rr.log(f"world/{label}", rr.Points3D(
                [pos], colors=[[255, 255, 0]], radii=[0.015], labels=[label]),
                static=True)

        # Head camera frustum
        if 'T_world_camF' in transforms:
            T_head = transforms['T_world_camF']
            head_pos = T_head[:3, 3]
            R_head = T_head[:3, :3]
            d, hw_, hh_ = 0.15, 0.09, 0.07
            corners = [[-hw_, -hh_, d], [hw_, -hh_, d], [hw_, hh_, d], [-hw_, hh_, d]]
            o = head_pos.tolist()
            cs = [(R_head @ np.array(c) + head_pos).tolist() for c in corners]
            lines = [[o, cs[0]], [o, cs[1]], [o, cs[2]], [o, cs[3]],
                     [cs[0], cs[1]], [cs[1], cs[2]], [cs[2], cs[3]], [cs[3], cs[0]]]
            rr.log("world/head_cam_frustum", rr.LineStrips3D(
                lines, colors=[[255, 50, 50]], radii=[0.003]), static=True)

        # Workspace box — centered between the two arm bases, offset in the average
        # forward direction (local Y of each base frame) so it matches actual calibration.
        base_mid = (self._T_world_baseL[:3, 3] + self._T_world_baseR[:3, 3]) / 2.0
        fwd_L = self._T_world_baseL[:3, 1]  # local Y axis of left base in world
        fwd_R = self._T_world_baseR[:3, 1]  # local Y axis of right base in world
        fwd_avg = (fwd_L + fwd_R) / 2.0
        fwd_norm = np.linalg.norm(fwd_avg)
        if fwd_norm > 1e-6:
            fwd_avg /= fwd_norm
        else:
            fwd_avg = np.array([0.0, 1.0, 0.0])
        ws_center = base_mid + fwd_avg * 0.30 + np.array([0.0, 0.0, 0.05])
        base_span = np.linalg.norm(self._T_world_baseL[:3, 3] - self._T_world_baseR[:3, 3])
        ws_width = max(1.0, base_span + 0.6)   # wider to cover full arm reach
        rr.log("world/workspace", rr.Boxes3D(
            centers=[ws_center.tolist()], sizes=[[ws_width, 0.9, 0.7]],
            colors=[[255, 255, 255, 40]]), static=True)

        # Load arm meshes — search multiple candidate paths (source layout, install layout, KAI0_ROOT)
        _mesh_suffix = os.path.join('train_deploy_alignment', 'inference', 'agilex',
                                     'Piper_ros_private-ros-noetic', 'src',
                                     'piper_description', 'meshes')
        mesh_dir = None
        for _mesh_base in [
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', '..', '..', 'kai0'),       # source layout
            _KAI0_ROOT,                                           # env / auto-detected
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', '..', '..', '..', 'kai0'),  # install layout
        ]:
            candidate = os.path.normpath(os.path.join(_mesh_base, _mesh_suffix))
            if os.path.isdir(candidate):
                mesh_dir = candidate
                break
        if mesh_dir is None:
            mesh_dir = os.path.normpath(os.path.join(_KAI0_ROOT, _mesh_suffix))
        mesh_names = ['base_link', 'link1', 'link2', 'link3', 'link4', 'link5',
                      'link6', 'gripper_base', 'link7', 'link8']
        self._arm_mesh_loaded = False
        if os.path.isdir(mesh_dir):
            try:
                import trimesh
                for arm_label in ('left', 'right'):
                    arm_color = [100, 100, 255] if arm_label == 'left' else [100, 255, 100]
                    for mesh_name in mesh_names:
                        stl_path = os.path.join(mesh_dir, f'{mesh_name}.STL')
                        if os.path.exists(stl_path):
                            m = trimesh.load(stl_path)
                            rr.log(f"world/{arm_label}/{mesh_name}", rr.Mesh3D(
                                vertex_positions=m.vertices,
                                triangle_indices=m.faces,
                                vertex_colors=np.full((len(m.vertices), 3),
                                                      arm_color, dtype=np.uint8),
                            ), static=True)
                self._arm_mesh_loaded = True
                self.get_logger().info(f'Loaded arm meshes from {mesh_dir}')
            except ImportError:
                self.get_logger().warn('trimesh not available, skipping arm mesh')
        else:
            self.get_logger().warn(f'Mesh dir not found: {mesh_dir}')

    def _rr_log_arm_fk(self, arm_label, q7, T_world_base):
        """Update arm link transforms via FK (drives mesh positions).

        Args:
            q7: 7-element array [j0..j5, gripper_open]. j0-j5 for FK, gripper_open for finger viz.
        """
        if self._fk is None or self._rr is None:
            return
        rr = self._rr

        q6_rad = q7[:6]
        gripper_val = float(q7[6]) if len(q7) > 6 else 0.0

        link_Ts = self._fk.fk_all_links(q6_rad)
        T_base = np.array(T_world_base)

        # Base link
        rr.log(f"world/{arm_label}/base_link", rr.Transform3D(
            translation=T_base[:3, 3], mat3x3=T_base[:3, :3]))

        # Links 1-6
        for idx, T_link in enumerate(link_Ts):
            T_world_link = T_base @ T_link
            rr.log(f"world/{arm_label}/link{idx+1}", rr.Transform3D(
                translation=T_world_link[:3, 3], mat3x3=T_world_link[:3, :3]))

        # Gripper base = link6 pose
        T_world_link6 = T_base @ link_Ts[5]
        rr.log(f"world/{arm_label}/gripper_base", rr.Transform3D(
            translation=T_world_link6[:3, 3], mat3x3=T_world_link6[:3, :3]))

        # Gripper fingers (link7, link8) — driven by gripper_val (q[6])
        # gripper_val ~0 = closed, ~0.04 = open. Map to lateral finger offset.
        finger_spread = np.clip(gripper_val, 0.0, 0.04) / 0.04  # normalize to [0, 1]
        max_lateral = 0.025  # max lateral offset per finger (m)
        for link_name, rpy, sign in [('link7', [np.pi/2, 0, 0], 1.0),
                                      ('link8', [np.pi/2, 0, -np.pi], -1.0)]:
            T_offset = np.eye(4)
            T_offset[:3, :3] = R_.from_euler('xyz', rpy).as_matrix()
            T_offset[:3, 3] = [sign * finger_spread * max_lateral, 0, 0.1358]
            T_w = T_world_link6 @ T_offset
            rr.log(f"world/{arm_label}/{link_name}", rr.Transform3D(
                translation=T_w[:3, 3], mat3x3=T_w[:3, :3]))

        # EE marker (link6 = end-effector, already computed above)
        ee_pos = T_world_link6[:3, 3]
        ee_color = [50, 100, 255] if arm_label == 'left' else [50, 255, 100]
        rr.log(f"world/{arm_label}/ee", rr.Points3D(
            [ee_pos], colors=[ee_color], radii=[0.008]))

        return ee_pos

    # Piper joint limits (rad) — used to validate FK inputs before visualization
    _JOINT_LIMITS_RAD = np.array([2.618, 1.571, 1.571, 1.745, 1.571, 2.618])  # ~[150,90,90,100,90,150]°

    def _rr_log_predicted_trajectory(self, actions):
        """Convert action chunk [50, 14] to EE paths via FK and log as LineStrips3D."""
        if self._fk is None or self._rr is None:
            return
        rr = self._rr

        limits = self._JOINT_LIMITS_RAD
        left_path, right_path = [], []
        for t in range(actions.shape[0]):
            q_l = actions[t, 0:6]
            q_r = actions[t, 7:13]
            # Skip steps with out-of-range joints to prevent wild FK output
            if np.any(np.abs(q_l) > limits * 1.5) or np.any(np.abs(q_r) > limits * 1.5):
                continue
            T_l = self._T_world_baseL @ self._fk.fk_homogeneous(q_l)
            T_r = self._T_world_baseR @ self._fk.fk_homogeneous(q_r)
            left_path.append(T_l[:3, 3])
            right_path.append(T_r[:3, 3])

        if len(left_path) < 2:
            return

        left_arr = np.array(left_path)
        right_arr = np.array(right_path)

        rr.log("world/predicted/left_traj", rr.LineStrips3D(
            [left_arr], colors=[[50, 100, 255]], radii=[0.003]))
        rr.log("world/predicted/right_traj", rr.LineStrips3D(
            [right_arr], colors=[[50, 255, 100]], radii=[0.003]))

        # Start/end markers
        rr.log("world/predicted/left_endpoints", rr.Points3D(
            [left_arr[0], left_arr[-1]], colors=[[50, 100, 255]],
            radii=[0.008], labels=["start", "end"]))
        rr.log("world/predicted/right_endpoints", rr.Points3D(
            [right_arr[0], right_arr[-1]], colors=[[50, 255, 100]],
            radii=[0.008], labels=["start", "end"]))

    # ── Frame sync (复刻原版 get_frame, 基于 min(timestamp) 对齐) ──

    def _get_synced_frame(self):
        """Return timestamp-aligned (img_front, img_left, img_right, joint_left, joint_right)
        or None if any sensor data is missing/stale."""
        with self._sensor_lock:
            if (len(self._img_front_deque) == 0
                    or len(self._img_left_deque) == 0
                    or len(self._img_right_deque) == 0
                    or len(self._joint_left_deque) == 0
                    or len(self._joint_right_deque) == 0):
                return None

            # Sync time = min of latest timestamps across 3 cameras
            frame_time = min(
                _stamp_to_sec(self._img_front_deque[-1].header.stamp),
                _stamp_to_sec(self._img_left_deque[-1].header.stamp),
                _stamp_to_sec(self._img_right_deque[-1].header.stamp),
            )

            # Check all sensors have data at or after frame_time
            for dq, name in [
                (self._img_front_deque, 'img_front'),
                (self._img_left_deque, 'img_left'),
                (self._img_right_deque, 'img_right'),
                (self._joint_left_deque, 'joint_left'),
                (self._joint_right_deque, 'joint_right'),
            ]:
                if len(dq) == 0 or _stamp_to_sec(dq[-1].header.stamp) < frame_time:
                    return None

            # Pop frames up to frame_time (discard stale data)
            def _pop_synced(dq):
                while len(dq) > 0 and _stamp_to_sec(dq[0].header.stamp) < frame_time:
                    dq.popleft()
                if len(dq) == 0:
                    return None
                return dq.popleft()

            img_front_msg = _pop_synced(self._img_front_deque)
            img_left_msg = _pop_synced(self._img_left_deque)
            img_right_msg = _pop_synced(self._img_right_deque)
            joint_left_msg = _pop_synced(self._joint_left_deque)
            joint_right_msg = _pop_synced(self._joint_right_deque)

            if any(m is None for m in (img_front_msg, img_left_msg, img_right_msg,
                                       joint_left_msg, joint_right_msg)):
                return None

        # 强制输出 BGR: RealSense ROS2 默认 rgb8, cv_bridge 会自动转换
        # 后续管线 jpeg_mapping + COLOR_BGR2RGB 假设输入为 BGR
        img_front = self.bridge.imgmsg_to_cv2(img_front_msg, 'bgr8')
        img_left = self.bridge.imgmsg_to_cv2(img_left_msg, 'bgr8')
        img_right = self.bridge.imgmsg_to_cv2(img_right_msg, 'bgr8')

        return img_front, img_left, img_right, joint_left_msg, joint_right_msg

    # ── Image preprocessing (严格复刻原版管线) ──────────────────────

    @staticmethod
    def _jpeg_mapping(img):
        """JPEG encode/decode 对齐训练数据的 MP4 视频压缩 artifacts.

        原版链路: passthrough → imencode (按 BGR 编码) → imdecode (固定输出 BGR)
        无论输入是 rgb8 还是 bgr8, 输出均为 BGR (OpenCV 约定).
        """
        img = cv2.imencode(".jpg", img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img

    def _get_observation(self):
        """Pack current sensor data into policy input.

        严格复刻原版 update_observation_window + inference_fn 的图像管线:
        1. get_synced_frame (timestamp 对齐)
        2. JPEG encode/decode (训练对齐)
        3. BGR → RGB
        4. resize_with_pad(224, 224) — 保持宽高比, 零填充
        5. HWC → CHW
        """
        frame = self._get_synced_frame()
        if frame is None:
            return None

        img_front, img_left, img_right, joint_left_msg, joint_right_msg = frame

        from openpi_client import image_tools

        # 原版顺序: front, right, left (camera_names = [front, right, left])
        imgs = [
            self._jpeg_mapping(img_front),
            self._jpeg_mapping(img_right),
            self._jpeg_mapping(img_left),
        ]
        imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
        # 与原版一致: 相同分辨率时走 batch resize, 否则逐张
        if imgs[0].shape == imgs[1].shape == imgs[2].shape:
            imgs = list(image_tools.resize_with_pad(np.array(imgs), 224, 224))
        else:
            imgs = [image_tools.resize_with_pad(im[np.newaxis], 224, 224)[0] for im in imgs]

        qpos = np.concatenate((
            np.array(joint_left_msg.position),
            np.array(joint_right_msg.position),
        ), axis=0)

        return {
            'state': qpos,
            'images': {
                'top_head':   imgs[0].transpose(2, 0, 1),   # CHW
                'hand_right': imgs[1].transpose(2, 0, 1),
                'hand_left':  imgs[2].transpose(2, 0, 1),
            },
            'prompt': self.prompt,
        }

    # ── Inference loop ──────────────────────────────────────────────

    def _inference_loop(self):
        """Background thread: continuously infer and push to buffer."""
        # Wait for policy to be loaded
        while self.policy is None and rclpy.ok():
            time.sleep(0.1)

        # Set thread-local Rerun recording so set_time calls don't clash with callbacks.
        # If set_thread_local is unavailable (old Rerun), disable Rerun in this thread
        # to prevent timeline corruption between inference thread and executor thread.
        self._rr_infer_thread = False
        if self._use_rerun and self._rr is not None and self._rr_rec is not None:
            if hasattr(self._rr_rec, 'set_thread_local'):
                self._rr_rec.set_thread_local()
                self._rr_infer_thread = True
            else:
                self.get_logger().warn(
                    'Rerun RecordingStream.set_thread_local() not available, '
                    'disabling Rerun logging in inference thread to prevent timeline corruption')

        self.get_logger().info('Inference loop started')
        rate_hz = self.inference_rate
        period = 1.0 / rate_hz

        # Warmup
        self.get_logger().info('Waiting for sensor data...')
        while rclpy.ok():
            obs = self._get_observation()
            if obs is not None:
                break
            time.sleep(0.1)

        self.get_logger().info('Running warmup inference...')
        t0 = time.monotonic()
        try:
            self.policy.infer(obs)
        except Exception as e:
            self.get_logger().warn(f'Warmup failed: {e}')
        self.get_logger().info(f'Warmup done in {(time.monotonic()-t0)*1000:.0f}ms')

        # Main loop
        while rclpy.ok():
            t_start = time.monotonic()
            try:
                obs = self._get_observation()
                if obs is None:
                    time.sleep(0.01)
                    continue

                result = self.policy.infer(obs)
                actions = result.get('actions', None)
                infer_ms = (time.monotonic() - t_start) * 1000

                if actions is not None and len(actions) > 0:
                    self.stream_buffer.integrate_new_chunk(
                        actions,
                        max_k=self.latency_k,
                        min_m=self.min_smooth_steps)

                    # Publish full chunk for rerun_viz to show predicted trajectory
                    try:
                        from std_msgs.msg import Float32MultiArray, MultiArrayDimension
                        chunk_msg = Float32MultiArray()
                        chunk_msg.layout.dim = [
                            MultiArrayDimension(label='steps', size=actions.shape[0],
                                                stride=actions.shape[0] * actions.shape[1]),
                            MultiArrayDimension(label='dims', size=actions.shape[1],
                                                stride=actions.shape[1]),
                        ]
                        chunk_msg.data = actions.astype(np.float32).flatten().tolist()
                        self.pub_action_chunk.publish(chunk_msg)
                    except Exception as e:
                        self.get_logger().debug(f'action_chunk publish error: {e}')

                    # Print inference result summary to console
                    a0 = actions[0]
                    a_last = actions[-1]
                    self.get_logger().info(
                        f'infer {infer_ms:.0f}ms | chunk={actions.shape} | '
                        f'L[0]=[{a0[0]:+.2f},{a0[1]:+.2f},{a0[2]:+.2f},{a0[3]:+.2f},'
                        f'{a0[4]:+.2f},{a0[5]:+.2f},g={a0[6]:.2f}] '
                        f'R[0]=[{a0[7]:+.2f},{a0[8]:+.2f},{a0[9]:+.2f},{a0[10]:+.2f},'
                        f'{a0[11]:+.2f},{a0[12]:+.2f},g={a0[13]:.2f}] '
                        f'| Δ(0→{len(actions)-1}): L={np.linalg.norm(a_last[:6]-a0[:6]):.2f} '
                        f'R={np.linalg.norm(a_last[7:13]-a0[7:13]):.2f}')

                    # Rerun: queue predicted trajectory for FK timer + log latency
                    # Pred queue is always safe (consumed by executor timer).
                    # Direct rr.log only if set_thread_local succeeded (_rr_infer_thread);
                    # otherwise route through the event queue so inference_ms still appears.
                    if self._use_rerun and self._rr is not None:
                        try:
                            ros_now_sec = self.get_clock().now().nanoseconds * 1e-9
                            # Queue FK-heavy predicted trajectory for the timer thread
                            if self._fk is not None:
                                self._rr_pred_queue.append((actions.copy(), ros_now_sec))
                            # Log inference latency
                            if self._rr_infer_thread:
                                self._rr.set_time("ros_time", timestamp=ros_now_sec)
                                self._rr.log("timeseries/inference_ms",
                                             self._rr.Scalar(float(infer_ms)))
                            else:
                                # Fallback: route through event queue for executor thread
                                ms = float(infer_ms)
                                self._rr_event_queue.append((
                                    "timeseries/inference_ms",
                                    lambda v=ms: self._rr.Scalar(v),
                                    ros_now_sec,
                                ))
                        except Exception as e:
                            self.get_logger().debug(f'Rerun prediction log error: {e}')

                elapsed = time.monotonic() - t_start
                sleep_time = max(0, period - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                import traceback
                self.get_logger().error(f'Inference error: {e}\n{traceback.format_exc()}')
                time.sleep(1.0)

    # ── Action publishing ───────────────────────────────────────────

    def _publish_action(self):
        """Timer callback: pop smoothed action and publish."""
        with self._exec_lock:
            enabled = self._execution_enabled
        if not enabled:
            self._last_published_action = None  # reset on observe so first execute re-seeds
            return
        act = self.stream_buffer.pop_next_action()
        if act is None:
            return

        # Jump protection: compare against current joint state (first action after
        # observe→execute) or last published action (subsequent actions)
        reference = self._last_published_action
        if reference is None:
            # First action after observe→execute: compare against current joints
            with self._sensor_lock:
                jl = self._joint_left_deque[-1] if self._joint_left_deque else None
                jr = self._joint_right_deque[-1] if self._joint_right_deque else None
            if jl is not None and jr is not None:
                ql = np.array(jl.position[:7])
                qr = np.array(jr.position[:7])
                if len(ql) >= 7 and len(qr) >= 7:
                    reference = np.concatenate([ql, qr])
        if reference is not None:
            max_jump = np.max(np.abs(act[:14] - reference[:14]))
            if max_jump > self._MAX_JOINT_JUMP_RAD:
                self.get_logger().warn(
                    f'Jump protection: rejected action with max Δ={np.degrees(max_jump):.1f}° '
                    f'(limit={np.degrees(self._MAX_JOINT_JUMP_RAD):.0f}°), flushing buffer')
                self._flush_stale_buffer()
                return
        self._last_published_action = act.copy()

        left = act[:7].copy()
        right = act[7:14].copy()
        left[6] = max(0.0, left[6] - self.gripper_offset)
        right[6] = max(0.0, right[6] - self.gripper_offset)

        now = self.get_clock().now().to_msg()

        # Publish to /master/joint_left and /master/joint_right
        for pub, values in [(self.pub_left, left), (self.pub_right, right)]:
            msg = JointState()
            msg.header.stamp = now
            msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            msg.position = values.tolist()
            pub.publish(msg)

        # Also publish combined action on /policy/actions
        msg = JointState()
        msg.header.stamp = now
        msg.name = [f'left_j{i}' for i in range(7)] + [f'right_j{i}' for i in range(7)]
        msg.position = act.tolist()
        self.pub_action.publish(msg)


def main():
    rclpy.init()
    node = PolicyInferenceNode()
    # Rerun set_time+log is not atomic — all ROS2 callbacks and timers MUST run
    # on the same thread. Use SingleThreadedExecutor unconditionally.
    executor = rclpy.executors.SingleThreadedExecutor()
    if node._rr_require_single_thread and isinstance(
            executor, rclpy.executors.MultiThreadedExecutor):
        node.get_logger().fatal(
            'Rerun is enabled but executor is MultiThreadedExecutor. '
            'Rerun set_time+log is non-atomic and WILL corrupt timelines. '
            'Use SingleThreadedExecutor or disable Rerun.')
        raise RuntimeError('Rerun requires SingleThreadedExecutor')
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
