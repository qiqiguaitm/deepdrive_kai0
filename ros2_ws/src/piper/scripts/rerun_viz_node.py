#!/usr/bin/python3
"""
ROS2 Rerun Visualization Node — standalone 3D scene viewer.

Subscribes to camera images, aligned depth, and joint states.
Renders: 3D point cloud (head depth), camera images, arm FK, predicted trajectories,
workspace box, arm meshes, and telemetry time series.

Runs independently from the policy inference node so visualization
never blocks or slows down inference.
"""

import os
import sys
import glob as _glob

# ── Ensure venv packages (rerun, numpy, scipy, trimesh, etc.) are importable ──
# When launched via ros2 launch, PYTHONPATH is set by the launch file.
# When launched via ros2 run, we need to add the paths ourselves.
def _setup_venv_paths():
    """Add kai0 venv site-packages and .pth entries to sys.path."""
    project_root = None
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(d, 'kai0')):
            project_root = d
            break
        d = os.path.dirname(d)
    if project_root is None:
        project_root = os.path.expanduser('~/workspace/deepdive_kai0')

    kai0_root = os.path.join(project_root, 'kai0')
    venv_lib = os.path.join(kai0_root, '.venv', 'lib')
    pydirs = sorted(_glob.glob(os.path.join(venv_lib, 'python3.*')))
    if not pydirs:
        return
    venv_sp = os.path.join(pydirs[-1], 'site-packages')
    if not os.path.isdir(venv_sp):
        return

    # Add site-packages
    if venv_sp not in sys.path:
        sys.path.insert(0, venv_sp)

    # Process .pth files (rerun_sdk, openpi, etc.)
    for pth in sorted(_glob.glob(os.path.join(venv_sp, '*.pth'))):
        with open(pth) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('import '):
                    continue
                resolved = line if os.path.isabs(line) else os.path.join(venv_sp, line)
                if os.path.isdir(resolved) and resolved not in sys.path:
                    sys.path.insert(0, resolved)

_setup_venv_paths()

import time
import numpy as np
import yaml

# GPU projection via JAX. The policy inference node owns GPU 0; initializing
# a second JAX context on the same device segfaults at first use. Pin this
# process to GPU 1 (sim01 has 2x RTX 5090) so it has its own CUDA context.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.20')
try:
    import jax
    import jax.numpy as jnp
    _gpu_devs = [d for d in jax.devices() if d.platform == 'gpu']
    _JAX_DEV = _gpu_devs[0] if _gpu_devs else None
    _GPU_OK = _JAX_DEV is not None
    # JIT-compiled projection core (fixed input shape -> cached compile).
    # Inputs: strided depth (uint16), scalar intrinsics, 4x4 world transform.
    # Outputs: per-pixel (x, y, z) in world frame + depth in meters.
    @jax.jit
    def _jax_project_core(depth_strided, fx, fy, cx, cy, step, T_wc):
        depth_m = depth_strided.astype(jnp.float32) * 0.001  # mm -> m
        hs, ws = depth_m.shape
        v_idx = jnp.arange(hs, dtype=jnp.float32) * step  # orig pixel y
        u_idx = jnp.arange(ws, dtype=jnp.float32) * step  # orig pixel x
        v_g, u_g = jnp.meshgrid(v_idx, u_idx, indexing='ij')
        x = (u_g - cx) * depth_m / fx
        y = (v_g - cy) * depth_m / fy
        pts_cam = jnp.stack([x, y, depth_m], axis=-1)  # (hs, ws, 3)
        R = T_wc[:3, :3]
        t = T_wc[:3, 3]
        pts_world = pts_cam @ R.T + t
        return pts_world, depth_m
    _jax_project_fn = _jax_project_core
except Exception as _gpu_err:
    import sys as _sys
    print(f'[rerun_viz] JAX GPU projection disabled: {_gpu_err}', file=_sys.stderr)
    jax = None
    jnp = None
    _JAX_DEV = None
    _GPU_OK = False
    _jax_project_fn = None
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool
from collections import deque
from scipy.spatial.transform import Rotation as R_


def _stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('true', '1', 'yes')
    return bool(value)


# ── Locate project root (same logic as launch file) ──
def _find_project_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(d, 'kai0')):
            return d
        d = os.path.dirname(d)
    return os.path.expanduser('~/workspace/deepdive_kai0')


_PROJECT_ROOT = _find_project_root()
_KAI0_ROOT = os.path.join(_PROJECT_ROOT, 'kai0')


class RerunVizNode(Node):
    def __init__(self):
        super().__init__('rerun_viz')

        # ── Parameters ──
        self.declare_parameter('calibration_config',
                               os.path.join(_PROJECT_ROOT, 'config', 'calibration.yml'))
        self.declare_parameter('img_front_topic', '/camera_f/camera/color/image_raw')
        self.declare_parameter('img_left_topic', '/camera_l/camera/color/image_raw')
        self.declare_parameter('img_right_topic', '/camera_r/camera/color/image_raw')
        self.declare_parameter('depth_front_topic',
                               '/camera_f/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('depth_left_topic',
                               '/camera_l/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('depth_right_topic',
                               '/camera_r/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('puppet_left_topic', '/puppet/joint_left')
        self.declare_parameter('puppet_right_topic', '/puppet/joint_right')
        self.declare_parameter('depth_step', 2)
        self.declare_parameter('depth_min', 0.05)
        self.declare_parameter('depth_max', 1.5)

        self._bridge = CvBridge()

        # ── Calibration ──
        self._T_world_camF = None
        self._T_world_baseL = None
        self._T_world_baseR = None
        self._cam_f_fx = self._cam_f_fy = self._cam_f_cx = self._cam_f_cy = None
        self._fk = None
        self._arm_mesh_loaded = False
        self._gpu_grid_cache = {}  # (h, w, step) -> (u_grid, v_grid) on GPU
        self._latest_front_rgb = None  # cached for RGB point cloud coloring
        self._latest_left_rgb = None
        self._latest_right_rgb = None
        # Cache-latest pattern: subscriptions only update these fields (cheap),
        # and fixed-rate timers read the latest snapshot for logging. This
        # guarantees we always render the newest data instead of processing
        # stale messages from a backed-up executor queue.
        self._latest_q_left = None        # np.ndarray[7] (full joints incl gripper)
        self._latest_q_right = None
        self._latest_q_left_stamp = 0.0
        self._latest_q_right_stamp = 0.0
        self._latest_depth_front = None   # (depth_u16, stamp)
        self._latest_depth_left = None
        self._latest_depth_right = None

        # ── FK state ──
        self._actual_trail_left = deque(maxlen=300)
        self._actual_trail_right = deque(maxlen=300)
        self._frame_idx_left = 0
        self._frame_idx_right = 0
        self._fk_skip = 3
        self._fk_queue = deque(maxlen=60)
        self._fk_drops = 0

        # ── Init Rerun ──
        import rerun as rr
        import rerun.blueprint as rrb
        self._rr = rr

        if hasattr(rr, 'new_recording'):
            try:
                rr.new_recording("inference_viz", make_default=True, make_thread_default=True)
            except TypeError:
                rr.new_recording("inference_viz", make_default=True)
        else:
            rr.init("inference_viz")

        if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
            try:
                rr.spawn()
            except Exception as e:
                from datetime import datetime
                rrd_path = f'/tmp/inference_viz_{datetime.now():%Y%m%d_%H%M%S}.rrd'
                self.get_logger().warn(f'rr.spawn() failed ({e}), saving to {rrd_path}')
                rr.save(rrd_path)
        else:
            from datetime import datetime
            rrd_path = f'/tmp/inference_viz_{datetime.now():%Y%m%d_%H%M%S}.rrd'
            self.get_logger().warn(f'No display, saving to {rrd_path}')
            rr.save(rrd_path)

        # Log origin entities before blueprint to avoid "Unknown entity" warnings
        rr.log("world", rr.Transform3D(), static=True)
        rr.log("images/top_head", rr.Clear(recursive=False), static=True)
        rr.log("images/hand_left", rr.Clear(recursive=False), static=True)
        rr.log("images/hand_right", rr.Clear(recursive=False), static=True)
        rr.log("timeseries", rr.Clear(recursive=False), static=True)

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

        # ── Load calibration + FK + meshes ──
        calib_path = self.get_parameter('calibration_config').value
        if calib_path and os.path.isfile(calib_path):
            self._load_calibration(calib_path)
        else:
            self.get_logger().warn(f'calibration_config not found: {calib_path!r}')

        # ── Subscribers ──
        # RealSense v4.56+ publishes all topics as RELIABLE + TRANSIENT_LOCAL.
        # Use RELIABLE subscriber to match (BEST_EFFORT sub silently drops RELIABLE pub).
        from rclpy.qos import DurabilityPolicy
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=5)
        depth_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=5)

        self.create_subscription(Image,
            self.get_parameter('img_front_topic').value,
            self._cb_img_front, img_qos)
        self.create_subscription(Image,
            self.get_parameter('img_left_topic').value,
            self._cb_img_left, img_qos)
        self.create_subscription(Image,
            self.get_parameter('img_right_topic').value,
            self._cb_img_right, img_qos)
        self.create_subscription(Image,
            self.get_parameter('depth_front_topic').value,
            self._cb_depth_front, depth_qos)
        self.create_subscription(Image,
            self.get_parameter('depth_left_topic').value,
            self._cb_depth_left, depth_qos)
        self.create_subscription(Image,
            self.get_parameter('depth_right_topic').value,
            self._cb_depth_right, depth_qos)
        self.create_subscription(JointState,
            self.get_parameter('puppet_left_topic').value,
            self._cb_joint_left, 10)
        self.create_subscription(JointState,
            self.get_parameter('puppet_right_topic').value,
            self._cb_joint_right, 10)
        self.create_subscription(Bool, '/policy/execute', self._cb_execute, 1)
        # Commanded (smoothed) actions sent to arms
        self.create_subscription(JointState, '/master/joint_left',
                                 self._cb_master_left, 10)
        self.create_subscription(JointState, '/master/joint_right',
                                 self._cb_master_right, 10)
        self._master_trail_left = deque(maxlen=300)
        self._master_trail_right = deque(maxlen=300)
        # Full predicted action chunk from policy (future trajectory, 50 steps × 14 dims)
        from std_msgs.msg import Float32MultiArray
        self.create_subscription(Float32MultiArray, '/policy/action_chunk',
                                 self._cb_action_chunk, 5)

        # Fixed-rate timers always process the NEWEST cached data.
        self._tick_arms_count = 0
        self._tick_pc_count = 0
        self._tick_img_count = 0
        self._tick_report_t = time.monotonic()
        self.create_timer(1.0 / 5.0, self._tick_arms)          # 5 Hz arm FK
        self.create_timer(1.0 / 5.0, self._tick_point_clouds)  # 5 Hz pt clouds
        self.create_timer(1.0 / 5.0, self._tick_images)        # 5 Hz 2D images
        self.create_timer(2.0, self._tick_report)              # diagnostic

        self.get_logger().info('Rerun visualization node ready')

    # ── Calibration loading ──────────────────────────────────────

    def _load_calibration(self, calib_path):
        rr = self._rr
        with open(calib_path) as f:
            calib = yaml.safe_load(f)

        transforms = calib['transforms']
        for key in transforms:
            transforms[key] = np.array(transforms[key])

        self._T_world_baseL = transforms['T_world_baseL']
        self._T_world_baseR = transforms['T_world_baseR']
        self._T_world_camF = transforms.get('T_world_camF')
        self._T_link6_camL = transforms.get('T_link6_camL')
        self._T_link6_camR = transforms.get('T_link6_camR')

        all_intr = calib.get('intrinsics', {})
        cam_f_intr = all_intr.get('cam_f', {})
        self._cam_f_fx = cam_f_intr.get('fx', 606.5)
        self._cam_f_fy = cam_f_intr.get('fy', 605.7)
        self._cam_f_cx = cam_f_intr.get('cx', 326.5)
        self._cam_f_cy = cam_f_intr.get('cy', 256.9)
        cam_l_intr = all_intr.get('cam_l', {})
        self._cam_l_fx = cam_l_intr.get('fx', 391.0)
        self._cam_l_fy = cam_l_intr.get('fy', 390.1)
        self._cam_l_cx = cam_l_intr.get('cx', 316.4)
        self._cam_l_cy = cam_l_intr.get('cy', 240.2)
        cam_r_intr = all_intr.get('cam_r', {})
        self._cam_r_fx = cam_r_intr.get('fx', 392.3)
        self._cam_r_fy = cam_r_intr.get('fy', 391.4)
        self._cam_r_cx = cam_r_intr.get('cx', 313.9)
        self._cam_r_cy = cam_r_intr.get('cy', 243.0)

        # FK
        calib_dir = None
        for candidate in [
            os.path.join(_PROJECT_ROOT, 'calib'),
            os.path.normpath(os.path.join(os.path.dirname(calib_path), '..', 'calib')),
        ]:
            if os.path.isfile(os.path.join(candidate, 'piper_fk.py')):
                calib_dir = candidate
                break
        if calib_dir is None:
            self.get_logger().warn('Cannot find calib/piper_fk.py, FK disabled')
        else:
            if calib_dir not in sys.path:
                sys.path.insert(0, calib_dir)
            from piper_fk import PiperFK
            self._fk = PiperFK()

        # ── Static Rerun elements ──

        # Base markers
        for label, T_key in [('baseL', 'T_world_baseL'), ('baseR', 'T_world_baseR')]:
            pos = transforms[T_key][:3, 3]
            rr.log(f"world/{label}", rr.Points3D(
                [pos], colors=[[255, 255, 0]], radii=[0.0075], labels=[label]),
                static=True)

        # Head camera frustum
        if self._T_world_camF is not None:
            T_head = self._T_world_camF
            head_pos = T_head[:3, 3]
            R_head = T_head[:3, :3]
            d, hw_, hh_ = 0.0375, 0.0225, 0.0175
            corners = [[-hw_, -hh_, d], [hw_, -hh_, d], [hw_, hh_, d], [-hw_, hh_, d]]
            o = head_pos.tolist()
            cs = [(R_head @ np.array(c) + head_pos).tolist() for c in corners]
            ground = [head_pos[0], head_pos[1], 0.0]
            lines = [[o, cs[0]], [o, cs[1]], [o, cs[2]], [o, cs[3]],
                     [cs[0], cs[1]], [cs[1], cs[2]], [cs[2], cs[3]], [cs[3], cs[0]],
                     [o, ground]]
            rr.log("world/head_cam_frustum", rr.LineStrips3D(
                lines, colors=[[255, 50, 50]], radii=[0.001]), static=True)
            rr.log("world/head_cam_pos", rr.Points3D(
                [o], colors=[[255, 50, 50]], radii=[0.005], labels=["D435 (head)"]),
                static=True)

        # Arm meshes
        self._load_meshes(transforms)

    @staticmethod
    def _load_binary_stl(path):
        """Load a binary STL file using only numpy. Returns (vertices [N,3], faces [M,3])."""
        with open(path, 'rb') as f:
            f.read(80)  # header
            n_tri = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            # Each triangle: 12 floats (normal + 3 vertices) + 2 bytes attribute
            dt = np.dtype([('normal', np.float32, 3), ('v', np.float32, (3, 3)), ('attr', np.uint16)])
            data = np.frombuffer(f.read(n_tri * dt.itemsize), dtype=dt)
        all_verts = data['v'].reshape(-1, 3)  # (n_tri*3, 3)
        # Deduplicate vertices for compact mesh
        verts_unique, inverse = np.unique(all_verts, axis=0, return_inverse=True)
        faces = inverse.reshape(-1, 3)
        return verts_unique, faces

    def _load_meshes(self, transforms):
        rr = self._rr
        _mesh_suffix = os.path.join('train_deploy_alignment', 'inference', 'agilex',
                                     'Piper_ros_private-ros-noetic', 'src',
                                     'piper_description', 'meshes')
        mesh_dir = None
        for _mesh_base in [_KAI0_ROOT,
                           os.path.join(_PROJECT_ROOT, 'kai0')]:
            candidate = os.path.normpath(os.path.join(_mesh_base, _mesh_suffix))
            if os.path.isdir(candidate):
                mesh_dir = candidate
                break
        if mesh_dir is None:
            mesh_dir = os.path.normpath(os.path.join(_KAI0_ROOT, _mesh_suffix))

        mesh_names = ['base_link', 'link1', 'link2', 'link3', 'link4', 'link5',
                      'link6', 'gripper_base', 'link7', 'link8']
        if os.path.isdir(mesh_dir):
            try:
                for arm_label in ('left', 'right'):
                    arm_rgba = ([100, 100, 255, 10] if arm_label == 'left'
                                else [100, 255, 100, 10])
                    for mesh_name in mesh_names:
                        stl_path = os.path.join(mesh_dir, f'{mesh_name}.STL')
                        if os.path.exists(stl_path):
                            verts, faces = self._load_binary_stl(stl_path)
                            # Rgba32 vertex colors (alpha in the 4th channel).
                            rr.log(f"world/{arm_label}/{mesh_name}/mesh", rr.Mesh3D(
                                vertex_positions=verts,
                                triangle_indices=faces,
                                vertex_colors=np.full((len(verts), 4),
                                                      arm_rgba, dtype=np.uint8),
                            ), static=True)
                self._arm_mesh_loaded = True
                self.get_logger().info(f'Loaded arm meshes from {mesh_dir}')
            except Exception as e:
                self.get_logger().warn(f'Failed to load arm meshes: {e}')
        else:
            self.get_logger().warn(f'Mesh dir not found: {mesh_dir}')

    # ── Image callbacks ──────────────────────────────────────────

    def _cb_img_front(self, msg):
        # Cache only; logging happens in _tick_images timer.
        try:
            self._latest_front_rgb = (
                self._bridge.imgmsg_to_cv2(msg, 'rgb8'),
                _stamp_to_sec(msg.header.stamp))
        except Exception as e:
            self.get_logger().debug(f'img_front error: {e}')

    def _cb_img_left(self, msg):
        try:
            self._latest_left_rgb = (
                self._bridge.imgmsg_to_cv2(msg, 'rgb8'),
                _stamp_to_sec(msg.header.stamp))
        except Exception as e:
            self.get_logger().debug(f'img_left error: {e}')

    def _cb_img_right(self, msg):
        try:
            self._latest_right_rgb = (
                self._bridge.imgmsg_to_cv2(msg, 'rgb8'),
                _stamp_to_sec(msg.header.stamp))
        except Exception as e:
            self.get_logger().debug(f'img_right error: {e}')

    def _tick_report(self):
        now = time.monotonic()
        dt = now - self._tick_report_t
        self._tick_report_t = now
        jl_age = (now - getattr(self, '_last_q_left_rx', 0))
        jr_age = (now - getattr(self, '_last_q_right_rx', 0))
        arms_ms_avg = (getattr(self, '_tick_arms_ms_sum', 0.0)
                       / max(1, self._tick_arms_count))
        arms_age_avg = (getattr(self, '_tick_arms_age_sum', 0.0)
                        / max(1, self._tick_arms_count))
        arms_age_max = getattr(self, '_tick_arms_age_max', 0.0)
        self.get_logger().info(
            f'ticks/2s: arms={self._tick_arms_count}(compute={arms_ms_avg:.1f}ms '
            f'data_age_avg={arms_age_avg:.0f}ms max={arms_age_max:.0f}ms) '
            f'pc={self._tick_pc_count} img={self._tick_img_count} | '
            f'cb/2s: jl={self._cb_jl_count} jr={self._cb_jr_count}')
        self._tick_arms_count = 0
        self._tick_arms_ms_sum = 0.0
        self._tick_arms_age_sum = 0.0
        self._tick_arms_age_max = 0.0
        self._tick_pc_count = 0
        self._tick_img_count = 0
        self._cb_jl_count = 0
        self._cb_jr_count = 0

    def _tick_images(self):
        """10 Hz timer: log the latest cached camera images to 2D views.
        Downsample 640x480 -> 320x240 to cut TCP bandwidth 4x (from ~40 MB/s
        to ~10 MB/s), preventing Rerun pipeline backpressure."""
        self._tick_img_count += 1
        rr = self._rr
        for entity, cached in [
            ("images/top_head", self._latest_front_rgb),
            ("images/hand_left", self._latest_left_rgb),
            ("images/hand_right", self._latest_right_rgb),
        ]:
            if cached is None:
                continue
            img, stamp = cached
            try:
                img_small = img[::2, ::2]  # 2x downsample
                rr.set_time("ros_time", timestamp=stamp)
                rr.log(entity, rr.Image(img_small))
            except Exception as e:
                self.get_logger().debug(f'img tick error ({entity}): {e}')

    # ── Depth → point cloud ──────────────────────────────────────

    _depth_log_count = 0

    def _project_depth_cpu(self, depth_u16, step, fx, fy, cx, cy,
                           T_world_cam, depth_min, depth_max, rgb_img):
        """Numpy fallback for depth -> world point cloud + RGB sampling."""
        h, w = depth_u16.shape
        depth_m = depth_u16[0:h:step, 0:w:step].astype(np.float32) * 0.001
        hs, ws = depth_m.shape
        v_g, u_g = np.mgrid[0:h:step, 0:w:step]
        valid = (depth_m > depth_min) & (depth_m < depth_max)
        if not valid.any():
            return None, None
        u = u_g[valid].astype(np.float32)
        v = v_g[valid].astype(np.float32)
        z = depth_m[valid]
        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        pts_cam = np.stack([x_cam, y_cam, z], axis=-1)
        R_wc = T_world_cam[:3, :3].astype(np.float32)
        t_wc = T_world_cam[:3, 3].astype(np.float32)
        pts_world = (R_wc @ pts_cam.T).T + t_wc
        colors = None
        if rgb_img is not None and rgb_img.shape[0] == h and rgb_img.shape[1] == w:
            u_idx = u.astype(np.int32).clip(0, w - 1)
            v_idx = v.astype(np.int32).clip(0, h - 1)
            colors = np.ascontiguousarray(rgb_img[v_idx, u_idx])
        return pts_world, colors

    def _project_depth_gpu(self, depth_u16, step, fx, fy, cx, cy,
                           T_world_cam, depth_min, depth_max, rgb_img):
        """JAX-accelerated depth -> world point cloud + RGB sampling.
        Falls back to numpy if JAX/CUDA unavailable. JIT compiles once per shape.
        Returns (pts_world_np, colors_np) or (None, None) if no valid points.
        """
        if not _GPU_OK or _jax_project_fn is None:
            return self._project_depth_cpu(
                depth_u16, step, fx, fy, cx, cy,
                T_world_cam, depth_min, depth_max, rgb_img)
        try:
            h, w = depth_u16.shape
            # Stride on CPU (cheap), upload strided slice to GPU.
            depth_strided = np.ascontiguousarray(depth_u16[0:h:step, 0:w:step])
            T_wc_f32 = np.ascontiguousarray(T_world_cam, dtype=np.float32)
            # JIT core — fixed input shape => cached compile after first call.
            pts_world_dev, depth_m_dev = _jax_project_fn(
                depth_strided, float(fx), float(fy), float(cx), float(cy),
                int(step), T_wc_f32)
            # Transfer back to CPU (async until we touch numpy).
            pts_world = np.asarray(pts_world_dev)  # (hs, ws, 3)
            depth_m = np.asarray(depth_m_dev)      # (hs, ws)
            # Mask on CPU (dynamic shape — cheap).
            valid = (depth_m > depth_min) & (depth_m < depth_max)
            if not valid.any():
                return None, None
            pts = pts_world[valid]  # (N, 3)
            colors = None
            if rgb_img is not None and rgb_img.shape[0] == h and rgb_img.shape[1] == w:
                vi, ui = np.where(valid)
                colors = np.ascontiguousarray(
                    rgb_img[vi * step, ui * step])
            return pts, colors
        except Exception as e:
            if not hasattr(self, '_jax_err_logged'):
                self._jax_err_logged = True
                self.get_logger().warn(f'JAX projection failed, falling back to CPU: {e}')
            return self._project_depth_cpu(
                depth_u16, step, fx, fy, cx, cy,
                T_world_cam, depth_min, depth_max, rgb_img)

    # ── Depth callbacks: cache-only (lightweight) ────────────────────

    def _cache_depth(self, msg):
        """Decode and return (depth_u16, stamp). Returns None on failure."""
        try:
            d = self._bridge.imgmsg_to_cv2(msg, 'passthrough')
            if d.dtype != np.uint16:
                d = d.astype(np.uint16)
            return (d, _stamp_to_sec(msg.header.stamp))
        except Exception:
            return None

    def _cb_depth_front(self, msg):
        cached = self._cache_depth(msg)
        if cached is not None:
            self._latest_depth_front = cached

    def _cb_depth_left(self, msg):
        cached = self._cache_depth(msg)
        if cached is not None:
            self._latest_depth_left = cached

    def _cb_depth_right(self, msg):
        cached = self._cache_depth(msg)
        if cached is not None:
            self._latest_depth_right = cached

    # ── Point cloud timer: process latest cached depth + color ───────

    def _tick_point_clouds(self):
        """Runs at ~15 Hz from a timer; always uses the newest cached data."""
        self._tick_pc_count += 1
        rr = self._rr
        step = self.get_parameter('depth_step').value
        depth_min = self.get_parameter('depth_min').value
        depth_max = self.get_parameter('depth_max').value

        # Head camera (fixed transform)
        if self._latest_depth_front is not None and self._T_world_camF is not None:
            depth_u16, stamp = self._latest_depth_front
            rgb_img = self._latest_front_rgb[0] if self._latest_front_rgb else None
            try:
                pts, colors = self._project_depth_gpu(
                    depth_u16, step, self._cam_f_fx, self._cam_f_fy,
                    self._cam_f_cx, self._cam_f_cy,
                    self._T_world_camF, depth_min, depth_max,
                    rgb_img)
                if pts is not None:
                    rr.set_time("ros_time", timestamp=stamp)
                    rr.log("world/head_rgb", rr.Points3D(
                        pts, radii=[0.001], colors=colors))
            except Exception as e:
                self.get_logger().debug(f'head pc error: {e}')

        # Wrist cameras (FK-transformed per latest joint state)
        _lrgb = self._latest_left_rgb[0] if self._latest_left_rgb else None
        _rrgb = self._latest_right_rgb[0] if self._latest_right_rgb else None
        for arm, dep_cache, q, T_base, T_l6_cam, fx, fy, cx, cy, rgb in [
            ('left', self._latest_depth_left, self._latest_q_left,
             self._T_world_baseL, self._T_link6_camL,
             self._cam_l_fx, self._cam_l_fy, self._cam_l_cx, self._cam_l_cy,
             _lrgb),
            ('right', self._latest_depth_right, self._latest_q_right,
             self._T_world_baseR, self._T_link6_camR,
             self._cam_r_fx, self._cam_r_fy, self._cam_r_cx, self._cam_r_cy,
             _rrgb),
        ]:
            if (dep_cache is None or self._fk is None or
                    T_l6_cam is None or q is None):
                continue
            depth_u16, stamp = dep_cache
            try:
                T_base_ee = self._fk.fk_homogeneous(q[:6])
                T_world_cam = np.array(T_base) @ T_base_ee @ np.array(T_l6_cam)
                pts, colors = self._project_depth_gpu(
                    depth_u16, step, fx, fy, cx, cy,
                    T_world_cam, 0.01, depth_max, rgb)
                if pts is not None:
                    rr.set_time("ros_time", timestamp=stamp)
                    if colors is not None:
                        rr.log(f"world/{arm}_rgb", rr.Points3D(
                            pts, colors=colors, radii=[0.001]))
                    else:
                        rr.log(f"world/{arm}_rgb", rr.Points3D(
                            pts, radii=[0.001]))
            except Exception as e:
                self.get_logger().debug(f'{arm} pc error: {e}')

    # ── Joint callbacks ──────────────────────────────────────────

    # Joint callbacks: cache latest only. A timer runs the expensive logging.
    _cb_jl_count = 0
    _cb_jr_count = 0

    def _cb_joint_left(self, msg):
        try:
            q = np.array(msg.position, dtype=np.float32)
            if len(q) >= 6:
                self._latest_q_left = q[:7] if len(q) >= 7 else np.append(q[:6], 0.0)
                self._latest_q_left_stamp = _stamp_to_sec(msg.header.stamp)
                self._last_q_left_rx = time.monotonic()
                self._cb_jl_count += 1
        except Exception as e:
            self.get_logger().debug(f'joint_left error: {e}')

    def _cb_joint_right(self, msg):
        try:
            q = np.array(msg.position, dtype=np.float32)
            if len(q) >= 6:
                self._latest_q_right = q[:7] if len(q) >= 7 else np.append(q[:6], 0.0)
                self._latest_q_right_stamp = _stamp_to_sec(msg.header.stamp)
                self._last_q_right_rx = time.monotonic()
                self._cb_jr_count += 1
        except Exception as e:
            self.get_logger().debug(f'joint_right error: {e}')

    # ── FK/timeseries timer: logs the latest snapshot ────────────────

    def _tick_arms(self):
        """Runs at ~30 Hz; always uses the newest cached joint state."""
        self._tick_arms_count += 1
        t0 = time.monotonic()
        # Measure joint-rx -> tick-entry latency (how old the data is now)
        if hasattr(self, '_last_q_left_rx'):
            age_ms = (t0 - self._last_q_left_rx) * 1000
            self._tick_arms_age_sum = getattr(
                self, '_tick_arms_age_sum', 0.0) + age_ms
            self._tick_arms_age_max = max(
                getattr(self, '_tick_arms_age_max', 0.0), age_ms)
        rr = self._rr
        for arm, q, stamp, T_base, trail, trail_color in [
            ('left', self._latest_q_left, self._latest_q_left_stamp,
             self._T_world_baseL, self._actual_trail_left, [50, 100, 255]),
            ('right', self._latest_q_right, self._latest_q_right_stamp,
             self._T_world_baseR, self._actual_trail_right, [50, 255, 100]),
        ]:
            if q is None or self._fk is None:
                continue
            try:
                rr.set_time("ros_time", timestamp=stamp)
                # Timeseries at 1/3 of FK rate (5Hz) — still smooth for plots.
                if self._tick_arms_count % 3 == 0:
                    for i in range(min(7, len(q))):
                        rr.log(f"timeseries/{arm}_j{i}", rr.Scalar(float(q[i])))
                ee_pos = self._log_arm_fk(arm, q, T_base)
                if ee_pos is not None:
                    trail.append(ee_pos.copy())
                    trail_arr = np.array(list(trail))
                    rr.log(f"world/actual/{arm}_trail", rr.Points3D(
                        trail_arr, colors=[trail_color], radii=[0.002]))
            except Exception as e:
                self.get_logger().warn(f'arm tick error ({arm}): {e}')
        self._tick_arms_ms_sum = getattr(self, '_tick_arms_ms_sum', 0.0) + (
            time.monotonic() - t0) * 1000.0

    def _cb_execute(self, msg):
        rr = self._rr
        rr.set_time("ros_time", timestamp=self.get_clock().now().nanoseconds * 1e-9)
        rr.log("timeseries/execute_mode", rr.Scalar(1.0 if msg.data else 0.0))

    def _log_master_arm(self, arm_label, msg, T_base, trail):
        """Log commanded (master) joint state: time series + EE marker + trail.

        Note: piper_start_ms_node ALSO publishes to /master/joint_* with all zeros
        (no physical master arm hardware). Filter those out so only real policy
        commands are visualized.
        """
        if self._fk is None:
            return
        try:
            q = np.array(msg.position, dtype=np.float32)
            if len(q) < 6:
                return
            # Skip piper_start_ms_node stub publishes (all zeros)
            if not np.any(np.abs(q[:6]) > 1e-6):
                return
            rr = self._rr
            stamp = _stamp_to_sec(msg.header.stamp) if msg.header.stamp.sec else \
                self.get_clock().now().nanoseconds * 1e-9
            rr.set_time("ros_time", timestamp=stamp)

            # Time series (7 joints)
            for i in range(min(7, len(q))):
                rr.log(f"timeseries/master_{arm_label}_j{i}", rr.Scalar(float(q[i])))

            # FK → world EE position
            q6 = q[:6]
            limits = self._JOINT_LIMITS_RAD
            if np.any(np.abs(q6) > limits * 1.5):
                return
            T_ee = T_base @ self._fk.fk_homogeneous(q6)
            ee = T_ee[:3, 3]
            trail.append(ee.copy())

            # Commanded EE marker (orange = left, yellow = right)
            color = [255, 150, 50] if arm_label == 'left' else [255, 220, 50]
            rr.log(f"world/master/{arm_label}/ee", rr.Points3D(
                [ee], colors=[color], radii=[0.01]))
            trail_arr = np.array(list(trail))
            rr.log(f"world/master/{arm_label}/trail", rr.LineStrips3D(
                [trail_arr], colors=[color], radii=[0.002]))
        except Exception as e:
            self.get_logger().debug(f'master {arm_label} error: {e}')

    def _cb_master_left(self, msg):
        self._log_master_arm('left', msg, self._T_world_baseL, self._master_trail_left)

    def _cb_master_right(self, msg):
        self._log_master_arm('right', msg, self._T_world_baseR, self._master_trail_right)

    # Piper joint limits (rad) — reject out-of-range FK inputs to avoid wild trajectories
    _JOINT_LIMITS_RAD = np.array([2.618, 1.571, 1.571, 1.745, 1.571, 2.618])

    _chunk_log_count = 0

    def _cb_action_chunk(self, msg):
        """Render the full predicted action chunk as a 3D trajectory line strip.

        Chunk shape: (N steps, 14 dims) where dims = [L_j0..L_j5, L_grip, R_j0..R_j5, R_grip].
        Runs FK per step → world EE positions → LineStrips3D.
        """
        self._chunk_log_count += 1
        if self._chunk_log_count <= 3:
            self.get_logger().info(
                f'action_chunk received: fk={self._fk is not None} '
                f'baseL={self._T_world_baseL is not None} baseR={self._T_world_baseR is not None}')
        if self._fk is None or self._T_world_baseL is None or self._T_world_baseR is None:
            return
        try:
            dims = msg.layout.dim
            if len(dims) < 2:
                return
            n_steps = dims[0].size
            n_dim = dims[1].size
            if n_dim != 14 or n_steps < 2:
                return
            actions = np.array(msg.data, dtype=np.float32).reshape(n_steps, n_dim)

            limits = self._JOINT_LIMITS_RAD
            left_path, right_path = [], []
            for t in range(n_steps):
                q_l = actions[t, 0:6]
                q_r = actions[t, 7:13]
                if np.any(np.abs(q_l) > limits * 1.5) or np.any(np.abs(q_r) > limits * 1.5):
                    continue
                T_l = self._T_world_baseL @ self._fk.fk_homogeneous(q_l)
                T_r = self._T_world_baseR @ self._fk.fk_homogeneous(q_r)
                left_path.append(T_l[:3, 3])
                right_path.append(T_r[:3, 3])

            if self._chunk_log_count <= 3:
                self.get_logger().info(
                    f'action_chunk: {n_steps} steps, {len(left_path)} L valid, {len(right_path)} R valid')

            if len(left_path) < 2 or len(right_path) < 2:
                return

            rr = self._rr
            left_arr = np.array(left_path)
            right_arr = np.array(right_path)
            stamp = self.get_clock().now().nanoseconds * 1e-9
            rr.set_time("ros_time", timestamp=stamp)
            rr.log("world/predicted/left_traj", rr.LineStrips3D(
                [left_arr], colors=[[50, 150, 255]], radii=[0.004]))
            rr.log("world/predicted/right_traj", rr.LineStrips3D(
                [right_arr], colors=[[50, 255, 150]], radii=[0.004]))
            rr.log("world/predicted/left_endpoints", rr.Points3D(
                [left_arr[0], left_arr[-1]], colors=[[50, 150, 255]],
                radii=[0.005], labels=["start", "end"]))
            rr.log("world/predicted/right_endpoints", rr.Points3D(
                [right_arr[0], right_arr[-1]], colors=[[50, 255, 150]],
                radii=[0.005], labels=["start", "end"]))
        except Exception as e:
            self.get_logger().warn(f'action_chunk error: {e}')

    # ── FK processing ────────────────────────────────────────────

    def _process_fk_queue(self):
        rr = self._rr
        t0 = time.monotonic()
        BUDGET = 0.030

        queue_len = len(self._fk_queue)
        if queue_len >= self._fk_queue.maxlen:
            self._fk_drops += 1
            self._fk_skip = min(self._fk_skip + 1, 15)
        elif queue_len <= self._fk_queue.maxlen // 4 and self._fk_skip > 3:
            self._fk_skip = max(self._fk_skip - 1, 3)

        processed = 0
        while self._fk_queue and processed < 10:
            if time.monotonic() - t0 > BUDGET:
                break
            try:
                arm_label, q7, T_base, stamp = self._fk_queue.popleft()
            except IndexError:
                break
            processed += 1
            try:
                rr.set_time("ros_time", timestamp=stamp)
                ee_pos = self._log_arm_fk(arm_label, q7, T_base)
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
                self.get_logger().debug(f'FK viz error ({arm_label}): {e}')

    def _log_arm_fk(self, arm_label, q7, T_world_base):
        if self._fk is None:
            return None
        rr = self._rr
        q6_rad = q7[:6]
        gripper_val = float(q7[6]) if len(q7) > 6 else 0.0

        link_Ts = self._fk.fk_all_links(q6_rad)
        T_base = np.array(T_world_base)

        rr.log(f"world/{arm_label}/base_link", rr.Transform3D(
            translation=T_base[:3, 3], mat3x3=T_base[:3, :3]))

        for idx, T_link in enumerate(link_Ts):
            T_world_link = T_base @ T_link
            rr.log(f"world/{arm_label}/link{idx+1}", rr.Transform3D(
                translation=T_world_link[:3, 3], mat3x3=T_world_link[:3, :3]))

        T_world_link6 = T_base @ link_Ts[5]
        rr.log(f"world/{arm_label}/gripper_base", rr.Transform3D(
            translation=T_world_link6[:3, 3], mat3x3=T_world_link6[:3, :3]))

        finger_spread = np.clip(gripper_val, 0.0, 0.04) / 0.04
        max_lateral = 0.025
        for link_name, rpy, sign in [('link7', [np.pi/2, 0, 0], 1.0),
                                      ('link8', [np.pi/2, 0, -np.pi], -1.0)]:
            T_offset = np.eye(4)
            T_offset[:3, :3] = R_.from_euler('xyz', rpy).as_matrix()
            T_offset[:3, 3] = [sign * finger_spread * max_lateral, 0, 0.1358]
            T_w = T_world_link6 @ T_offset
            rr.log(f"world/{arm_label}/{link_name}", rr.Transform3D(
                translation=T_w[:3, 3], mat3x3=T_w[:3, :3]))

        ee_pos = T_world_link6[:3, 3]
        base_pos = T_base[:3, 3]
        ee_color = [50, 100, 255] if arm_label == 'left' else [50, 255, 100]
        cam_color = [100, 100, 255] if arm_label == 'left' else [100, 255, 100]
        rr.log(f"world/{arm_label}/ee", rr.Points3D(
            [ee_pos], colors=[ee_color], radii=[0.008]))

        # Wrist camera frustum + EE-to-base line (like verify_calibration.py)
        T_link6_cam = self._T_link6_camL if arm_label == 'left' else self._T_link6_camR
        if T_link6_cam is not None:
            T_world_cam = T_world_link6 @ T_link6_cam
            R_cam = T_world_cam[:3, :3]
            cam_pos = T_world_cam[:3, 3]
            fd, fhw, fhh = 0.04, 0.02, 0.015
            fc = [(R_cam @ np.array(p) + cam_pos).tolist()
                  for p in [[0, 0, 0], [-fhw, -fhh, fd], [fhw, -fhh, fd],
                            [fhw, fhh, fd], [-fhw, fhh, fd]]]
            rr.log(f"world/{arm_label}_cam", rr.LineStrips3D(
                [[fc[0], fc[1]], [fc[0], fc[2]], [fc[0], fc[3]], [fc[0], fc[4]],
                 [fc[1], fc[2]], [fc[2], fc[3]], [fc[3], fc[4]], [fc[4], fc[1]],
                 [cam_pos.tolist(), ee_pos.tolist()],
                 [ee_pos.tolist(), base_pos.tolist()]],
                colors=[cam_color], radii=[0.002],
            ))
            rr.log(f"world/{arm_label}_cam_label", rr.Points3D(
                [cam_pos.tolist()], colors=[cam_color], radii=[0.002],
                labels=[f"D405 ({arm_label})"],
            ))

        return ee_pos


def main(args=None):
    rclpy.init(args=args)
    node = RerunVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
