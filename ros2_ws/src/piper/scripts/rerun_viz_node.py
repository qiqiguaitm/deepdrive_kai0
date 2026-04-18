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
#
# autonomy_launch.py sets these explicitly via additional_env when the node
# is spawned by the launch file (the authoritative source). The setdefault
# here is a fallback for standalone `ros2 run piper rerun_viz_node.py`
# invocations outside the launch file — if you change either side, keep
# them in sync.
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

# Optional background-mesh builder. Open3D is a heavy dependency so import
# failure must not crash the node — Step 5 is gracefully disabled instead.
try:
    import _bg_builder  # type: ignore
except Exception:
    # rerun_viz_node.py ships alongside _bg_builder.py in the same scripts
    # directory, but the ROS2 ament install sometimes places them under a
    # sibling lib/ dir. Adding __file__'s dir to sys.path is a no-op in the
    # normal case and lets `import _bg_builder` succeed post-install.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import _bg_builder  # type: ignore
    except Exception as _bg_err:
        _bg_builder = None
        _BG_IMPORT_ERROR = f'{type(_bg_err).__name__}: {_bg_err}'
    else:
        _BG_IMPORT_ERROR = ''
else:
    _BG_IMPORT_ERROR = ''


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
        self.declare_parameter('depth_step', 1)
        self.declare_parameter('depth_min', 0.05)
        self.declare_parameter('depth_max', 1.5)
        # Time sync tolerances (used to pair depth with the correct RGB frame
        # and interpolate joint state at the depth timestamp). Setting these
        # loosely only delays drop-outs; real-time drift normally stays below
        # 20 ms on sim01 with reliable QoS.
        self.declare_parameter('sync_rgb_tol_s', 0.03)
        self.declare_parameter('sync_joint_max_gap_s', 0.1)
        # Foreground mesh (Step 1–4 of docs/deployment/inference_visualization_mesh.md).
        # When fg_enable=True, _tick_point_clouds emits rr.Mesh3D per camera
        # instead of the legacy rr.Points3D path. Kept off by default so the
        # old behaviour is preserved until the mesh path is validated live.
        self.declare_parameter('fg_enable', False)
        self.declare_parameter('fg_edge_thresh_m', 0.02)
        # World-frame AABB of the dynamic workspace. Head camera pixels
        # outside this box are dropped (they belong to the static background
        # mesh in Step 5). Wrist cameras ignore the AABB — their field of
        # view is inherently inside the workspace.
        self.declare_parameter('fg_bbox_min', [-0.6, -0.6, 0.0])
        self.declare_parameter('fg_bbox_max', [0.6, 0.6, 0.8])
        # Optional downsample step for wrist cameras; head stays at depth_step.
        # Bump to 2 if rerun serialisation becomes the bottleneck.
        self.declare_parameter('fg_wrist_step', 1)
        # Static background mesh (Step 5). Built once via TSDF fusion of N
        # head-camera frames, then logged as a static rerun entity. A cache
        # file lets node restarts reuse the last rebuild without re-running
        # the fusion. Set bg_enable=True and call /rerun_viz/rebuild_bg (or
        # have a valid cache file on disk) to populate the mesh.
        self.declare_parameter('bg_enable', False)
        self.declare_parameter('bg_num_frames', 30)
        self.declare_parameter('bg_voxel_size', 0.005)
        self.declare_parameter('bg_sdf_trunc', 0.04)
        self.declare_parameter('bg_depth_trunc', 2.0)
        self.declare_parameter('bg_device', 'CUDA:0')
        self.declare_parameter(
            'bg_cache_path',
            os.path.expanduser('~/.cache/kai0_viz/bg_mesh.npz'))

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

        # ── Time-aligned ring buffers ──
        # Unlike the single-slot _latest_* fields above (which feed legacy
        # per-tick snapshots), these deques keep a short history so we can
        # pair depth frames with the RGB and joint state that actually
        # correspond to the depth timestamp. This is the prerequisite for
        # stable point-cloud / mesh rendering when the arms are moving:
        # otherwise wrist-camera FK uses "now" joints while depth is tens of
        # milliseconds old, visibly throwing the projection off.
        self._q_left_buf = deque(maxlen=200)    # list of (stamp, q[7])
        self._q_right_buf = deque(maxlen=200)
        self._rgb_front_buf = deque(maxlen=10)  # list of (stamp, rgb_hwc)
        self._rgb_left_buf = deque(maxlen=10)
        self._rgb_right_buf = deque(maxlen=10)
        self._sync_drop_count = 0  # diagnostic: frames dropped due to no sync match

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

        # Identity transform at world root so the 3D view has a valid origin.
        # An empty rr.Transform3D() is dropped on 0.31 ("chunk without
        # components"), so set an explicit zero translation.
        rr.log("world", rr.Transform3D(translation=[0, 0, 0]), static=True)
        # Do NOT pre-log static placeholder images — on rerun 0.31 a static
        # 1x1 image locks the 2D view size and prevents real-sized dynamic
        # frames from displaying. The views will populate once the first
        # real frame arrives from the cameras.

        # Layout:
        #   top (~70%): 3D scene | telemetry plot
        #   bottom (~30%): head | hand_left | hand_right (full-width horizontal)
        # Giving the 2D image row its own full-width strip ensures each of the
        # 3 image views is wide enough that Rerun doesn't collapse them.
        rr.send_blueprint(rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="world", contents="world/**",
                                       name="3D Scene"),
                    rrb.TimeSeriesView(origin="timeseries",
                                        contents="timeseries/**",
                                        name="Telemetry"),
                    column_shares=[3, 1],
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="images/top_head", name="Head"),
                    rrb.Spatial2DView(origin="images/hand_left", name="Left"),
                    rrb.Spatial2DView(origin="images/hand_right", name="Right"),
                ),
                row_shares=[7, 3],
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

        # ── Static background mesh (Step 5) ──
        # Service lets the user trigger a rebuild after repositioning cameras
        # or the workspace. The handler returns immediately; a daemon thread
        # does the fusion so the ROS spin loop isn't blocked. Mesh is logged
        # once built and cached to disk for instant reload on node restart.
        self._bg_building = False
        self._bg_thread = None
        if self.get_parameter('bg_enable').value:
            from std_srvs.srv import Trigger  # local import: optional path
            self.create_service(
                Trigger, '/rerun_viz/rebuild_bg', self._srv_rebuild_bg)
            # Try to restore the last build from disk so the user sees the
            # background mesh immediately on restart.
            self._try_load_cached_bg()

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
                [pos], colors=[[255, 255, 0]], radii=[0.0075],
                labels=[label], show_labels=False),
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
                [o], colors=[[255, 50, 50]], radii=[0.005],
                labels=["D435 (head)"], show_labels=False),
                static=True)

        # Arm meshes
        self._load_meshes(transforms)

    @staticmethod
    def _load_binary_stl(path):
        """Load a binary STL file. Returns (vertices [N,3], faces [M,3], vertex_normals [N,3]).

        Computes per-vertex normals by area-weighted averaging of adjacent face
        normals. Rerun's Mesh3D shader needs vertex_normals to use its PBR
        render path, which is what actually honors albedo_factor alpha —
        without normals the mesh renders fully opaque.
        """
        with open(path, 'rb') as f:
            f.read(80)  # header
            n_tri = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            dt = np.dtype([('normal', np.float32, 3), ('v', np.float32, (3, 3)), ('attr', np.uint16)])
            data = np.frombuffer(f.read(n_tri * dt.itemsize), dtype=dt)
        all_verts = data['v'].reshape(-1, 3)
        verts_unique, inverse = np.unique(all_verts, axis=0, return_inverse=True)
        faces = inverse.reshape(-1, 3)

        # Per-face normals (un-normalized = area-weighted)
        v0 = verts_unique[faces[:, 0]]
        v1 = verts_unique[faces[:, 1]]
        v2 = verts_unique[faces[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)  # (M, 3)

        # Accumulate face normals onto each vertex they touch
        vert_normals = np.zeros_like(verts_unique)
        for i in range(3):
            np.add.at(vert_normals, faces[:, i], face_normals)

        # Normalize
        lens = np.linalg.norm(vert_normals, axis=1, keepdims=True)
        lens[lens == 0] = 1.0
        vert_normals /= lens
        return verts_unique, faces, vert_normals.astype(np.float32)

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
                    arm_rgb = [100, 100, 255] if arm_label == 'left' else [100, 255, 100]
                    for mesh_name in mesh_names:
                        stl_path = os.path.join(mesh_dir, f'{mesh_name}.STL')
                        if os.path.exists(stl_path):
                            verts, faces, normals = self._load_binary_stl(stl_path)
                            # Mesh at the same entity as the dynamic Transform3D
                            # (like verify_calibration.py). vertex_normals is
                            # REQUIRED for Rerun to use its lit/PBR render path,
                            # which is the one that actually honors alpha in
                            # albedo_factor. Without normals, Rerun falls back
                            # to an unlit shader that draws fully opaque.
                            rr.log(f"world/{arm_label}/{mesh_name}", rr.Mesh3D(
                                vertex_positions=verts,
                                triangle_indices=faces,
                                vertex_normals=normals,
                                vertex_colors=np.full((len(verts), 3),
                                                      arm_rgb, dtype=np.uint8),
                                albedo_factor=arm_rgb + [20],
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
            img = self._bridge.imgmsg_to_cv2(msg, 'rgb8')
            stamp = _stamp_to_sec(msg.header.stamp)
            self._latest_front_rgb = (img, stamp)
            self._rgb_front_buf.append((stamp, img))
        except Exception as e:
            self.get_logger().debug(f'img_front error: {e}')

    def _cb_img_left(self, msg):
        try:
            img = self._bridge.imgmsg_to_cv2(msg, 'rgb8')
            stamp = _stamp_to_sec(msg.header.stamp)
            self._latest_left_rgb = (img, stamp)
            self._rgb_left_buf.append((stamp, img))
        except Exception as e:
            self.get_logger().debug(f'img_left error: {e}')

    def _cb_img_right(self, msg):
        try:
            img = self._bridge.imgmsg_to_cv2(msg, 'rgb8')
            stamp = _stamp_to_sec(msg.header.stamp)
            self._latest_right_rgb = (img, stamp)
            self._rgb_right_buf.append((stamp, img))
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
            f'cb/2s: jl={self._cb_jl_count} jr={self._cb_jr_count} | '
            f'sync_drops={self._sync_drop_count}')
        self._sync_drop_count = 0
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

    # ── Time sync helpers ────────────────────────────────────────

    @staticmethod
    def _interp_q(buf, t, max_gap):
        """Linearly interpolate joint state at timestamp t from a deque of
        (stamp, q) entries sorted ascending by stamp. Returns None if t falls
        outside the buffer range by more than max_gap on either side, or if
        the two bracketing samples are farther apart than max_gap (gap in data).

        buf : collections.deque[(float, np.ndarray)]
        t   : float, target timestamp (seconds)
        max_gap : float, maximum allowed time gap (seconds)
        """
        n = len(buf)
        if n == 0:
            return None
        if n == 1:
            s0, q0 = buf[0]
            return q0 if abs(s0 - t) <= max_gap else None
        # buf is small (≤200 entries @ 200 Hz ≈ 1 s); linear scan from the
        # end is fine and avoids converting to a list for bisect.
        s_last, _ = buf[-1]
        s_first, _ = buf[0]
        if t >= s_last:
            return buf[-1][1] if (t - s_last) <= max_gap else None
        if t <= s_first:
            return buf[0][1] if (s_first - t) <= max_gap else None
        # Walk back to find bracketing pair (hi is first entry with stamp > t).
        hi = n - 1
        while hi > 0 and buf[hi - 1][0] > t:
            hi -= 1
        s_hi, q_hi = buf[hi]
        s_lo, q_lo = buf[hi - 1]
        if (s_hi - s_lo) > max_gap:
            return None
        alpha = (t - s_lo) / max(s_hi - s_lo, 1e-9)
        return q_lo * (1.0 - alpha) + q_hi * alpha

    @staticmethod
    def _nearest_rgb(buf, t, tol):
        """Return the RGB frame whose stamp is closest to t, if within tol.
        Returns None (caller should skip) when no frame qualifies — do NOT
        fall back to the latest frame, that would reintroduce the misalignment
        the ring buffer exists to fix.
        """
        if not buf:
            return None
        best = None
        best_dt = tol
        # Search from newest to oldest — depth stamps are usually very recent.
        for stamp, img in reversed(buf):
            dt = abs(stamp - t)
            if dt <= best_dt:
                best_dt = dt
                best = img
            # Buffer is ordered ascending, so once dt starts growing past tol
            # we can stop. Guard with a tight bound to avoid worst-case scan.
            if stamp + tol < t:
                break
        return best

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

    # ── Depth → screen-space triangle mesh (foreground) ─────────────

    def _build_fg_mesh(self, depth_u16, rgb_img, step, fx, fy, cx, cy,
                       T_world_cam, depth_min, depth_max, edge_thresh,
                       bbox_min=None, bbox_max=None):
        """Build a screen-space triangle mesh from a depth frame.

        Mesh is expressed in world frame. Pixels outside [depth_min, depth_max]
        or (optionally) outside the world-frame AABB are masked out. Triangles
        whose four corner depths span more than ``edge_thresh`` metres are
        dropped — this kills the "rubber sheet" artefacts that would otherwise
        connect foreground and background across occlusion boundaries.

        Returns (verts, tris, colors) or None if no valid triangles. On first
        JAX failure falls back to None (caller simply skips this tick for that
        camera — the mesh path does not use CPU projection by design, because
        the CPU fallback is ~40× slower and would blow the 200 ms budget).
        """
        if not _GPU_OK or _jax_project_fn is None:
            return None

        h, w = depth_u16.shape
        try:
            # Stride on CPU, upload strided slice to GPU. This reuses the
            # JIT-compiled projection core and its shape-keyed compile cache
            # shared with _project_depth_gpu.
            depth_strided = np.ascontiguousarray(depth_u16[0:h:step, 0:w:step])
            T_wc_f32 = np.ascontiguousarray(T_world_cam, dtype=np.float32)
            pts_world_dev, depth_m_dev = _jax_project_fn(
                depth_strided, float(fx), float(fy), float(cx), float(cy),
                int(step), T_wc_f32)
            pts_world = np.asarray(pts_world_dev)  # (H, W, 3) strided
            depth_m   = np.asarray(depth_m_dev)    # (H, W)
        except Exception as e:
            if not hasattr(self, '_jax_mesh_err_logged'):
                self._jax_mesh_err_logged = True
                self.get_logger().warn(f'JAX mesh projection failed: {e}')
            return None

        H, W = depth_m.shape

        # Per-pixel validity: depth range + optional world-frame AABB.
        valid = (depth_m > depth_min) & (depth_m < depth_max)
        if bbox_min is not None and bbox_max is not None:
            xyz = pts_world
            valid &= (
                (xyz[..., 0] >= bbox_min[0]) & (xyz[..., 0] <= bbox_max[0]) &
                (xyz[..., 1] >= bbox_min[1]) & (xyz[..., 1] <= bbox_max[1]) &
                (xyz[..., 2] >= bbox_min[2]) & (xyz[..., 2] <= bbox_max[2]))
        if not valid.any():
            return None

        # Quad validity: all four corners valid AND corner depths agree within
        # edge_thresh. This runs on numpy — 2×(H-1)×(W-1) booleans at 640×480
        # is ~600k elements, vectorised reductions take <10 ms on modern CPUs.
        vq = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
        if not vq.any():
            return None
        d = depth_m
        dstack = np.stack([d[:-1, :-1], d[1:, :-1], d[:-1, 1:], d[1:, 1:]])
        vq &= (dstack.max(0) - dstack.min(0)) < edge_thresh
        if not vq.any():
            return None

        # Emit 2 triangles per valid quad. Vertex indices reference a flat
        # (H*W, 3) vertex array so winding stays consistent for lit shading.
        ii, jj = np.nonzero(vq)
        v00 = (ii * W + jj).astype(np.int32)
        v10 = ((ii + 1) * W + jj).astype(np.int32)
        v01 = (ii * W + (jj + 1)).astype(np.int32)
        v11 = ((ii + 1) * W + (jj + 1)).astype(np.int32)
        tris = np.concatenate([
            np.stack([v00, v10, v11], axis=-1),
            np.stack([v00, v11, v01], axis=-1),
        ], axis=0)

        # Compress to only the referenced vertices — rerun serialises the
        # full vertex array, so this shrinks the log payload 2–5× vs sending
        # the whole H*W grid.
        used, inverse = np.unique(tris, return_inverse=True)
        tris_c = inverse.reshape(-1, 3).astype(np.int32)
        verts = pts_world.reshape(-1, 3)[used].astype(np.float32)

        colors = None
        if (rgb_img is not None and rgb_img.shape[0] == h
                and rgb_img.shape[1] == w):
            rgb_strided = rgb_img[0:h:step, 0:w:step]
            colors = np.ascontiguousarray(
                rgb_strided.reshape(-1, 3)[used])

        return verts, tris_c, colors

    def _log_mesh3d(self, entity, verts, tris, colors):
        """Thin wrapper around rr.Mesh3D that handles optional colors."""
        rr = self._rr
        if colors is not None:
            rr.log(entity, rr.Mesh3D(
                vertex_positions=verts,
                triangle_indices=tris,
                vertex_colors=colors))
        else:
            rr.log(entity, rr.Mesh3D(
                vertex_positions=verts,
                triangle_indices=tris))

    # ── Static background mesh (Step 5) ─────────────────────────────

    def _log_bg_mesh(self, verts, tris, colors, normals):
        """Log the static background mesh. Logged once with static=True so
        it shows at every point on the rerun timeline without re-sending."""
        rr = self._rr
        kwargs = dict(vertex_positions=verts, triangle_indices=tris)
        if colors is not None:
            kwargs['vertex_colors'] = colors
        if normals is not None:
            kwargs['vertex_normals'] = normals
        rr.log('world/bg_mesh', rr.Mesh3D(**kwargs), static=True)

    def _try_load_cached_bg(self):
        """Load a previously-built background mesh from disk, if present."""
        if _bg_builder is None:
            self.get_logger().warn(
                f'bg_enable=True but _bg_builder import failed '
                f'({_BG_IMPORT_ERROR}); background mesh disabled')
            return
        path = self.get_parameter('bg_cache_path').value
        cached = _bg_builder.load_mesh_npz(path)
        if cached is None:
            self.get_logger().info(
                f'no cached bg mesh at {path} — call '
                f'/rerun_viz/rebuild_bg to build one')
            return
        verts, tris, colors, normals = cached
        self._log_bg_mesh(verts, tris, colors, normals)
        self.get_logger().info(
            f'loaded cached bg mesh ({len(verts)} verts, {len(tris)} tris) '
            f'from {path}')

    def _srv_rebuild_bg(self, request, response):
        """std_srvs/Trigger: spawn a worker thread to rebuild the bg mesh.

        Returns immediately. Progress is reported via get_logger(); the mesh
        appears in rerun once the worker finishes and writes the disk cache.
        """
        del request  # std_srvs/Trigger has no fields
        if _bg_builder is None or not _bg_builder.open3d_available():
            response.success = False
            err = (_BG_IMPORT_ERROR if _bg_builder is None
                   else _bg_builder.open3d_error())
            response.message = f'open3d unavailable: {err}'
            return response
        if self._bg_building:
            response.success = False
            response.message = 'background rebuild already in progress'
            return response
        if self._T_world_camF is None or self._cam_f_fx is None:
            response.success = False
            response.message = 'head camera calibration not loaded'
            return response
        import threading
        self._bg_building = True
        self._bg_thread = threading.Thread(
            target=self._bg_rebuild_worker, daemon=True)
        self._bg_thread.start()
        response.success = True
        response.message = 'background rebuild started'
        return response

    def _bg_rebuild_worker(self):
        """Daemon thread: collect N head-camera frames, fuse via TSDF, log.

        Runs on a background thread, so reads of the _latest_* caches must
        tolerate concurrent writes from the ROS callbacks. Python tuple
        assignment is atomic under GIL so grabbing `self._latest_depth_front`
        by reference is safe — we immediately copy the arrays to be sure
        nothing mutates them during integration.
        """
        try:
            N = int(self.get_parameter('bg_num_frames').value)
            rgb_tol = float(self.get_parameter('sync_rgb_tol_s').value)
            frames = []
            last_stamp = -1.0
            deadline = time.monotonic() + max(15.0, N * 0.5)
            self.get_logger().info(
                f'[bg] collecting {N} head-camera frames (timeout '
                f'{deadline - time.monotonic():.0f}s)')
            while len(frames) < N and time.monotonic() < deadline:
                dep = self._latest_depth_front
                if dep is None:
                    time.sleep(0.05)
                    continue
                depth_u16, stamp = dep
                if stamp <= last_stamp:  # haven't received a new frame yet
                    time.sleep(0.02)
                    continue
                rgb = self._nearest_rgb(self._rgb_front_buf, stamp, rgb_tol)
                if rgb is None:
                    time.sleep(0.02)
                    continue
                frames.append((depth_u16.copy(), rgb.copy()))
                last_stamp = stamp
                time.sleep(0.1)  # space frames out to decorrelate noise

            if len(frames) < max(5, N // 2):
                self.get_logger().warn(
                    f'[bg] only collected {len(frames)}/{N} frames — '
                    f'check camera topics')
                return

            K = np.array([
                [self._cam_f_fx, 0.0, self._cam_f_cx],
                [0.0, self._cam_f_fy, self._cam_f_cy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            bbox_min = list(self.get_parameter('fg_bbox_min').value)
            bbox_max = list(self.get_parameter('fg_bbox_max').value)
            self.get_logger().info(
                f'[bg] fusing {len(frames)} frames via TSDF '
                f'(voxel={self.get_parameter("bg_voxel_size").value} m, '
                f'device={self.get_parameter("bg_device").value})')
            t_fuse0 = time.monotonic()
            result = _bg_builder.build_bg_mesh_gpu(
                frames,
                K=K,
                T_world_cam=self._T_world_camF,
                bbox_exclude_min=bbox_min,
                bbox_exclude_max=bbox_max,
                voxel_size=float(self.get_parameter('bg_voxel_size').value),
                sdf_trunc=float(self.get_parameter('bg_sdf_trunc').value),
                depth_trunc=float(self.get_parameter('bg_depth_trunc').value),
                device_str=self.get_parameter('bg_device').value,
            )
            fuse_ms = (time.monotonic() - t_fuse0) * 1000.0
            if result is None:
                self.get_logger().warn('[bg] TSDF produced empty mesh')
                return
            verts, tris, colors, normals = result
            self.get_logger().info(
                f'[bg] built mesh: {len(verts)} verts, {len(tris)} tris '
                f'in {fuse_ms:.0f} ms')
            self._log_bg_mesh(verts, tris, colors, normals)
            try:
                path = self.get_parameter('bg_cache_path').value
                _bg_builder.save_mesh_npz(path, verts, tris, colors, normals)
                self.get_logger().info(f'[bg] cached to {path}')
            except Exception as e:
                self.get_logger().warn(f'[bg] cache save failed: {e}')
        except Exception as e:
            self.get_logger().error(f'[bg] rebuild failed: {e}')
        finally:
            self._bg_building = False

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
        """5 Hz timer. Dispatches to the mesh or legacy Points3D path.

        Step 0 added time-aligned ring buffers for RGB + joint state.
        Step 1–4 adds the mesh path (fg_enable=True); the legacy path is
        kept for A/B comparison and fallback when JAX is unavailable.
        """
        self._tick_pc_count += 1
        if self.get_parameter('fg_enable').value:
            self._tick_fg_mesh()
        else:
            self._tick_pcd_legacy()

    def _tick_fg_mesh(self):
        """Dynamic foreground: screen-space triangle mesh per camera.

        See docs/deployment/inference_visualization_mesh.md §6. Head camera is bbox
        clipped; wrist cameras use the full FOV (they are inherently inside
        the workspace). All three cameras pair depth/RGB/joint state via the
        Step 0 ring buffers; unpaired frames are dropped, never rendered
        with mismatched data.
        """
        rr = self._rr
        step_head = self.get_parameter('depth_step').value
        step_wrist = self.get_parameter('fg_wrist_step').value
        depth_min = self.get_parameter('depth_min').value
        depth_max = self.get_parameter('depth_max').value
        edge_thresh = self.get_parameter('fg_edge_thresh_m').value
        rgb_tol = self.get_parameter('sync_rgb_tol_s').value
        q_gap = self.get_parameter('sync_joint_max_gap_s').value
        bbox_min = np.asarray(
            self.get_parameter('fg_bbox_min').value, dtype=np.float32)
        bbox_max = np.asarray(
            self.get_parameter('fg_bbox_max').value, dtype=np.float32)

        # Head camera: bbox-clipped dynamic region only. Pixels outside the
        # workspace AABB belong to the static background mesh (Step 5).
        if self._latest_depth_front is not None and self._T_world_camF is not None:
            depth_u16, stamp = self._latest_depth_front
            rgb = self._nearest_rgb(self._rgb_front_buf, stamp, rgb_tol)
            try:
                result = self._build_fg_mesh(
                    depth_u16, rgb, step_head,
                    self._cam_f_fx, self._cam_f_fy,
                    self._cam_f_cx, self._cam_f_cy,
                    self._T_world_camF,
                    depth_min, depth_max, edge_thresh,
                    bbox_min, bbox_max)
                if result is not None:
                    verts, tris, colors = result
                    rr.set_time("ros_time", timestamp=stamp)
                    self._log_mesh3d(
                        "world/dynamic/head_fg_mesh", verts, tris, colors)
            except Exception as e:
                self.get_logger().debug(f'head mesh error: {e}')

        # Wrist cameras: FK transform with joint state interpolated to the
        # depth timestamp. No bbox clipping — wrist FOV is fully inside the
        # workspace and depth/FK errors grow quickly outside ~0.5 m anyway.
        for arm, dep_cache, q_buf, rgb_buf, T_base, T_l6_cam, \
                fx, fy, cx, cy in [
            ('left', self._latest_depth_left, self._q_left_buf,
             self._rgb_left_buf, self._T_world_baseL, self._T_link6_camL,
             self._cam_l_fx, self._cam_l_fy, self._cam_l_cx, self._cam_l_cy),
            ('right', self._latest_depth_right, self._q_right_buf,
             self._rgb_right_buf, self._T_world_baseR, self._T_link6_camR,
             self._cam_r_fx, self._cam_r_fy, self._cam_r_cx, self._cam_r_cy),
        ]:
            if dep_cache is None or self._fk is None or T_l6_cam is None:
                continue
            depth_u16, stamp = dep_cache
            q = self._interp_q(q_buf, stamp, q_gap)
            if q is None:
                self._sync_drop_count += 1
                continue
            rgb = self._nearest_rgb(rgb_buf, stamp, rgb_tol)
            try:
                T_base_ee = self._fk.fk_homogeneous(q[:6])
                T_world_cam = np.array(T_base) @ T_base_ee @ np.array(T_l6_cam)
                result = self._build_fg_mesh(
                    depth_u16, rgb, step_wrist, fx, fy, cx, cy,
                    T_world_cam,
                    0.01, depth_max, edge_thresh,
                    bbox_min=None, bbox_max=None)
                if result is not None:
                    verts, tris, colors = result
                    rr.set_time("ros_time", timestamp=stamp)
                    self._log_mesh3d(
                        f'world/dynamic/{arm}_fg_mesh', verts, tris, colors)
            except Exception as e:
                self.get_logger().debug(f'{arm} mesh error: {e}')

    def _tick_pcd_legacy(self):
        """Legacy Points3D path (fg_enable=False). Each camera pairs its
        depth frame with the RGB frame and (for wrist cameras) the joint
        state at that depth's timestamp. Frames that cannot be paired within
        the configured tolerance are skipped rather than rendered with
        mismatched data — see _interp_q / _nearest_rgb.
        """
        rr = self._rr
        step = self.get_parameter('depth_step').value
        depth_min = self.get_parameter('depth_min').value
        depth_max = self.get_parameter('depth_max').value
        rgb_tol = self.get_parameter('sync_rgb_tol_s').value
        q_gap = self.get_parameter('sync_joint_max_gap_s').value

        # Head camera (fixed transform)
        if self._latest_depth_front is not None and self._T_world_camF is not None:
            depth_u16, stamp = self._latest_depth_front
            rgb_img = self._nearest_rgb(self._rgb_front_buf, stamp, rgb_tol)
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

        # Wrist cameras: interpolate joint state to the depth timestamp, pair
        # RGB by stamp. Dropping a tick is cheaper than rendering wrong data.
        for arm, dep_cache, q_buf, rgb_buf, T_base, T_l6_cam, \
                fx, fy, cx, cy in [
            ('left', self._latest_depth_left, self._q_left_buf,
             self._rgb_left_buf, self._T_world_baseL, self._T_link6_camL,
             self._cam_l_fx, self._cam_l_fy, self._cam_l_cx, self._cam_l_cy),
            ('right', self._latest_depth_right, self._q_right_buf,
             self._rgb_right_buf, self._T_world_baseR, self._T_link6_camR,
             self._cam_r_fx, self._cam_r_fy, self._cam_r_cx, self._cam_r_cy),
        ]:
            if (dep_cache is None or self._fk is None or T_l6_cam is None):
                continue
            depth_u16, stamp = dep_cache
            q = self._interp_q(q_buf, stamp, q_gap)
            if q is None:
                self._sync_drop_count += 1
                continue
            rgb = self._nearest_rgb(rgb_buf, stamp, rgb_tol)
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
                q7 = q[:7] if len(q) >= 7 else np.append(q[:6], 0.0)
                stamp = _stamp_to_sec(msg.header.stamp)
                self._latest_q_left = q7
                self._latest_q_left_stamp = stamp
                self._q_left_buf.append((stamp, q7))
                self._last_q_left_rx = time.monotonic()
                self._cb_jl_count += 1
        except Exception as e:
            self.get_logger().debug(f'joint_left error: {e}')

    def _cb_joint_right(self, msg):
        try:
            q = np.array(msg.position, dtype=np.float32)
            if len(q) >= 6:
                q7 = q[:7] if len(q) >= 7 else np.append(q[:6], 0.0)
                stamp = _stamp_to_sec(msg.header.stamp)
                self._latest_q_right = q7
                self._latest_q_right_stamp = stamp
                self._q_right_buf.append((stamp, q7))
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
                        rr.log(f"timeseries/{arm}_j{i}", rr.Scalars([float(q[i])]))
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
        rr.log("timeseries/execute_mode", rr.Scalars([1.0 if msg.data else 0.0]))

    def _log_master_arm(self, arm_label, msg, T_base, trail):
        """Log commanded (master) joint state: time series + EE marker + trail.

        Note: arm_reader_node ALSO publishes to /master/joint_* with all zeros
        (no physical master arm hardware). Filter those out so only real policy
        commands are visualized.
        """
        if self._fk is None:
            return
        try:
            q = np.array(msg.position, dtype=np.float32)
            if len(q) < 6:
                return
            # Skip arm_reader_node stub publishes (all zeros)
            if not np.any(np.abs(q[:6]) > 1e-6):
                return
            rr = self._rr
            stamp = _stamp_to_sec(msg.header.stamp) if msg.header.stamp.sec else \
                self.get_clock().now().nanoseconds * 1e-9
            rr.set_time("ros_time", timestamp=stamp)

            # Time series (7 joints)
            for i in range(min(7, len(q))):
                rr.log(f"timeseries/master_{arm_label}_j{i}", rr.Scalars([float(q[i])]))

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
                radii=[0.005], labels=["start", "end"], show_labels=False))
            rr.log("world/predicted/right_endpoints", rr.Points3D(
                [right_arr[0], right_arr[-1]], colors=[[50, 255, 150]],
                radii=[0.005], labels=["start", "end"], show_labels=False))
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
        ee_color = [50, 100, 255] if arm_label == 'left' else [50, 255, 100]
        cam_color = [100, 100, 255] if arm_label == 'left' else [100, 255, 100]
        rr.log(f"world/{arm_label}/ee", rr.Points3D(
            [ee_pos], colors=[ee_color], radii=[0.008]))

        # Wrist camera frustum only (no link6->base line — visually noisy)
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
                 [cam_pos.tolist(), ee_pos.tolist()]],
                colors=[cam_color], radii=[0.002],
            ))
            rr.log(f"world/{arm_label}_cam_label", rr.Points3D(
                [cam_pos.tolist()], colors=[cam_color], radii=[0.002],
                labels=[f"D405 ({arm_label})"], show_labels=False,
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
