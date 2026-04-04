"""
一键启动全套推理栈 (ROS2 native 模式)

包含: 2x piper (mode=0 只读) + 3x 相机 (RGB+depth 对齐) + policy_inference_node

Usage:
  # 纯 ROS2 模式 (推荐, 最低延迟)
  ros2 launch piper inference_full_launch.py mode:=ros2

  # WebSocket 模式 (兼容旧 serve_policy.py, 需先启动 serve_policy)
  ros2 launch piper inference_full_launch.py mode:=websocket

  # 两者兼有模式
  ros2 launch piper inference_full_launch.py mode:=both
"""
import os
import glob
import yaml
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            SetEnvironmentVariable, TimerAction)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# ── Project paths ──
# Resolve from the *source* tree (not the install tree) so paths work after colcon install.
# Walk up from this file's real path to find the workspace root containing 'kai0/' and 'ros2_ws/'.
def _find_project_root():
    """Find workspace root by looking for kai0/ directory, starting from the source tree."""
    # Try source-tree location first (../../..)  from ros2_ws/src/piper/launch/
    candidate = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    if os.path.isdir(os.path.join(candidate, 'kai0')):
        return candidate
    # Fallback: walk up from __file__ until we find kai0/
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(d, 'kai0')):
            return d
        d = os.path.dirname(d)
    # Last resort: assume workspace is at a well-known location
    return os.path.expanduser('~/workspace/deepdive_kai0')

_PROJECT_ROOT = _find_project_root()
_KAI0_ROOT = os.path.join(_PROJECT_ROOT, 'kai0')
_CONFIG_DIR = os.path.join(_PROJECT_ROOT, 'config')

# 自动构建 CUDA LD_LIBRARY_PATH
_VENV_LIB = os.path.join(_KAI0_ROOT, '.venv', 'lib')
_VENV_PYDIR = sorted(glob.glob(os.path.join(_VENV_LIB, 'python3.*')))
_VENV = os.path.join(_VENV_PYDIR[-1], 'site-packages') if _VENV_PYDIR else os.path.join(_VENV_LIB, 'python3.12', 'site-packages')
_NVIDIA_LIBS = ':'.join(
    sorted(glob.glob(os.path.join(_VENV, 'nvidia', '*', 'lib')))
)
# .pth files aren't processed when using PYTHONPATH, so add their entries explicitly
_PTH_DIRS = []
for pth in sorted(glob.glob(os.path.join(_VENV, '*.pth'))):
    with open(pth) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith('#') or _line.startswith('import '):
                continue
            _resolved = _line if os.path.isabs(_line) else os.path.join(_VENV, _line)
            if os.path.isdir(_resolved):
                _PTH_DIRS.append(_resolved)
_PYTHONPATH = ':'.join([_VENV] + _PTH_DIRS + [os.path.join(_KAI0_ROOT, 'src')])

# ── Load hardware config ──
def _load_config(name):
    path = os.path.join(_CONFIG_DIR, name)
    if os.path.isfile(path):
        with open(path) as f:
            return yaml.safe_load(f)
    return {}

_cameras_cfg = _load_config('cameras.yml')
_calib_cfg = _load_config('calibration.yml')

# CAN port names from calibration (hardware section)
_hw_cfg = _calib_cfg.get('hardware', {})
_LEFT_CAN = _hw_cfg.get('left_arm_can', 'can_left_slave')
_RIGHT_CAN = _hw_cfg.get('right_arm_can', 'can_right_slave')

def _cam_serial(role):
    """Get camera serial from config/cameras.yml by role key."""
    cams = _cameras_cfg.get('cameras', {})
    entry = cams.get(role, {})
    return entry.get('serial_number', '')

_CAM_F_SERIAL = _cam_serial('top_head') or '254622070889'
_CAM_L_SERIAL = _cam_serial('hand_left') or '409122273074'
_CAM_R_SERIAL = _cam_serial('hand_right') or '409122271568'

_DEFAULT_CALIB = os.path.join(_CONFIG_DIR, 'calibration.yml')


def generate_launch_description():
    # ── Arguments ──
    mode_arg = DeclareLaunchArgument('mode', default_value='ros2',
        description='ros2 | websocket | both')
    gpu_arg = DeclareLaunchArgument('gpu_id', default_value='0')
    # ── Policy 配置 (参照 serve_policy.py 的 Checkpoint 模式) ──
    # config_name: 决定 transform 链 (图像预处理、归一化、action 后处理)
    # checkpoint_dir: 决定模型权重来源 (GCS 路径会自动下载到 $OPENPI_DATA_HOME)
    #
    # 常见组合:
    #   kai0 最佳模型: config=pi05_flatten_fold_normal  ckpt=.../checkpoints/Task_A/mixed_1
    #   自训练模型:    config=pi05_flatten_fold_normal  ckpt=.../checkpoints/<config>/<exp>/<step>
    config_arg = DeclareLaunchArgument('config_name',
        default_value='pi05_flatten_fold_normal',
        description='Training config name (determines transform pipeline)')
    ckpt_arg = DeclareLaunchArgument('checkpoint_dir',
        default_value=os.path.join(_KAI0_ROOT, 'checkpoints', 'Task_A', 'mixed_1'),
        description='Trained model checkpoint path (kai0 best model or your own trained checkpoint)')
    host_arg = DeclareLaunchArgument('host', default_value='localhost',
        description='WebSocket server host (only for mode=websocket)')
    port_arg = DeclareLaunchArgument('port', default_value='8000',
        description='WebSocket server port')
    prompt_arg = DeclareLaunchArgument('prompt',
        default_value='Flatten and fold the cloth.',
        description='Language prompt (must match training config default_prompt)')
    execute_mode_arg = DeclareLaunchArgument('execute_mode',
        default_value='false',
        description='Start in execute mode (true) or observe-only mode (false)')
    enable_rerun_arg = DeclareLaunchArgument('enable_rerun',
        default_value='false',
        description='Enable Rerun 3D visualization of trajectories')
    calib_arg = DeclareLaunchArgument('calibration_config',
        default_value=_DEFAULT_CALIB,
        description='Calibration YAML path (for FK visualization in Rerun)')

    # ── Piper 左臂 (can1, mode=0 只读) ──
    piper_left = Node(
        package='piper', executable='piper_start_ms_node.py',
        name='piper_left', output='screen',
        parameters=[{'can_port': _LEFT_CAN, 'mode': 0, 'auto_enable': False}],
        remappings=[
            ('/puppet/joint_states', '/puppet/joint_left'),
            ('/master/joint_states', '/master/joint_left'),
            ('/puppet/arm_status', '/puppet/arm_status_left'),
            ('/puppet/end_pose', '/puppet/end_pose_left'),
            ('/puppet/end_pose_euler', '/puppet/end_pose_euler_left'),
        ],
    )

    # ── Piper 右臂 (mode=0 只读) ──
    piper_right = Node(
        package='piper', executable='piper_start_ms_node.py',
        name='piper_right', output='screen',
        parameters=[{'can_port': _RIGHT_CAN, 'mode': 0, 'auto_enable': False}],
        remappings=[
            ('/puppet/joint_states', '/puppet/joint_right'),
            ('/master/joint_states', '/master/joint_right'),
            ('/puppet/arm_status', '/puppet/arm_status_right'),
            ('/puppet/end_pose', '/puppet/end_pose_right'),
            ('/puppet/end_pose_euler', '/puppet/end_pose_euler_right'),
        ],
    )

    # ── Multi-camera node: single process manages all 3 RealSense cameras ──
    # Avoids USB contention from multiple realsense2_camera_node processes
    # each doing independent device enumeration and USB resets.
    multi_cam = Node(
        package='piper', executable='multi_camera_node.py',
        name='multi_camera', output='screen',
        parameters=[{
            'cam_f_serial': _CAM_F_SERIAL,
            'cam_l_serial': _CAM_L_SERIAL,
            'cam_r_serial': _CAM_R_SERIAL,
            'fps': 30,
            'width': 640,
            'height': 480,
            'enable_head_depth': True,
            'enable_wrist_depth': True,
        }],
    )

    # ── Policy Inference Node ──
    policy_node = Node(
        package='piper', executable='policy_inference_node.py',
        name='policy_inference', output='screen',
        parameters=[{
            'mode': LaunchConfiguration('mode'),
            'config_name': LaunchConfiguration('config_name'),
            'checkpoint_dir': LaunchConfiguration('checkpoint_dir'),
            'host': LaunchConfiguration('host'),
            'port': LaunchConfiguration('port'),
            'prompt': LaunchConfiguration('prompt'),
            'gpu_id': LaunchConfiguration('gpu_id'),
            'img_front_topic': '/camera_f/camera/color/image_raw',
            'img_left_topic': '/camera_l/camera/color/image_raw',
            'img_right_topic': '/camera_r/camera/color/image_raw',
            'puppet_left_topic': '/puppet/joint_left',
            'puppet_right_topic': '/puppet/joint_right',
            'execute_mode': LaunchConfiguration('execute_mode'),
        }],
    )

    # ── Rerun Visualization Node (separate process, conditional) ──
    rerun_node = Node(
        package='piper', executable='rerun_viz_node.py',
        name='rerun_viz', output='screen',
        condition=IfCondition(LaunchConfiguration('enable_rerun')),
        parameters=[{
            'calibration_config': LaunchConfiguration('calibration_config'),
            'img_front_topic': '/camera_f/camera/color/image_raw',
            'img_left_topic': '/camera_l/camera/color/image_raw',
            'img_right_topic': '/camera_r/camera/color/image_raw',
            'depth_front_topic': '/camera_f/camera/aligned_depth_to_color/image_raw',
            'depth_left_topic': '/camera_l/camera/aligned_depth_to_color/image_raw',
            'depth_right_topic': '/camera_r/camera/aligned_depth_to_color/image_raw',
            'puppet_left_topic': '/puppet/joint_left',
            'puppet_right_topic': '/puppet/joint_right',
        }],
    )

    # 环境变量 (CUDA 库 + Python 路径 + venv bin for rerun CLI, 追加到现有值)
    _VENV_BIN = os.path.join(_KAI0_ROOT, '.venv', 'bin')
    existing_ld = os.environ.get('LD_LIBRARY_PATH', '')
    existing_py = os.environ.get('PYTHONPATH', '')
    existing_path = os.environ.get('PATH', '')
    set_ld = SetEnvironmentVariable('LD_LIBRARY_PATH',
        _NVIDIA_LIBS + ':' + existing_ld if existing_ld else _NVIDIA_LIBS)
    set_py = SetEnvironmentVariable('PYTHONPATH',
        _PYTHONPATH + ':' + existing_py if existing_py else _PYTHONPATH)
    set_path = SetEnvironmentVariable('PATH',
        _VENV_BIN + ':' + existing_path if existing_path else _VENV_BIN)
    set_cache = SetEnvironmentVariable('JAX_COMPILATION_CACHE_DIR', '/tmp/xla_cache')
    # 0.35 = ~11.4GB on 32GB GPU, leaves room for cuBLAS workspace and other processes
    set_mem_frac = SetEnvironmentVariable('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.35')

    # ── Cleanup: kill stale processes from previous launches ──
    # Stale Rerun viewer holds port 9876 -> rr.spawn() silently reconnects to
    # zombie with old blueprint state. Stale RealSense / policy / piper nodes
    # hold USB handles & GPU memory. Kill them all before starting.
    cleanup = ExecuteProcess(
        cmd=['bash', '-c',
             'pkill -9 -f "rerun_viz_node|multi_camera_node|policy_inference_node'
             '|piper_start_ms_node|piper_start_slave_node|realsense2_camera_node'
             '|rerun_sdk/rerun_cli/rerun" || true; '
             'sleep 2'],
        output='screen',
    )

    # Rerun viz node starts early (lightweight, waits for topics)
    rerun_delayed = TimerAction(period=4.0, actions=[rerun_node])
    # Cameras/arms start after cleanup settles
    multi_cam_delayed = TimerAction(period=3.0, actions=[multi_cam])
    piper_left_delayed = TimerAction(period=3.0, actions=[piper_left])
    piper_right_delayed = TimerAction(period=3.0, actions=[piper_right])
    # Policy node waits for cameras to stabilize
    policy_delayed = TimerAction(period=17.0, actions=[policy_node])

    return LaunchDescription([
        set_ld, set_py, set_path, set_cache, set_mem_frac,
        mode_arg, gpu_arg, config_arg, ckpt_arg, host_arg, port_arg, prompt_arg,
        execute_mode_arg, enable_rerun_arg, calib_arg,
        cleanup,
        piper_left_delayed, piper_right_delayed,
        multi_cam_delayed,
        rerun_delayed,
        policy_delayed,
    ])
