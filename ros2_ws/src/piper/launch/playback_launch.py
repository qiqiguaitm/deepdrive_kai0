"""ROS2 launch — playback stack: same architecture as autonomy_launch.py, but
the camera input + the joint output both come from a recorded LeRobot v2.1
episode rather than RealSense + policy_inference.

Replaces:
  multi_camera_node       → video_publisher_node    (mp4 + depth zarr)
  policy_inference_node   → replay_node             (parquet action[T,14])
Keeps:
  arm_reader_node ×2      (slave; skip when enable_real_arms:=false → sim)
  rerun_viz_node          (consumes the same Image + JointState topics)

Usage:
  ros2 launch piper playback_launch.py \\
      episode_path:=/data1/DATA_IMP/KAI0/Task_A_mirror/base/2026-04-28/data/chunk-000/episode_000000.parquet
  ros2 launch piper playback_launch.py episode_path:=... enable_real_arms:=true
  ros2 launch piper playback_launch.py episode_path:=... enable_rerun:=false

Designed to be wrapped by start_autonomy.sh `--replay [--sim] --episode <ep>`.
The wrapper writes the /tmp/kai0_deployment_mode = 'replay' marker that
replay_node's pre-flight checks.
"""
import glob
import os

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            SetEnvironmentVariable, TimerAction)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# ── Project paths (mirror autonomy_launch.py) ──────────────────────────────
def _find_project_root():
    candidate = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    if os.path.isdir(os.path.join(candidate, 'kai0')):
        return candidate
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(d, 'kai0')):
            return d
        d = os.path.dirname(d)
    return os.path.expanduser('~/workspace/deepdive_kai0')


_PROJECT_ROOT = _find_project_root()
_KAI0_ROOT = os.path.join(_PROJECT_ROOT, 'kai0')
_CONFIG_DIR = os.path.join(_PROJECT_ROOT, 'config')

# kai0 venv site-packages → PYTHONPATH so video_publisher (cv2, zarr) and
# replay_node (pyarrow, numpy) load. Matches autonomy_launch's pattern.
_VENV_LIB = os.path.join(_KAI0_ROOT, '.venv', 'lib')
_VENV_PYDIR = sorted(glob.glob(os.path.join(_VENV_LIB, 'python3.*')))
_VENV = (os.path.join(_VENV_PYDIR[-1], 'site-packages') if _VENV_PYDIR
         else os.path.join(_VENV_LIB, 'python3.12', 'site-packages'))
_PTH_DIRS = []
for pth in sorted(glob.glob(os.path.join(_VENV, '*.pth'))):
    try:
        with open(pth) as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith('#') or _line.startswith('import '):
                    continue
                _resolved = _line if os.path.isabs(_line) else os.path.join(_VENV, _line)
                if os.path.isdir(_resolved):
                    _PTH_DIRS.append(_resolved)
    except OSError:
        pass
_PYTHONPATH = ':'.join([_VENV] + _PTH_DIRS + [os.path.join(_KAI0_ROOT, 'src')])

_DEFAULT_CALIB = os.path.join(_CONFIG_DIR, 'calibration.yml')


def generate_launch_description():
    episode_path_arg = DeclareLaunchArgument(
        'episode_path', default_value='',
        description='absolute parquet path; videos derived from sibling videos/ tree')
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate', default_value='30',
        description='shared publish rate for replay_node + video_publisher (Hz)')
    enable_real_arms_arg = DeclareLaunchArgument(
        'enable_real_arms', default_value='false',
        description='true → arm_reader slave drives CAN; false → sim (no real arms move)')
    enable_rerun_arg = DeclareLaunchArgument(
        'enable_rerun', default_value='true',
        description='true → spawn rerun_viz_node alongside; false → headless')
    calib_file_arg = DeclareLaunchArgument(
        'calib_file', default_value=_DEFAULT_CALIB,
        description='hardware calibration yaml for FK / mesh visualization')
    auto_execute_arg = DeclareLaunchArgument(
        'auto_execute', default_value='true',
        description=("true → after a short delay automatically set "
                     "replay_mode=replay and publish /policy/execute=true so the "
                     "arm starts moving without manual ros2 commands. set false "
                     "if you want to fire manually (e.g. via start_replay_test.sh)."))

    enable_real_arms = LaunchConfiguration('enable_real_arms')
    enable_rerun = LaunchConfiguration('enable_rerun')
    auto_execute = LaunchConfiguration('auto_execute')

    # Inject venv site-packages into PYTHONPATH so spawned nodes see numpy/cv2/
    # zarr/pyarrow. Crucially we PREPEND to the inherited PYTHONPATH (which has
    # rclpy + ROS msg packages) — overwriting it strips rclpy and the nodes can't
    # import ros at all.
    _existing_pp = os.environ.get('PYTHONPATH', '')
    _full_pp = _PYTHONPATH + (':' + _existing_pp if _existing_pp else '')
    set_pp = SetEnvironmentVariable('PYTHONPATH', _full_pp)

    # ── Video publisher (replaces multi_camera_node) ──
    video_pub = Node(
        package='piper', executable='video_publisher_node.py',
        name='video_publisher', output='screen',
        parameters=[{
            'episode_path': LaunchConfiguration('episode_path'),
            'publish_rate': LaunchConfiguration('publish_rate'),
        }],
    )

    # ── Replay (replaces policy_inference_node) ──
    replay = Node(
        package='piper', executable='replay_node.py',
        name='replay', output='screen',
        parameters=[{
            'publish_rate': LaunchConfiguration('publish_rate'),
            # episode_path on replay_node is hot-set later by the caller (start_replay_test.sh
            # or `ros2 param set /replay replay_episode_path …`). We could pre-set here too,
            # but leaving it empty matches start_replay_stack.sh's existing UX.
            'replay_episode_path': LaunchConfiguration('episode_path'),
        }],
    )

    # ── Real-arm slave readers (skipped in sim) ──
    piper_left = Node(
        package='piper', executable='arm_reader_node.py',
        name='piper_left', output='screen',
        parameters=[{'can_port': 'can_left_slave', 'mode': 1, 'auto_enable': True}],
        remappings=[
            ('/puppet/joint_states', '/puppet/joint_left'),
            ('/master/joint_states', '/master/joint_left'),
            ('/puppet/arm_status', '/puppet/arm_status_left'),
            ('/puppet/end_pose', '/puppet/end_pose_left'),
            ('/puppet/end_pose_euler', '/puppet/end_pose_euler_left'),
        ],
        condition=IfCondition(enable_real_arms),
    )
    piper_right = Node(
        package='piper', executable='arm_reader_node.py',
        name='piper_right', output='screen',
        parameters=[{'can_port': 'can_right_slave', 'mode': 1, 'auto_enable': True}],
        remappings=[
            ('/puppet/joint_states', '/puppet/joint_right'),
            ('/master/joint_states', '/master/joint_right'),
            ('/puppet/arm_status', '/puppet/arm_status_right'),
            ('/puppet/end_pose', '/puppet/end_pose_right'),
            ('/puppet/end_pose_euler', '/puppet/end_pose_euler_right'),
        ],
        condition=IfCondition(enable_real_arms),
    )

    # ── Sim mode: feed /master/joint_* back as /puppet/joint_* so replay_node's
    #    pose-alignment / jump-protection logic sees a "current state". Without
    #    real arms there's no slave reader, so we run a single Python relay node
    #    instead of two topic_tools/relay processes (avoids that apt dependency). ──
    sim_loopback = Node(
        package='piper', executable='joint_state_loopback_node.py',
        name='joint_state_loopback', output='log',
        parameters=[{
            # action[0] seed → /puppet/joint_* so replay_node's start-pose
            # alignment passes before /master starts publishing.
            'episode_path': LaunchConfiguration('episode_path'),
        }],
        condition=UnlessCondition(enable_real_arms),  # only in sim mode
    )

    # ── Rerun visualization (same node as autonomy) ──
    rerun_node = Node(
        package='piper', executable='rerun_viz_node.py',
        name='rerun_viz', output='screen',
        parameters=[{'calib_file': LaunchConfiguration('calib_file')}],
        condition=IfCondition(enable_rerun),
    )

    # ── Auto-execute kicker ──
    # After ~5 s (replay_node graph-registered, action[0] seed in /puppet deque),
    # set replay_mode=replay (triggers in-node preflight + buffer fill) then fire
    # /policy/execute=true. Without this, the stack stays idle and the arm in
    # rerun never moves.
    kick_set_mode = ExecuteProcess(
        cmd=['ros2', 'param', 'set', '/replay', 'replay_mode', 'replay'],
        output='log', shell=False,
    )
    kick_execute = ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '--once', '/policy/execute',
             'std_msgs/Bool', '{data: true}'],
        output='log', shell=False,
    )
    auto_kicker = TimerAction(
        period=5.0,
        actions=[kick_set_mode, TimerAction(period=0.5, actions=[kick_execute])],
        condition=IfCondition(auto_execute),
    )

    return LaunchDescription([
        episode_path_arg, publish_rate_arg, enable_real_arms_arg,
        enable_rerun_arg, calib_file_arg, auto_execute_arg,
        set_pp,
        video_pub, replay,
        piper_left, piper_right,
        sim_loopback,
        rerun_node,
        auto_kicker,
    ])
