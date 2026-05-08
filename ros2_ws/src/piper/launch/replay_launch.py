"""ROS2 launch — replay-only stack (NO autonomy / NO cameras / NO JAX).

Brings up:
  - 2× arm_reader_node mode=1 (left + right slave) — drives arms via CAN
  - 1× replay_node — accepts replay params + drives /master/joint_*

Usage:
  ros2 launch piper replay_launch.py
  ros2 launch piper replay_launch.py publish_rate:=30

Wraps in start_replay_stack.sh for end users; that script also writes the
/tmp/kai0_deployment_mode marker. Direct ros2 launch usage works too — replay_node
will ALSO accept marker='replay' (start_replay_stack.sh) OR 'autonomy' (full stack).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    publish_rate_arg = DeclareLaunchArgument(
        "publish_rate", default_value="30",
        description="Rate for publishing /master/joint_* (Hz)",
    )

    # ── Piper left slave (mode=1: subs /master/joint_left → drives CAN) ──
    piper_left = Node(
        package="piper", executable="arm_reader_node.py",
        name="piper_left", output="screen",
        parameters=[{"can_port": "can_left_slave", "mode": 1, "auto_enable": True}],
        remappings=[
            ("/puppet/joint_states", "/puppet/joint_left"),
            ("/master/joint_states", "/master/joint_left"),
            ("/puppet/arm_status", "/puppet/arm_status_left"),
            ("/puppet/end_pose", "/puppet/end_pose_left"),
            ("/puppet/end_pose_euler", "/puppet/end_pose_euler_left"),
        ],
    )

    # ── Piper right slave (mode=1) ──
    piper_right = Node(
        package="piper", executable="arm_reader_node.py",
        name="piper_right", output="screen",
        parameters=[{"can_port": "can_right_slave", "mode": 1, "auto_enable": True}],
        remappings=[
            ("/puppet/joint_states", "/puppet/joint_right"),
            ("/master/joint_states", "/master/joint_right"),
            ("/puppet/arm_status", "/puppet/arm_status_right"),
            ("/puppet/end_pose", "/puppet/end_pose_right"),
            ("/puppet/end_pose_euler", "/puppet/end_pose_euler_right"),
        ],
    )

    # ── Replay node ──
    replay = Node(
        package="piper", executable="replay_node.py",
        name="replay", output="screen",
        parameters=[{
            "publish_rate": LaunchConfiguration("publish_rate"),
        }],
    )

    return LaunchDescription([publish_rate_arg, piper_left, piper_right, replay])
