"""ROS2 launch — replay stack (NO autonomy / NO cameras / NO JAX).

Modes:
  - real arms (default): replay_node + 2× arm_reader_node mode=1 → drives CAN.
  - sim         : replay_node only; /master/joint_* still publishes but no
                  arm_reader subscribes, so arms don't move physically. Useful
                  for kinematic replay / rerun visualization. Toggle with
                  `enable_real_arms:=false`.

Usage:
  ros2 launch piper replay_launch.py                            # real arms
  ros2 launch piper replay_launch.py enable_real_arms:=false    # sim only
  ros2 launch piper replay_launch.py publish_rate:=30

Both modes are wrapped by start_autonomy.sh (`--replay [--sim]`); the wrapper
writes the /tmp/kai0_deployment_mode marker the backend's preflight gate checks.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    publish_rate_arg = DeclareLaunchArgument(
        "publish_rate", default_value="30",
        description="Rate for publishing /master/joint_* (Hz)",
    )
    enable_real_arms_arg = DeclareLaunchArgument(
        "enable_real_arms", default_value="true",
        description="false skips arm_reader (sim mode: replay publishes joints "
                    "but no CAN drive). Default true for real-arm replay.",
    )
    enable_real_arms = LaunchConfiguration("enable_real_arms")

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
        condition=IfCondition(enable_real_arms),
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
        condition=IfCondition(enable_real_arms),
    )

    # ── Replay node ──
    replay = Node(
        package="piper", executable="replay_node.py",
        name="replay", output="screen",
        parameters=[{
            "publish_rate": LaunchConfiguration("publish_rate"),
        }],
    )

    return LaunchDescription([
        publish_rate_arg, enable_real_arms_arg,
        piper_left, piper_right, replay,
    ])
