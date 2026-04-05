"""ROS2 launch file for 4-arm teleoperation.

Launches left/right master (teach-handle) arms in mode=0 and left/right slave
(executor) arms in mode=1, using arm_teleop_node.py. Used for dual-arm teleop
data collection on the master/slave CAN topology (can_*_mas + can_*_slave).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare arguments
    mode_master_arg = DeclareLaunchArgument(
        'mode_master', default_value='true',
        description='Master arm initial mode (true=teach/drag, false=slave-follow)'
    )
    auto_enable_slave_arg = DeclareLaunchArgument(
        'auto_enable_slave', default_value='true',
        description='Auto-enable slave arms'
    )

    # --- Master arms: mode=0, publish control commands ---

    # Left master
    piper_master_left = Node(
        package='piper',
        executable='arm_teleop_node.py',
        name='piper_master_left',
        output='screen',
        parameters=[{
            'can_port': 'can_left_mas',
            'mode': 0,
            'auto_enable': False,
            'mode_master': LaunchConfiguration('mode_master'),
        }],
        remappings=[
            ('/puppet/arm_status', '/puppet_master/arm_status_left'),
            ('/puppet/joint_states', '/puppet_master/joint_left'),
            ('/master/joint_states', '/master/joint_left'),
            ('/puppet/end_pose', '/puppet_master/end_pose_left'),
            ('/pos_cmd', '/puppet_master/pos_cmd_left'),
            ('/puppet/end_pose_euler', '/puppet_master/end_pose_euler_left'),
            ('/master/arm_status', '/teach/master_control_status_left'),
            ('/master/mode_status', '/teach/master_mode_status_left'),
            ('/master/enable', '/teach/master_enable_left'),
            ('/master/linkage_config', '/teach/master_config_left'),
            ('/master/teach_mode', '/teach/teach_mode_left'),
            ('/master_controled/joint_states', '/master_controled/joint_left'),
        ],
    )

    # Right master
    piper_master_right = Node(
        package='piper',
        executable='arm_teleop_node.py',
        name='piper_master_right',
        output='screen',
        parameters=[{
            'can_port': 'can_right_mas',
            'mode': 0,
            'auto_enable': False,
            'mode_master': LaunchConfiguration('mode_master'),
        }],
        remappings=[
            ('/puppet/arm_status', '/puppet_master/arm_status_right'),
            ('/puppet/joint_states', '/puppet_master/joint_right'),
            ('/master/joint_states', '/master/joint_right'),
            ('/puppet/end_pose', '/puppet_master/end_pose_right'),
            ('/pos_cmd', '/puppet_master/pos_cmd_right'),
            ('/puppet/end_pose_euler', '/puppet_master/end_pose_euler_right'),
            ('/master/arm_status', '/teach/master_control_status_right'),
            ('/master/mode_status', '/teach/master_mode_status_right'),
            ('/master/enable', '/teach/master_enable_right'),
            ('/master/linkage_config', '/teach/master_config_right'),
            ('/master/teach_mode', '/teach/teach_mode_right'),
            ('/master_controled/joint_states', '/master_controled/joint_right'),
        ],
    )

    # --- Slave arms: mode=1, subscribe and execute control commands ---

    # Left slave
    piper_slave_left = Node(
        package='piper',
        executable='arm_teleop_node.py',
        name='piper_slave_left',
        output='screen',
        parameters=[{
            'can_port': 'can_left_slave',
            'mode': 1,
            'auto_enable': LaunchConfiguration('auto_enable_slave'),
        }],
        remappings=[
            ('/puppet/arm_status', '/puppet/arm_status_left'),
            ('/puppet/joint_states', '/puppet/joint_left'),
            ('/master/joint_states', '/master/joint_left'),
            ('/puppet/end_pose', '/puppet/end_pose_left'),
            ('/pos_cmd', '/puppet/pos_cmd_left'),
            ('/puppet/end_pose_euler', '/puppet/end_pose_euler_left'),
            ('/enable_flag', '/puppet/enable_left'),
        ],
    )

    # Right slave
    piper_slave_right = Node(
        package='piper',
        executable='arm_teleop_node.py',
        name='piper_slave_right',
        output='screen',
        parameters=[{
            'can_port': 'can_right_slave',
            'mode': 1,
            'auto_enable': LaunchConfiguration('auto_enable_slave'),
        }],
        remappings=[
            ('/puppet/arm_status', '/puppet/arm_status_right'),
            ('/puppet/joint_states', '/puppet/joint_right'),
            ('/master/joint_states', '/master/joint_right'),
            ('/puppet/end_pose', '/puppet/end_pose_right'),
            ('/pos_cmd', '/puppet/pos_cmd_right'),
            ('/puppet/end_pose_euler', '/puppet/end_pose_euler_right'),
            ('/enable_flag', '/puppet/enable_right'),
        ],
    )

    return LaunchDescription([
        mode_master_arg,
        auto_enable_slave_arg,
        piper_master_left,
        piper_master_right,
        piper_slave_left,
        piper_slave_right,
    ])
