"""
端到端推理测试 Launch — Piper (mode=0 只读) + 3 相机

piper 以 mode=0 启动 (arm_reader_node 被动读取模式): 只发布关节状态，不接受
控制命令 → 安全，不会控制臂运动。推理脚本发布的动作命令无接收者 → 不产生任何运动。

Topic 映射:
  /puppet/joint_left   ← piper 左臂关节反馈 (can_left_slave)
  /puppet/joint_right  ← piper 右臂关节反馈 (can_right_slave)
  /camera_f/color/image_raw  ← D435 RGB
  /camera_l/color/image_raw  ← D405-L RGB
  /camera_r/color/image_raw  ← D405-R RGB

Usage:
  ros2 launch scripts/launch_e2e_test.py
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # ── Piper 左臂 (can_left_slave, mode=0 只读) ──
    piper_left = Node(
        package='piper',
        executable='arm_reader_node.py',
        name='piper_left',
        output='screen',
        parameters=[{
            'can_port': 'can_left_slave',
            'mode': 0,          # 只读: 发布关节状态，不接受控制
            'auto_enable': False,
        }],
        remappings=[
            ('/puppet/joint_states', '/puppet/joint_left'),
            ('/master/joint_states', '/master/joint_left'),
            ('/puppet/arm_status', '/puppet/arm_status_left'),
            ('/puppet/end_pose', '/puppet/end_pose_left'),
            ('/puppet/end_pose_euler', '/puppet/end_pose_euler_left'),
        ],
    )

    # ── Piper 右臂 (can_right_slave, mode=0 只读) ──
    piper_right = Node(
        package='piper',
        executable='arm_reader_node.py',
        name='piper_right',
        output='screen',
        parameters=[{
            'can_port': 'can_right_slave',
            'mode': 0,
            'auto_enable': False,
        }],
        remappings=[
            ('/puppet/joint_states', '/puppet/joint_right'),
            ('/master/joint_states', '/master/joint_right'),
            ('/puppet/arm_status', '/puppet/arm_status_right'),
            ('/puppet/end_pose', '/puppet/end_pose_right'),
            ('/puppet/end_pose_euler', '/puppet/end_pose_euler_right'),
        ],
    )

    # ── D435 头顶相机 (namespace=camera_f, 但 topic 重映射到 /camera_f/*) ──
    cam_f = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='camera_f',
        namespace='',           # 不加额外 namespace
        output='screen',
        parameters=[{
            'serial_no': '254622070889',
            'camera_name': 'camera_f',
            'enable_color': True,
            'enable_depth': True,
            'enable_infra1': False,
            'enable_infra2': False,
            'enable_gyro': False,
            'enable_accel': False,
            'rgb_camera.color_profile': '640x480x30',
            'depth_module.depth_profile': '640x480x30',
            'align_depth.enable': False,
            'initial_reset': False,
        }],
        remappings=[
            # 重映射到推理脚本期望的 topic 名
            ('color/image_raw', '/camera_f/color/image_raw'),
            ('color/camera_info', '/camera_f/color/camera_info'),
            ('depth/image_rect_raw', '/camera_f/depth/image_raw'),
            ('depth/camera_info', '/camera_f/depth/camera_info'),
        ],
    )

    # ── D405-L 左腕相机 ──
    cam_l = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='camera_l',
        namespace='',
        output='screen',
        parameters=[{
            'serial_no': '409122273074',
            'camera_name': 'camera_l',
            'enable_color': True,
            'enable_depth': True,
            'enable_infra1': False,
            'enable_infra2': False,
            'enable_gyro': False,
            'enable_accel': False,
            'rgb_camera.color_profile': '640x480x30',
            'depth_module.depth_profile': '640x480x30',
            'align_depth.enable': False,
            'initial_reset': False,
        }],
        remappings=[
            ('color/image_raw', '/camera_l/color/image_raw'),
            ('color/image_rect_raw', '/camera_l/color/image_raw'),
            ('color/camera_info', '/camera_l/color/camera_info'),
            ('depth/image_rect_raw', '/camera_l/depth/image_raw'),
            ('depth/camera_info', '/camera_l/depth/camera_info'),
        ],
    )

    # ── D405-R 右腕相机 ──
    cam_r = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='camera_r',
        namespace='',
        output='screen',
        parameters=[{
            'serial_no': '409122271568',
            'camera_name': 'camera_r',
            'enable_color': True,
            'enable_depth': True,
            'enable_infra1': False,
            'enable_infra2': False,
            'enable_gyro': False,
            'enable_accel': False,
            'rgb_camera.color_profile': '640x480x30',
            'depth_module.depth_profile': '640x480x30',
            'align_depth.enable': False,
            'initial_reset': False,
        }],
        remappings=[
            ('color/image_raw', '/camera_r/color/image_raw'),
            ('color/image_rect_raw', '/camera_r/color/image_raw'),
            ('color/camera_info', '/camera_r/color/camera_info'),
            ('depth/image_rect_raw', '/camera_r/depth/image_raw'),
            ('depth/camera_info', '/camera_r/depth/camera_info'),
        ],
    )

    return LaunchDescription([piper_left, piper_right, cam_f, cam_l, cam_r])
