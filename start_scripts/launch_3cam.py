"""
Launch 3 RealSense cameras via ROS2 realsense2_camera nodes.

  D435 (top)    → namespace: camera_f  | RGB 640x480   + Depth 640x480  @ 15fps
  D405-A (left) → namespace: camera_l  | RGB 640x480   + Depth 640x480  @ 15fps
  D405-B (right)→ namespace: camera_r  | RGB 640x480   + Depth 640x480  @ 15fps

  Note: 15fps (not 30) 以缓解 3 相机共享 USB 3 hub 的带宽压力;
  之前 30fps 观察到 hand_left 触发 "Incomplete video frame" 频繁丢帧, 实际只有 1-10Hz.

Usage:
  ros2 launch scripts/launch_3cam.py
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node

_DEFAULT_FPS = int(os.environ.get('CAM_FPS', '15'))


def make_camera_node(name, namespace, serial,
                     rgb_w, rgb_h, depth_w, depth_h, fps=_DEFAULT_FPS):
    return Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name=name,
        namespace=namespace,
        output='screen',
        parameters=[{
            'serial_no': serial,
            'camera_name': name,
            'enable_color': True,
            'enable_depth': True,
            'enable_infra1': False,
            'enable_infra2': False,
            'enable_gyro': False,
            'enable_accel': False,
            # D435 has dedicated RGB module → 用 rgb_camera.color_profile
            # D405 把 color 共享在 stereo (depth) 模块 → 用 depth_module.color_profile
            # 同时设两个: 不适用的会被驱动忽略 (我们之前只设 rgb_camera.color_profile,
            # 结果 D405 的 color 没人管, 默认跑成 848x480x30)
            'rgb_camera.color_profile': f'{rgb_w}x{rgb_h}x{fps}',
            'depth_module.color_profile': f'{rgb_w}x{rgb_h}x{fps}',
            'depth_module.depth_profile': f'{depth_w}x{depth_h}x{fps}',
            'align_depth.enable': False,
        }],
    )


def generate_launch_description():
    cam_f = make_camera_node(
        name='camera_f', namespace='',
        serial='254622070889',
        rgb_w=640, rgb_h=480, depth_w=640, depth_h=480,
    )
    cam_l = make_camera_node(
        name='camera_l', namespace='',
        serial='409122273074',
        rgb_w=640, rgb_h=480, depth_w=640, depth_h=480,
    )
    cam_r = make_camera_node(
        name='camera_r', namespace='',
        serial='409122271568',
        rgb_w=640, rgb_h=480, depth_w=640, depth_h=480,
    )
    return LaunchDescription([cam_f, cam_l, cam_r])
