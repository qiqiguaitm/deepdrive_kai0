#!/usr/bin/python3
"""
Single-process multi-camera RealSense node.

Manages all 3 RealSense cameras (D435 head + 2x D405 wrist) in one process
via pyrealsense2, publishing ROS2 Image topics. Avoids USB contention caused
by multiple realsense2_camera_node processes each doing device enumeration.

Publishes:
  /camera_f/camera/color/image_raw          (head color)
  /camera_f/camera/aligned_depth_to_color/image_raw  (head aligned depth)
  /camera_l/camera/color/image_raw          (left wrist color)
  /camera_l/camera/aligned_depth_to_color/image_raw  (left wrist aligned depth)
  /camera_r/camera/color/image_raw          (right wrist color)
  /camera_r/camera/aligned_depth_to_color/image_raw  (right wrist aligned depth)
"""

import os
import sys
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from builtin_interfaces.msg import Time

try:
    import pyrealsense2 as rs
except ImportError:
    # Try from venv
    def _setup_venv():
        import glob as _g
        for root in [os.path.expanduser('~/workspace/deepdive_kai0/kai0')]:
            vlib = os.path.join(root, '.venv', 'lib')
            pydirs = sorted(_g.glob(os.path.join(vlib, 'python3.*')))
            if pydirs:
                sp = os.path.join(pydirs[-1], 'site-packages')
                if sp not in sys.path:
                    sys.path.insert(0, sp)
    _setup_venv()
    import pyrealsense2 as rs


class MultiCameraNode(Node):
    def __init__(self):
        super().__init__('multi_camera')

        # Parameters
        self.declare_parameter('cam_f_serial', '')
        self.declare_parameter('cam_l_serial', '')
        self.declare_parameter('cam_r_serial', '')
        self.declare_parameter('fps', 15)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('enable_head_depth', True)
        self.declare_parameter('enable_wrist_depth', True)

        fps = self.get_parameter('fps').value
        w = self.get_parameter('width').value
        h = self.get_parameter('height').value
        enable_head_depth = self.get_parameter('enable_head_depth').value
        enable_wrist_depth = self.get_parameter('enable_wrist_depth').value

        cam_f_serial = self.get_parameter('cam_f_serial').value
        cam_l_serial = self.get_parameter('cam_l_serial').value
        cam_r_serial = self.get_parameter('cam_r_serial').value

        # QoS: use RELIABLE for all topics so both RELIABLE and BEST_EFFORT
        # subscribers can receive. BEST_EFFORT sub accepts RELIABLE pub.
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=5)
        depth_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=5)

        # Publishers
        self._pub_f_color = self.create_publisher(Image, '/camera_f/camera/color/image_raw', img_qos)
        self._pub_f_depth = self.create_publisher(Image, '/camera_f/camera/aligned_depth_to_color/image_raw', depth_qos)
        self._pub_l_color = self.create_publisher(Image, '/camera_l/camera/color/image_raw', img_qos)
        self._pub_l_depth = self.create_publisher(Image, '/camera_l/camera/aligned_depth_to_color/image_raw', depth_qos)
        self._pub_r_color = self.create_publisher(Image, '/camera_r/camera/color/image_raw', img_qos)
        self._pub_r_depth = self.create_publisher(Image, '/camera_r/camera/aligned_depth_to_color/image_raw', depth_qos)

        # Depth post-processing filters (one set per camera that has depth)
        def _make_depth_filters():
            spatial = rs.spatial_filter()
            spatial.set_option(rs.option.filter_magnitude, 2)
            spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
            spatial.set_option(rs.option.filter_smooth_delta, 20)
            temporal = rs.temporal_filter()
            temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
            temporal.set_option(rs.option.filter_smooth_delta, 20)
            return spatial, temporal
        self._depth_filters = {}  # role -> (spatial, temporal)

        # Open cameras sequentially in one process
        self._pipelines = {}  # serial -> (pipeline, align_or_None)
        self._serial_role = {}  # serial -> 'head' | 'left' | 'right'

        cameras = [
            ('head', cam_f_serial, True, enable_head_depth),
            ('left', cam_l_serial, False, enable_wrist_depth),
            ('right', cam_r_serial, False, enable_wrist_depth),
        ]

        # Start cameras sequentially with warm-up frames (matches verify_calibration.py approach).
        # Use bgr8 (D405 native format) and convert to RGB when publishing.
        # D435 uses rgb8 natively via its separate RGB camera module.
        for role, serial, is_d435, need_depth in cameras:
            if not serial:
                self.get_logger().warn(f'{role} camera serial not configured, skipping')
                continue
            color_fmt = rs.format.rgb8 if is_d435 else rs.format.bgr8
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    pipeline = rs.pipeline()
                    config = rs.config()
                    config.enable_device(serial)
                    config.enable_stream(rs.stream.color, w, h, color_fmt, fps)
                    if need_depth:
                        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
                    profile = pipeline.start(config)

                    # Warm up: grab frames to stabilize the USB stream
                    # (critical for D405 on shared USB hubs)
                    for _ in range(10):
                        pipeline.wait_for_frames(timeout_ms=2000)

                    align = rs.align(rs.stream.color) if need_depth else None
                    self._pipelines[serial] = (pipeline, align, not is_d435)  # store needs_bgr2rgb flag
                    self._serial_role[serial] = role
                    if need_depth:
                        self._depth_filters[role] = _make_depth_filters()

                    name = profile.get_device().get_info(rs.camera_info.name)
                    self.get_logger().info(f'{role} camera started: {name} ({serial})')
                    time.sleep(3.0)
                    break
                except Exception as e:
                    self.get_logger().warn(
                        f'{role} camera ({serial}) attempt {attempt+1}/{max_retries}: {e}')
                    time.sleep(5.0)
            else:
                self.get_logger().error(f'Failed to start {role} camera ({serial}) after {max_retries} attempts')

        if not self._pipelines:
            self.get_logger().error('No cameras available!')
            return

        # Map role -> serial for quick lookup
        self._role_serial = {v: k for k, v in self._serial_role.items()}

        # Timer for grabbing frames
        timer_period = 1.0 / fps
        self.create_timer(timer_period, self._grab_and_publish)

        self.get_logger().info(
            f'Multi-camera node ready: {len(self._pipelines)} cameras at {w}x{h}@{fps}fps')

    def _make_header(self, frame_ts, frame_id='camera'):
        """Create ROS2 header from RealSense frame timestamp."""
        header = Header()
        # RS timestamp is in ms
        ts_sec = frame_ts / 1000.0
        header.stamp = Time(sec=int(ts_sec), nanosec=int((ts_sec % 1) * 1e9))
        header.frame_id = frame_id
        return header

    def _numpy_to_image_msg(self, arr, header, encoding):
        """Convert numpy array to sensor_msgs/Image."""
        msg = Image()
        msg.header = header
        msg.height = arr.shape[0]
        msg.width = arr.shape[1]
        msg.encoding = encoding
        msg.is_bigendian = False
        if arr.ndim == 3:
            msg.step = arr.shape[1] * arr.shape[2] * arr.dtype.itemsize
        else:
            msg.step = arr.shape[1] * arr.dtype.itemsize
        msg.data = arr.tobytes()
        return msg

    def _grab_and_publish(self):
        """Grab frames from all cameras and publish."""
        for serial, (pipeline, align, needs_bgr2rgb) in self._pipelines.items():
            role = self._serial_role[serial]
            try:
                frames = pipeline.wait_for_frames(timeout_ms=100)
            except RuntimeError:
                continue

            ts = frames.get_timestamp()

            if align is not None:
                # Camera with depth: aligned color + depth
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
            else:
                color_frame = frames.get_color_frame()
                depth_frame = None

            if color_frame:
                header = self._make_header(ts, 'camera_color_optical_frame')
                color = np.asanyarray(color_frame.get_data())
                if needs_bgr2rgb:
                    color = color[:, :, ::-1].copy()
                pub_color = {'head': self._pub_f_color,
                             'left': self._pub_l_color,
                             'right': self._pub_r_color}.get(role)
                if pub_color:
                    pub_color.publish(self._numpy_to_image_msg(color, header, 'rgb8'))

            if depth_frame and role in self._depth_filters:
                spatial, temporal = self._depth_filters[role]
                depth_frame = spatial.process(depth_frame)
                depth_frame = temporal.process(depth_frame)
                header = self._make_header(ts, 'camera_color_optical_frame')
                depth = np.asanyarray(depth_frame.get_data())
                pub_depth = {'head': self._pub_f_depth,
                             'left': self._pub_l_depth,
                             'right': self._pub_r_depth}.get(role)
                if pub_depth:
                    pub_depth.publish(self._numpy_to_image_msg(depth, header, '16UC1'))

    def destroy_node(self):
        for serial, (pipeline, _, _) in self._pipelines.items():
            try:
                pipeline.stop()
                self.get_logger().info(f'Stopped camera {serial}')
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MultiCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
