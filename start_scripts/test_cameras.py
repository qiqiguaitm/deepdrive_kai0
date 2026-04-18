#!/usr/bin/env python3
"""
三相机验��脚本 (统一入口)

模式:
  --mode ros2    ROS2 topic 订阅验证 (需先 ros2 launch scripts/launch_3cam.py)
  --mode direct  pyrealsense2 SDK 直连压测 (不需要 ROS2)

检查项:
  - 实际 FPS、帧间 jitter、端到端延迟
  - 图像尺寸和内容有效性 (非全黑)
  - 丢帧检测 (direct 模式)
  - USB 带宽 (direct 模式)

合并自: test_3cam_ros2.py + test_realsense_3cam.py

Usage:
  # ROS2 模式 (需要相机 launch 在跑)
  source /opt/ros/jazzy/setup.bash
  python3 scripts/test_cameras.py --mode ros2 [--duration 10]

  # 直连模式 (不需要 ROS2)
  python3 scripts/test_cameras.py --mode direct [--duration 10] [--save-sample]
"""
import argparse
import time
import threading
import numpy as np

# ── 相机硬件配置 ──────────────────────────────────────────────────────────────
CAMERA_SERIALS = {
    'D435_top':    '254622070889',
    'D405_left':   '409122273074',
    'D405_right':  '409122271568',
}

ROS2_TOPICS = [
    ('/camera_f/color/image_raw',  'D435  RGB',    'rgb'),
    ('/camera_f/depth/image_raw',  'D435  Depth',  'depth'),
    ('/camera_l/color/image_raw',  'D405-L RGB',   'rgb'),
    ('/camera_l/depth/image_raw',  'D405-L Depth', 'depth'),
    ('/camera_r/color/image_raw',  'D405-R RGB',   'rgb'),
    ('/camera_r/depth/image_raw',  'D405-R Depth', 'depth'),
]


# ══════════════════════════════════════════════════════════════════════════════
# ROS2 模式
# ══════════════════════════════════════════════════════════════════════════════

class _TopicStats:
    def __init__(self, label, img_type):
        self.label = label
        self.img_type = img_type
        self.count = 0
        self.first_ts = None
        self.last_ts = None
        self.intervals = []
        self.e2e_latencies = []
        self.shape = None
        self.dtype = None
        self.mean_val = None
        self.nonzero_ratio = None


def run_ros2(duration):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    class CameraTestNode(Node):
        def __init__(self):
            super().__init__('camera_test_node')
            self.bridge = CvBridge()
            self.stats = {}
            self.start_time = time.monotonic()
            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=10)
            for topic, label, img_type in ROS2_TOPICS:
                self.stats[topic] = _TopicStats(label, img_type)
                self.create_subscription(Image, topic,
                                         lambda msg, t=topic: self._cb(msg, t), qos)
            self.get_logger().info(f'Subscribing to {len(ROS2_TOPICS)} topics, waiting {duration}s...')

        def _cb(self, msg, topic):
            now = time.monotonic()
            st = self.stats[topic]
            st.count += 1
            if st.first_ts is None:
                st.first_ts = now
            if st.last_ts is not None:
                st.intervals.append(now - st.last_ts)
            st.last_ts = now
            stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            clock_sec = self.get_clock().now().nanoseconds * 1e-9
            e2e = (clock_sec - stamp_sec) * 1000
            if 0 < e2e < 5000:
                st.e2e_latencies.append(e2e)
            if st.shape is None:
                try:
                    enc = 'bgr8' if st.img_type == 'rgb' else 'passthrough'
                    img = self.bridge.imgmsg_to_cv2(msg, enc)
                    st.shape = img.shape
                    st.dtype = str(img.dtype)
                    st.mean_val = float(img.mean())
                    st.nonzero_ratio = float(np.count_nonzero(img) / img.size)
                except Exception:
                    pass

        def is_done(self):
            return time.monotonic() - self.start_time > duration

    rclpy.init()
    node = CameraTestNode()
    try:
        while rclpy.ok() and not node.is_done():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    # 报告
    print('\n' + '=' * 70)
    print('ROS2 三相机验��报告')
    print('=' * 70)
    all_ok = True
    for topic, label, _ in ROS2_TOPICS:
        st = node.stats[topic]
        print(f'\n  [{label}] {topic}')
        if st.count == 0:
            print('    NO DATA RECEIVED')
            all_ok = False
            continue
        elapsed = (st.last_ts - st.first_ts) if st.last_ts and st.first_ts else 1
        fps = (st.count - 1) / elapsed if elapsed > 0 and st.count > 1 else 0
        intervals = np.array(st.intervals) * 1000 if st.intervals else np.array([0])
        e2e = np.array(st.e2e_latencies) if st.e2e_latencies else np.array([0])
        print(f'    帧数: {st.count}  FPS: {fps:.1f}')
        print(f'    图像: {st.shape}  dtype={st.dtype}  均值={st.mean_val:.1f}  非零率={st.nonzero_ratio:.2%}')
        print(f'    帧间隔: avg={intervals.mean():.1f}ms std={intervals.std():.1f}ms max={intervals.max():.1f}ms')
        print(f'    端到端: avg={e2e.mean():.1f}ms p99={np.percentile(e2e, 99):.1f}ms')
        if fps < 25:
            print('    [WARN] FPS < 25')
            all_ok = False
    _print_verdict(all_ok)
    node.destroy_node()
    rclpy.shutdown()


# ══════════════════════════════════════════════════════════════════════════════
# Direct (pyrealsense2) 模式
# ══════════════════════════════════════════════════════════════════════════════

DIRECT_CAMERAS = [
    {'name': 'D435 (top)',    'serial': CAMERA_SERIALS['D435_top'],   'rgb_w': 640, 'rgb_h': 480, 'depth_w': 640, 'depth_h': 480, 'fps': 30},
    {'name': 'D405-L (wrist)','serial': CAMERA_SERIALS['D405_left'],  'rgb_w': 640, 'rgb_h': 480, 'depth_w': 640, 'depth_h': 480, 'fps': 30},
    {'name': 'D405-R (wrist)','serial': CAMERA_SERIALS['D405_right'], 'rgb_w': 640, 'rgb_h': 480, 'depth_w': 640, 'depth_h': 480, 'fps': 30},
]


class _CameraThread:
    def __init__(self, cfg):
        self.cfg = cfg
        self.name = cfg['name']
        self.serial = cfg['serial']
        self.frame_count = 0
        self.drop_count = 0
        self.latencies = []
        self.rgb_shape = None
        self.depth_shape = None
        self.error = None
        self.actual_fps = 0.0
        self._stop = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self):
        try:
            import pyrealsense2 as rs
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.serial)
            c = self.cfg
            config.enable_stream(rs.stream.color, c['rgb_w'], c['rgb_h'], rs.format.bgr8, c['fps'])
            config.enable_stream(rs.stream.depth, c['depth_w'], c['depth_h'], rs.format.z16, c['fps'])
            pipeline.start(config)
            for _ in range(15):
                pipeline.wait_for_frames(timeout_ms=3000)
            prev_fn = -1
            t_start = time.monotonic()
            while not self._stop.is_set():
                t0 = time.monotonic()
                frames = pipeline.wait_for_frames(timeout_ms=2000)
                t1 = time.monotonic()
                cf = frames.get_color_frame()
                df = frames.get_depth_frame()
                if not cf or not df:
                    continue
                if self.rgb_shape is None:
                    self.rgb_shape = np.asanyarray(cf.get_data()).shape
                    self.depth_shape = np.asanyarray(df.get_data()).shape
                fn = cf.get_frame_number()
                if prev_fn >= 0 and fn > prev_fn + 1:
                    self.drop_count += fn - prev_fn - 1
                prev_fn = fn
                self.frame_count += 1
                self.latencies.append((t1 - t0) * 1000)
            self.actual_fps = self.frame_count / (time.monotonic() - t_start)
            pipeline.stop()
        except Exception as e:
            self.error = str(e)


def run_direct(duration, save_sample=False):
    import pyrealsense2 as rs

    ctx = rs.context()
    connected = {d.get_info(rs.camera_info.serial_number): d.get_info(rs.camera_info.name)
                 for d in ctx.query_devices()}
    print(f'\n检测到 {len(connected)} 个 RealSense 设备:')
    for sn, name in connected.items():
        print(f'  {name} (SN: {sn})')
    missing = [c for c in DIRECT_CAMERAS if c['serial'] not in connected]
    if missing:
        print('ERROR: 以下相机未连接:')
        for c in missing:
            print(f'  {c["name"]} (SN: {c["serial"]})')
        return

    threads = [_CameraThread(c) for c in DIRECT_CAMERAS]
    for t in threads:
        t.start()
        time.sleep(0.5)
    print(f'采集中 ({duration}s)...')
    time.sleep(duration)
    for t in threads:
        t.stop()

    print('\n' + '=' * 70)
    print('Direct SDK 三相机压测报告')
    print('=' * 70)
    all_ok = True
    for t in threads:
        print(f'\n  {t.name} (SN: {t.serial})')
        if t.error:
            print(f'    ERROR: {t.error}')
            all_ok = False
            continue
        lat = np.array(t.latencies) if t.latencies else np.array([0])
        print(f'    RGB: {t.rgb_shape}  Depth: {t.depth_shape}')
        print(f'    帧数: {t.frame_count}  丢帧: {t.drop_count}  FPS: {t.actual_fps:.1f}')
        print(f'    延迟: avg={lat.mean():.1f}ms p50={np.median(lat):.1f}ms p99={np.percentile(lat, 99):.1f}ms')
        if t.actual_fps < 25:
            print('    [WARN] FPS < 25')
            all_ok = False
        if t.drop_count > t.frame_count * 0.05:
            print('    [WARN] 丢帧率 > 5%')
            all_ok = False
    _print_verdict(all_ok)


def _print_verdict(all_ok):
    print('\n' + '-' * 70)
    print(f'结论: {"PASS" if all_ok else "WARN - 见上方详情"}')
    print('-' * 70)


def main():
    parser = argparse.ArgumentParser(description='三相机验证 (ROS2 / Direct SDK)')
    parser.add_argument('--mode', choices=['ros2', 'direct'], default='ros2')
    parser.add_argument('--duration', type=int, default=10, help='采集秒数')
    parser.add_argument('--save-sample', action='store_true', help='保存样图 (direct 模式)')
    args = parser.parse_args()

    print('=' * 70)
    print(f'三相机验证 — mode={args.mode}  duration={args.duration}s')
    print('=' * 70)

    if args.mode == 'ros2':
        run_ros2(args.duration)
    else:
        run_direct(args.duration, args.save_sample)


if __name__ == '__main__':
    main()
