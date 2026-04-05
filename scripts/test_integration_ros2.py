#!/usr/bin/env python3
"""
kai0 ROS2 推理模块端到端测试

测试流程:
  1. 启动可用的传感器 ROS2 节点 (2x D405 + 1x Piper on can2)
  2. 对缺失的传感器 (D435 + 左臂) 发布模拟数据
  3. 启动 policy_inference_node (WebSocket 模式, 连接已运行的 serve_policy)
  4. 监听 /policy/actions 输出, 验证推理结果

前置条件:
  - serve_policy.py 已在 :8000 运行
  - ROS2 Jazzy 已安装
  - ros2_ws 已编译

用法:
  source /opt/ros/jazzy/setup.bash
  source /data1/tim/workspace/deepdive_kai0/ros2_ws/install/setup.bash
  python3.12 test_ros2_inference.py
"""

import os
import sys
import time
import subprocess
import signal
import threading
import json

# 颜色
G = "\033[0;32m"; R = "\033[0;31m"; Y = "\033[1;33m"; C = "\033[0;36m"; N = "\033[0m"
def ok(m):   print(f"{G}[PASS]{N} {m}")
def fail(m): print(f"{R}[FAIL]{N} {m}")
def warn(m): print(f"{Y}[WARN]{N} {m}")
def info(m): print(f"{C}[INFO]{N} {m}")

KAI0 = "/data1/tim/workspace/deepdive_kai0/kai0"
ROS2_WS = "/data1/tim/workspace/deepdive_kai0/ros2_ws"
VENV_PYTHON = f"{KAI0}/.venv/bin/python"

processes = []

def cleanup():
    for p in processes:
        try:
            p.terminate()
            p.wait(timeout=3)
        except:
            try: p.kill()
            except: pass
    # kill any remaining nodes we started
    subprocess.run("pkill -f 'test_ros2_fake_pub|policy_inference_node' 2>/dev/null", shell=True)

signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(0)))


def wait_for_topic(topic, timeout=15):
    """Wait until a ROS2 topic has at least one publisher."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        result = subprocess.run(
            ["ros2", "topic", "info", topic],
            capture_output=True, text=True, timeout=5
        )
        if "Publisher count: 0" not in result.stdout and "Publisher count:" in result.stdout:
            return True
        time.sleep(1)
    return False


def main():
    print(f"\n{'='*60}")
    print(f"  kai0 ROS2 推理模块端到端测试")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # ── 0. 前置检查 ──
    info("前置检查...")

    # serve_policy 是否运行
    result = subprocess.run("ss -tlnp 2>/dev/null | grep ':8000 '", shell=True, capture_output=True, text=True)
    if ":8000" not in result.stdout:
        fail("serve_policy.py 未在 :8000 运行, 请先启动")
        return
    ok("serve_policy.py 在 :8000 运行")

    # ROS2 环境
    if subprocess.run(["which", "ros2"], capture_output=True).returncode != 0:
        fail("ros2 命令不可用, 请先 source /opt/ros/jazzy/setup.bash")
        return
    ok("ROS2 环境可用")

    # ── 1. 启动 Piper 右臂节点 (can_right_slave) ──
    print(f"\n{'─'*60}")
    info("Step 1: 启动 Piper 右臂节点 (can_right_slave, mode=0 只读)...")

    # 检查 can_right_slave
    can_check = subprocess.run(["ip", "link", "show", "can_right_slave"], capture_output=True, text=True)
    if "UP" in can_check.stdout:
        p_arm = subprocess.Popen(
            ["ros2", "run", "piper", "arm_reader_node.py",
             "--ros-args",
             "-p", "can_port:=can_right_slave", "-p", "mode:=0",
             "-r", "/puppet/joint_states:=/puppet/joint_right",
             "-r", "/puppet/arm_status:=/puppet/arm_status_right",
             "-r", "/puppet/end_pose:=/puppet/end_pose_right",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        processes.append(p_arm)
        time.sleep(2)

        if wait_for_topic("/puppet/joint_right", timeout=10):
            ok("右臂节点启动, /puppet/joint_right 有数据")
        else:
            warn("右臂节点启动但 topic 未就绪")
    else:
        warn("can_right_slave 未 UP, 跳过右臂节点")

    # ── 2. 启动相机节点 ──
    print(f"\n{'─'*60}")
    info("Step 2: 启动 RealSense 相机节点 (D405-L, D405-R)...")

    cameras = [
        ("camera_l", "409122273074", "/camera_l/camera/color/image_raw"),
        ("camera_r", "409122271568", "/camera_r/camera/color/image_raw"),
    ]

    for cam_name, serial, topic in cameras:
        p_cam = subprocess.Popen(
            ["ros2", "run", "realsense2_camera", "realsense2_camera_node",
             "--ros-args",
             "-p", f"serial_no:={serial}",
             "-p", f"camera_name:={cam_name}",
             "-p", "enable_color:=true",
             "-p", "enable_depth:=false",
             "-p", "enable_infra1:=false", "-p", "enable_infra2:=false",
             "-p", "enable_gyro:=false", "-p", "enable_accel:=false",
             "-p", "rgb_camera.color_profile:=640x480x30",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        processes.append(p_cam)
        info(f"  {cam_name} (SN:{serial}) 启动中...")
        time.sleep(4)  # 相机间隔启动

    # 验证相机 topics
    for cam_name, _, topic in cameras:
        if wait_for_topic(topic, timeout=15):
            ok(f"  {cam_name}: {topic} 有数据")
        else:
            warn(f"  {cam_name}: {topic} 未就绪")

    # ── 3. 发布缺失传感器的模拟数据 ──
    print(f"\n{'─'*60}")
    info("Step 3: 发布模拟数据 (D435 头顶相机 + 左臂关节)...")

    # 用 Python 脚本在后台发布模拟数据
    fake_pub_code = '''
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header
import numpy as np
import time

class FakeSensorPub(Node):
    def __init__(self):
        super().__init__("test_ros2_fake_pub")
        self.pub_img = self.create_publisher(Image, "/camera_f/camera/color/image_raw", 10)
        self.pub_joint = self.create_publisher(JointState, "/puppet/joint_left", 10)
        self.timer = self.create_timer(1.0/30.0, self.publish)
        self.get_logger().info("Fake publisher started (D435 + left arm)")

    def publish(self):
        now = self.get_clock().now().to_msg()
        # 模拟 D435 图像 (640x480 RGB)
        img_msg = Image()
        img_msg.header.stamp = now
        img_msg.header.frame_id = "camera_f_color_optical_frame"
        img_msg.height = 480
        img_msg.width = 640
        img_msg.encoding = "rgb8"
        img_msg.step = 640 * 3
        img_msg.data = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8).tobytes()
        self.pub_img.publish(img_msg)

        # 模拟左臂关节 (7 DOF)
        js = JointState()
        js.header.stamp = now
        js.name = ["joint0","joint1","joint2","joint3","joint4","joint5","joint6"]
        js.position = [0.1, -0.5, 0.3, 0.8, 1.2, -0.4, 0.0]
        js.velocity = [0.0] * 7
        js.effort = [0.0] * 7
        self.pub_joint.publish(js)

rclpy.init()
node = FakeSensorPub()
try:
    rclpy.spin(node)
except:
    pass
finally:
    node.destroy_node()
    rclpy.shutdown()
'''

    p_fake = subprocess.Popen(
        ["python3.12", "-c", fake_pub_code],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    processes.append(p_fake)
    time.sleep(2)

    if wait_for_topic("/camera_f/camera/color/image_raw", timeout=5):
        ok("模拟 D435 图像: /camera_f/camera/color/image_raw 发布中")
    else:
        warn("模拟 D435 topic 未就绪")

    if wait_for_topic("/puppet/joint_left", timeout=5):
        ok("模拟左臂关节: /puppet/joint_left 发布中")
    else:
        warn("模拟左臂 topic 未就绪")

    # ── 4. 列出所有活跃 topic ──
    print(f"\n{'─'*60}")
    info("Step 4: 当前活跃 topics...")
    result = subprocess.run(["ros2", "topic", "list"], capture_output=True, text=True, timeout=10)
    relevant = [t for t in result.stdout.strip().split("\n")
                if any(k in t for k in ["camera", "puppet", "master", "policy"])]
    for t in relevant:
        info(f"  {t}")

    # ── 5. 启动 policy_inference_node (WebSocket 模式) ──
    print(f"\n{'─'*60}")
    info("Step 5: 启动 policy_inference_node (WebSocket 模式)...")

    # 设置环境
    env = os.environ.copy()
    venv_site = f"{KAI0}/.venv/lib/python3.12/site-packages"
    import glob
    nvidia_libs = ":".join(sorted(glob.glob(os.path.join(venv_site, "nvidia", "*", "lib"))))
    env["LD_LIBRARY_PATH"] = nvidia_libs + ":" + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = venv_site + ":" + f"{KAI0}/src:" + env.get("PYTHONPATH", "")

    p_policy = subprocess.Popen(
        ["ros2", "run", "piper", "policy_inference_node.py",
         "--ros-args",
         "-p", "mode:=websocket",
         "-p", "host:=localhost",
         "-p", "port:=8000",
         "-p", "img_front_topic:=/camera_f/camera/color/image_raw",
         "-p", "img_left_topic:=/camera_l/camera/color/image_raw",
         "-p", "img_right_topic:=/camera_r/camera/color/image_raw",
         "-p", "puppet_left_topic:=/puppet/joint_left",
         "-p", "puppet_right_topic:=/puppet/joint_right",
         "-p", "publish_rate:=30",
         "-p", "inference_rate:=3.0",
         "-p", "chunk_size:=50",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
    )
    processes.append(p_policy)

    # 等待节点启动并输出
    info("等待 policy 节点连接 WebSocket 并开始推理...")
    info("(首次推理需要等待传感器同步, 约 5-10s)")

    # ── 6. 监听 /policy/actions 输出 ──
    print(f"\n{'─'*60}")
    info("Step 6: 监听 /policy/actions (等待推理输出, 最长 60s)...")

    action_received = False
    action_data = None

    def listen_actions():
        nonlocal action_received, action_data
        try:
            result = subprocess.run(
                ["ros2", "topic", "echo", "/policy/actions", "--once"],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0 and "position:" in result.stdout:
                action_received = True
                action_data = result.stdout
        except subprocess.TimeoutExpired:
            pass

    listener = threading.Thread(target=listen_actions, daemon=True)
    listener.start()

    # 同时监控 policy node 输出
    t0 = time.time()
    policy_lines = []
    while time.time() - t0 < 65:
        if p_policy.poll() is not None:
            fail("policy_inference_node 异常退出!")
            remaining = p_policy.stdout.read().decode(errors="replace")
            for line in remaining.split("\n")[-20:]:
                if line.strip():
                    info(f"  {line.strip()}")
            break

        # 非阻塞读 stdout
        import select
        if select.select([p_policy.stdout], [], [], 0.5)[0]:
            line = p_policy.stdout.readline().decode(errors="replace").strip()
            if line:
                policy_lines.append(line)
                # 打印关键日志
                if any(k in line.lower() for k in ["error", "fail", "connected", "ready", "warmup", "inference", "loaded", "waiting"]):
                    info(f"  [node] {line}")

        if action_received:
            break

    listener.join(timeout=2)

    # ── 7. 结果验证 ──
    print(f"\n{'─'*60}")
    info("Step 7: 结果验证")
    print()

    if action_received and action_data:
        ok("推理输出已收到 (/policy/actions)")
        # 解析 position 数据
        lines = action_data.split("\n")
        pos_lines = [l for l in lines if l.strip().startswith("- ")]
        if pos_lines:
            values = [float(l.strip().strip("- ")) for l in pos_lines[:14]]
            info(f"  action 维度: {len(pos_lines)} (期望 14)")
            info(f"  left arm (7):  {[f'{v:.4f}' for v in values[:7]]}")
            info(f"  right arm (7): {[f'{v:.4f}' for v in values[7:14]]}")
            if len(pos_lines) == 14:
                ok("Action 维度正确 (14 = 7 left + 7 right)")
            else:
                warn(f"Action 维度 {len(pos_lines)} ≠ 14")
        else:
            info("  (无法解析 position 数据)")
            info(f"  原始输出前 10 行:")
            for l in lines[:10]:
                info(f"    {l}")
    else:
        fail("未收到推理输出 (60s 超时)")
        info("可能原因:")
        info("  1. 传感器数据未同步 (缺少某个 topic)")
        info("  2. WebSocket 连接失败")
        info("  3. 模型推理异常")
        print()
        info("policy node 最后输出:")
        for line in policy_lines[-15:]:
            info(f"  {line}")

    # ── 8. 附加: topic 帧率检查 ──
    print(f"\n{'─'*60}")
    info("Step 8: Topic 帧率检查 (2 秒采样)...")

    topics_to_check = [
        "/camera_f/camera/color/image_raw",
        "/camera_l/camera/color/image_raw",
        "/camera_r/camera/color/image_raw",
        "/puppet/joint_left",
        "/puppet/joint_right",
        "/policy/actions",
    ]
    for topic in topics_to_check:
        try:
            result = subprocess.run(
                ["ros2", "topic", "hz", topic],
                capture_output=True, text=True, timeout=4
            )
            out = result.stdout + result.stderr
            hz_lines = [l for l in out.split("\n") if "average rate" in l]
            if hz_lines:
                info(f"  {topic}: {hz_lines[-1].strip()}")
            else:
                warn(f"  {topic}: 无数据或帧率太低")
        except subprocess.TimeoutExpired:
            warn(f"  {topic}: 测量超时")

    # ── 清理 ──
    print(f"\n{'─'*60}")
    info("清理进程...")
    cleanup()
    ok("测试完成")

    print(f"\n{'='*60}")
    print(f"  测试总结")
    print(f"{'='*60}")
    if action_received:
        ok("ROS2 推理模块端到端: PASS")
        info("  serve_policy (:8000) → WebSocket → policy_inference_node → /policy/actions")
        info("  传感器: 2x D405 (真实) + 1x D435 (模拟) + 1x 右臂 (真实) + 1x 左臂 (模拟)")
    else:
        fail("ROS2 推理模块端到端: FAIL (未收到推理输出)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        import traceback
        fail(f"异常: {e}")
        traceback.print_exc()
        cleanup()
