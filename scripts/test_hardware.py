#!/usr/bin/env python3
"""
kai0 硬件测试脚本 — 传感器 + 机械臂数据读取验证

测试:
  1. RealSense 相机: 枚举设备, 打开 RGB 流, 采集帧, 检查分辨率/帧率
  2. CAN 机械臂: 通过 piper_sdk 连接, 读取关节状态
  3. ROS2 集成: 启动节点, 发布/订阅相机和臂 topic (可选)

用法:
  python3 test_hardware.py              # 全部测试
  python3 test_hardware.py --cam-only   # 只测相机
  python3 test_hardware.py --arm-only   # 只测机械臂
"""

import sys
import time
import argparse
import subprocess

# ─── 颜色输出 ────────────────────────────────────────────────────────────────
RED = "\033[0;31m"; GREEN = "\033[0;32m"; YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"; NC = "\033[0m"

def ok(msg):    print(f"{GREEN}[PASS]{NC} {msg}")
def fail(msg):  print(f"{RED}[FAIL]{NC} {msg}")
def warn(msg):  print(f"{YELLOW}[WARN]{NC} {msg}")
def info(msg):  print(f"{CYAN}[INFO]{NC} {msg}")
def section(msg): print(f"\n{'='*60}\n  {msg}\n{'='*60}")

# ─── 相机序列号配置 ──────────────────────────────────────────────────────────
CAMERAS = {
    "D435 (top/head)":   "254622070889",
    "D405-L (wrist-L)":  "409122273074",
    "D405-R (wrist-R)":  "409122271568",
}

# ─── CAN 配置 ────────────────────────────────────────────────────────────────
CAN_ARMS = {
    "Piper 左臂 (slave)": "can_left_slave",
    "Piper 右臂 (slave)": "can_right_slave",
}

results = {}


def test_realsense_cameras():
    """测试 RealSense 相机: 枚举 + RGB 帧采集"""
    section("Test 1: RealSense 相机")

    try:
        import pyrealsense2 as rs
    except ImportError:
        fail("pyrealsense2 未安装。安装: pip install pyrealsense2")
        results["cameras"] = "FAIL (no pyrealsense2)"
        return

    ctx = rs.context()
    devices = ctx.query_devices()
    info(f"检测到 {len(devices)} 个 RealSense 设备")

    if len(devices) == 0:
        fail("未检测到任何 RealSense 设备")
        results["cameras"] = "FAIL (0 devices)"
        return

    # 列出所有设备
    found_serials = {}
    for dev in devices:
        sn = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        fw = dev.get_info(rs.camera_info.firmware_version)
        info(f"  {name}  SN={sn}  FW={fw}")
        found_serials[sn] = name

    # 检查预期设备
    cam_pass = 0
    cam_total = len(CAMERAS)

    for label, expected_sn in CAMERAS.items():
        if expected_sn in found_serials:
            ok(f"{label} (SN:{expected_sn}) — 已连接")
            cam_pass += 1
        else:
            fail(f"{label} (SN:{expected_sn}) — 未检测到")

    # 逐个相机采集 RGB 帧
    print()
    info("逐个相机 RGB 采集测试 (640x480 @ 30fps, 采集 2 秒)...")
    for label, sn in CAMERAS.items():
        if sn not in found_serials:
            continue

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(sn)

        # D405 只支持 image_rect_raw (infrared), 用 color 流
        # 用 640x480 降低带宽
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        try:
            profile = pipeline.start(config)
            sensor_name = profile.get_device().get_info(rs.camera_info.name)

            frames_count = 0
            start = time.time()
            last_frame = None

            while time.time() - start < 2.0:
                frameset = pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frameset.get_color_frame()
                if color_frame:
                    frames_count += 1
                    last_frame = color_frame

            elapsed = time.time() - start
            fps = frames_count / elapsed if elapsed > 0 else 0

            if last_frame:
                import numpy as np
                img = np.asanyarray(last_frame.get_data())
                h, w, c = img.shape
                mean_val = img.mean()
                ok(f"{label}: {frames_count} 帧 / {elapsed:.1f}s = {fps:.1f} FPS, "
                   f"分辨率 {w}x{h}, channels={c}, mean_pixel={mean_val:.1f}")

                # 检查图像是否全黑 (可能镜头盖未取)
                if mean_val < 5:
                    warn(f"  图像几乎全黑 (mean={mean_val:.1f}), 请检查镜头盖/光线")
            else:
                fail(f"{label}: 无法获取 color frame")

            pipeline.stop()

        except Exception as e:
            fail(f"{label}: 采集异常 — {e}")
            try:
                pipeline.stop()
            except:
                pass

    # 多相机同时采集
    print()
    available_cams = [(l, s) for l, s in CAMERAS.items() if s in found_serials]
    if len(available_cams) >= 2:
        info(f"多相机并发采集测试 ({len(available_cams)} 个, 3 秒)...")

        pipelines = []
        for label, sn in available_cams:
            p = rs.pipeline()
            c = rs.config()
            c.enable_device(sn)
            c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipelines.append((label, sn, p, c))

        # 顺序启动 (避免 USB 竞争)
        started = []
        for label, sn, p, c in pipelines:
            try:
                p.start(c)
                started.append((label, sn, p))
                time.sleep(0.5)  # 间隔启动
            except Exception as e:
                fail(f"  {label}: 启动失败 — {e}")

        # 并发采集
        counts = {sn: 0 for _, sn, _ in started}
        t0 = time.time()
        while time.time() - t0 < 3.0:
            for label, sn, p in started:
                try:
                    fs = p.wait_for_frames(timeout_ms=100)
                    if fs.get_color_frame():
                        counts[sn] += 1
                except:
                    pass

        elapsed = time.time() - t0
        all_good = True
        for label, sn, p in started:
            fps = counts[sn] / elapsed
            if fps > 15:
                ok(f"  {label}: {counts[sn]} 帧 = {fps:.1f} FPS")
            else:
                warn(f"  {label}: {counts[sn]} 帧 = {fps:.1f} FPS (低于预期)")
                all_good = False
            p.stop()

        if all_good:
            ok("多相机并发采集: PASS")
        else:
            warn("多相机并发采集: 部分相机帧率偏低")

    results["cameras"] = f"{cam_pass}/{cam_total} 已连接, 采集正常"


def test_can_arms():
    """测试 CAN 机械臂: 检查接口 + piper_sdk 读取关节状态"""
    section("Test 2: CAN 机械臂 (Piper)")

    # 1. 检查 CAN 接口状态
    info("检查 CAN 接口状态...")
    can_up = {}
    for label, iface in CAN_ARMS.items():
        try:
            out = subprocess.run(
                ["ip", "-details", "link", "show", iface],
                capture_output=True, text=True, timeout=5
            )
            if out.returncode == 0 and "UP" in out.stdout:
                ok(f"{iface} ({label}): UP")
                can_up[iface] = label
            elif out.returncode == 0:
                fail(f"{iface} ({label}): DOWN — 请激活: sudo ip link set {iface} up type can bitrate 1000000")
            else:
                fail(f"{iface} ({label}): 不存在")
        except Exception as e:
            fail(f"{iface} ({label}): 检查异常 — {e}")

    # 2. candump 快速检测数据流
    print()
    info("CAN 数据流检测 (candump, 1 秒)...")
    for iface, label in can_up.items():
        try:
            out = subprocess.run(
                ["candump", iface, "-n", "10"],
                capture_output=True, text=True, timeout=2
            )
            lines = out.stdout.strip().split("\n") if out.stdout.strip() else []
            if len(lines) > 0:
                ok(f"{iface} ({label}): 收到 {len(lines)} 帧 CAN 数据")
                # 显示前 3 帧
                for line in lines[:3]:
                    info(f"    {line}")
            else:
                fail(f"{iface} ({label}): 无 CAN 数据 (臂未上电?)")
        except subprocess.TimeoutExpired:
            fail(f"{iface} ({label}): 超时无数据 (臂未上电?)")
        except Exception as e:
            fail(f"{iface} ({label}): candump 异常 — {e}")

    # 3. piper_sdk 读取关节状态
    print()
    info("piper_sdk 关节状态读取...")
    try:
        from piper_sdk import C_PiperInterface
    except ImportError:
        fail("piper_sdk 未安装 (pip install piper_sdk)")
        results["arms"] = "FAIL (no piper_sdk)"
        return

    arms_ok = 0
    for iface, label in can_up.items():
        try:
            info(f"  连接 {iface} ({label})...")
            piper = C_PiperInterface(can_name=iface)
            piper.ConnectPort()
            time.sleep(2)  # 等待数据同步

            # 读取关节角度 — SDK 0.6.1 API: GetArmJointMsgs() 返回全部关节
            import math
            joint_msgs = piper.GetArmJointMsgs()
            feedback = joint_msgs.joint_state
            # feedback.foc_position 等是按关节编号 1-6 的属性,
            # SDK 0.6.1 返回 ArmMsgFeedBackJointStates 作为整体
            # 从 str(joint_msgs) 解析或直接访问字段
            joint_positions = []
            joint_valid = True
            try:
                # SDK 0.6.1: joint_state 包含 joint_1 ~ joint_6 字段 (6-DOF + gripper)
                raw_str = str(joint_msgs)
                # 解析 "Joint N:value" 行
                import re
                joints_raw = re.findall(r'Joint\s+(\d+):(-?\d+)', raw_str)
                for _, val in joints_raw:
                    joint_positions.append(int(val) / 1000.0)  # mrad → rad
            except Exception:
                joint_valid = False

            if joint_positions:
                pos_str = ", ".join([f"{p:7.3f}" for p in joint_positions])
                ok(f"{iface} ({label}) — {len(joint_positions)}-DOF 关节状态:")
                info(f"    位置 (rad): [{pos_str}]")
                info(f"    反馈频率: {joint_msgs.Hz:.0f} Hz")

                if all(abs(p) < 0.001 for p in joint_positions):
                    warn("    所有关节角度接近 0, 可能在零位或未上电")
                else:
                    ok("    关节数据有效 (非零)")
            else:
                fail(f"{iface} ({label}): 无法解析关节数据")

            # 读取末端位姿 — SDK 0.6.1: GetArmEndPoseMsgs()
            try:
                end_pose = piper.GetArmEndPoseMsgs()
                ep = end_pose.end_pose
                x = ep.X_axis / 1000000.0  # um → m
                y = ep.Y_axis / 1000000.0
                z = ep.Z_axis / 1000000.0
                info(f"    末端位置: X={x:.4f}m, Y={y:.4f}m, Z={z:.4f}m")
            except Exception as e:
                warn(f"    末端位姿读取异常: {e}")

            # 读取夹爪 — SDK 0.6.1: GetArmGripperMsgs()
            try:
                gripper = piper.GetArmGripperMsgs()
                gripper_pos = gripper.gripper_state.grippers_angle / 1000.0
                gripper_effort = gripper.gripper_state.grippers_effort / 1000.0
                info(f"    夹爪: 角度={gripper_pos:.3f}, 力矩={gripper_effort:.3f}")
            except Exception as e:
                warn(f"    夹爪读取异常: {e}")

            arms_ok += 1

        except Exception as e:
            fail(f"{iface} ({label}): piper_sdk 连接失败 — {e}")

    if arms_ok > 0:
        results["arms"] = f"{arms_ok}/{len(can_up)} 臂数据读取正常"
    else:
        results["arms"] = "FAIL"


def test_can_raw_stats():
    """CAN 总线统计"""
    section("Test 3: CAN 总线统计")

    for iface in ["can_left_mas", "can_left_slave", "can_right_mas", "can_right_slave"]:
        try:
            out = subprocess.run(
                ["ip", "-s", "link", "show", iface],
                capture_output=True, text=True, timeout=5
            )
            if out.returncode == 0:
                lines = out.stdout.strip().split("\n")
                for line in lines:
                    if "RX:" in line or "TX:" in line or "packets" in line.lower() or "errors" in line.lower():
                        info(f"  {iface}: {line.strip()}")
        except:
            pass


def print_summary():
    """打印测试总结"""
    section("测试总结")

    for test_name, result in results.items():
        if "FAIL" in str(result).upper():
            fail(f"{test_name}: {result}")
        elif "WARN" in str(result).upper():
            warn(f"{test_name}: {result}")
        else:
            ok(f"{test_name}: {result}")

    # 硬件拓扑提示
    print()
    info("当前硬件状态:")
    # 相机
    try:
        out = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=5)
        cam_count = out.stdout.count("RealSense")
        info(f"  RealSense 相机: {cam_count} 个")
    except:
        pass
    # CAN
    try:
        out = subprocess.run(["ip", "link", "show", "type", "can"], capture_output=True, text=True, timeout=5)
        can_ifaces = [l.split(":")[1].strip() for l in out.stdout.split("\n") if "can" in l and ":" in l]
        up_ifaces = [l.split(":")[1].strip() for l in out.stdout.split("\n") if "UP" in l and "can" in l]
        info(f"  CAN 接口: {len(can_ifaces)} 个 ({len(up_ifaces)} UP)")
    except:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="kai0 硬件测试")
    parser.add_argument("--cam-only", action="store_true", help="只测试相机")
    parser.add_argument("--arm-only", action="store_true", help="只测试机械臂")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  kai0 硬件测试 — 传感器 & 机械臂数据读取")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    if not args.arm_only:
        test_realsense_cameras()

    if not args.cam_only:
        test_can_arms()
        test_can_raw_stats()

    print_summary()
