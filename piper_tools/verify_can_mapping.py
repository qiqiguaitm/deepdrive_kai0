#!/usr/bin/env python3
"""
CAN-臂映射校验脚本

用法: python3 piper_tools/verify_can_mapping.py

原理:
  - slave 臂有伺服环, 通过 SDK 读关节值变化检测运动
  - master 臂无伺服, 通过 candump 计帧数检测运动 (静止时无帧, 晃动时有帧)

操作步骤:
  1. 先运行 activate_can.sh 激活并重命名 CAN 接口
  2. 运行本脚本
  3. 依次晃动每个臂, 观察输出中哪个接口标记了 "<<< MOVING"
  4. 核对是否与预期映射一致
  5. Ctrl+C 退出
"""

import os
import re
import select
import subprocess
import sys
import time

from piper_sdk import C_PiperInterface

MOVE_THRESHOLD = 50  # 关节原始值变化超过此阈值视为运动 (slave)
CANDUMP_THRESHOLD = 5  # 帧数超过此阈值视为运动 (master)


def get_can_interfaces():
    """自动检测所有 UP 状态的 CAN 接口"""
    out = subprocess.run(
        ["ip", "-br", "link", "show", "type", "can"],
        capture_output=True, text=True, timeout=5,
    )
    interfaces = []
    for line in out.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        name = parts[0]
        state = parts[1] if len(parts) > 1 else ""
        if "UP" in state or "UP" in line:
            interfaces.append(name)
    return sorted(interfaces)


def classify_interfaces(interfaces):
    """分类 slave/master: slave 有伺服环, 静止时关节值非零且 Hz > 0"""
    slaves = []
    masters = []
    for iface in interfaces:
        try:
            p = C_PiperInterface(can_name=iface)
            p.ConnectPort()
            time.sleep(1.5)
            raw = str(p.GetArmJointMsgs())
            vals = [int(v) for _, v in re.findall(r"Joint\s+(\d+):(-?\d+)", raw)]
            # slave: 关节值不全为零
            if vals and any(v != 0 for v in vals):
                slaves.append((iface, p))
            else:
                masters.append(iface)
        except Exception:
            masters.append(iface)
    return slaves, masters


def read_joints(piper):
    raw = str(piper.GetArmJointMsgs())
    return [int(v) for _, v in re.findall(r"Joint\s+(\d+):(-?\d+)", raw)]


def main():
    interfaces = get_can_interfaces()
    if not interfaces:
        print("未检测到活跃的 CAN 接口, 请先运行: bash piper_tools/activate_can.sh")
        sys.exit(1)

    print(f"检测到 {len(interfaces)} 个 CAN 接口: {', '.join(interfaces)}")
    print("正在连接并分类 slave/master ...")
    print()

    slaves, masters = classify_interfaces(interfaces)

    slave_names = [iface for iface, _ in slaves]
    print(f"  Slave  ({len(slaves)}): {', '.join(slave_names) if slave_names else '无'}")
    print(f"  Master ({len(masters)}): {', '.join(masters) if masters else '无'}")
    print()

    if not slaves and not masters:
        print("所有接口连接失败")
        sys.exit(1)

    # 初始读数 (slave)
    prev = {}
    for iface, piper in slaves:
        prev[iface] = read_joints(piper)

    # 启动 candump 进程 (master)
    candump_procs = {}
    for iface in masters:
        p = subprocess.Popen(
            ["candump", iface],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        candump_procs[iface] = p

    print("开始监控, 请依次晃动每个臂的关节 (Ctrl+C 退出)")
    print("  slave 臂: 通过关节值变化检测")
    print("  master 臂: 通过 CAN 帧计数检测 (每 0.3s 窗口)")
    print("=" * 70)

    # 打印占位行
    all_monitored = slave_names + masters
    for iface in all_monitored:
        kind = "slave" if iface in slave_names else "master"
        print(f"  {iface:20s} [{kind:6s}]: ---")

    try:
        while True:
            time.sleep(0.3)
            lines = []

            # Slave: SDK 关节值变化
            for iface, piper in slaves:
                cur = read_joints(piper)
                if prev[iface] and cur:
                    diffs = [abs(a - b) for a, b in zip(cur, prev[iface])]
                    max_diff = max(diffs)
                    marker = " <<< MOVING" if max_diff > MOVE_THRESHOLD else ""
                    lines.append(f"  {iface:20s} [slave ]: max_delta={max_diff:6d}{marker}")
                else:
                    lines.append(f"  {iface:20s} [slave ]: 读取失败")
                prev[iface] = cur

            # Master: candump 帧计数 (非阻塞读, 统计本周期帧数)
            for iface in masters:
                count = 0
                p = candump_procs[iface]
                while select.select([p.stdout], [], [], 0)[0]:
                    line = p.stdout.readline()
                    if line:
                        count += 1
                    else:
                        break
                marker = " <<< MOVING" if count > CANDUMP_THRESHOLD else ""
                lines.append(f"  {iface:20s} [master]: frames={count:6d}{marker}")

            # 覆盖上一轮输出
            sys.stdout.write(f"\033[{len(lines)}A")
            for line in lines:
                sys.stdout.write(f"\033[2K{line}\n")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n已退出。")
    finally:
        for p in candump_procs.values():
            p.terminate()
            p.wait()


if __name__ == "__main__":
    main()
