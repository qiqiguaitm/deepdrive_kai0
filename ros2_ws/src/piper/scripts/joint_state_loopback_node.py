#!/usr/bin/python3
"""ROS2 sim helper — relays /master/joint_<arm> → /puppet/joint_<arm>, and
seeds /puppet with the episode's action[0] so replay_node's pose-alignment
preflight passes in sim mode (no real arm_reader to feed back current state).

Topology when used by playback_launch.py with enable_real_arms:=false:

  ┌──────────┐  /master/joint_left   ┌─────────┐  /puppet/joint_left   ┌──────┐
  │ replay   ├──────────────────────▶│ loopback├──────────────────────▶│replay│
  └──────────┘                       │ (relay) │                       └──────┘
                                     │ + seed  │
                                     │ action0 │ on startup, repeat 5Hz
                                     └─────────┘

Real-arm mode doesn't use this node — arm_reader_node mode=1 publishes
/puppet/joint_<arm> from CAN telemetry as usual.
"""
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

JOINT_NAMES = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def _load_action0(parquet_path: str):
    """Returns ([left7], [right7]) from action[0] of the parquet, or (None, None)."""
    if not parquet_path or not Path(parquet_path).is_file():
        return None, None
    try:
        import pyarrow.parquet as pq
        tbl = pq.read_table(parquet_path, columns=["action"])
        a0 = tbl.column("action").to_pylist()[0]
        if len(a0) < 14:
            return None, None
        return list(a0[:7]), list(a0[7:14])
    except Exception as e:  # noqa: BLE001
        print(f"[loopback] action[0] load failed: {e}", file=sys.stderr)
        return None, None


class _Loopback(Node):
    def __init__(self):
        super().__init__("joint_state_loopback")

        self.declare_parameter("episode_path", "")
        ep = str(self.get_parameter("episode_path").value)

        self._left0, self._right0 = _load_action0(ep)
        if self._left0 is not None:
            self.get_logger().info(
                f"loaded action[0] from {ep}: "
                f"L={[round(x, 3) for x in self._left0]}, "
                f"R={[round(x, 3) for x in self._right0]}")
        else:
            self.get_logger().warn(
                f"no action[0] seed (episode_path empty or unreadable: {ep!r}); "
                f"replay_node preflight may fail until /master starts publishing")

        self._pub_left = self.create_publisher(JointState, "/puppet/joint_left", 100)
        self._pub_right = self.create_publisher(JointState, "/puppet/joint_right", 100)

        # ── Relay: /master/joint_* → /puppet/joint_* (1-hop) ──
        self.create_subscription(
            JointState, "/master/joint_left",
            lambda m: self._pub_left.publish(m), 100)
        self.create_subscription(
            JointState, "/master/joint_right",
            lambda m: self._pub_right.publish(m), 100)

        # ── Seed: publish action[0] to /puppet/joint_* at 5 Hz ──
        # Once /master starts streaming, the relay path overwrites at 30 Hz so
        # the seed becomes a no-op. Before that, this lets the alignment check
        # see "current pose == action[0]" (trivially aligned).
        if self._left0 is not None:
            self.create_timer(0.2, self._tick_seed)

    def _tick_seed(self):
        now = self.get_clock().now().to_msg()
        for pub, vals in ((self._pub_left, self._left0),
                          (self._pub_right, self._right0)):
            msg = JointState()
            msg.header.stamp = now
            msg.name = JOINT_NAMES
            msg.position = list(vals)
            pub.publish(msg)


def main():
    rclpy.init()
    n = _Loopback()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
