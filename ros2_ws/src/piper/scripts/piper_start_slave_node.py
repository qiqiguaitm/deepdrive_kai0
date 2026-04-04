#!/usr/bin/python3
# -*-coding:utf8-*-
# ROS2 Jazzy version of piper_start_slave_node.py
# 本文件为打开从臂控制
# 默认认为从臂有夹爪
# mode为0时为发送从臂消息
# mode为1时为控制从臂，此时如果要控制从臂，需要给主臂的topic发送消息
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import time
import threading
import math
from piper_sdk import *
from piper_sdk import C_PiperInterface
from piper_msgs.msg import PiperStatusMsg, PosCmd
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler  # 用于欧拉角到四元数的转换


class C_PiperRosNode(Node):
    """机械臂ros节点
    """
    def __init__(self) -> None:
        super().__init__('piper_start_all_node')

        # Parameters
        self.declare_parameter('can_port', 'can0')
        self.declare_parameter('mode', 0)
        self.declare_parameter('auto_enable', False)

        self.can_port = self.get_parameter('can_port').value
        self.get_logger().info(f"can_port is {self.can_port}")

        # 模式，模式为1的时候，才能够控制从臂
        self.mode = self.get_parameter('mode').value
        self.get_logger().info(f"mode is {self.mode}")

        # 是否自动使能，默认不自动使能，只有模式为1的时候才能够被设置为自动使能
        self.auto_enable = False
        if self.get_parameter('auto_enable').value and self.mode == 1:
            self.auto_enable = True
        self.get_logger().info(f"auto_enable is {self.auto_enable}")

        # publish
        self.joint_std_pub_puppet = self.create_publisher(JointState, '/puppet/joint_states', 1)
        self.arm_status_pub = self.create_publisher(PiperStatusMsg, '/puppet/arm_status', 1)
        self.end_pose_pub = self.create_publisher(PoseStamped, '/puppet/end_pose', 1)

        self.__enable_flag = False
        # 从臂消息
        self.joint_state_slave = JointState()
        self.joint_state_slave.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.joint_state_slave.position = [0.0] * 7
        self.joint_state_slave.velocity = [0.0] * 7
        self.joint_state_slave.effort = [0.0] * 7

        self.piper = C_PiperInterface(can_name=self.can_port)
        self.piper.ConnectPort()
        # 模式为1的时候，订阅控制消息 (ROS2: 直接创建订阅，后台spin线程处理回调)
        if self.mode == 1:
            self.create_subscription(PosCmd, 'pos_cmd', self.pos_callback, 1)
            self.create_subscription(JointState, '/master/joint_states', self.joint_callback, 1)
            self.create_subscription(Bool, '/enable_flag', self.enable_callback, 1)

        # 后台spin线程处理所有回调
        self._spin_thread = threading.Thread(target=self._spin_background, daemon=True)
        self._spin_thread.start()

    def GetEnableFlag(self):
        return self.__enable_flag

    def _spin_background(self):
        """Background thread to process all subscription/service callbacks."""
        rclpy.spin(self)

    def Pubilsh(self):
        """机械臂消息发布
        """
        rate = self.create_rate(200)  # 200 Hz
        enable_flag = False
        # 设置超时时间（秒）
        timeout = 5
        # 记录进入循环前的时间
        start_time = time.time()
        elapsed_time_flag = False
        while rclpy.ok():
            if self.auto_enable and self.mode == 1:
                while not enable_flag:
                    elapsed_time = time.time() - start_time
                    print("--------------------")
                    enable_flag = self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                        self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                        self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                        self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                        self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                        self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
                    print("使能状态:", enable_flag)
                    if enable_flag:
                        self.__enable_flag = True
                    self.piper.EnableArm(7)
                    self.piper.GripperCtrl(0, 1000, 0x01, 0)
                    print("--------------------")
                    # 检查是否超过超时时间
                    if elapsed_time > timeout:
                        print("超时....")
                        elapsed_time_flag = True
                        enable_flag = True
                        break
                    time.sleep(1)
            if elapsed_time_flag:
                print("程序自动使能超时,退出程序")
                exit(0)
            # 发布消息
            self.PublishSlaveArmJointAndGripper()
            self.PublishSlaveArmState()
            self.PublishSlaveArmEndPose()

            rate.sleep()

    def PublishSlaveArmState(self):
        arm_status = PiperStatusMsg()
        arm_status.ctrl_mode = self.piper.GetArmStatus().arm_status.ctrl_mode
        arm_status.arm_status = self.piper.GetArmStatus().arm_status.arm_status
        arm_status.mode_feedback = self.piper.GetArmStatus().arm_status.mode_feed
        arm_status.teach_status = self.piper.GetArmStatus().arm_status.teach_status
        arm_status.motion_status = self.piper.GetArmStatus().arm_status.motion_status
        arm_status.trajectory_num = self.piper.GetArmStatus().arm_status.trajectory_num
        arm_status.err_code = self.piper.GetArmStatus().arm_status.err_code
        arm_status.joint_1_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_1_angle_limit
        arm_status.joint_2_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_2_angle_limit
        arm_status.joint_3_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_3_angle_limit
        arm_status.joint_4_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_4_angle_limit
        arm_status.joint_5_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_5_angle_limit
        arm_status.joint_6_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_6_angle_limit
        arm_status.communication_status_joint_1 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_1
        arm_status.communication_status_joint_2 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_2
        arm_status.communication_status_joint_3 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_3
        arm_status.communication_status_joint_4 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_4
        arm_status.communication_status_joint_5 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_5
        arm_status.communication_status_joint_6 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_6
        self.arm_status_pub.publish(arm_status)

    def PublishSlaveArmEndPose(self):
        # 末端位姿
        endpos = PoseStamped()
        endpos.pose.position.x = self.piper.GetArmEndPoseMsgs().end_pose.X_axis / 1000000
        endpos.pose.position.y = self.piper.GetArmEndPoseMsgs().end_pose.Y_axis / 1000000
        endpos.pose.position.z = self.piper.GetArmEndPoseMsgs().end_pose.Z_axis / 1000000
        roll = self.piper.GetArmEndPoseMsgs().end_pose.RX_axis / 1000
        pitch = self.piper.GetArmEndPoseMsgs().end_pose.RY_axis / 1000
        yaw = self.piper.GetArmEndPoseMsgs().end_pose.RZ_axis / 1000
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
        quaternion = quaternion_from_euler(roll, pitch, yaw)
        endpos.pose.orientation.x = quaternion[0]
        endpos.pose.orientation.y = quaternion[1]
        endpos.pose.orientation.z = quaternion[2]
        endpos.pose.orientation.w = quaternion[3]
        # 为末端位姿增加时间戳
        endpos.header.stamp = self.get_clock().now().to_msg()
        self.end_pose_pub.publish(endpos)

    def PublishSlaveArmJointAndGripper(self):
        # 从臂反馈消息
        self.joint_state_slave.header.stamp = self.get_clock().now().to_msg()
        joint_0: float = (self.piper.GetArmJointMsgs().joint_state.joint_1 / 1000) * 0.017444
        joint_1: float = (self.piper.GetArmJointMsgs().joint_state.joint_2 / 1000) * 0.017444
        joint_2: float = (self.piper.GetArmJointMsgs().joint_state.joint_3 / 1000) * 0.017444
        joint_3: float = (self.piper.GetArmJointMsgs().joint_state.joint_4 / 1000) * 0.017444
        joint_4: float = (self.piper.GetArmJointMsgs().joint_state.joint_5 / 1000) * 0.017444
        joint_5: float = (self.piper.GetArmJointMsgs().joint_state.joint_6 / 1000) * 0.017444
        joint_6: float = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000000
        vel_0: float = self.piper.GetArmHighSpdInfoMsgs().motor_1.motor_speed / 1000
        vel_1: float = self.piper.GetArmHighSpdInfoMsgs().motor_2.motor_speed / 1000
        vel_2: float = self.piper.GetArmHighSpdInfoMsgs().motor_3.motor_speed / 1000
        vel_3: float = self.piper.GetArmHighSpdInfoMsgs().motor_4.motor_speed / 1000
        vel_4: float = self.piper.GetArmHighSpdInfoMsgs().motor_5.motor_speed / 1000
        vel_5: float = self.piper.GetArmHighSpdInfoMsgs().motor_6.motor_speed / 1000
        effort_6: float = self.piper.GetArmGripperMsgs().gripper_state.grippers_effort / 1000
        self.joint_state_slave.position = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]  # Example values
        self.joint_state_slave.velocity = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5, 0.0]  # Example values
        self.joint_state_slave.effort[6] = effort_6
        self.joint_std_pub_puppet.publish(self.joint_state_slave)

    def pos_callback(self, pos_data):
        """机械臂末端位姿订阅回调函数

        Args:
            pos_data ():
        """
        factor = 180 / 3.1415926
        x = round(pos_data.x * 1000) * 1000
        y = round(pos_data.y * 1000) * 1000
        z = round(pos_data.z * 1000) * 1000
        rx = round(pos_data.roll * 1000 * factor)
        ry = round(pos_data.pitch * 1000 * factor)
        rz = round(pos_data.yaw * 1000 * factor)
        self.get_logger().info(f"Received PosCmd: x={x} y={y} z={z} roll={rx} pitch={ry} yaw={rz}")
        self.get_logger().info(f"gripper: {pos_data.gripper}")
        self.get_logger().info(f"mode1: {pos_data.mode1}")
        self.get_logger().info(f"mode2: {pos_data.mode2}")
        if self.GetEnableFlag():
            self.piper.MotionCtrl_1(0x00, 0x00, 0x00)
            self.piper.MotionCtrl_2(0x01, 0x00, 50)
            self.piper.EndPoseCtrl(x, y, z,
                                    rx, ry, rz)
            gripper = round(pos_data.gripper * 1000 * 1000)
            if pos_data.gripper > 80000:
                gripper = 80000
            if pos_data.gripper < 0:
                gripper = 0
            if self.girpper_exist:
                self.piper.GripperCtrl(abs(gripper), 1000, 0x01, 0)
            self.piper.MotionCtrl_2(0x01, 0x00, 50)

    def joint_callback(self, joint_data):
        """机械臂关节角回调函数

        Args:
            joint_data ():
        """
        factor = 57324.840764  # 1000*180/3.14
        factor1 = 57.32484
        self.get_logger().info("Received Joint States:")
        self.get_logger().info(f"joint_0: {joint_data.position[0] * 1}")
        self.get_logger().info(f"joint_1: {joint_data.position[1] * 1}")
        self.get_logger().info(f"joint_2: {joint_data.position[2] * 1}")
        self.get_logger().info(f"joint_3: {joint_data.position[3] * 1}")
        self.get_logger().info(f"joint_4: {joint_data.position[4] * 1}")
        self.get_logger().info(f"joint_5: {joint_data.position[5] * 1}")
        self.get_logger().info(f"joint_6: {joint_data.position[6] * 1}")
        joint_0 = round(joint_data.position[0] * factor)
        joint_1 = round(joint_data.position[1] * factor)
        joint_2 = round(joint_data.position[2] * factor)
        joint_3 = round(joint_data.position[3] * factor)
        joint_4 = round(joint_data.position[4] * factor)
        joint_5 = round(joint_data.position[5] * factor)
        joint_6 = round(joint_data.position[6] * 1000 * 1000)
        if joint_6 > 80000:
            joint_6 = 80000
        if joint_6 < 0:
            joint_6 = 0
        if self.GetEnableFlag():
            self.piper.MotionCtrl_2(0x01, 0x01, 100)
            self.piper.JointCtrl(joint_0, joint_1, joint_2,
                                    joint_3, joint_4, joint_5)
            self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
            self.piper.MotionCtrl_2(0x01, 0x01, 100)

    def enable_callback(self, enable_flag: Bool):
        """机械臂使能回调函数

        Args:
            enable_flag ():
        """
        self.get_logger().info(f"Received enable flag: {enable_flag.data}")
        if enable_flag.data:
            self.__enable_flag = True
            self.piper.EnableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x02, 0)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)
        else:
            self.__enable_flag = False
            self.piper.DisableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x00, 0)


def main():
    rclpy.init()
    node = C_PiperRosNode()
    try:
        node.Pubilsh()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
