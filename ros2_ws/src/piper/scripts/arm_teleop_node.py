#!/usr/bin/python3
# -*-coding:utf8-*-
# Active teleop arm node (master/slave split-CAN topology).
# mode=0: physical teach-handle side. Initializes MasterSlaveConfig(0xFA),
#         auto-homes, exposes runtime linkage/teach-mode switching. WARNING:
#         actively moves the arm at init. Use only on can_*_mas interfaces.
# mode=1: physical executor side. Subscribes to /master/joint_states and
#         drives the arm. Use on can_*_slave interfaces.
from typing import (
    Optional,
)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String, Int32
import time
import threading
import math
from piper_sdk import *
from piper_sdk import C_PiperInterface
from piper_msgs.msg import PiperStatusMsg, PosCmd
from geometry_msgs.msg import Pose, PoseStamped, Twist
from std_srvs.srv import Trigger
from tf_transformations import quaternion_from_euler  # 用于欧拉角到四元数的转换


class C_PiperRosNode(Node):
    """机械臂ros节点
    """
    def __init__(self) -> None:
        super().__init__('piper_start_all_node')

        # Parameters
        self.declare_parameter('can_port', 'can0')
        self.declare_parameter('mode', 0)
        self.declare_parameter('mode_master', True)
        self.declare_parameter('auto_enable', False)

        self.can_port = self.get_parameter('can_port').value
        self.get_logger().info(f"can_port is {self.can_port}")

        # 模式，模式为1的时候，才能够控制从臂
        raw_mode = self.get_parameter('mode').value
        self.mode = int(raw_mode) if isinstance(raw_mode, str) else raw_mode
        self.get_logger().info(f"mode is {self.mode}")

        # 新增：mode_master参数，只对mode=0有效，控制主臂的起始状态
        self.mode_master = True  # 默认主臂模式
        if self.mode == 0:
            raw = self.get_parameter('mode_master').value
            # LaunchConfiguration 传入的是字符串, 需要转布尔
            if isinstance(raw, str):
                self.mode_master = raw.lower() in ('true', '1', 'yes')
            else:
                self.mode_master = bool(raw)
            self.get_logger().info(f"mode_master is {self.mode_master} (raw={raw!r})")

        # 新增：当前主从状态 (只对mode=0有效)
        if self.mode == 0:
            self.current_linkage_config = 0xFA if self.mode_master else 0xFC  # 0xFA=主臂示教, 0xFC=从臂跟随
            self.in_teach_mode = self.mode_master  # 主臂模式默认进入示教
            self.get_logger().info(f"mode=0 初始配置: {'主臂示教模式(0xFA)' if self.mode_master else '从臂跟随模式(0xFC)'}")
        else:
            # mode=1 固定为从臂，不需要切换
            self.get_logger().info("mode=1 固定为从臂模式，不支持主从切换")

        # 是否自动使能，默认不自动使能，只有模式为1的时候才能够被设置为自动使能
        self.auto_enable = False
        raw_ae = self.get_parameter('auto_enable').value
        ae_val = raw_ae.lower() in ('true', '1', 'yes') if isinstance(raw_ae, str) else bool(raw_ae)
        if ae_val and self.mode == 1:
            self.auto_enable = True
        self.get_logger().info(f"auto_enable is {self.auto_enable}")
        self.gripper_exist = True

        # 新增：状态变量
        self.is_enabled = False
        self.new_config_ = None

        # publish
        self.joint_std_pub_puppet = self.create_publisher(JointState, '/puppet/joint_states', 1)
        # 默认模式为0，读取主从臂消息
        if self.mode == 0:
            self.joint_std_pub_master = self.create_publisher(JointState, '/master/joint_states', 1)
            # 新增：mode=0时的状态发布器
            self.master_status_pub = self.create_publisher(Twist, '/master/arm_status', 1)
            self.mode_status_pub = self.create_publisher(String, '/master/mode_status', 1)

        self.arm_status_pub = self.create_publisher(PiperStatusMsg, '/puppet/arm_status', 1)
        self.end_pose_pub = self.create_publisher(PoseStamped, '/puppet/end_pose', 1)
        self.end_pose_euler_pub = self.create_publisher(PosCmd, '/puppet/end_pose_euler', 1)

        self.__enable_flag = False
        # 从臂消息
        self.joint_state_slave = JointState()
        self.joint_state_slave.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.joint_state_slave.position = [0.0] * 7
        self.joint_state_slave.velocity = [0.0] * 7
        self.joint_state_slave.effort = [0.0] * 7
        # 主臂消息
        self.joint_state_master = JointState()
        self.joint_state_master.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.joint_state_master.position = [0.0] * 7
        self.joint_state_master.velocity = [0.0] * 7
        self.joint_state_master.effort = [0.0] * 7

        self.piper = C_PiperInterface(can_name=self.can_port)
        self.piper.ConnectPort()

        # 新增：mode=0时初始化机械臂配置
        if self.mode == 0:
            self._initialize_master_arm_config()

        # service
        str_can_port = str(self.can_port)
        # 主臂单独回零
        self.master_go_zero_service = self.create_service(
            Trigger, '/' + str_can_port + '/go_zero_master', self.handle_master_go_zero_service)
        # 主从臂一起回零
        self.master_slave_go_zero_service = self.create_service(
            Trigger, '/' + str_can_port + '/go_zero_master_slave', self.handle_master_slave_go_zero_service)
        # Restoring the master and slave
        self.restore_ms_mode_service = self.create_service(
            Trigger, '/' + str_can_port + '/restore_ms_mode', self.handle_restore_ms_mode_service)

        # 模式为1的时候，订阅控制消息 (ROS2: 直接创建订阅，后台spin线程处理回调)
        if self.mode == 1:
            self.create_subscription(PosCmd, '/pos_cmd', self.pos_callback, 1)
            self.create_subscription(JointState, '/master/joint_states', self.joint_callback, 1)
            self.create_subscription(Bool, '/enable_flag', self.enable_callback, 1)

        # 新增：mode=0时启动主从模式控制订阅器 (ROS2: 直接创建订阅)
        if self.mode == 0:
            # mode=0时，主臂被控模式使用专门的控制话题
            self.create_subscription(JointState, '/master_controled/joint_states', self.joint_callback, 1)
            self.get_logger().info("主臂被控模式订阅: /master_controled/joint_states")

            # 在线切换主从配置订阅器
            self.create_subscription(String, '/master/linkage_config', self.linkage_config_callback, 1)
            # 示教模式切换订阅器
            self.create_subscription(Int32, '/master/teach_mode', self.teach_mode_callback, 1)
            # 使能控制订阅器
            self.create_subscription(Bool, '/master/enable', self.master_enable_callback, 1)

        # 后台spin线程处理所有回调
        self._spin_thread = threading.Thread(target=self._spin_background, daemon=True)
        self._spin_thread.start()

    # 新增：初始化主臂配置 (mode=0专用)
    def _initialize_master_arm_config(self):
        """初始化主臂配置：根据mode_master参数设置初始状态"""
        try:
            self.get_logger().info("正在初始化主臂配置...")

            # 先回零，确保初始位置一致
            self.get_logger().info("主从臂回零中...")
            self.piper.EnableArm(7)
            time.sleep(1)
            self.piper.ReqMasterArmMoveToHome(2)
            time.sleep(5)
            self.get_logger().info("主从臂回零完成")

            # 恢复主从模式
            self.get_logger().info("恢复主从模式...")
            self.piper.ReqMasterArmMoveToHome(0)
            time.sleep(2)

            # 设置主从配置
            self.get_logger().info(f"设置初始模式: {'主臂示教模式' if self.mode_master else '从臂跟随模式'}")
            self.piper.MasterSlaveConfig(
                linkage_config=self.current_linkage_config,
                feedback_offset=0x00,
                ctrl_offset=0x00,
                linkage_offset=0x00
            )
            time.sleep(2)

            # 使能
            self.get_logger().info("使能机械臂...")
            self.piper.EnableArm(7)
            time.sleep(2)
            self.is_enabled = True
            self.__enable_flag = True

            if self.mode_master:
                # 主臂模式：进入示教模式
                self.get_logger().info("主臂模式：进入拖动示教模式")
                self.piper.MotionCtrl_1(grag_teach_ctrl=0x01)  # 进入拖动示教模式
            else:
                # 从臂模式：设置CAN控制模式
                self.get_logger().info("从臂模式：设置CAN控制模式")
                self.piper.MotionCtrl_2(
                    ctrl_mode=0x01,     # CAN控制模式
                    move_mode=0x01,     # MOVE J
                    move_spd_rate_ctrl=30  # 30%速度
                )
                # 确保退出示教模式
                self.piper.MotionCtrl_1(grag_teach_ctrl=0x02)  # 退出拖动示教模式

            self.get_logger().info("主臂配置初始化完成")

        except Exception as e:
            self.get_logger().error(f"主臂配置初始化失败: {e}")

    # 新增：在线切换主从配置回调 (mode=0专用)
    def linkage_config_callback(self, msg):
        """在线切换主从配置回调"""
        if self.mode != 0:
            self.get_logger().warn("只有mode=0的节点才支持主从配置切换")
            return

        config_str = msg.data.lower().strip()

        # 支持的配置
        config_map = {
            "master": 0xFA,    # 主臂示教模式
            "slave": 0xFC,     # 从臂跟随模式
            "0xfa": 0xFA,
            "0xfc": 0xFC,
            "fa": 0xFA,
            "fc": 0xFC
        }

        if config_str not in config_map:
            self.get_logger().warn(f"不支持的配置: {config_str}")
            self.get_logger().warn("   支持的配置: master/0xFA (主臂示教), slave/0xFC (从臂跟随)")
            return

        new_config = config_map[config_str]

        if new_config == self.current_linkage_config:
            self.get_logger().info(f"当前已经是配置 0x{new_config:02X}")
            return

        try:
            self.get_logger().info(f"切换主从配置: 0x{self.current_linkage_config:02X} -> 0x{new_config:02X}")

            # 记录当前使能状态
            was_enabled = self.is_enabled

            # 发送新配置
            self.piper.MasterSlaveConfig(
                linkage_config=new_config,
                feedback_offset=0x00,
                ctrl_offset=0x00,
                linkage_offset=0x00
            )

            time.sleep(2)  # 等待配置生效

            # 如果之前是使能状态，确保切换后也是使能的
            if was_enabled:
                self.get_logger().info("确保切换后保持使能状态...")
                self.piper.EnableArm(7)
                time.sleep(2)
                self.is_enabled = True
                self.__enable_flag = True

            # 根据新配置初始化相应模式
            if new_config == 0xFA:
                # 主臂示教模式
                self._init_master_teach_mode()
            elif new_config == 0xFC:
                # 从臂跟随模式
                self._init_slave_follow_mode()

            self.current_linkage_config = new_config
            self.get_logger().info(f"主从配置切换成功: 0x{new_config:02X}")

            # 发布状态更新
            self._publish_mode_status()

            self.new_config_ = new_config  # 更新全局变量
        except Exception as e:
            self.get_logger().error(f"主从配置切换失败: {e}")

    # 新增：初始化主臂示教模式
    def _init_master_teach_mode(self):
        """初始化主臂示教模式 (0xFA)"""
        self.get_logger().info("初始化主臂示教模式 (0xFA)")

        if self.is_enabled:
            # 重新使能以确保状态正确
            self.get_logger().info("重新使能机械臂...")
            self.piper.EnableArm(7)
            time.sleep(2)

            # 进入拖动示教模式
            self.piper.MotionCtrl_1(grag_teach_ctrl=0x01)  # 进入拖动示教模式
            self.in_teach_mode = True
            self.get_logger().info("主臂示教模式：已进入拖动示教模式，可以手动拖拽")

    # 新增：初始化从臂跟随模式
    def _init_slave_follow_mode(self):
        """初始化从臂跟随模式 (0xFC)"""
        self.get_logger().info("初始化从臂跟随模式 (0xFC)")

        if self.is_enabled:
            # 从臂模式需要退出拖动示教模式
            self.piper.MotionCtrl_1(grag_teach_ctrl=0x02)  # 退出拖动示教模式
            self.in_teach_mode = False

            time.sleep(1)

            # 重新使能以确保状态正确
            self.get_logger().info("重新使能机械臂...")
            self.piper.EnableArm(7)
            time.sleep(2)

            # 设置为CAN控制模式
            self.piper.MotionCtrl_2(
                ctrl_mode=0x01,     # CAN控制模式
                move_mode=0x01,     # MOVE J
                move_spd_rate_ctrl=30
            )
            self.get_logger().info("从臂跟随模式：已退出示教模式，等待接收控制指令")

    # 新增：示教模式切换回调 (mode=0专用)
    def teach_mode_callback(self, msg):
        """示教模式切换回调"""
        if self.mode != 0:
            self.get_logger().warn("只有mode=0的节点才支持示教模式切换")
            return

        try:
            if msg.data == 1:
                # 进入拖动示教模式
                self.get_logger().info("切换到拖动示教模式")
                self.piper.MotionCtrl_1(grag_teach_ctrl=0x01)
                self.in_teach_mode = True

            elif msg.data == 0:
                # 退出拖动示教模式
                self.get_logger().info("退出拖动示教模式，进入控制模式")
                self.piper.MotionCtrl_1(grag_teach_ctrl=0x02)
                self.in_teach_mode = False

                # 重新设置为CAN控制模式
                time.sleep(0.5)
                self.piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=30)

        except Exception as e:
            self.get_logger().error(f"示教模式切换失败: {e}")

    # 新增：主臂使能回调 (mode=0专用)
    def master_enable_callback(self, msg):
        """主臂使能/失能回调"""
        if self.mode != 0:
            self.get_logger().warn("只有mode=0的节点才支持使能控制")
            return

        try:
            if msg.data and not self.is_enabled:
                # 使能主臂
                self.piper.EnableArm(7)
                self.is_enabled = True
                self.__enable_flag = True
                self.get_logger().info("主臂已使能")

                # 使能后根据当前配置进行初始化
                time.sleep(2)
                if self.current_linkage_config == 0xFA:
                    self._init_master_teach_mode()
                elif self.current_linkage_config == 0xFC:
                    self._init_slave_follow_mode()

            elif not msg.data and self.is_enabled:
                # 失能主臂
                self.piper.DisableArm(7)
                self.is_enabled = False
                self.__enable_flag = False
                self.get_logger().info("主臂已失能")

        except Exception as e:
            self.get_logger().error(f"使能操作失败: {e}")

    # 新增：发布模式状态 (mode=0专用)
    def _publish_mode_status(self):
        """发布当前模式状态"""
        if self.mode == 0:
            mode_msg = String()
            config_desc = "主臂示教模式" if self.current_linkage_config == 0xFA else "从臂跟随模式"
            mode_msg.data = f"0x{self.current_linkage_config:02X}:{config_desc}"
            self.mode_status_pub.publish(mode_msg)

    # 新增：发布主臂状态 (mode=0专用)
    def _publish_master_status(self):
        """发布主臂状态"""
        if self.mode == 0:
            try:
                # 使用Twist消息传递状态信息
                status_msg = Twist()

                # 用Twist的线性和角速度字段来传递状态信息
                status_msg.linear.x = 1.0  # 连接状态（固定为1）
                status_msg.linear.y = 1.0 if self.is_enabled else 0.0
                status_msg.linear.z = 1.0 if self.in_teach_mode else 0.0

                # 添加时间戳信息
                now_msg = self.get_clock().now().to_msg()
                status_msg.angular.x = float(now_msg.sec) + float(now_msg.nanosec) / 1e9

                # 添加配置信息
                status_msg.angular.y = float(self.current_linkage_config)  # 0xFA或0xFC

                self.master_status_pub.publish(status_msg)

                # 发布模式状态
                self._publish_mode_status()

            except Exception as e:
                self.get_logger().error(f"发布主臂状态失败: {e}")

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
                    self.piper.GripperCtrl(0, 1000, 0x02, 0)
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
            # 模式为0的时候，发布主臂消息
            if self.mode == 0:
                if self.mode_master or self.new_config_ == 0xFA:
                    self.PublishMasterArmJointAndGripper()
                    self.get_logger().info("Publishing master arm joint states (mode_master=True)")
                else:
                    self.get_logger().info("Master joint states publishing disabled (mode_master=False)")
            self._publish_master_status()
            rate.sleep()

    # 其他原有方法保持不变...
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

        end_pose_euler = PosCmd()
        end_pose_euler.x = self.piper.GetArmEndPoseMsgs().end_pose.X_axis / 1000000
        end_pose_euler.y = self.piper.GetArmEndPoseMsgs().end_pose.Y_axis / 1000000
        end_pose_euler.z = self.piper.GetArmEndPoseMsgs().end_pose.Z_axis / 1000000
        end_pose_euler.roll = roll
        end_pose_euler.pitch = pitch
        end_pose_euler.yaw = yaw
        end_pose_euler.gripper = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000000
        end_pose_euler.mode1 = 0
        end_pose_euler.mode2 = 0
        self.end_pose_euler_pub.publish(end_pose_euler)

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

    def PublishMasterArmJointAndGripper(self):
        # 主臂控制消息
        self.joint_state_master.header.stamp = self.get_clock().now().to_msg()

        # 读取控制指令（0x155/0x156/0x157 CAN消息）
        joint_0: float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_1 / 1000) * 0.017444
        joint_1: float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_2 / 1000) * 0.017444
        joint_2: float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_3 / 1000) * 0.017444
        joint_3: float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_4 / 1000) * 0.017444
        joint_4: float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_5 / 1000) * 0.017444
        joint_5: float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_6 / 1000) * 0.017444

        # 夹爪：示教模式(0xFA)下主臂通过CAN 0x159发出控制指令，用GetArmGripperCtrl读取
        joint_6: float = self.piper.GetArmGripperCtrl().gripper_ctrl.grippers_angle / 1000000

        # 防护：如果在示教模式下检测到全零（异常情况），则使用反馈位置
        if self.current_linkage_config == 0xFA and abs(joint_0) < 0.001 and abs(joint_1) < 0.001 and abs(joint_2) < 0.001:
            self.get_logger().warn("GetArmJointCtrl返回零值，使用反馈位置替代")
            joint_0 = (self.piper.GetArmJointMsgs().joint_state.joint_1 / 1000) * 0.017444
            joint_1 = (self.piper.GetArmJointMsgs().joint_state.joint_2 / 1000) * 0.017444
            joint_2 = (self.piper.GetArmJointMsgs().joint_state.joint_3 / 1000) * 0.017444
            joint_3 = (self.piper.GetArmJointMsgs().joint_state.joint_4 / 1000) * 0.017444
            joint_4 = (self.piper.GetArmJointMsgs().joint_state.joint_5 / 1000) * 0.017444
            joint_5 = (self.piper.GetArmJointMsgs().joint_state.joint_6 / 1000) * 0.017444

        self.joint_state_master.position = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]  # Example values
        self.joint_std_pub_master.publish(self.joint_state_master)

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
        self.get_logger().info("Received PosCmd:")
        self.get_logger().info("x: %f" % x)
        self.get_logger().info("y: %f" % y)
        self.get_logger().info("z: %f" % z)
        self.get_logger().info("roll: %f" % rx)
        self.get_logger().info("pitch: %f" % ry)
        self.get_logger().info("yaw: %f" % rz)
        self.get_logger().info("gripper: %f" % pos_data.gripper)
        self.get_logger().info("mode1: %d" % pos_data.mode1)
        self.get_logger().info("mode2: %d" % pos_data.mode2)
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
            if self.gripper_exist:
                self.piper.GripperCtrl(abs(gripper), 1000, 0x01, 0)
            self.piper.MotionCtrl_2(0x01, 0x00, 50)

    def joint_callback(self, joint_data):
        """机械臂关节角回调函数

        Args:
            joint_data ():
        """
        import os
        import datetime

        factor = 57324.840764  # 1000*180/3.14
        factor1 = 57.32484

        # # 显示信息（保持原有的）
        # self.get_logger().info("Received Joint States:")
        # self.get_logger().info("joint_0: %f" % (joint_data.position[0]*1))
        # self.get_logger().info("joint_1: %f" % (joint_data.position[1]*1))
        # self.get_logger().info("joint_2: %f" % (joint_data.position[2]*1))
        # self.get_logger().info("joint_3: %f" % (joint_data.position[3]*1))
        # self.get_logger().info("joint_4: %f" % (joint_data.position[4]*1))
        # self.get_logger().info("joint_5: %f" % (joint_data.position[5]*1))
        # self.get_logger().info("joint_6: %f" % (joint_data.position[6]*1))

        # 计算转换后的关节值
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

        # # 新增：保存数据到txt文件
        # try:
        #     # 创建保存目录（如果不存在）
        #     save_dir = os.path.expanduser("/home/agilex/cobot_magic/Piper_ros_private-ros-noetic_debug/traj")
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #
        #     # 文件路径
        #     txt_file_path = os.path.join(save_dir, "joint_states.txt")
        #
        #     # 获取当前时间戳
        #     timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        #
        #     # 准备写入的数据
        #     raw_data = "RAW: %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f" % (
        #         joint_data.position[0], joint_data.position[1], joint_data.position[2],
        #         joint_data.position[3], joint_data.position[4], joint_data.position[5], joint_data.position[6]
        #     )
        #
        #     converted_data = "CONVERTED: %d, %d, %d, %d, %d, %d, %d" % (
        #         joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6
        #     )
        #
        #     degrees_data = "DEGREES: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.6f" % (
        #         joint_data.position[0]*57.32484, joint_data.position[1]*57.32484, joint_data.position[2]*57.32484,
        #         joint_data.position[3]*57.32484, joint_data.position[4]*57.32484, joint_data.position[5]*57.32484, joint_data.position[6]
        #     )
        #
        #     # 写入文件
        #     with open(txt_file_path, 'a', encoding='utf-8') as f:
        #         f.write(f"[{timestamp}] {raw_data}\n")
        #         f.write(f"[{timestamp}] {converted_data}\n")
        #         f.write(f"[{timestamp}] {degrees_data}\n")
        #         f.write(f"[{timestamp}] ENABLE_FLAG: {self.GetEnableFlag()}\n")
        #         f.write("-" * 80 + "\n")
        #
        #     # 每100次记录打印一次保存信息（避免日志过多）
        #     if not hasattr(self, '_save_count'):
        #         self._save_count = 0
        #     self._save_count += 1
        #
        #     if self._save_count % 100 == 1:  # 第1次和每100次
        #         self.get_logger().info("Joint data saved to: %s (count: %d)" % (txt_file_path, self._save_count))
        #
        # except Exception as e:
        #     self.get_logger().warn("Failed to save joint data to txt file: %s" % str(e))

        # 原有的控制逻辑（保持不变）
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
        self.get_logger().info("Received enable flag:")
        self.get_logger().info("enable_flag: %s" % enable_flag.data)
        if enable_flag.data:
            self.__enable_flag = True
            self.piper.EnableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x02, 0)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)
        else:
            self.__enable_flag = False
            self.piper.DisableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x00, 0)

    def handle_master_go_zero_service(self, request, response):
        self.get_logger().info(f"-----------------------RESET---------------------------")
        self.get_logger().info(f"{self.can_port} send piper master go zero service")
        self.get_logger().info(f"-----------------------RESET---------------------------")
        self.piper.ReqMasterArmMoveToHome(1)
        response.success = True
        response.message = str({self.can_port}) + "send piper master go zero service success"
        self.get_logger().info(f"Returning resetResponse: {response.success}, {response.message}")
        return response

    def handle_master_slave_go_zero_service(self, request, response):
        self.get_logger().info(f"-----------------------RESET---------------------------")
        self.get_logger().info(f"{self.can_port} send piper master slave go zero service")
        self.get_logger().info(f"-----------------------RESET---------------------------")
        self.piper.ReqMasterArmMoveToHome(2)
        response.success = True
        response.message = str({self.can_port}) + "send piper master slave go zero service success"
        self.get_logger().info(f"Returning resetResponse: {response.success}, {response.message}")
        return response

    def handle_restore_ms_mode_service(self, request, response):
        self.get_logger().info(f"-----------------------RESET---------------------------")
        self.get_logger().info(f"{self.can_port} send piper restore master slave mode service")
        self.get_logger().info(f"-----------------------RESET---------------------------")
        self.piper.ReqMasterArmMoveToHome(0)
        response.success = True
        response.message = str({self.can_port}) + "send piper restore master slave mode service success"
        self.get_logger().info(f"Returning resetResponse: {response.success}, {response.message}")
        return response


def main():
    rclpy.init()
    node = C_PiperRosNode()

    # 新增：启动时打印使用说明
    if node.mode == 0:
        node.get_logger().info("=" * 60)
        node.get_logger().info("增强版主从控制功能已集成!")
        node.get_logger().info("=" * 60)
        node.get_logger().info("Launch参数:")
        node.get_logger().info(f"  - mode: {node.mode} (0=主臂读取模式)")
        node.get_logger().info(f"  - mode_master: {node.mode_master} ({'主臂示教模式' if node.mode_master else '从臂跟随模式'})")
        node.get_logger().info(f"  - 当前配置: 0x{node.current_linkage_config:02X}")
        node.get_logger().info("=" * 60)
        node.get_logger().info("ROS2控制指令:")
        node.get_logger().info("  1. 使能: ros2 topic pub /master/enable std_msgs/msg/Bool '{data: true}'")
        node.get_logger().info("  2. 失能: ros2 topic pub /master/enable std_msgs/msg/Bool '{data: false}'")
        node.get_logger().info("  3. 示教: ros2 topic pub /master/teach_mode std_msgs/msg/Int32 '{data: 1}'")
        node.get_logger().info("  4. 控制: ros2 topic pub /master/teach_mode std_msgs/msg/Int32 '{data: 0}'")
        node.get_logger().info("  在线切换主从配置:")
        node.get_logger().info("     - 主臂示教: ros2 topic pub /master/linkage_config std_msgs/msg/String '{data: master}'")
        node.get_logger().info("     - 从臂跟随: ros2 topic pub /master/linkage_config std_msgs/msg/String '{data: slave}'")
        node.get_logger().info("     - 十六进制: ros2 topic pub /master/linkage_config std_msgs/msg/String '{data: 0xFA}'")
        node.get_logger().info("=" * 60)
    elif node.mode == 1:
        node.get_logger().info("从臂模式启动 - 等待接收控制指令")

    try:
        node.Pubilsh()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
