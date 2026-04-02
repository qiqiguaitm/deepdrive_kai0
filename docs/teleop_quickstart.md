# 机械臂遥操快速操作手册

本文档描述从硬件接入到启动遥操的完整步骤。

## 前置条件

- 4 个 Piper 机械臂已上电（2 master + 2 slave）
- 4 个 USB-CAN 适配器已物理连接到 IPC
- 3 个 RealSense 相机已连接（1× D435 俯视 + 2× D405 腕部）

---

## Step 1: 接入 CAN-to-USB 适配器

将 4 个机械臂的 CAN-to-USB 适配器插入 IPC 的 USB 口。

确认系统识别到适配器：

```bash
lsusb | grep -i can
# 应看到 4 个 CAN 设备

ip -br link show type can
# 应看到 4 个 canX 接口 (如 can0 ~ can3)
```

> **注意**: 如果接口数不足 4 个，检查 USB 线连接和适配器指示灯，必要时 `sudo modprobe gs_usb`。

---

## Step 2: 激活 CAN 接口

```bash
cd piper_tools
bash find_n_activate.sh
```

该脚本会扫描所有 CAN 接口、设置波特率 (1000000) 并激活为 `can0` ~ `can3`。此时接口名称是随机分配的，尚未对应到具体的臂。

---

## Step 3: 校准主从臂的 CAN 映射关系（交互式）

```bash
python calibrate_can_mapping.py
```

脚本会依次提示你晃动指定的臂（左 master → 左 slave → 右 master → 右 slave），自动检测哪个 CAN 接口产生了数据变化，建立 bus-info 到臂角色的映射。

校准结果会自动写入：
- `config/pipers.yml` — 映射配置
- `activate_can.sh` — 更新脚本中的 bus-info 映射数组

---

## Step 4: 按映射关系重命名并激活 CAN 接口

```bash
bash activate_can.sh
```

该脚本根据 Step 3 校准的 bus-info 映射，将 `canX` 重命名为 ROS2 launch 期望的符号名：

| 符号名 | 角色 |
|--------|------|
| `can_left_mas` | 左臂 master（示教） |
| `can_left_slave` | 左臂 slave（执行） |
| `can_right_mas` | 右臂 master（示教） |
| `can_right_slave` | 右臂 slave（执行） |

输出末尾会自动验证各接口是否 UP 并有数据流。

---

## Step 5: 验证映射正确性

### 5a. 验证 CAN-臂对应关系

```bash
python verify_can_mapping.py
```

脚本持续监控所有 CAN 接口的关节数据。依次晃动每个臂，确认输出中标记 `<<< MOVING` 的接口与预期一致。`Ctrl+C` 退出。

### 5b. 验证相机

打开 RealSense Viewer 确认 3 个相机均正常工作且 serial number 对应正确：

```bash
realsense-viewer
```

核对相机序列号与配置一致：

| 角色 | 型号 | Serial No. | 说明 |
|------|------|-----------|------|
| top_head | D435 | 254622070889 | 俯视全局 |
| hand_left | D405 | 409122273074 | 左腕 |
| hand_right | D405 | 409122271568 | 右腕 |

> 相机配置见 `config/cameras.yml`。如序列号不匹配需更新配置文件。

---

## Step 6: 启动遥操

```bash
cd scripts
bash start_teleop.sh
```

启动后 master 臂进入拖拽示教模式，拖动 master 臂，对应的 slave 臂会实时跟随。

### 验证遥操正常

另开终端检查 topic 频率：

```bash
source ros2_ws/install/setup.bash
ros2 topic hz /master/joint_left     # 应 ~200 Hz
ros2 topic hz /puppet/joint_left     # 应 ~200 Hz
```

---

## 故障排查

| 现象 | 排查 |
|------|------|
| `ip link show type can` 接口数 < 4 | 检查 USB 连接，`sudo modprobe gs_usb` |
| `candump canX` 无数据 | 对应臂未上电或 USB-CAN 适配器故障 |
| launch 报错找不到 `can_left_slave` 等 | 未执行 Step 4 `activate_can.sh` |
| 关节读取 NaN | SDK API 变更，确认使用 `GetArmJointMsgs()` |
| 相机在 realsense-viewer 中不显示 | 重新插拔 USB，检查 Hub 供电/带宽 |
| slave 臂不跟随 master | 检查 master 是否进入示教模式，确认映射无误 |
