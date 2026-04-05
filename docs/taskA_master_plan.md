# kai0 Task A 主规划 (训练 + 部署 Roadmap)

> **作用**: 项目级主规划 —— 训练集群 (gf0/gf1) + 部署机 (sim01) 的整体算力/流程/交付路线图。
> **关联文档**: sim01 本地具体部署步骤详见 [sim01_deployment.md](sim01_deployment.md)。

> 硬件配置:
> - 机器人: 双臂 Piper + 2×D405 (hand) + 1×D435 (top)
> - 工控机/推理机 (**sim01**): tim@sim01, 双卡 RTX 5090 32GB, 工作目录 `/data1/tim/workspace/deepdive_kai0`
> - 训练机 **gf0**: `ssh -p 55555 -R 29290:localhost:29290 tim@14.103.44.161`, 8×A100 80GB, 工作目录 `~/workspace/deepdive_kai0`
> - 训练机 **gf1**: `ssh -p 11111 -R 29290:localhost:29290 tim@14.103.44.161`, 8×A100 80GB, 工作目录 `~/workspace/deepdive_kai0`
> - 所有机器密码: tim | 工作目录同构: `~/workspace/deepdive_kai0`
> - gf0/gf1 访问外网通过 SSH 反向隧道代理 (端口 29290), 登录即自动生效
> 目标: 最快速度复现 Task A (T恤展平&折叠) 全流程

---

## 一、GPU 算力规划

### 1.1 算力分配总览

```
┌──────────────────────────────────────────────────────────────────┐
│              训练集群 (远端, 14.103.44.161)                        │
│              gf0/gf1 访问外网: SSH 反向隧道代理 (:29290), 登录即生效  │
│                                                                  │
│  ┌───────────────────────────┐  ┌───────────────────────────┐    │
│  │  gf0: 8×A100 80GB         │  │  gf1: 8×A100 80GB         │    │
│  │  ssh -p 55555 tim@...     │  │  ssh -p 11111 tim@...     │    │
│  │  ~/workspace/deepdive_kai0│  │  ~/workspace/deepdive_kai0│    │
│  │                           │  │                           │    │
│  │  · Full 微调              │  │  · Full 微调 (并行)        │    │
│  │  · AWBC 训练              │  │  · Advantage 估计器        │    │
│  │  · Model Arithmetic       │  │  · 数据子集训练            │    │
│  │    (需要加载模型)           │  │    (用于 MA 合并)          │    │
│  └───────────────────────────┘  └───────────────────────────┘    │
│              gf0 <──内网互通──> gf1                                │
│  训练完成后 → checkpoint 传到 sim01                                │
└──────────────────────────────────────────────────────────────────┘
                         │ rsync -e "ssh -p 55555"
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│              sim01 (工控机 + 推理机, 桌边, 连接机器人)               │
│              tim@sim01 | /data1/tim/workspace/deepdive_kai0      │
│                                                                  │
│  ┌─────────────────────────────────────────────────────┐         │
│  │  双卡 RTX 5090 (32GB × 2)                            │         │
│  │                                                     │         │
│  │  GPU 0: serve_policy.py (:8000) — 推理 ~8GB         │         │
│  │  GPU 1: 备用 / Model Arithmetic (小规模)             │         │
│  │         或 DAgger 时同时推理第二个策略                  │         │
│  │                                                     │ ←USB→ 机器人
│  │  同时兼任 IPC: ROS + 相机 + CAN + Piper SDK          │         │
│  └─────────────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 各任务算力需求 vs 硬件

| 任务 | 需求 | 分配到 | 说明 |
|------|------|--------|------|
| **推理部署** | >8 GB | sim01 GPU 0 | 单卡 32GB 绰绰有余 |
| **Full 微调** | >70 GB, batch=256 | gf0 / gf1 | **与 kai0 原版完全一致** |
| **AWBC 训练** | 同上 | gf0 / gf1 | JAX 训练, 配置相同 |
| **Advantage 估计器** | PyTorch 单/多卡 | gf1 | DDP 多卡加速 |
| **Model Arithmetic** | 加载多 ckpt, 单卡 A100 | gf0 | 或 sim01 5090 (32GB 够用) |
| **compute_norm_stats** | CPU 为主 | gf0 / gf1 任一 | 几分钟完成 |
| **DAgger 推理** | >8 GB | sim01 GPU 0 | 实时推理 |

### 1.3 关键优势: 训练配置与 kai0 原版完全一致

kai0 原版使用 **8×A100**, 你有 **8×A100 × 2**. 这意味着:
- `batch_size=256` **不需要改**
- `fsdp_devices=1` **不需要改** (单卡 A100 80GB 即可 Full 微调)
- 甚至可以同时在两个节点**并行训练**多个实验 (不同超参/数据子集)
- Model Arithmetic 需要的多个 checkpoint 可以在两个节点上并行训练, 速度翻倍

```python
# config.py 完全不需要修改! 直接使用原版配置:
TrainConfig(
    name="pi05_flatten_fold_normal",
    batch_size=256,      # 原版, 无需改动
    fsdp_devices=1,      # 单卡 A100 80GB 即可
    num_train_steps=100_000,
    ...
)
```

### 1.4 训练集群 vs 部署机 工作流

```
开发循环:

1. [gf0/gf1] 训练 → 生成 checkpoint
        │
        ↓ rsync -e "ssh -p 55555" tim@14.103.44.161:~/workspace/deepdive_kai0/kai0/checkpoints/ ...
        ↓ (或 gf1: -p 11111)

2. [sim01] serve_policy.py → 推理验证
        │
        ↓ 如果效果不够好

3. [sim01] DAgger 采集纠正数据
        │
        ↓ rsync -e "ssh -p 55555" ./data/ tim@14.103.44.161:~/workspace/deepdive_kai0/kai0/data/

4. [gf0/gf1] 合并数据 → 重新训练 → 回到 step 1
```

---

## 二、物理布局与安装

### 2.1 桌面布局 (严格复现 Task A 尺寸)

```
                         桌面 (推荐 120cm × 80cm, 高度 ~75cm)
                         浅色不反光桌面 (建议铺浅色桌布)

                    ┌─────────────────────────────────────────┐
                    │                                         │
                    │        ┌───────────────────┐            │
                    │        │   操作区域          │            │
                    │        │  (~60×50cm)        │            │
                    │        │  T恤放置区          │            │
                    │        └───────────────────┘            │
                    │                                         │
              18cm  │  ┌───────┐      39cm       ┌───────┐   │ 12cm
          ←───────→ │  │ 左Piper│ ←────────────→ │ 右Piper│   │←───→
                    │  │ slave │                  │ slave │   │
                    │  └───┬───┘                  └───┬───┘   │
                    │      │                          │       │
    桌前沿 ─────────┤──────┼──────────────────────────┼───────┤
                    │      │     34cm       34cm      │       │
                    │      │   ←──────┼──────→       │       │
                    │      │         ┌┴┐             │       │
                    │      │         │支│ D435        │       │
                    │      │         │架│ 第三视角     │       │
                    │      │         └─┘             │       │
                    │      │          ↕ 76cm高        │       │
                    │      │          30° 俯角        │       │
                    │      │                          │       │
                    │  ┌───────┐                  ┌───────┐   │
              18cm  │  │ 左Piper│                  │ 右Piper│   │ 18cm
                    │  │ master│                  │ master│   │
                    │  └───────┘                  └───────┘   │
                    └─────────────────────────────────────────┘

                              ▲ 操作员站位方向
```

**精确尺寸 checklist**:

- [ ] 左主臂 (slave) 底座中心距桌前沿: 18 cm
- [ ] 右主臂 (slave) 底座中心距桌前沿: 12 cm (注意不对称!)
- [ ] 左右主臂底座中心距: 39 cm
- [ ] 中心相机支架底座距左副臂中心: 34 cm
- [ ] 中心相机支架底座距右副臂中心: 34 cm
- [ ] 中心相机 D435 高度 (桌面到光心): 76 cm
- [ ] 中心相机俯角: 30°
- [ ] 副臂 (master, DAgger 用) 各距桌前沿 18 cm

### 2.2 相机安装

#### D435 (第三视角/头顶)

打印 kai0 提供的支架:
```
setup/FlattenFold/camera-bottom-mount-bracket.STEP  → 3D 打印
```

安装要点:
- 用金属管或三脚架固定在桌面后方中央
- D435 用螺丝固定在打印支架上
- 光心高度 76cm, 向下倾斜 30°
- 视野应覆盖整个 60×50cm 操作区域
- USB-C 线从背面走线, 避免被机械臂钩到

#### D405 (左右腕部) — 使用 Cobot Magic ALOHA 一体式支架

当前配置: **Cobot Magic 随附的 ALOHA 夹爪 + D405 一体式支架, 带可调高度**。
这是最理想的方案, 无需自制任何安装件。

**与 kai0 原版对比**:

| 维度 | kai0 原版 | 你的配置 | 差异影响 |
|------|----------|---------|---------|
| 相机 | D435i (90×25mm 长条) | D405 (42×42mm 方形) | FOV 不同, 微调解决 |
| 安装件 | 自制 3D 打印件 | **Cobot Magic 原装一体式** | 更稳固, 更标准 |
| 光心对齐 | "centered" 居中 | **双目居中** | 对称性更好 |
| 高度 | 固定 | **可调** | 更灵活, 可以调到最优 |
| 夹爪 | 自制 3D 打印 (agilex-gripper.3mf) | **ALOHA 原装夹爪** | 直接用 |

**D405 高度调节建议**:

```
      正视图 (从正前方看):

      Piper 腕部
           │
      ┌────┴────┐
      │ ┌─────┐ │
      │ │D405 │ │ ← 双目居中, 光心对准夹爪操作中心
      │ │ ○ ○ │ │   (○ = 左右 stereo 镜头)
      │ └─────┘ │
      │         │
      │  ALOHA  │
      │  夹 爪   │
      └─────────┘
```

高度调节目标:
- D405 能看到**夹爪尖端** + **正前方 10-30cm 范围的布料**
- 不被夹爪本体遮挡视线
- 建议先默认位置试跑, 根据推理效果再微调

**USB 走线**: D405 USB-C 线沿机械臂背面走线, 用扎带固定,
留够长度避免在关节运动时拉扯。每个关节处留弧形余量。

### 2.3 夹爪

**直接使用 Cobot Magic ALOHA 原装夹爪**, 无需 3D 打印 kai0 的夹爪。

ALOHA 夹爪相比 kai0 自制夹爪的优势:
- 工厂品质, 精度和一致性更好
- 与 D405 支架一体化设计, 无额外装配
- Piper SDK 直接支持

**增强抓取**: 叠衣服需要稳定抓取布料, 建议在夹爪内侧贴橡胶垫或硅胶条增加摩擦力。

> 备用: 如果 ALOHA 夹爪对叠衣服效果不好, kai0 提供了专为叠衣设计的夹爪:
> - `setup/FlattenFold/agilex-gripper.3mf` (标准版, 直接打印)
> - `setup/FlattenFold/agilex-gripper-modified.STEP` (改进版)
> - `setup/FlattenFold/agilex-gripper-soft.SLDPRT` (软夹爪, 更适合布料)

---

## 三、接线与通信

### 3.1 接线清单

```
┌───────────────────────────────────────────────────────────────┐
│                   桌面 / 机械臂侧                               │
│                                                               │
│  ┌──────────┐  USB-CAN #1  ┌────────────────────────────────┐│
│  │左Piper   │ ────────────→│                                ││
│  │ slave    │              │  sim01 (工控机 + 推理机一体)      ││
│  └──────────┘              │  tim@sim01 | 密码: tim          ││
│                            │  双卡 RTX 5090 32GB             ││
│  ┌──────────┐  USB-CAN #2  │                                ││
│  │右Piper   │ ────────────→│  GPU 0: serve_policy.py :8000  ││
│  │ slave    │              │  GPU 1: 备用 / MA              ││
│  └──────────┘              │                                ││
│                            │  ROS + 相机 + CAN + 推理        ││
│  ┌──────────┐  USB-CAN #3  │  全部在本机完成                   ││
│  │左Piper   │ ────────────→│                                ││
│  │ master   │              │  工作目录:                       ││
│  └──────────┘              │  /data1/tim/workspace/          ││
│                            │    deepdive_kai0/kai0           ││
│  ┌──────────┐  USB-CAN #4  │                                ││
│  │右Piper   │ ────────────→│                                ││
│  │ master   │              │                                ││
│  └──────────┘              │                                ││
│                            │                                ││
│  ┌──────┐  USB-C #1       │                                ││
│  │D405左│ ────────────────→│                                ││
│  └──────┘                  │                                ││
│  ┌──────┐  USB-C #2       │                                ││
│  │D405右│ ────────────────→│                                ││
│  └──────┘                  │                                ││
│  ┌──────┐  USB-C #3       │                                ││
│  │D435  │ ────────────────→│                                ││
│  └──────┘                  └────────────────────────────────┘│
└───────────────────────────────────────────────────────────────┘
```

**USB 带宽注意**: 3 个 RealSense 同时 640×480@30fps 需要 ~500MB/s USB 带宽。
- 将 3 个相机分散接在**不同的 USB 控制器**上 (不同物理 USB 口)
- 避免全部接同一个 USB hub
- 可用 `lsusb -t` 查看 USB 拓扑

### 3.2 IPC 配置 (sim01 兼任)

**当前方案: sim01 同时作为工控机 (IPC) 和推理机**, 无独立 IPC。

| 组件 | sim01 实际配置 |
|------|--------------|
| 主机名 | sim01 (tim@sim01, 密码 tim) |
| GPU | 2× RTX 5090 32GB |
| USB | 需 ≥5 个 USB 3.0 口 (3 相机 + 2~4 CAN) |
| 工作目录 | `/data1/tim/workspace/deepdive_kai0/kai0` |

注意事项:
- ROS Noetic 依赖 Python 3.8, 与 kai0 训练环境 (Python 3.10+) 冲突
- 建议用 conda 隔离: `kai0_inference` (Python 3.10) 和 `ros_env`
- 或用 Docker 隔离 ROS 环境
- 推理 (serve_policy.py) 用 GPU 0, 其余 ROS/相机/CAN 不需要 GPU

---

## 四、软件环境搭建

### 4.1 训练集群: gf0 + gf1 (各 8×A100 80GB)

```bash
# === 登录方式 (从 sim01) ===
# gf0:
ssh -p 55555 -R 29290:localhost:29290 tim@14.103.44.161
# gf1:
ssh -p 11111 -R 29290:localhost:29290 tim@14.103.44.161
# 密码: tim
# 反向隧道代理已自动生效, 可直接访问 GitHub / HuggingFace / PyPI

# === 在 gf0 和 gf1 上都执行 ===
cd ~/workspace/deepdive_kai0

# 1. 克隆仓库
git clone --recurse-submodules https://github.com/OpenDriveLab/kai0.git
cd kai0

# 2. 安装 uv 包管理器
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 安装环境
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 4. 验证 JAX 能看到 8 张 A100
uv run python -c "import jax; print(jax.devices())"
# 应输出: [CudaDevice(id=0), ..., CudaDevice(id=7)]

# 5. 下载数据和检查点
uv run python scripts/download_dataset.py --tasks Task_A
uv run python scripts/download_checkpoints.py --tasks Task_A

# 训练配置完全使用 kai0 原版, 不需任何修改:
# batch_size=256, fsdp_devices=1, num_train_steps=100000
```

**双节点并行训练策略**:
```
gf0: 主训练 (normal fine-tune, AWBC)
gf1: 并行训练 (数据子集 → Model Arithmetic, Advantage 估计器)
两个节点可以同时训练不同实验, 速度翻倍
gf0 <──内网互通──> gf1 (数据/checkpoint 互传走内网)
```

**gf0 <-> gf1 数据互传** (走内网, 需确认内网 IP):
```bash
# 示例 (假设 gf1 内网 IP 为 192.168.x.x):
# 从 gf0 传 checkpoint 到 gf1:
rsync -avP ./checkpoints/ tim@<gf1内网IP>:~/workspace/deepdive_kai0/kai0/checkpoints/
```

### 4.2 sim01 工控机/推理机 (双 RTX 5090, 兼 IPC)

```bash
# === sim01 本机操作 ===
cd /data1/tim/workspace/deepdive_kai0

# 1. 克隆仓库
git clone --recurse-submodules https://github.com/OpenDriveLab/kai0.git
cd kai0

# 2. 安装 uv + 基本环境 (用于 serve_policy)
curl -LsSf https://astral.sh/uv/install.sh | sh
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 3. 从训练集群拉取 checkpoint
# 从 gf0:
rsync -avP -e "ssh -p 55555" tim@14.103.44.161:~/workspace/deepdive_kai0/kai0/checkpoints/ ./checkpoints/
# 从 gf1:
rsync -avP -e "ssh -p 11111" tim@14.103.44.161:~/workspace/deepdive_kai0/kai0/checkpoints/ ./checkpoints/

# 4. 启动推理 (单卡即可, 仅需 ~8GB)
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_flatten_fold_normal \
  --policy.dir=checkpoints/Task_A/best \
  --port=8000
```

### 4.3 sim01 IPC 环境 (ROS + 相机 + Piper SDK)

> sim01 同时兼任 IPC, 在同一台机器上跑推理 + ROS + 相机 + CAN。
> 用 conda 隔离环境, 避免 Python 版本冲突。

```bash
# === sim01 本机操作 ===
cd /data1/tim/workspace/deepdive_kai0/kai0

# 1. 创建 conda 环境
conda create -n kai0_inference python=3.10 -y
conda activate kai0_inference

# 2. 安装 PyTorch
pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. 安装推理依赖
pip install -r train_deploy_alignment/inference/agilex/requirements_inference_ipc.txt

# 4. 安装 openpi 客户端
cd packages/openpi-client && pip install -e . && cd ../..

# 5. 安装 Piper SDK
pip install piper_sdk python-can

# 6. 安装 ROS Noetic (Ubuntu 20.04) 或用 Docker
# 参考: http://wiki.ros.org/noetic/Installation/Ubuntu
sudo apt install ros-noetic-realsense2-camera

# 7. 编译 Piper ROS 包
cd train_deploy_alignment/inference/agilex/Piper_ros_private-ros-noetic
catkin_make
source devel/setup.bash
```

---

## 五、D405 适配 (最小改动)

### 5.1 核心事实: 不需要改模型代码

D405 和 D435 在软件层面完全兼容:
- 都支持 640×480 RGB8 @ 30fps
- 都用 pyrealsense2 / realsense2_camera ROS 包
- 模型只用 RGB (不用深度)
- 图像最终 resize 到 224×224

### 5.2 需要改的地方

**改动 1: ROS 相机 launch 文件**

创建自己的 `my_multi_camera.launch`:

```xml
<launch>
  <!-- 头顶 D435 -->
  <include file="$(find realsense2_camera)/launch/rs_camera.launch">
    <arg name="camera"       value="camera_f"/>
    <arg name="serial_no"    value="你的D435序列号"/>
    <arg name="color_width"  value="640"/>
    <arg name="color_height" value="480"/>
    <arg name="color_fps"    value="30"/>
    <arg name="enable_depth" value="false"/>
  </include>

  <!-- 左手 D405 -->
  <include file="$(find realsense2_camera)/launch/rs_camera.launch">
    <arg name="camera"       value="camera_l"/>
    <arg name="serial_no"    value="你的左D405序列号"/>
    <arg name="color_width"  value="640"/>
    <arg name="color_height" value="480"/>
    <arg name="color_fps"    value="30"/>
    <arg name="enable_depth" value="false"/>
  </include>

  <!-- 右手 D405 -->
  <include file="$(find realsense2_camera)/launch/rs_camera.launch">
    <arg name="camera"       value="camera_r"/>
    <arg name="serial_no"    value="你的右D405序列号"/>
    <arg name="color_width"  value="640"/>
    <arg name="color_height" value="480"/>
    <arg name="color_fps"    value="30"/>
    <arg name="enable_depth" value="false"/>
  </include>
</launch>
```

获取 serial number:
```bash
rs-enumerate-devices | grep "Serial Number"
```

**改动 2: URDF mesh 路径 (如需 RViz 可视化)**

```bash
# piper_description.urdf 中 mesh 路径硬编码为 /home/agilex/...
# 全局替换为你的路径:
sed -i 's|/home/agilex/cobot_magic/Piper_ros_private-ros-noetic|你的绝对路径|g' \
  train_deploy_alignment/inference/agilex/Piper_ros_private-ros-noetic/src/piper_description/urdf/piper_description.urdf
```

**其他代码不需要改**: ROS topic 名称、图像分辨率、推理参数等全部可用默认值。

---

## 六、分阶段复现计划

### Phase 1: 硬件验证 (1-2 天)

#### Step 1.1: CAN 通信验证

```bash
# sim01 上
cd /data1/tim/workspace/deepdive_kai0/kai0/train_deploy_alignment/dagger/agilex

# 识别 USB-CAN 端口
./find_all_can_port.sh

# 编辑 activate_can_arms.sh, 填入你的 bus-info
# 如果只有 slave 臂 (2个CAN), 只配置 can_left_slave 和 can_right_slave
vim activate_can_arms.sh

# 激活 CAN
./activate_can_arms.sh

# 验证
ip link show | grep can
# 应看到 can_left_slave, can_right_slave UP
```

#### Step 1.2: 相机验证

```bash
# 验证 3 个相机都能识别
realsense-viewer
# 检查 3 个设备都显示, 都能出 640×480 RGB 画面

# ROS 方式验证
roscore &
roslaunch my_multi_camera.launch
# 新终端:
rostopic list | grep camera
# 应看到:
# /camera_f/color/image_raw
# /camera_l/color/image_raw
# /camera_r/color/image_raw

# 可视化
rosrun image_view image_view image:=/camera_f/color/image_raw
```

#### Step 1.3: 机械臂验证

```bash
source devel/setup.bash
roslaunch piper start_ms_piper.launch mode:=0 auto_enable:=true

# 新终端: 检查话题
rostopic echo /puppet/joint_left   # 应看到 7 个关节角度
rostopic echo /puppet/joint_right
```

### Phase 2: 预训练模型推理 (1 天)

#### Step 2.1: 启动 Policy Server (sim01)

```bash
# sim01 上
cd /data1/tim/workspace/deepdive_kai0/kai0

# 下载 Task_A best checkpoint (如果还没下载)
uv run python scripts/download_checkpoints.py --tasks Task_A

# 启动服务 (单卡推理)
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_flatten_fold_normal \
  --policy.dir=checkpoints/Task_A/best \
  --port=8000
```

#### Step 2.2: 启动推理 (sim01, 多个终端)

```bash
# === 全部在 sim01 上操作 ===
REPO=/data1/tim/workspace/deepdive_kai0/kai0

# 终端 1: roscore
roscore

# 终端 2: CAN + 相机
cd $REPO/train_deploy_alignment/dagger/agilex && ./activate_can_arms.sh
roslaunch my_multi_camera.launch

# 终端 3: Piper 臂 (mode:=1 允许外部控制)
source devel/setup.bash
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true

# 终端 4: 推理!! (sim01 本机推理, host 用 localhost)
conda activate kai0_inference
cd $REPO/train_deploy_alignment/inference/agilex/inference
python agilex_inference_openpi_temporal_smoothing.py \
  --host localhost \
  --port 8000 \
  --ctrl_type joint \
  --use_temporal_smoothing \
  --chunk_size 50 \
  --publish_rate 30 \
  --inference_rate 3.0 \
  --latency_k 8 \
  --min_smooth_steps 8
```

#### Step 2.3: 预期结果与调试

**正常表现**: 机器人开始尝试操作桌上的 T 恤, 可能不够精确但有明显的展平/折叠意图。

**常见问题排查**:

| 现象 | 原因 | 解决 |
|------|------|------|
| 机器人完全不动 | CAN 未激活或 mode 设置错误 | 检查 `ip link show`, 确认 mode:=1 |
| 动作剧烈抖动 | 未启用时序平滑 | 确保 `--use_temporal_smoothing` |
| 动作方向错误 | 左右臂 CAN 接反 | 交换 can_left/right 的 bus-info |
| 图像全黑 | 相机 topic 不匹配 | `rostopic hz /camera_f/color/image_raw` |
| 超时无响应 | serve_policy 未启动 | 确认 `CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py` 正在运行 |
| 动作偏移大 | D405 vs D435 视觉域差异 | 正常, Phase 3 微调解决 |

### Phase 3: 数据采集与微调 (3-5 天)

#### Step 3.1: 遥操作采集演示数据

**使用 DAgger 脚本采集** (需要 master 臂):

```bash
# sim01: 已在运行 serve_policy.py (GPU 0)

# sim01 另一个终端:
conda activate kai0_inference
cd /data1/tim/workspace/deepdive_kai0/kai0/train_deploy_alignment/dagger/agilex
python agilex_openpi_dagger_collect.py \
  --host localhost \
  --port 8000 \
  --ctrl_type joint \
  --use_temporal_smoothing \
  --chunk_size 50 \
  --dataset_name my_d405_flatfold_v1
```

键盘操作:
1. 让策略自动推理 (观察效果)
2. 按 **d** 进入 DAgger 模式 (人工接管)
3. 按 **Space** 开始录制
4. 移动 master 臂演示正确动作
5. 按 **s** 保存
6. 按 **r** 恢复自动推理
7. 重复, 目标: 采集 **50-200 条**有效 episode

**如果没有 master 臂**: 使用独立的遥操作采集脚本 `collect_data.py`, 或考虑购买 master 臂 (Piper SDK 支持主从模式)。

#### Step 3.2: 数据转换

```bash
# HDF5 → LeRobot 格式
cd train_deploy_alignment/data_augment/utils
export PYTHONPATH="${PYTHONPATH}:$(pwd)/mini_lerobot"

python convert_h5_lerobot.py \
  /path/to/my_d405_flatfold_v1 \
  /path/to/output \
  my_d405_flatfold_v1 \
  --prompt "fold the cloth" \
  --max-workers 8
```

#### Step 3.3: 数据增强 (可选但推荐)

```bash
# 空间镜像: 将数据量翻倍
python train_deploy_alignment/data_augment/space_mirroring.py full \
  --src-path /path/to/my_lerobot_data \
  --mirror-path /path/to/mirrored \
  --merge-path /path/to/merged \
  --repo-id my_d405_merged
```

#### Step 3.4: 训练 (gf0)

```bash
# === gf0 上执行 ===
cd ~/workspace/deepdive_kai0/kai0

# 0. 从 sim01 传输采集的数据到 gf0
# (在 sim01 上执行 push 方式):
rsync -avP -e "ssh -p 55555" \
  /data1/tim/workspace/deepdive_kai0/kai0/data/Task_A/my_d405/ \
  tim@14.103.44.161:~/workspace/deepdive_kai0/kai0/data/Task_A/my_d405/

# 1. 修改 config.py 中 repo_id 为:
#    ~/workspace/deepdive_kai0/kai0/data/Task_A/my_d405 (绝对路径)

# 2. 计算归一化统计
uv run python scripts/compute_norm_states_fast.py \
  --config-name pi05_flatten_fold_normal

# 3. 训练 (单卡 A100 80GB, 与 kai0 原版完全一致)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_flatten_fold_normal \
  --exp_name=d405_finetune_v1

# 4. 训练完成后传回 sim01 (在 sim01 上执行 pull 方式):
rsync -avP -e "ssh -p 55555" \
  tim@14.103.44.161:~/workspace/deepdive_kai0/kai0/checkpoints/ \
  /data1/tim/workspace/deepdive_kai0/kai0/checkpoints/
```

### Phase 4: 进阶优化 (1-2 周)

#### Step 4.1: Model Arithmetic

训练多个 checkpoint 后合并:

```bash
# 1. 切分数据为 4 份
python model_arithmetic/split_data.py \
  --source_path /path/to/merged_data \
  --dst_path /path/to/splits \
  --split_num 4

# 2. 分别训练 4 个模型 (可以串行, 每个 ~20K steps)

# 3. 导出验证数据
python model_arithmetic/dump_data.py \
  --dataset pi05_flatten_fold_normal \
  --output flatfold_val.pkl

# 4. 合并 (推荐 greedy)
CUDA_VISIBLE_DEVICES=0 python model_arithmetic/arithmetic.py \
  --config pi05_flatten_fold_normal \
  --data-path flatfold_val.pkl \
  --checkpoints ckpt1/80000 ckpt2/80000 ckpt3/80000 ckpt4/80000 \
  --output /path/to/mixed_ckpt \
  --optimize_method greedy \
  --use_gpu --gpu_ids "0"
```

#### Step 4.2: Stage Advantage (AWBC)

**捷径: 直接用 kai0 开源的 Task_A/advantage 数据集**:

```bash
# 1. 下载 advantage 数据 (已包含所有标签)
# Task_A/advantage/ 包含 relative_advantage, absolute_value, absolute_advantage, task_index

# 2. 修改 config.py 中 pi05_flatten_fold_awbc 的 repo_id
# 指向 data/Task_A/advantage

# 3. 计算归一化
uv run python scripts/compute_norm_states_fast.py \
  --config-name pi05_flatten_fold_awbc

# 4. AWBC 训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_flatten_fold_awbc \
  --exp_name=awbc_v1

# 5. 推理时使用 positive prompt
# 在推理脚本中设置:
# lang_embeddings = "fold the cloth, Advantage: positive"
```

**如果要标注自己的数据 (完整流水线)**:

```
Step 0: 人工标注 stage_progress_gt (工具待自建)
Step 1: uv run python scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD --exp_name=run1
Step 2: uv run python stage_advantage/annotation/eval.py Task-A KAI0 /path/to/dataset
Step 3: python stage_advantage/annotation/discretize_advantage.py <dataset> --threshold 30
Step 4: AWBC 训练 (同上)
```

---

## 七、关键配置参数速查

### 7.1 训练配置 (config.py, 在 gf0/gf1 上)

```python
# 需要修改的参数 (仅路径, 其他不改):
TrainConfig(
    name="pi05_flatten_fold_normal",
    data=LerobotAgilexDataConfig(
        # gf0/gf1 上的绝对路径:
        repo_id="/home/tim/workspace/deepdive_kai0/kai0/data/Task_A/base",  # ← 改这里
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/home/tim/workspace/deepdive_kai0/kai0/checkpoints/Task_A/best"    # ← 改这里
    ),
    # A100 训练: 与 kai0 原版完全一致, 不需改!
    batch_size=256,         # 原版, 不改
    fsdp_devices=1,         # 单卡 A100 80GB, 不改
    num_train_steps=100_000,
)
```

> 注: 如果在 sim01 (双 5090) 上训练, 需要调整:
> `fsdp_devices=2, batch_size=128` (5090 32GB 显存不如 A100 80GB)

### 7.2 推理参数

```bash
python agilex_inference_openpi_temporal_smoothing.py \
  --host localhost      # sim01 本机推理, 用 localhost
  --port 8000          # Policy Server 端口
  --ctrl_type joint    # 关节空间控制 (不要改成 eef)
  --chunk_size 50      # 动作块长度
  --publish_rate 30    # 控制发布频率 Hz
  --use_temporal_smoothing  # 启用时序平滑 (必须!)
  --inference_rate 3.0      # 推理线程频率 Hz
  --latency_k 8             # 延迟补偿步数
  --min_smooth_steps 8      # 最小平滑重叠
  --exp_decay_alpha 0.25    # 指数衰减权重
```

### 7.3 ROS Topic 映射

```
相机 RGB:
  /camera_f/color/image_raw  → top_head (D435)
  /camera_l/color/image_raw  → hand_left (D405 左)
  /camera_r/color/image_raw  → hand_right (D405 右)

机械臂:
  /puppet/joint_left   → 左臂状态反馈
  /puppet/joint_right  → 右臂状态反馈
  /master/joint_left   → 左臂控制命令
  /master/joint_right  → 右臂控制命令
```

### 7.4 Prompt 设置

```python
# 普通模型:
lang_embeddings = "fold the cloth"

# AWBC 模型 (必须包含 advantage 标签):
lang_embeddings = "fold the cloth, Advantage: positive"
```

---

## 八、与 kai0 原版的差异总结

| 差异点 | kai0 原版 | 当前配置 | 影响 | 处理方式 |
|--------|----------|---------|------|---------|
| 腕部相机 | D435i | **D405 (Cobot Magic 原装)** | FOV 和色彩差异 | 采集少量数据微调 |
| 头部相机 | D435i | D435 | 仅少 IMU, 无差异 | 不需处理 |
| 腕部安装 | 3D 打印件 (D435i) | **ALOHA 一体式 + 可调高度** | 光心位置不同 | 实际更优, 微调解决 |
| 夹爪 | 3D 打印 agilex-gripper | **ALOHA 原装夹爪** | 形状微差 | 直接用, 效果不好再换 |
| 训练 GPU | 8×A100 80GB | **gf0 + gf1 各 8×A100** | **完全一致, 甚至更强** | **不需任何改动** |
| 推理 GPU | RTX 4090 | **sim01 双 RTX 5090** | 更强 | 不需处理 |
| IPC 架构 | IPC + GPU 主机分离 | **sim01 一体机** | 零网络延迟 | host 用 localhost |
| batch_size | 256 | **256 (不需改!)** | **零差异** | 原版配置直接用 |
| config.py | 原版 | **原版, 仅改路径** | **零差异** | 只改 repo_id 和 weight_loader |

**核心结论**: 训练侧与 kai0 原版**完全一致** (gf0/gf1 同为 8×A100, batch_size=256, 无需任何妥协)。
sim01 兼任 IPC + 推理机, 推理延迟比原版 IPC+GPU 双机方案更低 (localhost vs 千兆网)。
唯一差异是 D405 vs D435i 的视觉域差距, 通过采集少量自己的数据微调即可解决。

---

## 九、时间线总结

```
Day 1-2:  硬件搭建 [sim01]
  ├── 按尺寸图放置双臂 Piper + ALOHA 夹爪/D405 已就位
  ├── D435 头顶相机安装 (76cm高, 30°俯角)
  ├── CAN 接线 + USB 接线 (全部接 sim01)
  └── 验证: CAN 通信 + 3路相机出图 + 臂控制

Day 3:    软件环境 (三台机器并行)
  ├── [gf0] ssh -p 55555 登录 → uv sync + 下载数据/checkpoint
  ├── [gf1] ssh -p 11111 登录 → uv sync + 下载数据/checkpoint
  └── [sim01] uv sync + conda kai0_inference + ROS + openpi-client

Day 4:    首次推理 [sim01]
  ├── serve_policy.py 启动 (GPU 0, :8000)
  ├── 多终端流程跑通 (roscore + CAN + 相机 + Piper + 推理)
  └── 推理 host=localhost, 观察机器人动作

Day 5-7:  数据采集 [sim01]
  ├── DAgger 采集 50-200 条 episode
  ├── HDF5 → LeRobot 转换
  ├── 空间镜像增强
  └── rsync -e "ssh -p 55555" 数据到 gf0

Day 8-10: 微调训练 [gf0], 与原版完全一致
  ├── compute_norm_stats
  ├── 单卡 A100 Full 微调 (batch=256, 与 kai0 原版相同)
  ├── rsync -e "ssh -p 55555" checkpoint 回 sim01
  └── [sim01] 推理验证效果

Day 11+:  进阶优化 [gf0 + gf1 双节点并行]
  ├── [gf0] Model Arithmetic (4个数据子集并行训练)
  ├── [gf1] Advantage 估计器训练 + AWBC
  ├── [sim01] 更多 DAgger 迭代
  └── 调优 temporal smoothing 参数
```
