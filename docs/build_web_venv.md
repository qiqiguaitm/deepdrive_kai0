# 构建 data_manager 后端 venv

本文说明如何在 sim01 上正确构建 `web/data_manager/backend/.venv`，使后端能 `import rclpy` 并连通真实 ROS2 / CAN / 相机。

> **看 UI 上 CAN / 相机 / teleop 全红？** —— 基本就是 venv 解释器版本不对,
> backend 启动时静默回落到 `MockBridge`。读完本文按任一路径重建即可。

## 1. 为什么 venv 必须跑 Python 3.12

- 后端 `ros_bridge.py` 启动时 `import rclpy`，失败则回落 `MockBridge`，其 `get_health()` 恒返回 `{ros2:False, can_*:False, teleop:False}`，UI 表现为 "服务在跑但红灯闪"。
- ROS2 Jazzy **只提供 Python 3.12 的 rclpy C 扩展**:
  ```
  /opt/ros/jazzy/lib/python3.12/site-packages/rclpy/_rclpy_pybind11.cpython-312-x86_64-linux-gnu.so
  ```
  任何非 3.12 版本 `import rclpy._rclpy_pybind11` 都会报
  `No module named 'rclpy._rclpy_pybind11'`，然后 `ros_bridge._make_bridge()` catch 住异常 → Mock。
- 所以: `.venv/bin/python -V` 必须打印 `Python 3.12.x`，否则整条链路都是假的。

## 2. sim01 的三个坑

1. 默认 `python3` 是 miniconda 的 3.13 (`/data1/miniconda3/bin/python3`) —— **版本不对**。
2. `/usr/bin/python3.12` 存在，但系统 **没装** `python3.12-venv`，直接
   `python3.12 -m venv .venv` 会报 `ensurepip is not available`。
3. 不一定有 sudo 装 `python3.12-venv`。

因此下面给出两条路径：

- **A** — 有 sudo，最干净。
- **B** — 无 sudo，用 "miniconda 3.13 建骨架 + 符号链接指 3.12 + `get-pip.py` 装 pip" 的 workaround（`/data1/gzllll/deepdive_kai0` 的 venv 就是这样搭的，实测可用）。

## 3. 路径 A —— 有 sudo

```bash
sudo apt install -y python3.12-venv

cd /data1/tim/workspace/deepdive_kai0/web/data_manager/backend
rm -rf .venv
/usr/bin/python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

完成后直接跳到 §5 验证。

## 4. 路径 B —— 无 sudo (workaround)

原理: `venv` 模块要求 Python 自带 `ensurepip`，3.12 缺它我们就用 3.13 先搭骨架，再把 `bin/python` 换成 3.12，最后用 `get-pip.py` 给 3.12 引导 pip。这样:

- `pyvenv.cfg` 里 `version = 3.13.x`（骨架来源不改）
- `.venv/bin/python -V` → `Python 3.12.x`（实际跑的解释器）
- 依赖装进 `lib/python3.12/site-packages/`（pip 会按当前解释器版本选路径）

```bash
cd /data1/tim/workspace/deepdive_kai0/web/data_manager/backend
rm -rf .venv

# 1) 用 miniconda 3.13 搭 venv 骨架 (activate / pyvenv.cfg / include / ...)
/data1/miniconda3/bin/python3 -m venv .venv

# 2) 把 python 符号链接全部指向系统 3.12
rm .venv/bin/python .venv/bin/python3 .venv/bin/python3.13
ln -sf /usr/bin/python3.12 .venv/bin/python
ln -sf python               .venv/bin/python3
ln -sf python               .venv/bin/python3.12
ln -sf python               .venv/bin/python3.13   # 有些工具仍会找 python3.13, 指回本 venv 的 3.12

# 3) 给 3.12 引导 pip (ensurepip 不可用, 只能用官方 bootstrap)
curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py

# 4) 装依赖 —— pip 这时已经以 3.12 身份跑, 会写入 lib/python3.12/site-packages
.venv/bin/pip install -r requirements.txt
```

## 5. 验证

### 5.1 venv 本身

```bash
cd /data1/tim/workspace/deepdive_kai0/web/data_manager/backend
.venv/bin/python -V                       # => Python 3.12.x
.venv/bin/pip --version                   # => pip ... from .../lib/python3.12/site-packages/pip (python 3.12)
```

### 5.2 rclpy 能否导入 (关键步骤)

```bash
source /opt/ros/jazzy/setup.bash
source /data1/tim/workspace/deepdive_kai0/ros2_ws/install/setup.bash
.venv/bin/python -c "import rclpy; print(rclpy.__file__)"
# => /opt/ros/jazzy/lib/python3.12/site-packages/rclpy/__init__.py
```

**必须看到上面这行**。如果仍然 `No module named 'rclpy._rclpy_pybind11'`，
大概率是 `.venv/bin/python` 没真的指向 3.12 —— `readlink -f .venv/bin/python` 检查一下。

### 5.3 端到端

```bash
cd /data1/tim/workspace/deepdive_kai0/web/data_manager
./run.sh stop
./run.sh start
grep ros_bridge logs/backend.log
# 期望:
#   [ros_bridge] RclpyBridge online
# 绝对不能出现:
#   [ros_bridge] rclpy unavailable (...) ; using MockBridge
```

UI (admin 角色) 右下角大牌应该变绿 (`ALL OK`)，`CAN 左/右`、`teleop`、三路
RealSense fps 全部亮绿；`/api/joints` 返回真实零位/夹爪数据（而不是 mock 的正弦波）。

## 6. FAQ / 常见坑

**Q: `pip install` 报 numpy / pyarrow 找不到 wheel？**
A: 3.12 wheels 对这些包都齐全，大概率是 `.venv/bin/python` 还是 3.13，
再跑一次 `.venv/bin/python -V` 确认；若是 3.13 说明第 2 步符号链接没成功，重做。

**Q: 启动后 `logs/backend.log` 报 `failed to start RclpyBridge (...)`？**
A: 说明 rclpy 本身能 import，但初始化节点失败（通常是 `RMW_IMPLEMENTATION`
和 `ROS_DOMAIN_ID` 环境缺失）。`run.sh` 已经在启动命令里 source 两份 setup.bash,
所以更可能是 `ros2_ws` 没 build，先在仓库根跑 `colcon build`。

**Q: 以后升级依赖？**
A: 直接 `.venv/bin/pip install --upgrade <pkg>` 即可，不用重建 venv。重建只在
换 Python 版本 / 损坏时才需要。

**Q: 能不能直接 copy gzllll 的 .venv 过来？**
A: 不行，里面所有可执行 / 脚本写死了绝对路径 `/data1/gzllll/...`，换目录后 `activate`、`pip` 等会找不到 python。

## 7. 未来清理建议

长期把 workaround 清掉的最简办法:
```bash
sudo apt install -y python3.12-venv
```
之后 `install.sh` 里可以直接 `python3.12 -m venv ...`, README / 本文 §4 就可以删掉。
