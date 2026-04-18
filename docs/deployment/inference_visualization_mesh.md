# 在线推理可视化：点云 → Mesh 化升级方案

> 日期: 2026-04-05
> 相关文件: `ros2_ws/src/piper/scripts/rerun_viz_node.py`, `ros2_ws/src/piper/launch/autonomy_launch.py`
> 依赖新增: `open3d`（tensor API，CUDA 后端）
> 前置文档: `docs/deployment/inference_visualization.md`（rerun 基础集成）

---

## 1. 背景与目标

### 1.1 现状问题
当前 `rerun_viz_node.py` 的 `_tick_point_clouds()` 以 5 Hz 把三路 RealSense 深度反投影为
`rr.Points3D` 后显示。实测存在以下问题：

1. **抖动闪烁**
   - depth / rgb / joint_state 三个回调各自独立缓存"最新一帧"，tick 时取 latest_of_each，
     不做时间配对。
   - 腕部相机的 FK 使用"当前"`_latest_q_*`，而 depth 本身可能是几十毫秒前采集的 →
     手臂移动时点云"甩出去"。
   - RealSense 原始 depth 的 mm 级量化噪声未经时间滤波，静态物体也有沙粒感。
   - RGB/depth 分辨率不一致时 `colors=None`，在"有色/无色"之间跳变。

2. **观感粗糙**
   - 离散点 + 固定 `radii=0.001` 形成"颗粒感"，不够接近 Isaac Sim viewport 的连续表面观感。

### 1.2 目标
把可视化从"散点云"升级为"全场景 mesh"，具体：

- **静态背景**（桌面、支架、远景）→ TSDF 融合一次成型，常驻显示；
- **动态前景**（布料、双臂、手腕近景）→ 每 tick 按相机做 screen-space 三角化；
- 两者通过 **workspace bbox** 互斥切分，无重叠、无拖影。
- 保持 **5 Hz** 刷新率、**640×480 step=1** 分辨率。
- 视觉稳定性接近 Isaac Sim viewport，同时不牺牲动态跟随性。

---

## 2. 总体架构

```
        depth_head ─┐                    ┌─► static/background (TSDF mesh, 常驻)
        rgb_head   ─┼── bbox 切分 ──────┤
                    │                    └─► dynamic/head  (screen-space mesh, 5 Hz)
depth_left + rgb_left + q_left(插值)  ──────► dynamic/left  (screen-space mesh, 5 Hz)
depth_right+ rgb_right+ q_right(插值) ──────► dynamic/right (screen-space mesh, 5 Hz)
```

Rerun entity 层次：

```
world/
├── bg_mesh                        # rr.Mesh3D, static=True（全时间轴常驻）
└── dynamic/
    ├── head_fg_mesh               # rr.Mesh3D, 每 tick 覆写
    ├── left_fg_mesh               # rr.Mesh3D, 每 tick 覆写
    └── right_fg_mesh              # rr.Mesh3D, 每 tick 覆写
```

关键设计点：

| # | 决策 | 理由 |
|---|---|---|
| 1 | 背景用 TSDF 融合、前景用 screen-space | 背景可以靠多帧平均压噪声；前景必须每帧独立避免拖影 |
| 2 | 动静分离用 world-frame AABB | 向量化，无需 arm-mask / nearest-neighbor，~2 ms |
| 3 | 头部相机同时出前景 + 背景 | 头部视野覆盖整个工作区，天然适合做背景源 |
| 4 | 腕部相机只出前景 | 近距高动态，无背景贡献；且姿态依赖 FK，不适合做累积融合 |
| 5 | 三角形按深度跳变阈值裁剪 | 消除遮挡边界的"橡皮膜"伪面 |

---

## 3. 前置修复：时间同步（必须先做）

点云 mesh 化后，任何 FK / 颜色错位都会变成"整块面片错位"，比散点更刺眼。第一步必须把
三路缓存改成按时间戳对齐。

### 3.1 环形缓冲
```python
# __init__ 新增
from collections import deque
self._q_left_buf    = deque(maxlen=200)   # (stamp, q[7])
self._q_right_buf   = deque(maxlen=200)
self._rgb_front_buf = deque(maxlen=10)    # (stamp, rgb_hwc)
self._rgb_left_buf  = deque(maxlen=10)
self._rgb_right_buf = deque(maxlen=10)
```

- `_cb_joint_*` 里：`self._q_*_buf.append((stamp, q))`（`_latest_q_*` 仍保留给 FK 可视化节点兼容）
- `_cb_*_image` 里：`self._rgb_*_buf.append((stamp, rgb))`

### 3.2 取值工具
```python
def _interp_q(buf, t, max_gap=0.1):
    """按 t 在 buf 中二分 + 线性插值；超出范围/gap 过大返回 None。"""

def _nearest_rgb(buf, t, tol=0.03):
    """返回 |stamp - t| ≤ tol 的 RGB，找不到返回 None（不要用错位帧）。"""
```

### 3.3 tick 中的使用
- 头部相机：`rgb = _nearest_rgb(self._rgb_front_buf, depth_stamp)`。
- 腕部相机：`q = _interp_q(self._q_left_buf, depth_stamp)`，然后才做 FK。
- 任一项返回 None → 该相机这一 tick 跳过，不要用错位数据兜底。

> 这一步即使不做 mesh 化也会显著降低现有 Points3D 的抖动；强烈建议作为独立 commit 先合入。

---

## 4. Workspace bbox（动静分离的核心）

在 `world` 坐标系下定义前景 AABB，覆盖"所有可能动"的区域（桌面上方 + 双臂工作区）：

```python
self.declare_parameter('fg_bbox_min', [-0.6, -0.6, 0.00])   # x, y, z (m)
self.declare_parameter('fg_bbox_max', [ 0.6,  0.6, 0.80])
```

- bbox **内** → 动态前景，每 tick 重新 mesh 化
- bbox **外** → 静态背景的一部分，只重建一次
- 桌面正好是 z 下边界；远处墙壁 / 支架 / 吊顶自动归入静态
- bbox 扩展 margin（默认 5 cm）给 TSDF integrate 用，避免布料/手臂污染背景

纯向量化 bbox 测试 ~2 ms，不需要 arm-mask 或 nearest-neighbor。

---

## 5. 静态背景构建（TSDF，Open3D CUDA）

### 5.1 触发方式
提供一个 ROS2 service，按需重建：

```python
from std_srvs.srv import Trigger
self.create_service(Trigger, '/rerun_viz/rebuild_bg', self._srv_rebuild_bg)
```

节点启动时若存在缓存文件 `~/.cache/kai0_viz/bg_mesh.ply` → 直接加载并 log 为 static；
否则用户摆好静态场景（双臂 rest pose）后调用 service 触发一次重建。

### 5.2 重建流程（只用 head camera）
```
1. 收集 N=30 帧 (depth_u16, rgb) from head 相机（6 s @ 5 Hz）
2. 创建 o3d.t.geometry.VoxelBlockGrid:
       attr_names   = ('tsdf', 'weight', 'color')
       attr_dtypes  = (float32, float32, float32)
       attr_channels= ((1), (1), (3))
       voxel_size   = 0.005         # 5 mm
       block_resolution = 16
       block_count  = 50000         # ~1.4 GB 上限
       device       = CUDA:0
3. 对每帧：
       depth_t = o3d.t.geometry.Image(Tensor(depth_u16, device=CUDA:0))
       color_t = o3d.t.geometry.Image(Tensor(rgb,       device=CUDA:0))
       frustum = vbg.compute_unique_block_coordinates(
                     depth_t, K, inv(T_world_camF),
                     depth_scale=1000.0, depth_max=2.0)
       vbg.integrate(frustum, depth_t, color_t, K, inv(T_world_camF),
                     depth_scale=1000.0, depth_max=2.0)
4. mesh = vbg.extract_triangle_mesh().to_legacy()
5. 把 fg_bbox（外扩 margin）内部的顶点裁掉，保证前景区域由动态 mesh 负责
6. mesh.compute_vertex_normals()
7. 存盘 bg_cache_path（.ply）
8. log 到 rerun:
       rr.log("world/bg_mesh",
              rr.Mesh3D(vertex_positions, triangle_indices,
                        vertex_colors, vertex_normals),
              static=True)
9. del vbg; o3d.core.cuda.release_cache()
```

### 5.3 参数
| 参数 | 默认 | 说明 |
|---|---|---|
| `bg_voxel_size` | 0.005 | TSDF 体素（5 mm）|
| `bg_sdf_trunc`  | 0.04  | 截断距离 |
| `bg_depth_trunc`| 2.0   | 深度上限（m）|
| `bg_num_frames` | 30    | 积分帧数 |
| `bg_bbox_margin`| 0.05  | fg_bbox 外扩裁剪 |
| `bg_cache_path` | `~/.cache/kai0_viz/bg_mesh.ply` | 持久化路径 |

### 5.4 耗时
GPU tensor API：积分 30 帧 ~150 ms + extract ~50 ms ≈ **0.2 s 总时长**。
一次性操作，不在 5 Hz tick 里，完全可接受。

---

## 6. 动态前景 mesh（screen-space，每 tick）

替换现有 `_tick_point_clouds` → `_tick_meshes`。三路相机走同一条管线。

### 6.1 单相机函数
```python
def _build_fg_mesh(self, depth_u16, rgb, fx, fy, cx, cy,
                   T_world_cam, bbox_min, bbox_max,
                   depth_min=0.05, depth_max=1.5, edge_thresh=0.02):
    # 1) JAX 反投影（复用现有 _jax_project_fn, step=1）
    pts_world, depth_m = _jax_project_fn(
        depth_u16, fx, fy, cx, cy, 1, T_world_cam.astype(np.float32))
    pts_world = np.asarray(pts_world)           # (H, W, 3)
    depth_m   = np.asarray(depth_m)             # (H, W)
    H, W = depth_m.shape

    # 2) 像素有效性：深度范围 + workspace bbox
    valid = (depth_m > depth_min) & (depth_m < depth_max)
    xyz = pts_world
    valid &= ((xyz[..., 0] >= bbox_min[0]) & (xyz[..., 0] <= bbox_max[0]) &
              (xyz[..., 1] >= bbox_min[1]) & (xyz[..., 1] <= bbox_max[1]) &
              (xyz[..., 2] >= bbox_min[2]) & (xyz[..., 2] <= bbox_max[2]))
    if not valid.any():
        return None

    # 3) 四邻接 quad 有效性 + 深度跳变剔除（消除遮挡边界伪面）
    m = valid
    vq = m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]         # (H-1, W-1)
    d = depth_m
    dstack = np.stack([d[:-1, :-1], d[1:, :-1], d[:-1, 1:], d[1:, 1:]])
    vq &= (dstack.max(0) - dstack.min(0)) < edge_thresh

    if not vq.any():
        return None

    # 4) 生成三角形索引（每个有效 quad 出 2 个三角形）
    ii, jj = np.nonzero(vq)
    v00 = ii * W + jj
    v10 = (ii + 1) * W + jj
    v01 = ii * W + (jj + 1)
    v11 = (ii + 1) * W + (jj + 1)
    tris = np.concatenate([
        np.stack([v00, v10, v11], axis=-1),
        np.stack([v00, v11, v01], axis=-1)], axis=0).astype(np.int32)

    # 5) 顶点压缩：只保留被三角形引用的顶点
    used, inverse = np.unique(tris, return_inverse=True)
    tris_c = inverse.reshape(-1, 3).astype(np.int32)
    verts  = pts_world.reshape(-1, 3)[used]
    colors = (rgb.reshape(-1, 3)[used]
              if rgb is not None and rgb.shape[:2] == (H, W) else None)

    return verts, tris_c, colors
```

### 6.2 tick 逻辑
```python
def _tick_meshes(self):
    # 头部：固定 T_world_camF，RGB 按时间戳取
    if self._latest_depth_front is not None and self._T_world_camF is not None:
        depth_u16, t_dep = self._latest_depth_front
        rgb = self._nearest_rgb(self._rgb_front_buf, t_dep)
        self._log_mesh('head', depth_u16, t_dep, rgb,
                       self._cam_f_fx, self._cam_f_fy,
                       self._cam_f_cx, self._cam_f_cy,
                       self._T_world_camF)

    # 腕部：q 插值到 depth_stamp，再做 FK
    for side in ('left', 'right'):
        dep_cache = getattr(self, f'_latest_depth_{side}')
        if dep_cache is None:
            continue
        depth_u16, t_dep = dep_cache
        q = self._interp_q(getattr(self, f'_q_{side}_buf'), t_dep)
        if q is None:
            continue
        T_base   = getattr(self, f'_T_world_base{side[0].upper()}')
        T_l6_cam = getattr(self, f'_T_link6_cam{side[0].upper()}')
        if T_base is None or T_l6_cam is None:
            continue
        T_base_ee = self._fk.fk_homogeneous(q[:6])
        T_wc = np.array(T_base) @ T_base_ee @ np.array(T_l6_cam)
        rgb = self._nearest_rgb(getattr(self, f'_rgb_{side}_buf'), t_dep)
        fx, fy, cx, cy = self._intrinsics(side)
        self._log_mesh(side, depth_u16, t_dep, rgb, fx, fy, cx, cy, T_wc)

def _log_mesh(self, name, depth_u16, t_dep, rgb, fx, fy, cx, cy, T_wc):
    result = self._build_fg_mesh(
        depth_u16, rgb, fx, fy, cx, cy, T_wc,
        self._bbox_min, self._bbox_max,
        depth_min=0.05, depth_max=1.5,
        edge_thresh=self._edge_thresh)
    if result is None:
        return
    verts, tris, colors = result
    self._rr.set_time("ros_time", timestamp=t_dep)
    self._rr.log(f"world/dynamic/{name}_fg_mesh",
                 rr.Mesh3D(vertex_positions=verts,
                           triangle_indices=tris,
                           vertex_colors=colors))
```

### 6.3 定时器
- `_tick_meshes`：5 Hz，放在独立 `MutuallyExclusiveCallbackGroup`
- 原 `_tick_point_clouds` 可通过 `fg_enable` 参数切换（false 时退回 Points3D，便于 A/B 对比）
- Node executor 用 `MultiThreadedExecutor`，允许 mesh tick 与相机 / joint 回调并发

---

## 7. GPU 分配与进程隔离

### 7.1 sim01 双卡现状（Step 6 实施时核对修正）
- `policy_inference_node.py:445` 通过 ROS 参数 `gpu_id` 设置 `CUDA_VISIBLE_DEVICES`；
  `autonomy_launch.py:102` 将 `gpu_id` 默认值设为 `0` → policy 节点跑在 **GPU0**。
- `rerun_viz_node.py:65` 通过 `os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')`
  把 viz 节点固定到 **GPU1**（注释里说明是历史上解决"同卡双 JAX 上下文首次调用段错误"
  的 workaround）。
- `scripts/start_autonomy.sh:155` 只是 websocket 模式下 serve_policy 未运行时的
  *提示文本*，不实际启动进程；用户在 websocket 模式下手动在 GPU1 起 serve_policy。
- 因此 **ros2/both 模式**：policy=GPU0，viz=GPU1，零冲突。**websocket 模式**：
  viz 和 serve_policy 共用 GPU1，靠 viz 侧的 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.20`
  \+ `PREALLOCATE=false` 避免 JAX 抢空间。

### 7.2 Step 6 落地的改动
把 viz 节点的 GPU 绑定从 "setdefault 隐式默认" 升级为 "launch 文件显式声明"，
避免依赖外层 shell 环境，并留出 Open3D 的 CUDA caching allocator 空间：

```python
rerun_node = Node(
    package='piper', executable='rerun_viz_node.py',
    name='rerun_viz', output='screen',
    condition=IfCondition(LaunchConfiguration('enable_rerun')),
    parameters=[{
        ...,
        'fg_enable': LaunchConfiguration('fg_enable'),
        'bg_enable': LaunchConfiguration('bg_enable'),
    }],
    additional_env={
        'CUDA_VISIBLE_DEVICES': '1',              # viz 独占 GPU1（policy=GPU0）
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', # 给 Open3D 留 CUDA 空间
        'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.20', # JAX 硬上限 20%
    },
)
```

- `rerun_viz_node.py:65` 的 `setdefault` 保留作为 `ros2 run` 独立启动时的兜底，
  和 launch 侧保持同值。
- Open3D 侧使用 `o3d.core.Device("CUDA:0")`：由于 `CUDA_VISIBLE_DEVICES=1`
  重映射，这里的 `CUDA:0` 实际指向物理 GPU1。`_bg_builder.py` 里不需要改动。
- 背景重建完毕后：`del vbg; o3c.cuda.release_cache()` 把 Open3D caching allocator
  释放，避免长期占用显存影响后续动态 tick 的 JAX 投影。
- websocket 模式下如果 serve_policy 也在 GPU1，两个 JAX 进程共享同卡仍有风险；
  若要完全隔离，可以把 serve_policy 改到 GPU0（需要评估是否和 policy_inference
  的默认 `gpu_id=0` 冲突，或者切到 ros2/both 模式不跑 serve_policy）。

### 7.3 为什么不走 CPU legacy TSDF
- CPU `ScalableTSDFVolume` 30 帧积分 + extract ≈ 2 s；
- GPU tensor API 同样任务 ≈ 0.2 s；
- 未来若想扩展到"慢速持续更新"（每秒 integrate 一次），CPU 版会和 5 Hz dynamic tick
  抢 Python GIL，GPU 版则完全走旁路。
- 一次性引入 tensor API 的复杂度很低（~30 行代码），收益明确，推荐直接上 GPU。

---

## 8. 端到端时延预算（5 Hz = 200 ms/tick）

| 阶段 | 单相机 | 3 相机合计 |
|---|---|---|
| JAX depth → world points（已缓存编译） | ~5 ms | ~15 ms |
| bbox + edge quad mask | ~10 ms | ~30 ms |
| 三角索引生成 + 顶点压缩（`np.unique`） | ~15 ms | ~45 ms |
| `rr.log Mesh3D` 序列化 | ~20 ms | ~60 ms |
| **单 tick 合计** | ~50 ms | **~150 ms** |

余量 ~50 ms 留给 FK、时间同步查找、joint/image log 等现有逻辑。

TSDF 重建：一次性 ~200 ms，不在 tick 内。

Rerun 数据量：有效三角形通常远少于理论上限 `2×639×479=612k`，实测多在 150k 左右/相机，
三路合计 ~450k tris × (12 B pos + 12 B normal + 4 B color) ≈ 12 MB/tick → 60 MB/s。
本机 rerun viewer 通过 gRPC over localhost 可承受。

若实测超预算，按优先级优化：
1. `np.unique` 换为 `bincount`-based 压缩（省 ~5 ms/相机）
2. 三路相机用 `ThreadPoolExecutor` 并发（`rr.log` 释放 GIL，串→并省 ~40%）
3. 腕部相机降到 step=2，头部保持 step=1（近距降采样肉眼几乎无差）

---

## 9. 参数清单（全部通过 ROS2 参数暴露）

### 9.1 Mesh 质量
| 参数 | 默认 | 说明 |
|---|---|---|
| `fg_enable` | true | 关闭时退回旧的 `rr.Points3D` 路径 |
| `bg_enable` | true | 关闭时完全不显示背景 |
| `fg_edge_thresh_m` | 0.02 | quad 深度跳变阈值，遮挡边界用 |
| `fg_depth_min` | 0.05 | 有效深度下限 (m) |
| `fg_depth_max` | 1.5  | 有效深度上限 (m) |
| `fg_wrist_step` | 1 | 腕部相机降采样步长（性能兜底）|

### 9.2 Workspace bbox
| 参数 | 默认 | 说明 |
|---|---|---|
| `fg_bbox_min` | `[-0.6, -0.6, 0.00]` | 前景 AABB 下界 (world frame) |
| `fg_bbox_max` | `[ 0.6,  0.6, 0.80]` | 前景 AABB 上界 (world frame) |
| `bg_bbox_margin` | 0.05 | TSDF mask 外扩 |

### 9.3 TSDF
| 参数 | 默认 | 说明 |
|---|---|---|
| `bg_voxel_size` | 0.005 | 5 mm |
| `bg_sdf_trunc` | 0.04 | 4 cm |
| `bg_depth_trunc` | 2.0 | m |
| `bg_num_frames` | 30 | 积分帧数 |
| `bg_cache_path` | `~/.cache/kai0_viz/bg_mesh.ply` | 持久化路径 |
| `bg_device` | `CUDA:0` | Open3D tensor 后端 |

### 9.4 时间同步
| 参数 | 默认 | 说明 |
|---|---|---|
| `sync_rgb_tol_s` | 0.03 | RGB-depth 配对容差 |
| `sync_joint_max_gap_s` | 0.1 | joint 插值最大空隙 |

---

## 10. 代码改动清单

| 文件 | 改动 |
|---|---|
| `ros2_ws/src/piper/scripts/rerun_viz_node.py` | 加环形缓冲、`_interp_q` / `_nearest_rgb`、`_build_fg_mesh`、`_tick_meshes`、`_srv_rebuild_bg`、新参数；保留 `_tick_point_clouds` 走 `fg_enable=false` 分支便于 A/B |
| `ros2_ws/src/piper/scripts/_bg_builder.py`（新建）| Open3D VoxelBlockGrid 封装：`build_bg_mesh_gpu(frames, K, Tc2w, bbox_exclude_min, bbox_exclude_max) -> o3d.geometry.TriangleMesh` |
| `ros2_ws/src/piper/launch/autonomy_launch.py` | `rerun_node` 加 `additional_env={'CUDA_VISIBLE_DEVICES': '0', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'}`；新参数透传 |
| `kai0/pyproject.toml` 或 viz venv | 加 `open3d>=0.18` 依赖 |

不需要改 `kai0/src/openpi/` 主工程，不影响训练 / 推理 / policy server。

---

## 11. 实施顺序（建议分 commit）

1. **Step 0 — 时间同步修复**
   环形缓冲 + 插值 + 最近邻 RGB 配对。替换现有 `_tick_point_clouds` 里的
   `self._latest_*` 直接取值。**独立收益**：不做 mesh 化也能显著降低现有 Points3D 抖动。

2. **Step 1 — `_build_fg_mesh` 基础实现**
   先只跑 head 一路，`rr.log` 时仍用 `rr.Points3D` 显示 `verts`，验证几何正确。

3. **Step 2 — 切换到 `rr.Mesh3D`**
   添加三角索引 + 顶点压缩 + `vertex_colors`。此时已能看到连续表面，无 shading。

4. **Step 3 — bbox 分割**
   加 workspace bbox 裁剪，头部只留动态区域，确认背景位置符合预期（会出现"空洞"）。

5. **Step 4 — 三路全开**
   接入腕部相机（插值后的 FK），测单 tick 耗时：
   ```python
   t0 = time.perf_counter()
   ...
   self.get_logger().info(f'mesh tick {(time.perf_counter()-t0)*1000:.1f} ms')
   ```
   超预算时按 §8 优化路径依次应用。

6. **Step 5 — TSDF 背景**
   新建 `_bg_builder.py`，实现 `build_bg_mesh_gpu`；加 `_srv_rebuild_bg` service +
   cache 加载逻辑。摆 rest pose 调用一次，确认 `world/bg_mesh` 稳定、bbox 内空洞正确。

7. **Step 6 — Launch 环境隔离**
   `autonomy_launch.py` 的 `rerun_node` 加 `additional_env`；确认 `nvidia-smi`
   下 viz node 只出现在 GPU0、policy server 只出现在 GPU1。

8. **Step 7 — 现场调参**
   `edge_thresh`（0.01 / 0.02 / 0.04）、`fg_bbox`、`bg_voxel_size`；保存一份
   `config/rerun_viz.yaml` 作为现场默认值。

---

## 12. 风险与回退

| 风险 | 表现 | 缓解 |
|---|---|---|
| Rerun Mesh3D 序列化超预算 | 单 tick > 180 ms，viewer 卡顿 | 先 bbox 裁剪（真实三角 < 150k/相机）；兜底 `fg_wrist_step=2`；再极端则 `ThreadPoolExecutor` 三路并发 |
| Open3D CUDA 与 JAX 同卡 OOM | 节点启动或 TSDF 首次 integrate 时 `cuMemAlloc` 失败 | `CUDA_VISIBLE_DEVICES=0` + `XLA_PYTHON_CLIENT_PREALLOCATE=false` + `MEM_FRACTION=0.35`；extract 后 `release_cache()` |
| `open3d` 装不上 / 版本不兼容 5090 | import 失败 | 运行时 try-import：失败则自动退回 CPU legacy `ScalableTSDFVolume`（2 s 完成，tick 路径不受影响） |
| 深度 edge 毛刺 | mesh 边缘有锯齿/小片 | `edge_thresh` 自适应 `0.01 + 0.01*z`；RealSense launch 侧开 temporal/spatial filter（已有 commit `152c47f`） |
| 腕部 FK 误差放大 | mesh 整块偏移 | §3 插值是必要前提；另外复核 hand-eye（commit `52e8d5d`）；极端时对腕部用略宽 edge_thresh |
| 重启后 bg mesh 丢失 | 每次都要 service 触发 | 持久化 `bg_cache_path`，启动时自动加载 |
| policy server 换 GPU | `CUDA_VISIBLE_DEVICES=0` 后和 viz node 撞卡 | launch 里把 policy server 的 GPU 也暴露为参数，文档里锁定 "viz=GPU0, policy=GPU1" |

---

## 13. 验收指标

完成后应满足：

1. **静态场景**：双臂停在 rest pose，布料静止 → `world/bg_mesh` 连续无空洞，
   `world/dynamic/*` 只覆盖 bbox 内的物体；肉眼观察 dynamic mesh 无明显"呼吸"抖动
   （RealSense temporal filter 打开前提下）。
2. **动态场景**：执行一次折衣任务 → 布料 mesh 跟手无拖影（screen-space 保证），
   背景保持稳定不动（TSDF 一次成型）。
3. **性能**：单 tick < 180 ms（5 Hz 稳定），`nvidia-smi` 下 viz node 只在 GPU0，
   显存占用 < 4 GB（JAX ~1.5 GB + Open3D ~1.5 GB）。
4. **退化路径**：`fg_enable:=false` 启动 → 退回旧 Points3D；`bg_enable:=false` → 无背景；
   `open3d` 缺失 → 自动走 CPU TSDF。三条路径都不能让节点崩溃。
5. **时间同步**：手动晃动左腕，观察 `world/dynamic/left_fg_mesh` 跟随末端运动，
   不应出现"点云甩出去"的错位；和 URDF 骨架叠加显示时应贴合。

---

## 14. 与既有文档的关系

- 本方案不改动 `docs/deployment/inference_visualization.md` 描述的 rerun 基础集成（交互执行控制、
  FK 可视化、轨迹显示、图像 log 等），只替换点云一条路径。
- 与 `docs/rerun_mesh_transparency_lesson.md` 的 mesh alpha 经验兼容：dynamic mesh 默认
  不透明，若需半透明叠加（例如显示规划路径），按 lesson 里推荐的 rerun 0.31+ 用法设置
  vertex_colors 的 alpha 通道。
- RealSense 深度滤波依赖 commit `152c47f`（spatial + temporal filter 已加到驱动 launch）。
- hand-eye 标定依赖 commit `52e8d5d`（腕部相机外参精度直接影响动态 mesh 对齐）。
