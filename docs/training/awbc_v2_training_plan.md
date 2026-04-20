# awbc_v2 训练计划

**状态**: 🔄 **实施中** (2026-04-19)
**决策**: Option 1 — gf0 继续跑 v1 作 baseline，gf1 切换到 v2 全量数据 + Mirror，获得干净 A/B 对比
**弃用的平行路线**: [awbc_pi07style_experiment.md](awbc_pi07style_experiment.md) (v1/v2/v3 all 失败)

## 实施进度

| 步骤 | 状态 |
|---|---|
| 1. 生成 Task_A/advantage_mirror（3055 ep Mirror 翻转）| ⏳ 待办 |
| 2. 生成 Task_A/dagger_mirror（3457 ep Mirror 翻转）| ⏳ 待办 |
| 3. 合并 4 份 → Task_A/advantage_v2 (12,024 ep)| ⏳ 待办 |
| 4. compute_norm_states_fast 生成 norm_stats | ⏳ 待办 |
| 5. 新 TrainConfig `pi05_flatten_fold_awbc_v2` | ⏳ 待办 |
| 6. CameraAugConfig 参数化（可延后）| ⏳ 可选 |
| 7. 启动脚本 + 启动 gf1_awbc_v2_v1 | ⏳ 待办 |

## 背景与决策依据

### 为什么不做 Model Arithmetic

从前期实验经验来看，split-merge（Model Arithmetic）对模型效果提升无明显新概念。本方案放弃 MA，聚焦于：
- **数据覆盖**：base（人类示教）+ dagger（policy 执行轨迹）
- **优势条件化**：AWBC prompt 区分高/低优势帧
- **离线数据增强**：space_mirroring 翻倍数据
- **在线增强参数化**：适配新机器人硬件配置

### 为什么不做 time_scaling

time_scaling（2× 帧抽取）会破坏 advantage label 的语义：
- `absolute_advantage(n)` 定义为"未来 50 帧"的进度差
- 抽帧后该窗口语义改变，task_index 与实际帧内容对应关系失效
- 引入噪声标签，风险高于收益
- dagger 数据平均仅 699 帧/episode，本已较短，抽帧损失关键过渡帧

### 为什么只训练一个模型（awbc_v2_d405）

`awbc_v2_d405` 使用更激进的在线增强（D405 参数），让模型接触更宽的视觉分布。
D435 的图像（清晰、特定色彩）是该分布的子集，**awbc_v2_d405 可直接部署在 D435 上，不会回退**。
因此无需单独训练 D435 版本，节省 4.7 天 GPU 时间。

---

## 一、数据现状

| 数据集 | Episodes | Frames | stage_progress_gt | task_index | 状态 |
|--------|----------|--------|-------------------|------------|------|
| `data/Task_A/advantage/` | 3,055 | 3.36M | ✅ | ✅ | awbc_v1 训练集 |
| `data/Task_A/dagger/` | 3,457 | 2.41M | ❌ | ❌ | 待处理 |
| `data/Task_A/advantage_v2/`（目标） | **12,024** | **~11.5M** | ✅/N/A | ✅ | awbc_v2 训练集 |

awbc_v1 vs awbc_v2 数据来源对比：

| | awbc_v1 | awbc_v2 |
|--|---------|---------|
| 训练数据 | advantage/ (base only) | advantage/ + dagger/ + 各自镜像 |
| Episodes | 3,055 | 12,024 |
| Frames | 3.36M | ~11.5M |
| OOD 覆盖 | ❌ 无 policy 执行分布 | ✅ 覆盖 policy 真实轨迹 |

---

## 二、新机器人配置变化分析

### 2.1 变化列表

| 变化 | 具体影响 | 严重程度 | 应对策略 |
|------|---------|---------|---------|
| 手部相机 D435 → D405 | FOV 变窄 (86°→69°)；近距景深更浅；色彩/白平衡特性不同 | **高** | 在线增强加强 + 新数据采集 |
| Top camera 安装高度/角度轻微变化 | 物体在视野中位置偏移；尺度轻微缩放 | **中** | 更激进的 Crop + 更大 Rotate 范围 |
| 双臂间距轻微变化 | 关节角空间 (14D) 不变；任务空间工作区轻微偏移 | **低** | 少量新示教数据（50-100 episodes） |

### 2.2 D435 vs D405 相机差异

| 参数 | D435 | D405 |
|------|------|------|
| FOV (H×V) | 86°×57° | 69°×44° |
| 工作距离 | 0.3m-3m | 0.07m-0.5m |
| 景深特性 | 较深（远景） | 较浅（近距清晰，远处模糊） |
| 应对增强 | 当前基线 | GaussianBlur + 更强 ColorJitter + 更大 Crop |

### 2.3 awbc_v2_d405 能否部署在 D435 上

**可以，不会回退。** 在线增强只在训练时生效，推理时直接使用原始相机图像。
`_AUGMENT_D405` 让模型训练时接触更宽的视觉分布（更强色彩扰动、模糊、更大裁剪），
D435 图像（清晰、D435 色彩）是该分布的子集，推理时完全能处理。
更激进的增强提升泛化，不会损害原始分布上的性能。

---

## 三、增强策略

### 3.1 离线增强

| 策略 | 决策 | 理由 |
|------|------|------|
| **space_mirroring** | ✅ 做，对 advantage/ 和 dagger/ | 数据翻倍；task_index 与臂的方向无关，label 完全保持有效 |
| time_scaling | ❌ 不做 | 破坏 absolute_advantage 语义，引入噪声标签 |

space_mirroring 操作：左右臂关节角互换（各 7 维）+ 所有摄像头视频水平翻转

### 3.2 在线增强（参数配置化）

新增 `AugmentConfig` / `CameraAugConfig` 数据类，替代 model.py 中硬编码参数。
本次只使用一套方案 `_AUGMENT_D405`，同时兼容新旧两套硬件。

**`_AUGMENT_D405`**（兼容 D435 + D405，唯一部署方案）

| 相机 | crop_ratio | rotate_deg | brightness | contrast | saturation | hue | gaussian_blur |
|------|-----------|-----------|-----------|---------|-----------|-----|--------------|
| `hand`（D405） | 0.90 | ±5° | 0.5 | 0.5 | 0.6 | 0.15 | σ≤2.0, p=0.4 |
| `top`（安装偏差） | 0.85 | ±8° | 0.3 | 0.4 | 0.5 | 0 | 关闭 |
| `_default` | 0.95 | ±5° | 0.3 | 0.4 | 0.5 | 0 | 关闭 |
| `wrist` | — (skip crop/rotate) | — | 0.3 | 0.4 | 0.5 | 0 | 关闭 |

---

## 四、代码变更

### 4.1 新增数据类（`src/openpi/training/config.py`）

在 `DataConfig` 之前插入：

```python
@dataclasses.dataclass(frozen=True)
class CameraAugConfig:
    """Per-camera augmentation parameters."""
    crop_ratio: float = 0.95
    rotate_deg: float = 5.0
    brightness: float = 0.3
    contrast: float = 0.4
    saturation: float = 0.5
    hue: float = 0.0
    gaussian_blur_sigma: float = 0.0   # 0 = disabled
    gaussian_blur_prob: float = 0.0


@dataclasses.dataclass(frozen=True)
class AugmentConfig:
    """
    Training-time image augmentation config. Keyed by camera name substring.
    Fallback key "_default" applies to unmatched cameras.
    Cameras matching skip_crop_rotate_keys skip crop/rotate transforms.
    """
    camera_configs: dict[str, CameraAugConfig] = dataclasses.field(
        default_factory=lambda: {"_default": CameraAugConfig()}
    )
    skip_crop_rotate_keys: tuple[str, ...] = ("wrist",)

    def get(self, camera_key: str) -> CameraAugConfig:
        for k, cfg in self.camera_configs.items():
            if k != "_default" and k in camera_key:
                return cfg
        return self.camera_configs.get("_default", CameraAugConfig())
```

`TrainConfig` 增加字段：

```python
@dataclasses.dataclass(frozen=True)
class TrainConfig:
    ...
    augment_config: AugmentConfig | None = None  # None = legacy hardcoded behavior
```

预定义增强方案（模块级常量）：

```python
_AUGMENT_D405 = AugmentConfig(
    camera_configs={
        "hand": CameraAugConfig(
            crop_ratio=0.90, rotate_deg=5.0,
            brightness=0.5, contrast=0.5, saturation=0.6, hue=0.15,
            gaussian_blur_sigma=2.0, gaussian_blur_prob=0.4,
        ),
        "top": CameraAugConfig(
            crop_ratio=0.85, rotate_deg=8.0,
            brightness=0.3, contrast=0.4, saturation=0.5,
        ),
        "_default": CameraAugConfig(
            crop_ratio=0.95, rotate_deg=5.0,
            brightness=0.3, contrast=0.4, saturation=0.5,
        ),
    },
    skip_crop_rotate_keys=("wrist",),
)
```

### 4.2 修改 `preprocess_observation()`（`src/openpi/models/model.py`）

修改签名：

```python
def preprocess_observation(
    rng,
    observation,
    *,
    train: bool = False,
    image_keys=IMAGE_KEYS,
    image_resolution=IMAGE_RESOLUTION,
    augment_config=None,   # AugmentConfig | None
) -> Observation:
```

替换增强逻辑（在 `if train:` 块内）：

```python
if augment_config is not None:
    cam_cfg = augment_config.get(key)
    skip_spatial = any(s in key for s in augment_config.skip_crop_rotate_keys)
    transforms = []
    if not skip_spatial:
        h, w = image.shape[1:3]
        transforms += [
            augmax.RandomCrop(int(w * cam_cfg.crop_ratio),
                              int(h * cam_cfg.crop_ratio)),
            augmax.Resize(w, h),
            augmax.Rotate((-cam_cfg.rotate_deg, cam_cfg.rotate_deg)),
        ]
    if cam_cfg.gaussian_blur_sigma > 0:
        transforms += [
            augmax.GaussianBlur(
                sigma=(0.0, cam_cfg.gaussian_blur_sigma),
                p=cam_cfg.gaussian_blur_prob,
            )
        ]
    transforms += [
        augmax.ColorJitter(
            brightness=cam_cfg.brightness,
            contrast=cam_cfg.contrast,
            saturation=cam_cfg.saturation,
            hue=cam_cfg.hue,
        )
    ]
else:
    # Legacy hardcoded behavior (backward compatible, augment_config=None)
    transforms = []
    if "wrist" not in key:
        h, w = image.shape[1:3]
        transforms += [
            augmax.RandomCrop(int(w * 0.95), int(h * 0.95)),
            augmax.Resize(w, h),
            augmax.Rotate((-5, 5)),
        ]
    transforms += [
        augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
    ]
```

训练循环透传（调用 `preprocess_observation` 处补充参数）：

```python
preprocess_observation(rng, obs, train=True,
                       augment_config=config.augment_config)
```

### 4.3 新增训练 Config（`src/openpi/training/config.py`）

```python
# awbc_v2：base + dagger + 镜像，_AUGMENT_D405 兼容新旧两套硬件
TrainConfig(
    name="pi05_flatten_fold_awbc_v2",
    model=pi0_config.Pi0Config(pi05=True),
    data=LerobotAgilexDataConfig(
        repo_id="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/advantage_v2",
        default_prompt="Flatten and fold the cloth.",
        use_delta_joint_actions=False,
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/vePFS/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=200_000,   # 数据量 ×3.4 → 对齐约 4.5 epochs（awbc_v1 为 7.6 epochs）
    keep_period=5000,
    num_workers=8,
    batch_size=256,
    augment_config=_AUGMENT_D405,
),
```

---

## 五、完整 Step 序列

### 阶段一：数据准备

```
Step 0  KAI0 推理 dagger/                             ~1.5h (8 GPU)
        stage_advantage/annotation/eval.py
        输入:  data/Task_A/dagger/ (视频 + parquet)
        输出:  data/Task_A/dagger/data_KAI0_100000/
               新列: absolute_advantage, relative_advantage, absolute_value

Step 1  离散化 dagger → task_index                     ~15min
        stage_advantage/annotation/discretize_advantage.py \
            data/Task_A/dagger \
            --threshold 30 \
            --discretion-type binary \
            --advantage-source absolute_advantage \
            --model-name KAI0 \
            --ckpt-steps 100000
        输出:  每帧 task_index ∈ {0,1}，meta/tasks.jsonl 更新

Step 2a Space mirroring: advantage/ → advantage_sym/   ~2h (CPU, 8 workers)
        train_deploy_alignment/data_augment/space_mirroring.py full \
            --src-path  data/Task_A/advantage \
            --mirror-path data/Task_A/advantage_mirror \
            --merge-path  data/Task_A/advantage_sym \
            --repo-id Task_A_advantage_sym \
            --num-workers 8

Step 2b Space mirroring: dagger（已标注）→ dagger_sym/  ~2h (CPU, 8 workers)
        train_deploy_alignment/data_augment/space_mirroring.py full \
            --src-path  data/Task_A/dagger \
            --mirror-path data/Task_A/dagger_mirror \
            --merge-path  data/Task_A/dagger_sym \
            --repo-id Task_A_dagger_sym \
            --num-workers 8
        ※ Step 2a 和 2b 可并行执行

Step 3  合并四路数据 → advantage_v2/                  ~30min
        train_deploy_alignment/data_augment/merge_lerobot.py \
            --src_paths \
                data/Task_A/advantage_sym \
                data/Task_A/dagger_sym \
            --tgt_path data/Task_A/advantage_v2 \
            --repo_id Task_A_advantage_v2
        输出:  12,024 episodes, ~11.5M frames
               tasks.jsonl 保持 2 条（negative/positive prompt）
```

### 阶段二：代码变更

```
Step 4  新增 AugmentConfig / CameraAugConfig          ~1h
        src/openpi/training/config.py
        + CameraAugConfig, AugmentConfig 数据类
        + _AUGMENT_D405 常量

Step 5  修改 preprocess_observation()                 ~1h
        src/openpi/models/model.py
        + augment_config 参数 + 分相机逻辑（向后兼容）
        + 训练循环透传 augment_config

Step 6  新增 TrainConfig                              ~30min
        src/openpi/training/config.py
        pi05_flatten_fold_awbc_v2（augment_config=_AUGMENT_D405）
```

### 阶段三：训练与评估

```
Step 7  训练 awbc_v2                                  ~4.7 天 (8 GPU)
        export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
        uv run scripts/train.py pi05_flatten_fold_awbc_v2 \
            --exp_name=awbc_v2 \
            --fsdp-devices 8 \
            --batch-size 256
        ※ 200K steps，约每 5K 存一次 checkpoint，共 40 个 checkpoint
        ※ 150K checkpoint 自动保留，如已收敛可提前取用

Step 8  离线对比评估                                   ~1h
        model_arithmetic/eval_awbc_with_advantage_prompt.py
        对比: awbc_v1 vs awbc_v2
        指标: loss / MSE-to-GT / L1-to-GT

Step 9  （强烈建议）采集 D405 配置新示教数据           50-100 episodes
        → 加入 advantage_v2 重训（awbc_v2_plus）
        → 视觉域偏移靠增强只能缓解，新数据是根本解法
```

---

## 六、模型对比

| | awbc_v1 | **awbc_v2**（本计划） |
|--|---------|----------------------|
| **训练数据** | advantage/ 3,055 ep | advantage_v2/ 12,024 ep |
| **Frames** | 3.36M | ~11.5M |
| **Space mirroring** | ❌ | ✅ |
| **DAgger 分布** | ❌ | ✅ |
| **在线增强** | 硬编码（旧） | `_AUGMENT_D405`（配置化） |
| **hand 相机** | ColorJitter(b=0.3) | +GaussianBlur(σ≤2, p=0.4), hue=0.15 |
| **top 相机** | Crop 95%, ±5° | Crop 85%, ±8° |
| **部署硬件** | 旧机器人 (D435) | **D435 + D405 均可** |
| **预期收益** | 基线 | 更少分布偏移 + 相机域适配 |

---

## 七、时间估算

### Steps 选择依据

| | awbc_v1 | awbc_v2 |
|--|---------|---------|
| 总帧数 | 3.36M | ~11.5M |
| 每 epoch steps（batch=256） | ~13,125 | ~44,921 |
| 100K steps ≈ | 7.6 epochs | 2.2 epochs（欠拟合） |
| 150K steps ≈ | — | 3.3 epochs（偏少） |
| **200K steps ≈** | — | **4.5 epochs（推荐）** |
| 341K steps ≈ | — | 7.6 epochs（完全对齐，成本过高） |

数据量 ×3.4 倍，steps 调整至 **200,000**，对齐约 4.5 个完整 epoch。
dagger 数据含 OOD advantage 估计噪声，更多 epoch 有助于平均噪声；
每 5K 存一次 checkpoint，150K 节点自动保留，如提前收敛可直接取用，无额外风险。

### 多节点加速方案

#### 网络实测结论（2026-04-17）

**硬件规格**：每台机器 4 × Mellanox mlx5 EFA 网卡，每张 **200 Gbps (4X HDR)**，总计 **800 Gbps/机**（官方 200 Gbps×4 即此含义）。
**连接入口**：`ssh -p 2222 root@192.168.0.161`（gf1），`ssh -p 2222 root@192.168.0.144`（gf0）。

| 测试项 | 实测值 | 说明 |
|--------|--------|------|
| 原始 TCP 单流（33.201.x.x EFA 接口） | 28.5 Gbps | Python socket，无加密 |
| NCCL TCP 默认（6 GB all_reduce） | 5.5 Gbps | NCCL 默认单 socket |
| **NCCL TCP 调优（8×4 sockets）** | **11.4 Gbps** | NCCL_SOCKET_NTHREADS=8, NSOCKS_PERTHREAD=4 |
| NCCL IB/RoCE RDMA | ❌ 不可用 | 报错 "Could not find NET with id 0"（缺 aws-ofi-nccl） |

> 两台机器的 mlx5 EFA 设备有 RoCEv2 GID（index 6），对应 33.201.x.x 接口；
> 但 NCCL IB 传输层初始化失败，跨子网 RDMA 不可用。
> 实际 NCCL 只能走 TCP，调优后约 **11.4 Gbps**（远低于 800 Gbps 物理规格）。

#### 训练开销实测估算

```
Pi0.5 梯度大小：~6 GB（3B params × bfloat16）
NCCL TCP all_reduce（6 GB，调优）：实测 ~4.2s
单机 step 计算时间：~4s

FSDP backward 与 all_reduce 重叠分析：
  - 乐观（80% 重叠）：额外开销 +0.8s → step 时间 ~4.8s
  - 保守（50% 重叠）：额外开销 +2.1s → step 时间 ~6.1s

双机步数减半（batch 翻倍）：200K × 256 / 512 = 100K steps
双机总时间：
  乐观：100K × 4.8s ≈ 5.6 天
  保守：100K × 6.1s ≈ 7.1 天
单机总时间：200K × 4s ≈ 9.3 天
实际加速比：1.3×～1.7×（取决于 FSDP overlap 效率）
```

> **结论**：多节点训练有收益但有限（**1.3-1.7×**），节省 2-4 天。
> 若安装 `aws-ofi-nccl` 插件启用 RDMA，带宽可大幅提升（理论 ~100 Gbps+），加速比可达 ~1.85×。

#### mesh 拓扑

`make_mesh(fsdp_devices=8)` 在 16 GPU 下 → `(2 batch, 8 fsdp)`
- FSDP axis（8）：节点内 NVLink，~600 GB/s，无瓶颈
- Batch axis（2）：跨节点 EFA，梯度 reduce-scatter，与反向传播重叠

#### 多节点启动命令

**Prerequisites：两台机器 SSH 互信（root@192.168.0.x:2222），数据集路径一致或各自同步。**

**train.py 增加 5 行（在 `main()` 开头）：**

```python
import os
# ...
def main(config: _config.TrainConfig):
    init_logging()
    if "JAX_COORDINATOR_ADDRESS" in os.environ:
        jax.distributed.initialize()   # 读取环境变量自动初始化
    logging.info(f"Running on: {platform.node()}")
    ...
```

**Node 1 = gf0（coordinator，用 192.168.0.144 作为稳定地址）：**

```bash
export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_COORDINATOR_ADDRESS=192.168.0.144:1234
export JAX_NUM_PROCESSES=2
export JAX_PROCESS_ID=0
# NCCL TCP 调优（EFA 接口，IB 不可用）
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth2,eth3,eth4,eth5
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=4

uv run scripts/train.py pi05_flatten_fold_awbc_v2 \
    --exp_name=awbc_v2 \
    --fsdp-devices 8 \
    --batch-size 512
```

**Node 2 = gf1（通过 `ssh -p 2222 root@192.168.0.161` 启动）：**

```bash
export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_COORDINATOR_ADDRESS=192.168.0.144:1234
export JAX_NUM_PROCESSES=2
export JAX_PROCESS_ID=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth2,eth3,eth4,eth5
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=4

uv run scripts/train.py pi05_flatten_fold_awbc_v2 \
    --exp_name=awbc_v2 \
    --fsdp-devices 8 \
    --batch-size 512
```

> **注意**：两台机器的数据集路径必须一致（共享存储 `/vePFS/` 或各自同步一份）。
> checkpoint 由 Node 1（process_id=0）写出。
> 建议先跑 1K steps 验证多节点通信正常再开始完整训练。

### 时间明细

参照基准：25,000 steps ≈ 28h（8 × A100，batch_size=256），步速约 **893 steps/h**。

| Step | 内容 | 单机 8 GPU | 双机 16 GPU（EFA TCP） |
|------|------|-----------|----------------------|
| 0 | KAI0 推理 dagger | ~1.5h | ~1.5h（单机即可） |
| 1 | 离散化 dagger | ~15min | ~15min |
| 2a+2b | Space mirroring × 2 | ~2h | **~1h（两台并行）** |
| 3 | 合并数据集 | ~30min | ~30min |
| 4+5+6 | 代码变更 | ~2.5h | ~2.5h |
| 7 | 训练 awbc_v2（200K/100K steps） | **~9.3 天** | **~5.8 天**（估算） |
| 8 | 离线评估 | ~1h | ~1h |
| **总计** | | **~10.5 天** | **~6.5 天** |

---

## 八、checklist

### 数据准备
- [ ] `data/Task_A/advantage/` 存在，tasks.jsonl 含 positive/negative 两条
- [ ] `data/Task_A/dagger/` 完整（3,457 episodes, videos/ + data/）
- [ ] KAI0 checkpoint 路径：`experiment/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/run1/100000/`
- [ ] `space_mirroring.py` 可用，`merge_lerobot.py` schema 兼容

### 代码变更
- [ ] `CameraAugConfig` / `AugmentConfig` 新增完成（含 `get()` 方法）
- [ ] `preprocess_observation()` 修改，`augment_config=None` 走旧逻辑（向后兼容）
- [ ] `TrainConfig` 增加 `augment_config` 字段
- [ ] 训练循环正确透传 `augment_config`
- [ ] `pi05_flatten_fold_awbc_v2` config 可通过 `get_config()` 查到

### 训练前验证
- [ ] `uv run python -c "from openpi.training.config import get_config; get_config('pi05_flatten_fold_awbc_v2')"` 无报错
- [ ] 单步 dry-run（`--num_train_steps 1`）验证数据 pipeline 通畅
- [ ] `advantage_v2/meta/info.json` 的 `total_episodes == 12024`

---

## 九、风险说明

| 风险 | 概率 | 应对 |
|------|------|------|
| KAI0 对 dagger OOD 退化 | 低（已验证 std ratio=0.98×） | 接受；dagger task_index 质量略低但可用 |
| space_mirroring 后优势标签失效 | 低（优势与臂方向无关） | 抽样验证 50 个镜像 episode 的 absolute_value 单调性 |
| D405 视觉域偏移在增强后仍过大 | 中 | 靠 Step 9 新数据采集根本解决 |
| 数据量 ×3.4 但 steps 不变导致欠拟合 | 低-中 | 观察 loss 曲线；如未收敛延长到 150K steps |
| awbc_v2_d405 增强在 D435 上回退 | 极低（更宽分布包含 D435 子集） | 无需处理 |

---

## 参考

- [stage_advantage/README.md](../stage_advantage/README.md)
- [train_deploy_alignment/data_augment/](../train_deploy_alignment/data_augment/)
- [training_plans.md](training_plans.md) — kai0_mixed_1 / kai0_full 方案（含 MA 路线）
