# Task A Official-Dynamic Mixed 实验记录

> 日期：2026-04-25 ~ 2026-04-26
> 配置：`pi05_flatten_fold_mixed_visrobot01`
> 实验名：`mixed_visrobot01_1500`
> 启动脚本：`train_scripts/launch/run_task_a_official_dynamic.sh`
> 状态：✅ 已完成

---

## 一、实验背景

这次实验是 Task A (`Flatten and fold the cloth.`) 的一条 full fine-tuning + dynamic mixed dataset 训练线：

- 8×A100 80GB，JAX + FSDP=8
- 复用已有 `Task_A_mixed_gf1`
- 首次启动由 launcher 完成，后续由 watcher 负责数据增长后的自动 rebuild + `--resume`

### 环境配置

| 项目 | 值 |
|------|-----|
| GPU | 8× A100 80GB |
| W&B project | `kai0_policy_exp` |
| W&B run id | `dnafjz77` |
| W&B 本地目录 | `kai0/wandb/run-20260425_083446-dnafjz77/` |

---

## 二、启动命令

> secret 已脱敏。

```bash
export http_proxy=http://<proxy-host>:3128
export https_proxy=http://<proxy-host>:3128

export WANDB_API_KEY='<redacted>'
wandb login --relogin "$WANDB_API_KEY"

unset HF_ENDPOINT
hf auth login --token <redacted>

export REPO_ROOT=/VLA-Data/scripts/xyh/deepdive_kai0
export KAI0_DIR=/VLA-Data/scripts/xyh/deepdive_kai0/kai0
export VIS_ROOT=/VLA-Data/scripts/lianqing/data/bipiper_dataset/Task_A
export OLD_ROOT=/VLA-Data/scripts/lianqing/data/OpenDriveLab-org/Kai0/Task_A
export MIX_ROOT=/VLA-Data/scripts/xyh/deepdive_kai0/kai0/data/Task_A_mixed_gf1

export KAI0_DATA_ROOT=/VLA-Data/scripts/xyh/deepdive_kai0/kai0
export OPENPI_DATA_HOME=/VLA-Data/scripts/xyh/deepdive_kai0/kai0/.cache/openpi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /VLA-Data/scripts/xyh/deepdive_kai0

bash train_scripts/launch/run_task_a_official_dynamic.sh \
  pi05_flatten_fold_mixed_visrobot01 \
  --exp-name mixed_visrobot01_1500 \
  --project-name kai0_policy_exp \
  --batch-size 64 \
  --num-train-steps 50000 \
  --save-interval 5000 \
  --log-interval 100 \
  --num-workers 8 \
  --reuse-mix \
  --foreground
```

---

## 三、数据与配置

### 3.1 数据快照口径

本次 run 使用：

```text
kai0/data/Task_A_mixed_gf1/
```

| 项目 | 值 |
|------|-----|
| train_episodes | 1036 |
| train_frames | 929,822 |
| val_episodes | 20 |
| val_frames | 19,197 |

### 3.2 source buckets（本次实验记录对应的 mixed 快照）

| bucket | 原始 episode 数 | train | val |
|------|------------------:|------:|----:|
| `visroot/all` | 264 | 259 | 5 |
| `existing/base` | 3055 | 259 | 5 |
| `existing/dagger` | 3457 | 259 | 5 |
| `existing/advantage` | 3055 | 259 | 5 |

> 说明：这张表描述的是**本次实验记录对应的 mixed 快照**。  
> watcher 后续若检测到新数据，会整包重建 `Task_A_mixed_gf1`；因此当前磁盘上的 manifest 可能与这次实验启动时的快照不同。

### 3.3 实际训练参数

| 参数 | 值 |
|------|-----|
| config | `pi05_flatten_fold_mixed_visrobot01` |
| exp_name | `mixed_visrobot01_1500` |
| init | `pi05_base` 本机缓存 |
| mode | 全参数微调 |
| batch_size | 64 |
| num_train_steps | 50,000 |
| save_interval | 5,000 |
| log_interval | 100 |
| num_workers | 8 |
| fsdp_devices | 8 |
| peak_lr / warmup / decay | `1.5e-5` / `500` / cosine to `1.5e-6` |
| ema_decay | `0.999` |

---

## 四、训练结果

### 4.1 时间与速度

| 项目 | 值 |
|------|-----|
| step 0 | 2026-04-25 08:36:29 |
| final inline-eval | 2026-04-26 04:23:21 |
| 总时长 | 19:46:52 |
| 训练阶段速率 | 稳定约 `1.3 s/it` |
| 含 save + inline-eval 折算 | 约 `1.42 s/step` |
| 单次 inline-eval 耗时 | 620s ~ 652s |

### 4.2 训练曲线摘要

| step | loss | grad_norm | param_norm |
|---:|---:|---:|---:|
| 0 | 0.6948 | 5.0607 | 1802.3865 |
| 100 | 0.2638 | 1.8973 | 1802.3854 |
| 200 | 0.0501 | 0.3015 | 1802.3843 |
| 300 | 0.0371 | 0.2337 | 1802.3846 |
| 49999 | 0.004593 | 0.099333 | 1803.1390 |

### 4.3 Per-step inline-eval

| step | MAE@1 | @10 | @25 | @50 |
|---:|---:|---:|---:|---:|
| 5000 | 0.0147 | 0.0307 | 0.0549 | 0.0840 |
| 10000 | 0.0136 | 0.0281 | 0.0502 | 0.0769 |
| 15000 | 0.0132 | 0.0275 | 0.0490 | 0.0754 |
| 20000 | 0.0130 | 0.0272 | 0.0486 | 0.0748 |
| 25000 | 0.0129 | 0.0271 | 0.0484 | 0.0746 |
| 30000 | 0.0128 | 0.0269 | 0.0482 | 0.0741 |
| 35000 | 0.0127 | 0.0268 | 0.0480 | 0.0739 |
| 40000 | 0.0126 | 0.0268 | 0.0479 | 0.0737 |
| 45000 | 0.0126 | 0.0267 | 0.0477 | 0.0736 |
| 49999 | 0.0126 | 0.0267 | 0.0477 | 0.0736 |

### 4.4 最终结论

| 项目 | 值 |
|------|-----|
| best MAE@1 | **0.0126** |
| best MAE@10 | **0.0267** |
| best MAE@25 | **0.0477** |
| best MAE@50 | **0.0736** |
| best step | `MAE@1` 显示精度下 `40000 / 45000 / 49999` 打平；`@10/@25/@50` 最优为 `45000 / 49999` |
| plateau 区间 | `40000` → `49999` |
| 推荐 checkpoint | **`49999`** |

### 4.5 关键观察

1. `5000 → 30000` 持续稳定下降，`40000` 后进入 plateau。
2. `40000 / 45000 / 49999` 三个后期点几乎重合，40k 之后边际收益很小。
3. 这条 `mixed_visrobot01` 线明显强于此前单源 visrobot01-only，已经进入 Task A mixed 系列的最优梯队。
4. 如果只看部署，`49999` 最稳；如果想更早停训，`45000` 也足够接近最终表现。

---

## 五、Checkpoint

路径：

```text
kai0/checkpoints/pi05_flatten_fold_mixed_visrobot01/mixed_visrobot01_1500/
```

已确认存在：

```text
5000/
10000/
15000/
20000/
25000/
30000/
35000/
40000/
45000/
49999/
norm_stats.json
wandb_id.txt
```

总目录体积约：

```text
417G
```
