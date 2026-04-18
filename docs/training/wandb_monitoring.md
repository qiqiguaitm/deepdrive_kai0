# wandb 训练监控指南

## 当前配置

训练脚本 `run_gf2.sh` 已配置 `WANDB_MODE=offline`，wandb 在训练过程中将 loss、grad_norm、param_norm、learning_rate 等指标写入本地文件，不需要网络。

offline run 文件位于：`kai0/wandb/offline-run-<timestamp>-<run_id>/`

## 记录的指标

### AWBC 训练 (JAX train.py)
- `loss` — action MSE loss
- `grad_norm` — 梯度范数
- `param_norm` — 参数范数（kernel weights）

### Advantage Estimator 训练 (PyTorch train_pytorch.py)
- `loss` — 总 loss (value_loss × weight + action_loss × weight)
- `learning_rate` — 当前学习率
- `grad_norm` — 梯度范数
- `time_per_step` — 每步耗时

---

## 方式 1：本地 wandb server（Docker，无需联网）

适用于无外网或不想上传数据到云端的场景。

### 1.1 启动 server

```bash
wandb server start
# 默认启动在 http://localhost:8080
# 首次访问浏览器页面，创建本地账号（随意填写用户名/密码）
```

### 1.2 同步 offline runs

```bash
export WANDB_BASE_URL=http://localhost:8080

# 同步所有 offline runs
wandb sync kai0/wandb/offline-run-*

# 或同步特定 run
wandb sync kai0/wandb/offline-run-20260329_XXXXXX-XXXXXXXX
```

### 1.3 浏览器查看

打开 `http://localhost:8080`，在 dashboard 中找到对应 project 和 run，查看 loss 曲线等。

### 1.4 停止 server

```bash
wandb server stop
```

### 注意事项
- 需要 Docker (`docker --version` 确认已安装)
- 首次启动需拉取镜像，可能较慢
- server 数据存储在 Docker volume 中，停止后数据保留

---

## 方式 2：同步到 wandb.ai 云端（最简单）

适用于有外网访问的场景。

### 2.1 登录

```bash
wandb login
# 输入 API key（从 https://wandb.ai/authorize 获取）
# 或直接设置环境变量：
export WANDB_API_KEY=<your_api_key>
```

### 2.2 同步 offline runs

```bash
# 同步所有 offline runs
wandb sync kai0/wandb/offline-run-*

# 指定 project 名称
wandb sync --project kai0-reproduce kai0/wandb/offline-run-*

# 指定 entity（团队/用户名）
wandb sync --entity <your_username> --project kai0-reproduce kai0/wandb/offline-run-*
```

### 2.3 浏览器查看

同步完成后终端会输出 URL，直接点击打开。或访问 `https://wandb.ai/<entity>/<project>` 查看。

---

## 训练中实时查看（无需 wandb UI）

如果只想快速看 loss 数值，不需要启动 wandb：

```bash
# 从训练日志直接 grep loss
# JAX (train.py) 格式: "Step 1000: loss=0.1234, grad_norm=1.23, param_norm=45.6"
grep "^Step" logs/gf2_run.log | tail -20

# PyTorch (train_pytorch.py) 格式: "step=1000 loss=0.1234 lr=1.00e-04 grad_norm=1.23"
grep "step=" logs/gf2_run.log | tail -20

# 提取 loss 曲线数据
grep "^Step" logs/gf2_run.log | awk -F'[=,]' '{print $1, $2}' > /tmp/loss_curve.txt
```

也可以用验证脚本解析：

```bash
python scripts/validate_awbc.py loss --log logs/gf2_run.log
```
