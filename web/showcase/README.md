# deepdive_kai0 · showcase

对外展示页（不是生产监控、不是采集 UI）—— 一个轻量 FastAPI + 单页 HTML 站，介绍 deepdive_kai0 已经做到了什么。

参考 `GraspForge/web_demo` 的紫色主题与 tab 化布局，内容固定从本地 `docs/deployment/` 和 `web/showcase/content/` 读取。

> **定位**：先在内网 (`sim01:8765`) 跑起来给协作者看；后续可整体迁移到公网。代码里不含任何 IP / 主机名 / 密钥。

---

## 一键启动

```bash
cd web/showcase
./start_srv.sh start        # 后台启动，http://0.0.0.0:8765/
./start_srv.sh status       # 查看
./start_srv.sh logs         # 跟日志
./start_srv.sh restart
./start_srv.sh stop
./start_srv.sh fg           # 前台启动（调试用）
```

依赖：`python3 -m pip install fastapi uvicorn`（除此之外零依赖；不需要 ROS / CUDA / 任何模型库）。

---

## 目录结构

```
web/showcase/
├── server.py              FastAPI 单入口，~150 LoC
├── start_srv.sh           启停脚本
├── templates/
│   └── index.html         单页 HTML，紫色主题，4 tab
├── static/
│   ├── i18n.js            中英双语字典 + setLang()
│   ├── app.js             tab 切换 + 内容拉取调度
│   ├── feature_matrix.js  能力矩阵渲染
│   ├── timeline.js        里程碑时间轴
│   └── docs.js            文档浏览器 + marked.js 渲染
├── content/
│   ├── features.json      33 个能力卡片（5 分类，每条带 zh+en）
│   ├── milestones.json    12 条里程碑（zh+en）
│   └── docs_index.json    docs/deployment 分组与中文标题
├── logs/                  运行日志
└── README.md
```

---

## 四个 Tab

| Tab | 内容 |
|---|---|
| **概览** | 项目简介 + 4 个统计数字 + 技术栈 + ASCII 架构图 + 时间轴 |
| **能力矩阵** | 33 个能力卡片，按 5 类 (Hardware / Inference / Data / Training / Tooling) 过滤 |
| **端到端管线** | 数据采集 → 模型训练 → 真机部署 三栏并排 |
| **文档** | 24 份 `docs/deployment/*.md` 按主题分组，右侧 markdown 渲染 |

---

## 加新能力 / 里程碑

直接改 `content/features.json` 或 `content/milestones.json`，刷新页面即可（页面是纯前端拉 JSON 的，**无须重启服务器**）。

每条 feature 至少要给：
- `id`（kebab-case）
- `category`（5 个之一: `hardware` / `inference` / `data` / `training` / `tooling`）
- `title_zh` + `title_en`
- `summary_zh` + `summary_en`
- `status`（`DONE` / `IN-PROGRESS` / `PLANNED`）
- `completed_date`（YYYY-MM-DD 或 null）
- `docs`（数组，文件名带 `.md`，会被 `/api/doc/{name}` 解析）

---

## API

| 端点 | 用途 |
|---|---|
| `GET /` | 主页 |
| `GET /api/health` | 健康检查 |
| `GET /api/features` | 返回 features.json |
| `GET /api/milestones` | 返回 milestones.json |
| `GET /api/docs/index` | 返回 docs_index.json |
| `GET /api/doc/{name}` | 返回 `docs/{deployment,training,security,}/{name}` 原文 |
| `GET /api/readme?lang=zh\|en` | 项目根 README |

文件名解析受 `^[a-zA-Z0-9_\-]+\.md$` 限制，无路径遍历风险。

---

## 部署到公网（未来）

当前没有任何生产硬化。迁移到公网前需要至少：

1. 反代到 HTTPS（Caddy / nginx）+ 改 `--host 127.0.0.1`
2. 启用 CORS 白名单（FastAPI `CORSMiddleware`，仅放行展示域名）
3. `--workers N` + 用 gunicorn / hypercorn
4. content/*.json 与 docs/ 再过一遍敏感信息扫除（IPs / hostnames / paths / 内部 URL）
5. 加 `robots.txt` + Open Graph meta tag
6. 抽换 header 的 repo-link `href` 指到真实 GitHub URL

---

## 与 GraspForge web_demo 的差异

- **更轻**：无 viser、无 model loading、无 GPU 依赖、不需要 conda
- **更窄**：只展示 deepdive_kai0 一个项目，没有 research / talks / playground 等多模块切片
- **样式复用**：CSS 调色板与 grid + glow 背景完全复刻
- **i18n**：内置双语切换（GraspForge 是英文 README + 中文 README 各一份，本站合并到 dict）
