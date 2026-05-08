from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from .episodes import (
    delete_episode,
    episode_depth_zarr_path,
    episode_meta,
    episode_video_path,
)
from .models import (
    EpisodeMeta,
    Role,
    SaveRecordingReq,
    StartRecordingReq,
    StatsResponse,
    Template,
)
from . import replay as replay_mod
from .recorder import recorder
from .ros_bridge import bridge
from .stats_service import service as stats
from .status_hub import hub
from .templates import store as templates


class ReplayPreflightReq(BaseModel):
    task: str
    subset: str
    date: str
    episode_id: int


class ReplayExecuteReq(BaseModel):
    task: str
    subset: str
    date: str
    episode_id: int
    rate: float = 1.0
    loop: bool = False


def require_admin(x_role: Role = Header(default="collector")) -> Role:
    if x_role != "admin":
        raise HTTPException(status_code=403, detail="admin required")
    return x_role


@asynccontextmanager
async def lifespan(app: FastAPI):
    n = stats.full_rescan()
    stats.start_watcher()
    hub.start()
    print(f"[startup] rescanned {n} episodes; watcher + status hub running")
    yield
    stats.stop_watcher()


app = FastAPI(title="data_manager", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True}


# -------- templates --------
@app.get("/api/templates", response_model=list[Template])
def list_templates(only_enabled: bool = False):
    return templates.list(only_enabled=only_enabled)


@app.put("/api/templates/{tid}", response_model=Template)
def upsert_template(tid: str, t: Template, _: Role = Depends(require_admin)):
    if t.id != tid:
        raise HTTPException(status_code=400, detail="path id mismatch body")
    return templates.upsert(t)


@app.delete("/api/templates/{tid}")
def del_template(tid: str, _: Role = Depends(require_admin)):
    return {"deleted": templates.delete(tid)}


# -------- recorder --------
@app.get("/api/recorder")
def get_recorder():
    return recorder.snapshot()


@app.post("/api/recorder/start")
def start_rec(req: StartRecordingReq):
    try:
        return recorder.start(req)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/recorder/save")
def save_rec(req: SaveRecordingReq):
    try:
        return recorder.save(req)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/recorder/discard")
def discard_rec():
    return recorder.discard()


@app.post("/api/recorder/estop")
def estop():
    bridge.emergency_stop()
    recorder.discard()
    return {"ok": True}


@app.post("/api/recorder/toggle")
def toggle_rec():
    """鼠标按钮之外的第二路启停 (踏板 / 远程脚本可用).

    IDLE→start(用上次记住的 template/operator, 先跑和前端一致的 preflight);
    RECORDING→save(success=True, note=pedal);
    SAVING/ERROR / 无记忆/preflight 失败 → 409 + reason (+failures).
    """
    res = recorder.toggle(hub.snapshot)
    if res.get("action") == "rejected":
        return JSONResponse(status_code=409, content=res)
    return res


@app.get("/api/recorder/preflight")
def get_preflight():
    """返回当前 snapshot 下的 failure 列表 (调试 / 踏板日志用)."""
    from .preflight import collect_failures
    snap = hub.snapshot()
    return {"failures": collect_failures(snap), "state": snap.get("recorder", {}).get("state")}


# -------- session (pedal / remote 冷启动依赖) --------
# 前端每次改 template / operator 就 PUT 一次, 后端记到 recorder.last_* 里。
# 这样踏板第一次就能触发启动, 不必先用鼠标点一次"开始"。
class SessionReq(BaseModel):
    template_id: Optional[str] = None
    operator: Optional[str] = None


@app.get("/api/session")
def get_session():
    return {
        "template_id": recorder.last_template_id,
        "operator": recorder.last_operator,
    }


@app.put("/api/session")
def put_session(req: SessionReq):
    # None = 不改; 空串 = 清除 (避免"选了但没保存"的幽灵状态)
    if req.template_id is not None:
        recorder.last_template_id = req.template_id or None
    if req.operator is not None:
        recorder.last_operator = req.operator or None
    return {
        "template_id": recorder.last_template_id,
        "operator": recorder.last_operator,
    }


# -------- data sync (gf0/gf1) --------
@app.get("/api/sync/status")
def get_sync_status():
    from . import sync as _sync
    return _sync.status()


@app.post("/api/sync/all")
def post_sync_all(only_task: Optional[str] = None, _: Role = Depends(require_admin)):
    """管理员触发: 把 DATA_ROOT 下所有 (task,date,subset) 批量推到远端。
    每个 subset 一个后台线程, 本接口立即返回排队数。"""
    from . import sync as _sync
    n = _sync.sync_all(only_task=only_task)
    return {"queued": n}


# -------- stats --------
@app.get("/api/stats", response_model=StatsResponse)
def get_stats():
    return stats.stats()


@app.post("/api/stats/rescan")
def rescan(_: Role = Depends(require_admin)):
    n = stats.full_rescan()
    return {"rescanned": n}


# -------- episodes --------
@app.get("/api/episodes", response_model=list[EpisodeMeta])
def list_episodes(
    task_id: Optional[str] = Query(None),
    subset: Optional[str] = Query(None),
    operator: Optional[str] = Query(None),
    success: Optional[bool] = Query(None),
    prompt_kw: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=2000),
):
    return stats.list_episodes(task_id, subset, operator, success, prompt_kw, limit)


@app.get("/api/episodes/{task_id}/{subset}/{episode_id}/meta")
def get_episode_meta(task_id: str, subset: str, episode_id: int):
    return episode_meta(task_id, subset, episode_id)


@app.get("/api/episodes/{task_id}/{subset}/{episode_id}/video/{camera}")
def get_episode_video(task_id: str, subset: str, episode_id: int, camera: str,
                      raw: bool = Query(False, description="1=返回原始 AV1 mp4；默认转码为 H.264 以便浏览器直接播放")):
    p = episode_video_path(task_id, subset, episode_id, camera)
    if not p.exists():
        raise HTTPException(status_code=404, detail="video missing")
    if raw:
        return FileResponse(p, media_type="video/mp4", filename=p.name)

    import shutil, subprocess
    if not shutil.which("ffmpeg"):
        return FileResponse(p, media_type="video/mp4", filename=p.name)

    # 已是 H.264 的直接回源（双击/浏览器都能播，省去转码）
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "default=nk=1:nw=1", str(p)],
            capture_output=True, text=True, timeout=3,
        )
        if probe.stdout.strip() == "h264":
            return FileResponse(p, media_type="video/mp4", filename=p.name)
    except Exception:
        pass

    # 在线转 H.264/fragmented mp4（frag_keyframe+empty_moov 让浏览器可流式播放）
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(p),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "frag_keyframe+empty_moov+faststart",
        "-f", "mp4", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def gen():
        try:
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(64 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            try:
                proc.kill()
            except Exception:
                pass

    return StreamingResponse(gen(), media_type="video/mp4")


# ---- depth (zarr) ----------------------------------------------------------
# 用 JET 风格 LUT 把 uint16(mm) → RGB PNG, 浏览器直接 <img> 显示。
# 为什么不返回 raw uint16: 浏览器不能原生显示 16-bit 单通道, 还要 JS 写 colormap shader,
# 而我们这边一次预算固定 1 帧(480x640) ≈ <2 ms numpy + ~5 ms PNG, 完全够实时拖动。
def _build_jet_lut() -> "np.ndarray":  # type: ignore[name-defined]
    import numpy as np
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        v = i / 255.0
        if v < 0.125:
            r, g, b = 0, 0, int(255 * (0.5 + 4 * v))
        elif v < 0.375:
            r, g, b = 0, int(255 * (4 * v - 0.5)), 255
        elif v < 0.625:
            r, g, b = int(255 * (4 * v - 1.5)), 255, int(255 * (-4 * v + 2.5))
        elif v < 0.875:
            r, g, b = 255, int(255 * (-4 * v + 3.5)), 0
        else:
            r, g, b = int(255 * (-4 * v + 4.5)), 0, 0
        lut[i] = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    return lut


_JET_LUT = None  # lazy: 等到第一次有 depth 请求时再算 + 装 numpy/PIL


def _zarr_open_readonly(path):
    import zarr
    return zarr.open(str(path), mode="r")


@app.get("/api/episodes/{task_id}/{subset}/{episode_id}/depth/{camera}/info")
def get_episode_depth_info(task_id: str, subset: str, episode_id: int, camera: str):
    """返回 depth zarr 的形状, 供前端按 frame_index 拖动。"""
    p = episode_depth_zarr_path(task_id, subset, episode_id, camera)
    if not p.exists():
        raise HTTPException(status_code=404, detail="depth zarr missing")
    z = _zarr_open_readonly(p)
    return {"frames": int(z.shape[0]), "height": int(z.shape[1]), "width": int(z.shape[2])}


@app.get("/api/episodes/{task_id}/{subset}/{episode_id}/depth/{camera}/frame/{frame_index}")
def get_episode_depth_frame(
    task_id: str, subset: str, episode_id: int, camera: str, frame_index: int,
    min_mm: int = Query(200, description="depth 下限 mm, 此值映射到 LUT 索引 0"),
    max_mm: int = Query(2000, description="depth 上限 mm, 此值映射到 LUT 索引 255"),
):
    """读 depth zarr 的第 N 帧, JET 上色, 返回 PNG。
    min_mm/max_mm 让前端可调节窗位 (常见 0.2m – 2m 桌面工作距离)."""
    global _JET_LUT
    p = episode_depth_zarr_path(task_id, subset, episode_id, camera)
    if not p.exists():
        raise HTTPException(status_code=404, detail="depth zarr missing")
    z = _zarr_open_readonly(p)
    n = int(z.shape[0])
    if n == 0 or frame_index < 0 or frame_index >= n:
        raise HTTPException(status_code=404, detail=f"frame {frame_index} out of range (n={n})")

    import io
    import numpy as np
    from PIL import Image
    if _JET_LUT is None:
        _JET_LUT = _build_jet_lut()

    frame = np.asarray(z[frame_index])  # (H, W) uint16
    # 0 表示 RealSense "无效深度" — 单独标黑, 否则会被映射成最近距离色
    valid = frame > 0
    span = max(1, max_mm - min_mm)
    norm = np.clip((frame.astype(np.int32) - min_mm) / span * 255.0, 0, 255).astype(np.uint8)
    rgb = _JET_LUT[norm]
    rgb[~valid] = 0  # 无效像素纯黑, 视觉上区分

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=False, compress_level=1)
    return Response(content=buf.getvalue(), media_type="image/png")


@app.delete("/api/episodes/{task_id}/{subset}/{episode_id}")
def del_episode(task_id: str, subset: str, episode_id: int, _: Role = Depends(require_admin)):
    delete_episode(task_id, subset, episode_id)
    return {"deleted": True}


# -------- ros2 ----------
@app.get("/api/joints")
def get_joints():
    return bridge.get_joint_state()


@app.get("/api/camera/{cam}/mjpeg")
def camera_mjpeg(cam: str):
    """浏览器直接用 <img src> 消费的 multipart/x-mixed-replace MJPEG 流。"""
    if not hasattr(bridge, "get_latest_jpeg"):
        raise HTTPException(status_code=503, detail="camera stream not available (mock bridge)")

    def gen():
        boundary = b"--frame"
        while True:
            jpeg = bridge.get_latest_jpeg(cam, wait_timeout=2.0)
            if not jpeg:
                continue
            yield boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " \
                  + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/camera/{cam}/snapshot")
def camera_snapshot(cam: str):
    """取最近一帧 JPEG。"""
    if not hasattr(bridge, "get_latest_jpeg"):
        raise HTTPException(status_code=503, detail="snapshot not available (mock bridge)")
    jpeg = bridge.get_latest_jpeg(cam, wait_timeout=2.0)
    if not jpeg:
        raise HTTPException(status_code=404, detail="no frame yet")
    return StreamingResponse(iter([jpeg]), media_type="image/jpeg")


# -------- Replay (P2) --------
@app.post("/api/replay/preflight")
def api_replay_preflight(req: ReplayPreflightReq):
    """Read-only: parquet metadata + pose alignment + deployment / publisher gates.
    UI calls this when toggle is flipped ON, shows result, lets user confirm."""
    return replay_mod.preflight(req.task, req.subset, req.date, req.episode_id)


@app.post("/api/replay/execute")
def api_replay_execute(req: ReplayExecuteReq, _: Role = Depends(require_admin)):
    """Actually fire replay: param set + execute=true. Admin-only."""
    return replay_mod.execute(req.task, req.subset, req.date, req.episode_id,
                              rate=req.rate, loop=req.loop)


@app.post("/api/replay/stop")
def api_replay_stop(_: Role = Depends(require_admin)):
    """Cleanup: execute=false + replay_mode=inference. Idempotent."""
    return replay_mod.stop()


@app.get("/api/replay/progress")
def api_replay_progress():
    """Latest /replay_progress (cached by ros_bridge subscriber)."""
    return replay_mod.progress()


# -------- WS --------
@app.websocket("/ws/status")
async def ws_status(ws: WebSocket):
    await hub.register(ws)
    try:
        while True:
            await ws.receive_text()  # 心跳/忽略
    except WebSocketDisconnect:
        pass
    finally:
        await hub.unregister(ws)
