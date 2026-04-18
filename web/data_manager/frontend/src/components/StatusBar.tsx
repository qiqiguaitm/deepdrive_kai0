import type { Role, StatusPayload } from "../types";

interface Props { status: StatusPayload | null; role: Role; operator: string; connected: boolean; }

function Light({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className={`seg light ${ok ? "ok" : "bad"}`}>
      <span className={`dot ${ok ? "ok" : "bad"}`}></span>{label}
    </span>
  );
}

export function collectFailures(status: StatusPayload): string[] {
  const f: string[] = [];
  if (!status.health.ros2) f.push("ROS2");
  if (!status.health.can_left) f.push("CAN-L");
  if (!status.health.can_right) f.push("CAN-R");
  if (!status.health.teleop) f.push("Teleop");
  const cams = status.cameras || {};
  const expected = ["top_head", "hand_left", "hand_right"] as const;
  const camLabel: Record<string, string> = { top_head: "俯视相机", hand_left: "左手相机", hand_right: "右手相机" };
  for (const c of expected) {
    const s = cams[c];
    if (!s) f.push(`${camLabel[c]}(缺失)`);
    else {
      // 阈值跟随 target_fps (来自 cameras.yml), 不再硬编码 25;
      // 之前把 target 从 30 调到 15 后整面都飘红就是因为这里写死了。
      const target = s.target_fps ?? 30;
      const slack = 3;  // 允许真实 fps 比 target 低 3 之内 (jitter / 进度条平均误差)
      if ((s.fps ?? 0) < target - slack) f.push(`${camLabel[c]}(fps ${s.fps ?? 0}/${target})`);
    }
  }
  if (status.recorder.state === "ERROR") f.push(`录制错误: ${status.recorder.error || ""}`);
  for (const w of status.warnings || []) f.push(w);
  return f;
}

/** 右下角常驻大号状态显示：采集员只看这个即可。 */
export function FloatingHealth({ status, connected }: { status: StatusPayload | null; connected: boolean }) {
  if (!status) {
    return (
      <div className={`floating-health ${connected ? "warn" : "bad"}`}>
        <div className="fh-icon">{connected ? "…" : "✗"}</div>
        <div className="fh-main">
          <div className="fh-label">{connected ? "等待数据" : "未连接"}</div>
        </div>
      </div>
    );
  }
  const failures = collectFailures(status);
  const ok = failures.length === 0;
  return (
    <div className={`floating-health ${ok ? "ok" : "bad"}`}>
      <div className="fh-icon">{ok ? "✓" : "✗"}</div>
      <div className="fh-main">
        <div className="fh-label">{ok ? "正常" : "不正常"}</div>
        {!ok && (
          <ul className="fh-fails">
            {failures.map(x => <li key={x}>{x}</li>)}
          </ul>
        )}
      </div>
    </div>
  );
}

export function StatusBar({ status, role, operator, connected }: Props) {
  if (!status) {
    return (
      <div className="statusbar">
        <span className={`badge ${role}`}>{role === "admin" ? "管理员" : "采集员"}</span>
        {role === "admin" && (
          <span className="seg light bad"><span className="dot bad"></span>{connected ? "等待数据…" : "未连接"}</span>
        )}
      </div>
    );
  }
  const r = status.recorder;
  const camCount = Object.keys(status.cameras).length;
  const recLight =
    r.state === "RECORDING" ? <span className="seg light rec"><span className="dot rec"></span>● REC {r.elapsed_s.toFixed(1)}s</span>
    : r.state === "SAVING" ? <span className="seg light ok"><span className="dot ok"></span>SAVING…</span>
    : r.state === "ERROR" ? <span className="seg light bad"><span className="dot bad"></span>ERROR: {r.error}</span>
    : <span className="seg light ok"><span className="dot ok"></span>IDLE</span>;

  // 采集员精简版：只展示身份、操作员、当前录制状态、任务/Episode；硬件细节留给右下角大牌显示。
  if (role !== "admin") {
    return (
      <div className="statusbar">
        <span className={`badge ${role}`}>采集员</span>
        <span className="seg">操作员: <b>{operator || "—"}</b></span>
        <span className="seg divider">|</span>
        {recLight}
        <span className="seg">任务: <b>{r.task_id || "—"}/{r.subset || "—"}</b></span>
        <span className="seg">Episode #: <b>{status.next_episode_id ?? r.episode_id ?? "—"}</b></span>
      </div>
    );
  }

  return (
    <div className="statusbar">
      <span className={`badge ${role}`}>管理员</span>
      <span className="seg">操作员: <b>{operator || "—"}</b></span>
      <Light ok={status.health.ros2} label="ROS2" />
      <Light ok={status.health.can_left} label="CAN-L" />
      <Light ok={status.health.can_right} label="CAN-R" />
      <Light ok={camCount === 3} label={`相机 ${camCount}/3`} />
      <Light ok={status.health.teleop} label="Teleop" />
      <span className="seg divider">|</span>
      {recLight}
      <span className="seg">任务: <b>{r.task_id || "—"}/{r.subset || "—"}</b></span>
      <span className="seg">Episode #: <b>{status.next_episode_id ?? r.episode_id ?? "—"}</b></span>
      <span className="seg">磁盘: <b>{status.disk_free_gb} GB</b></span>
      <span className="seg">写入: <b>{status.write_mbps} MB/s</b></span>
      <span className="spacer" style={{ flex: 1 }}></span>
      {status.warnings.map(w => <span key={w} className="warn-pill">⚠ {w}</span>)}
    </div>
  );
}
