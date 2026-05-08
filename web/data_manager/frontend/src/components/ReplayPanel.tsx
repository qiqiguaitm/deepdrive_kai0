import { useEffect, useRef, useState } from "react";
import { api, splitCompoundTaskId } from "../api/client";
import type { ReplayPreflight, ReplayProgress } from "../api/client";
import type { EpisodeMeta, Role } from "../types";

interface Props {
  ep: EpisodeMeta | null;
  role: Role;
  onCloned: (taskKey: string) => void;
  onDeleted: () => void;
}

const CAMS = ["hand_left", "top_head", "hand_right"] as const;
type Cam = (typeof CAMS)[number];

export function ReplayPanel({ ep, role, onCloned, onDeleted }: Props) {
  const videoRefs = useRef<Record<Cam, HTMLVideoElement | null>>({} as any);
  const depthRefs = useRef<Record<Cam, HTMLImageElement | null>>({} as any);

  // ── Depth viewer state (existing) ──
  const [depthInfo, setDepthInfo] = useState<Record<Cam, { frames: number } | null>>(
    { hand_left: null, top_head: null, hand_right: null });
  const [minMm, setMinMm] = useState(200);
  const [maxMm, setMaxMm] = useState(2000);
  const lastFrameRef = useRef<Record<Cam, number>>({ hand_left: -1, top_head: -1, hand_right: -1 });

  // ── Replay/execute state (P3) ──
  // executeOnRobot: true → toggle ON, will sync video to action progress + drive arm.
  // preflight: latest preflight result (re-fetched on toggle ON or before execute).
  // progress: latest /api/replay/progress while running.
  // running: true after execute fired and not yet done/stopped.
  const [executeOnRobot, setExecuteOnRobot] = useState(false);
  const [preflight, setPreflight] = useState<ReplayPreflight | null>(null);
  const [preflightLoading, setPreflightLoading] = useState(false);
  const [progress, setProgress] = useState<ReplayProgress | null>(null);
  const [running, setRunning] = useState(false);
  const [errMsg, setErrMsg] = useState<string | null>(null);
  // Cached split (task, date) for the current ep, plus 'replayable' flag.
  // Episodes with non-date task_id suffix (e.g. kai0_official_base) can't be replayed
  // via UI (CLI only); show toggle but disable.
  const epSplit = ep ? splitCompoundTaskId(ep.task_id) : null;
  const epReplayable = !!epSplit;

  // ── Depth panel sync (existing) ──
  useEffect(() => {
    if (!ep) return;
    let cancelled = false;
    setDepthInfo({ hand_left: null, top_head: null, hand_right: null });
    lastFrameRef.current = { hand_left: -1, top_head: -1, hand_right: -1 };
    Promise.all(CAMS.map(async cam => {
      try {
        const info = await api.depthInfo(ep.task_id, ep.subset, ep.episode_id, cam);
        return [cam, info] as const;
      } catch {
        return [cam, null] as const;
      }
    })).then(results => {
      if (cancelled) return;
      const next: any = {};
      for (const [cam, info] of results) next[cam] = info;
      setDepthInfo(next);
    });
    return () => { cancelled = true; };
  }, [ep?.task_id, ep?.subset, ep?.episode_id]);

  // Reset replay state when episode changes
  useEffect(() => {
    setExecuteOnRobot(false);
    setPreflight(null);
    setProgress(null);
    setRunning(false);
    setErrMsg(null);
  }, [ep?.task_id, ep?.subset, ep?.episode_id]);

  const hasAnyDepth = Object.values(depthInfo).some(d => d && d.frames > 0);

  const syncDepthFrame = (cam: Cam) => {
    const v = videoRefs.current[cam];
    const img = depthRefs.current[cam];
    const info = depthInfo[cam];
    if (!v || !img || !info || !ep) return;
    const total = isFinite(v.duration) && v.duration > 0 ? v.duration : (ep.duration_s || 1);
    const idx = Math.min(info.frames - 1, Math.max(0, Math.floor((v.currentTime / total) * info.frames)));
    if (idx === lastFrameRef.current[cam]) return;
    lastFrameRef.current[cam] = idx;
    img.src = api.depthFrameUrl(ep.task_id, ep.subset, ep.episode_id, cam, idx, minMm, maxMm);
  };

  useEffect(() => {
    for (const cam of CAMS) {
      lastFrameRef.current[cam] = -1;
      syncDepthFrame(cam);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [minMm, maxMm, depthInfo]);

  // ── Replay progress polling (action-only, no video sync) ──
  // 5Hz polls /api/replay/progress and updates the progress bar.
  // Videos are NOT auto-synced — user's call whether to play them in browser.
  // Why: video↔action sync was finicky (stale buffer cache, race on confirm)
  // and not the actual goal — we only need to drive the arm + show progress.
  // Stale-progress guard: ignore progress msgs older than 2s (= leftover from
  // a previous session that backend's clear_replay_progress missed).
  useEffect(() => {
    if (!running) return;
    let cancelled = false;
    const tStart = Date.now();
    const tick = async () => {
      try {
        const p = await api.replayProgress();
        if (cancelled) return;
        // Discard stale msg from a prior session (age_s>2 in first 1s of run).
        const age = p.age_s ?? 0;
        const sessionRunningMs = Date.now() - tStart;
        if (sessionRunningMs < 1000 && age > 2.0) {
          // Still warming up; the cached msg is from before clear_replay_progress.
          return;
        }
        setProgress(p);
        if (p.done) setRunning(false);
      } catch (e: any) {
        if (cancelled) return;
        setErrMsg(`progress error: ${e?.message || e}`);
        setRunning(false);
      }
    };
    tick();
    const tm = setInterval(tick, 200);
    return () => { cancelled = true; clearInterval(tm); };
  }, [running]);

  // ── Toggle handler ──
  // ON  → fetch preflight (no side effects on robot). Show diag + arm "确认执行" button.
  // OFF → flip back, no robot side effects.
  const onToggleExecute = async (next: boolean) => {
    setErrMsg(null);
    if (!next || !ep || !epSplit) {
      setExecuteOnRobot(next);
      setPreflight(null);
      return;
    }
    setExecuteOnRobot(true);
    setPreflightLoading(true);
    try {
      const pf = await api.replayPreflight(epSplit.task, ep.subset, epSplit.date, ep.episode_id);
      setPreflight(pf);
      if (!pf.ok) setErrMsg(`preflight: ${pf.reason || JSON.stringify({step: pf.step, ...pf})}`);
    } catch (e: any) {
      setErrMsg(`preflight failed: ${e?.message || e}`);
    } finally {
      setPreflightLoading(false);
    }
  };

  const onExecuteConfirm = async () => {
    if (!ep || !epSplit || !preflight) return;
    setErrMsg(null);
    try {
      const r = await api.replayExecute(epSplit.task, ep.subset, epSplit.date, ep.episode_id, 1.0);
      if (!r.ok) {
        setErrMsg(`execute: ${r.reason || JSON.stringify(r.trace)}`);
        return;
      }
      setProgress(null);
      setRunning(true);
    } catch (e: any) {
      setErrMsg(`execute failed: ${e?.message || e}`);
    }
  };

  const onStop = async () => {
    try {
      await api.replayStop();
    } catch (e: any) {
      setErrMsg(`stop failed: ${e?.message || e}`);
    } finally {
      setRunning(false);
    }
  };

  // ── Manual playback (existing, when toggle is OFF) ──
  const playAll = () => Object.values(videoRefs.current).forEach(r => r && r.play());
  const pauseAll = () => Object.values(videoRefs.current).forEach(r => r && r.pause());

  if (!ep) return <div className="panel area-replay"><h3>回放</h3>从左侧选择一条 episode。</div>;

  // Compute progress display values
  const pIdx = progress?.idx ?? 0;
  const pTotal = progress?.total ?? (preflight?.expected_buffer_total ?? 0);
  const pPct = pTotal > 0 ? Math.min(100, (pIdx / pTotal) * 100) : 0;
  const phaseLabel = (() => {
    if (!running && !progress) return "";
    const homeN = preflight?.home_n_planned ?? 0;
    if (progress?.done) return "完成";
    if (pIdx < homeN) return `home (${pIdx}/${homeN})`;
    return `episode (${pIdx - homeN}/${pTotal - homeN})`;
  })();

  return (
    <div className="panel area-replay">
      <h3>回放: {ep.task_id}/{ep.subset} #{ep.episode_id} · {ep.duration_s.toFixed(1)}s · {ep.success ? "成功" : "失败"}</h3>
      <div style={{ marginBottom: 6, color: "var(--muted)" }}>prompt: <b>{ep.prompt || "—"}</b></div>

      <div className="replay-vids">
        {CAMS.map(cam => (
          <video key={cam} ref={el => (videoRefs.current[cam] = el)}
                 src={api.videoUrl(ep.task_id, ep.subset, ep.episode_id, cam)}
                 onTimeUpdate={() => syncDepthFrame(cam)}
                 onSeeked={() => syncDepthFrame(cam)}
                 onLoadedMetadata={() => syncDepthFrame(cam)}
                 controls={false} muted preload="metadata" />
        ))}
      </div>

      {hasAnyDepth && (
        <>
          <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "6px 0", color: "var(--muted)" }}>
            <span>深度 (JET)</span>
            <label>min mm <input type="number" value={minMm} step={50} min={0}
                                 onChange={e => setMinMm(parseInt(e.target.value || "0"))}
                                 style={{ width: 70 }} /></label>
            <label>max mm <input type="number" value={maxMm} step={100} min={1}
                                 onChange={e => setMaxMm(parseInt(e.target.value || "1"))}
                                 style={{ width: 80 }} /></label>
            <span style={{ fontSize: 11 }}>(0.2m – 2m 桌面默认)</span>
          </div>
          <div className="replay-vids">
            {CAMS.map(cam => (
              depthInfo[cam] && depthInfo[cam]!.frames > 0
                ? <img key={cam} ref={el => (depthRefs.current[cam] = el)}
                       alt={`${cam} depth`}
                       style={{ background: "#000", width: "100%", aspectRatio: "640/480" }} />
                : <div key={cam} style={{ background: "#222", color: "#888",
                       display: "flex", alignItems: "center", justifyContent: "center",
                       aspectRatio: "640/480" }}>无深度数据</div>
            ))}
          </div>
        </>
      )}

      <div className="controls" style={{ flexWrap: "wrap" }}>
        {/* Manual video controls — always available (videos are NOT synced to robot
            anymore; user plays/pauses the visual track independently of the arm). */}
        <button onClick={playAll}>▶ 播放</button>
        <button onClick={pauseAll}>⏸ 暂停</button>
        {running && (
          <button className="btn-discard" onClick={onStop}>⏹ 停止真机回放</button>
        )}
        <button onClick={() => onCloned(`${ep.task_id}/${ep.subset}`)} disabled={running}>以此配置新建采集</button>
        {role === "admin" && !running && (
          <button className="btn-discard"
            onClick={async () => {
              if (!confirm(`删除 ${ep.task_id}/${ep.subset} #${ep.episode_id}? 不可恢复。`)) return;
              await api.delEpisode(ep.task_id, ep.subset, ep.episode_id);
              onDeleted();
            }}>删除</button>
        )}
      </div>

      {/* ── 真实执行 toggle (admin only, episodes with parseable date suffix only) ── */}
      {role === "admin" && (
        <div style={{
          marginTop: 12, padding: 10,
          border: "1px solid var(--border)", borderRadius: 6,
          background: executeOnRobot ? "rgba(255,80,80,0.05)" : "transparent",
        }}>
          <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: epReplayable ? "pointer" : "not-allowed" }}>
            <input type="checkbox" checked={executeOnRobot}
                   disabled={!epReplayable || running}
                   onChange={e => onToggleExecute(e.target.checked)} />
            <strong>真实执行 (将动作下发到机械臂)</strong>
            {!epReplayable && <span style={{ color: "var(--bad)", fontSize: 12 }}>
              ⚠ 该 episode task_id 无日期后缀, UI 暂不支持; 用 CLI start_replay_test.sh
            </span>}
            {executeOnRobot && epReplayable && (
              <span style={{ color: "var(--bad)", fontSize: 12 }}>⚠ 机器臂会动 (视频独立, 不同步)</span>
            )}
          </label>

          {executeOnRobot && preflightLoading && <p style={{ margin: "6px 0", color: "var(--muted)" }}>preflight 检查中…</p>}

          {executeOnRobot && preflight && (
            <div style={{ marginTop: 8, fontSize: 13 }}>
              <div>
                <span style={{ color: "var(--muted)" }}>状态:</span>{" "}
                {preflight.ok
                  ? <span style={{ color: "var(--good)" }}>✓ 可执行</span>
                  : <span style={{ color: "var(--bad)" }}>✗ 不可执行</span>}
                {" · "}
                <span style={{ color: "var(--muted)" }}>部署:</span>{" "}
                <code>{preflight.deployment_mode}</code>
                {" · "}
                <span style={{ color: "var(--muted)" }}>replay node:</span>{" "}
                {preflight.policy_inference_alive
                  ? <code>{preflight.target_node}</code>
                  : <span style={{ color: "var(--bad)" }}>✗ 无活节点</span>}
                {preflight.publisher_conflict?.length > 0 &&
                  <> {" · "} <span style={{ color: "var(--bad)" }}>publisher 冲突: {preflight.publisher_conflict.join(", ")}</span></>}
              </div>
              <div style={{ marginTop: 4 }}>
                <span style={{ color: "var(--muted)" }}>起点对齐:</span>{" "}
                {preflight.aligned
                  ? <span style={{ color: "var(--good)" }}>✓ 当前姿态在 5° 内</span>
                  : <>
                      <span style={{ color: "var(--warn, #b80)" }}>偏 {preflight.max_diff_deg.toFixed(1)}°</span>
                      {preflight.auto_home_will_trigger &&
                        <span style={{ color: "var(--muted)" }}> · auto-home 会先 prepend {preflight.home_n_planned} 帧 ({(preflight.home_n_planned / preflight.publish_rate).toFixed(1)}s) 慢挪到起点</span>
                      }
                    </>
                }
              </div>
              <div style={{ marginTop: 4, color: "var(--muted)" }}>
                episode {preflight.frames} 帧 / {preflight.duration_s.toFixed(1)}s @ {preflight.fps.toFixed(1)} Hz
                · 总播放约 {(preflight.expected_buffer_total / preflight.publish_rate).toFixed(1)}s
              </div>
              {!running && preflight.ok && (
                <button onClick={onExecuteConfirm}
                        style={{ marginTop: 8, background: "#c33", color: "#fff", padding: "6px 12px" }}>
                  ▶ 确认下发到机械臂
                </button>
              )}
            </div>
          )}

          {/* Progress bar (visible while running OR after done if no nav) */}
          {(running || progress) && (
            <div style={{ marginTop: 8 }}>
              <div style={{ height: 8, background: "#333", borderRadius: 4, overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${pPct}%`,
                              background: progress?.done ? "var(--good)" : "#39c",
                              transition: "width 0.2s" }} />
              </div>
              <div style={{ marginTop: 4, fontSize: 12, color: "var(--muted)" }}>
                phase: <b>{phaseLabel}</b> · {pIdx}/{pTotal} · {pPct.toFixed(1)}%
              </div>
            </div>
          )}

          {errMsg && <p style={{ color: "var(--bad)", marginTop: 6, fontSize: 12 }}>{errMsg}</p>}
        </div>
      )}

      {ep.incomplete && <p style={{ color: "var(--bad)" }}>⚠ {ep.incomplete_reason}</p>}
    </div>
  );
}
