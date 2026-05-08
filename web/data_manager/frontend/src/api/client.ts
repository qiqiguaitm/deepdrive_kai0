import type {
  EpisodeMeta, JointState, RecorderSnap, Role, StatsResponse, Template,
} from "../types";

export function getRole(): Role {
  return (localStorage.getItem("role") as Role) || "collector";
}
export function setRole(r: Role) { localStorage.setItem("role", r); }
export function getOperator(): string { return localStorage.getItem("operator") || ""; }
export function setOperator(o: string) { localStorage.setItem("operator", o); }

async function req<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("X-Role", getRole());
  if (init.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const r = await fetch(path, { ...init, headers });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export const api = {
  templates: (only_enabled = false) =>
    req<Template[]>(`/api/templates?only_enabled=${only_enabled}`),
  upsertTemplate: (t: Template) =>
    req<Template>(`/api/templates/${t.id}`, { method: "PUT", body: JSON.stringify(t) }),
  delTemplate: (id: string) =>
    req<{ deleted: boolean }>(`/api/templates/${id}`, { method: "DELETE" }),

  recorder: () => req<RecorderSnap>(`/api/recorder`),
  // 前端把当前选择同步给后端, 供踏板等"无鼠标"入口冷启动用。
  // template_id/operator 传 null 表示不改这一项; 传 "" 表示清空。
  setSession: (template_id: string | null, operator: string | null) =>
    req<{ template_id: string | null; operator: string | null }>(
      `/api/session`,
      { method: "PUT", body: JSON.stringify({ template_id, operator }) },
    ),
  startRec: (template_id: string, operator: string) =>
    req<RecorderSnap>(`/api/recorder/start`, { method: "POST", body: JSON.stringify({ template_id, operator }) }),
  saveRec: (success: boolean, note: string, scene_tags: string[]) =>
    req<{ saved_episode_id: number }>(`/api/recorder/save`, { method: "POST", body: JSON.stringify({ success, note, scene_tags }) }),
  discardRec: () => req<RecorderSnap>(`/api/recorder/discard`, { method: "POST" }),
  estop: () => req<{ ok: boolean }>(`/api/recorder/estop`, { method: "POST" }),

  stats: () => req<StatsResponse>(`/api/stats`),
  rescan: () => req<{ rescanned: number }>(`/api/stats/rescan`, { method: "POST" }),

  episodes: (q: Record<string, string | undefined>) => {
    const usp = new URLSearchParams();
    Object.entries(q).forEach(([k, v]) => v != null && v !== "" && usp.set(k, v));
    return req<EpisodeMeta[]>(`/api/episodes?${usp}`);
  },
  delEpisode: (task: string, subset: string, ep: number) =>
    req<{ deleted: boolean }>(`/api/episodes/${task}/${subset}/${ep}`, { method: "DELETE" }),
  videoUrl: (task: string, subset: string, ep: number, cam: string) =>
    `/api/episodes/${task}/${subset}/${ep}/video/${cam}`,
  // 深度: 一帧一张 PNG (后端 JET 上色), 前端按视频时间戳 → frame_index 拉
  depthFrameUrl: (task: string, subset: string, ep: number, cam: string,
                  frame: number, minMm = 200, maxMm = 2000) =>
    `/api/episodes/${task}/${subset}/${ep}/depth/${cam}/frame/${frame}?min_mm=${minMm}&max_mm=${maxMm}`,
  depthInfo: (task: string, subset: string, ep: number, cam: string) =>
    req<{ frames: number; height: number; width: number }>(
      `/api/episodes/${task}/${subset}/${ep}/depth/${cam}/info`),

  joints: () => req<JointState>(`/api/joints`),

  // ── Replay (P3) ──
  replayPreflight: (task: string, subset: string, date: string, ep: number) =>
    req<ReplayPreflight>(`/api/replay/preflight`, {
      method: "POST",
      body: JSON.stringify({ task, subset, date, episode_id: ep }),
    }),
  replayExecute: (task: string, subset: string, date: string, ep: number, rate = 1.0) =>
    req<ReplayExecute>(`/api/replay/execute`, {
      method: "POST",
      body: JSON.stringify({ task, subset, date, episode_id: ep, rate, loop: false }),
    }),
  replayProgress: () => req<ReplayProgress>(`/api/replay/progress`),
  replayStop: () => req<ReplayStop>(`/api/replay/stop`, { method: "POST" }),
};

export interface ReplayPreflight {
  ok: boolean;
  parquet_path: string;
  frames: number;
  fps: number;
  duration_s: number;
  action0?: number[];
  current_state?: number[] | null;
  max_diff_deg: number;
  per_joint_diff_deg: number[];
  aligned: boolean;
  deployment_mode: string;
  policy_inference_alive: boolean;
  target_node?: string | null;   // '/replay' or '/policy_inference', null if neither alive
  publisher_conflict: string[];
  auto_home_will_trigger: boolean;
  home_n_planned: number;
  publish_rate: number;
  expected_buffer_total: number;
  step?: string;
  reason?: string;
}

export interface ReplayExecute {
  ok: boolean;
  started?: boolean;
  step?: string;
  reason?: string;
  trace?: { step: string; ok: boolean; msg: string }[];
}

export interface ReplayProgress {
  ok: boolean;
  reason?: string;
  idx?: number;
  total?: number;
  done?: boolean;
  fraction?: number;
  age_s?: number;
}

export interface ReplayStop {
  ok: boolean;
  stopped?: boolean;
  trace?: { step: string; ok: boolean; msg: string }[];
}

// Compound task_id `Task_A_2026-04-16` → `{task: 'Task_A', date: '2026-04-16'}`.
// Returns null if compound has no YYYY-MM-DD suffix (e.g. our kai0_official_base
// test data, which is replay-only via CLI not via UI episode browser).
export function splitCompoundTaskId(compound: string): { task: string; date: string } | null {
  const m = compound.match(/^(.+)_(\d{4}-\d{2}-\d{2})$/);
  return m ? { task: m[1], date: m[2] } : null;
}
