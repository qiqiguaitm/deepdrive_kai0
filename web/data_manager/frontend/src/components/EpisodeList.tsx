import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { EpisodeMeta, Role } from "../types";

interface Props {
  selected: EpisodeMeta | null;
  onSelect: (e: EpisodeMeta) => void;
  refreshKey: number;
  role: Role;
  onDeleted: (e: EpisodeMeta) => void;
}

const epKey = (e: EpisodeMeta) => `${e.task_id}/${e.subset}/${e.episode_id}`;

export function EpisodeList({ selected, onSelect, refreshKey, role, onDeleted }: Props) {
  const [items, setItems] = useState<EpisodeMeta[]>([]);
  const [task, setTask] = useState("");
  const [subset, setSubset] = useState("");
  const [okFilter, setOkFilter] = useState("");
  const [kw, setKw] = useState("");
  const [checked, setChecked] = useState<Set<string>>(new Set());
  const [bulkBusy, setBulkBusy] = useState(false);

  const load = async () => {
    try {
      const r = await api.episodes({
        task_id: task || undefined,
        subset: subset || undefined,
        success: okFilter === "" ? undefined : okFilter,
        prompt_kw: kw || undefined,
      });
      setItems(r);
      setChecked(prev => {
        const keep = new Set<string>();
        const alive = new Set(r.map(epKey));
        prev.forEach(k => { if (alive.has(k)) keep.add(k); });
        return keep;
      });
    } catch {}
  };
  useEffect(() => { load(); }, [task, subset, okFilter, kw, refreshKey]);

  const toggle = (k: string) => setChecked(prev => {
    const next = new Set(prev);
    next.has(k) ? next.delete(k) : next.add(k);
    return next;
  });
  const allChecked = items.length > 0 && items.every(e => checked.has(epKey(e)));
  const toggleAll = () => setChecked(allChecked ? new Set() : new Set(items.map(epKey)));

  const bulkDelete = async () => {
    const targets = items.filter(e => checked.has(epKey(e)));
    if (targets.length === 0) return;
    if (!confirm(`删除 ${targets.length} 条 episode? 不可恢复。`)) return;
    setBulkBusy(true);
    const failed: string[] = [];
    for (const e of targets) {
      try {
        await api.delEpisode(e.task_id, e.subset, e.episode_id);
        onDeleted(e);
      } catch (err) {
        failed.push(`${epKey(e)}: ${err}`);
      }
    }
    setBulkBusy(false);
    if (failed.length) alert(`部分删除失败:\n${failed.join("\n")}`);
  };

  return (
    <div className="panel area-list">
      <h3>历史 Episode</h3>
      <div style={{ display: "grid", gap: 4, marginBottom: 8 }}>
        <input placeholder="task_id (Task_A)" value={task} onChange={e => setTask(e.target.value)} />
        <select value={subset} onChange={e => setSubset(e.target.value)}>
          <option value="">所有 subset</option>
          <option value="base">base</option>
          <option value="dagger">dagger</option>
        </select>
        <select value={okFilter} onChange={e => setOkFilter(e.target.value)}>
          <option value="">所有结果</option>
          <option value="true">成功</option>
          <option value="false">失败</option>
        </select>
        <input placeholder="prompt 关键词" value={kw} onChange={e => setKw(e.target.value)} />
      </div>
      {role === "admin" && (
        <div style={{
          display: "flex", alignItems: "center", gap: 8, marginBottom: 6,
          padding: "4px 6px", background: "var(--panel2)", borderRadius: 3, fontSize: 12,
        }}>
          <label style={{ display: "flex", alignItems: "center", gap: 4, cursor: items.length ? "pointer" : "default" }}>
            <input type="checkbox"
                   checked={allChecked}
                   disabled={items.length === 0}
                   onChange={toggleAll} />
            全选 ({checked.size}/{items.length})
          </label>
          <span style={{ flex: 1 }} />
          <button
            className="btn-discard"
            disabled={checked.size === 0 || bulkBusy}
            onClick={bulkDelete}
            style={{ padding: "3px 10px", fontSize: 12 }}>
            {bulkBusy ? "删除中…" : `批量删除 ${checked.size || ""}`}
          </button>
        </div>
      )}
      <div className="ep-list">
        {items.length === 0 && <div style={{ color: "var(--muted)" }}>暂无</div>}
        {items.map(e => {
          const k = epKey(e);
          const isChecked = checked.has(k);
          return (
          <div key={k}
               className={`ep-row ${selected?.episode_id === e.episode_id && selected?.task_id === e.task_id ? "active" : ""}`}
               onClick={() => onSelect(e)}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              {role === "admin" && (
                <input type="checkbox"
                       checked={isChecked}
                       onClick={ev => ev.stopPropagation()}
                       onChange={() => toggle(k)}
                       title="选中以便批量删除"
                       style={{ margin: 0, cursor: "pointer" }} />
              )}
              <span style={{ flex: 1, minWidth: 0 }}>
                <b>{e.task_id}/{e.subset}</b> #{e.episode_id.toString().padStart(6, "0")}
                {e.success ? " ✅" : " ❌"} {e.incomplete ? " ⚠残缺" : ""}
              </span>
              {role === "admin" && (
                <button
                  title="删除该 episode (不可恢复)"
                  onClick={async ev => {
                    ev.stopPropagation();
                    if (!confirm(`删除 ${e.task_id}/${e.subset} #${e.episode_id}? 不可恢复。`)) return;
                    try {
                      await api.delEpisode(e.task_id, e.subset, e.episode_id);
                      onDeleted(e);
                    } catch (err) {
                      alert(`删除失败: ${err}`);
                    }
                  }}
                  style={{
                    background: "transparent", color: "var(--bad)", border: "1px solid var(--border)",
                    borderRadius: 3, padding: "0 6px", fontSize: 12, cursor: "pointer", lineHeight: "18px",
                  }}>×</button>
              )}
            </div>
            <div className="meta">
              {e.duration_s.toFixed(1)}s · {(e.size_bytes / 1024).toFixed(1)} KB · {e.operator || "—"}
            </div>
            <div className="meta" title={e.prompt} style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {e.prompt}
            </div>
            <div className="meta">{new Date(e.created_at * 1000).toLocaleString()}</div>
          </div>
          );
        })}
      </div>
    </div>
  );
}
