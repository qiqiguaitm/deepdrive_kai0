#!/usr/bin/env python3
"""Task E 4-experiment progress dashboard + health check + resume suggester.

Shows:
  - live training step / rate / ETA from log tail
  - loss curve snapshots at key steps (from wandb offline LevelDB)
  - inline-eval val MAE @1/@10/@25/@50 per saved ckpt
  - GPU util / VRAM + RAM snapshot
  - HEALTH: alive? progressing? last heartbeat? if dead, diagnosed cause + suggested resume cmd

Usage:
  scripts/check_task_e_progress.py                      # one-shot dashboard
  scripts/check_task_e_progress.py --watch 30           # refresh every 30s
  scripts/check_task_e_progress.py --auto-resume        # print health, auto-relaunch dead ones
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_SITE = PROJECT_ROOT / "kai0/.venv/lib/python3.11/site-packages"
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))

LOGS_DIR = PROJECT_ROOT / "logs"
WANDB_DIR = PROJECT_ROOT / "kai0/wandb"

EXPERIMENTS = [
    # Phase 2 — training-side levers beyond Phase 1 best (E2/14999 @1=0.0262)
    # (label,  exp_name,             gpu, pretty,            config_name)
    ("T1-1", "t1_lora16",           0,   "visionLoRA r16",  "pi05_stand_box_vision_lora16"),
    ("T1-2", "t1_lora32",           3,   "visionLoRA r32",  "pi05_stand_box_vision_lora32"),
    ("T2",   "t2_e2_ft",            1,   "E2+ultralowLR",   "pi05_stand_box_e2_ft"),
    ("T6",   "t6_kai0_allgood",     2,   "kai0+allgood",    "pi05_stand_box_kai0_allgood"),
]

CKPT_ROOT = PROJECT_ROOT / "kai0/checkpoints"
# GPUs with healthy NUMA (sim01 sockets 1,2 have 0 MB RAM)
HEALTHY_GPUS = {0, 3}

LOSS_STEPS = [100, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 14999]


def latest_wandb_run(exp_name: str) -> Path | None:
    """Most recent wandb offline-run-* dir matching --exp_name=<exp_name>."""
    runs = sorted(WANDB_DIR.glob("offline-run-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in runs:
        meta = d / "files" / "wandb-metadata.json"
        if not meta.exists():
            continue
        try:
            args = json.loads(meta.read_text()).get("args", [])
        except Exception:
            continue
        if f"--exp_name={exp_name}" in args:
            return d
    return None


def scan_wandb_history(wandb_file: Path) -> dict[int, dict]:
    """step -> {key: value} for every history record."""
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2
    ds = DataStore()
    ds.open_for_scan(str(wandb_file))
    out: dict[int, dict] = {}
    while True:
        try:
            data = ds.scan_data()
        except AssertionError:
            break
        if data is None:
            break
        pb = wandb_internal_pb2.Record()
        pb.ParseFromString(data)
        if pb.WhichOneof("record_type") != "history":
            continue
        d = {}
        for it in pb.history.item:
            k = it.nested_key[0] if it.nested_key else it.key
            try:
                d[k] = json.loads(it.value_json)
            except Exception:
                pass
        s = d.get("_step")
        if s is not None:
            out[s] = d
    return out


PROGRESS_RE = re.compile(
    r"Progress on: ([\d.]+)k?it/([\d.]+)k?it rate:([\d.]+)it/s remaining:([\d:]+)"
)


def tail_progress(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    try:
        out = subprocess.check_output(
            ["tac", str(log_path)], stderr=subprocess.DEVNULL, text=True
        )
    except Exception:
        try:
            out = log_path.read_text()[-20000:][::-1]
            out = "\n".join(reversed(log_path.read_text().splitlines()[-500:]))
        except Exception:
            return None
    for line in out.splitlines()[:5000]:
        m = PROGRESS_RE.search(line)
        if m:
            cur = float(m.group(1))
            total = float(m.group(2))
            if "kit/" in line:
                cur *= 1000
                total *= 1000
            return {"step": int(cur), "total": int(total), "rate": float(m.group(3)), "eta": m.group(4)}
    return None


INLINE_EVAL_RE = re.compile(
    r"\[inline-eval\] step=(\d+)\s+MAE@1=([\d.]+)\s+@10=([\d.]+)\s+@25=([\d.]+)\s+@50=([\d.]+)"
)


def inline_eval_records(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    out = []
    for line in log_path.read_text().splitlines():
        m = INLINE_EVAL_RE.search(line)
        if m:
            out.append({
                "step": int(m.group(1)),
                "mae_1": float(m.group(2)),
                "mae_10": float(m.group(3)),
                "mae_25": float(m.group(4)),
                "mae_50": float(m.group(5)),
            })
    return out


def find_process_for_exp(exp_name: str) -> int | None:
    """Return python PID running this exp_name, or None if not found."""
    try:
        out = subprocess.check_output(["pgrep", "-fa", f"scripts/train.py .* --exp_name={exp_name}"], text=True)
    except subprocess.CalledProcessError:
        return None
    for line in out.strip().splitlines():
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        pid = int(parts[0])
        cmdline = parts[1]
        if "python3" in cmdline or ".venv" in cmdline:
            return pid
    # fall back: any matching (uv wrapper PID)
    for line in out.strip().splitlines():
        parts = line.split(None, 1)
        if parts:
            return int(parts[0])
    return None


def last_progress_age_seconds(log_path: Path) -> float | None:
    """Seconds since the last 'Progress on:' line timestamp, or None if no progress line."""
    if not log_path.exists():
        return None
    try:
        out = subprocess.check_output(["tac", str(log_path)], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    # look for most recent "HH:MM:SS.ms [I] Progress on: ..." line
    ts_re = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.\d+\s+\[I\].*Progress on: ")
    for line in out.splitlines()[:2000]:
        m = ts_re.match(line)
        if m:
            h, m_, s = map(int, m.groups())
            log_mtime = log_path.stat().st_mtime
            import datetime as _dt
            log_time = _dt.datetime.fromtimestamp(log_mtime)
            line_time = log_time.replace(hour=h, minute=m_, second=s, microsecond=0)
            # if line_time > log_time, rollover — use log mtime
            return max(0.0, (log_time - line_time).total_seconds())
    return None


def list_saved_steps(config_name: str, exp_name: str) -> list[int]:
    d = CKPT_ROOT / config_name / exp_name
    if not d.exists():
        return []
    return sorted(int(p.name) for p in d.iterdir() if p.name.isdigit())


def diagnose_failure(log_path: Path) -> str:
    """Inspect log tail and return one-line diagnosis of the crash cause."""
    if not log_path.exists():
        return "no log file"
    txt = log_path.read_text()[-30000:]
    tail_lines = txt.splitlines()[-100:]
    hay = "\n".join(tail_lines)
    # Check for common failure modes, in priority order
    if re.search(r"Traceback \(most recent call last\)", hay):
        # extract last ~3 lines of traceback
        err_lines = [ln for ln in tail_lines if re.search(r"Error|Exception|Killed|Traceback|\[inline-eval\] failed", ln)]
        if err_lines:
            return "python-exception: " + err_lines[-1].strip()[:120]
        return "python-exception (see log)"
    if "Killed" in tail_lines[-5:] or any("OOM" in ln for ln in tail_lines[-20:]):
        return "oom-killed (out of RAM)"
    if re.search(r"Failed to find hwloc NUMA node [12]", hay):
        # inline-eval-then-silence is the v3/v4 silent death pattern
        if any("[inline-eval]" in ln for ln in tail_lines[-5:]):
            return "silent crash post-inline-eval on bad NUMA node (GPU1/GPU2) — numactl membind required"
    if any("[inline-eval]" in ln for ln in tail_lines[-3:]) and not any(
        "Progress on:" in ln for ln in tail_lines[-3:]
    ):
        return "silent crash post-inline-eval (no traceback, likely SIGKILL/SIGBUS)"
    if "CUDA error" in hay or "illegal memory access" in hay:
        return "CUDA error (illegal memory access — likely NUMA / NCCL)"
    return "unknown — last log line: " + (tail_lines[-1][:120] if tail_lines else "(empty)")


def health_check(label: str, exp_name: str, gpu: int, pretty: str, config_name: str) -> dict:
    log_path = LOGS_DIR / f"train_{exp_name}.log"
    pid = find_process_for_exp(exp_name)
    progress = tail_progress(log_path) if log_path.exists() else None
    age = last_progress_age_seconds(log_path)
    saved_steps = list_saved_steps(config_name, exp_name)
    status = "unknown"
    diag = ""

    if pid is not None and progress is not None:
        if age is not None and age < 60:
            status = "OK"
        elif age is not None and age < 300:
            status = "STALLED"
        else:
            status = "HANG"
    elif pid is None and progress is not None:
        if progress["step"] >= progress["total"] - 1:
            status = "DONE"
        else:
            status = "DEAD"
            diag = diagnose_failure(log_path)
    elif pid is None and progress is None:
        status = "NEVER-STARTED"

    return {
        "label": label, "exp": exp_name, "gpu": gpu, "pretty": pretty,
        "config": config_name, "pid": pid, "progress": progress, "age": age,
        "saved_steps": saved_steps, "status": status, "diag": diag,
    }


def resume_command(h: dict) -> str | None:
    """Emit the shell command to resume this experiment from its last saved ckpt."""
    if not h["saved_steps"]:
        return None
    target_gpu = h["gpu"]
    note = ""
    if h["gpu"] not in HEALTHY_GPUS:
        # find a free healthy GPU
        free_healthy = [g for g in (0, 3) if _gpu_idle(g)]
        if free_healthy:
            target_gpu = free_healthy[0]
            note = f"  # switched from GPU{h['gpu']} (bad NUMA) to GPU{target_gpu}"
        else:
            note = f"  # WARN: original GPU{h['gpu']} has bad NUMA; no healthy GPU free"
    last_step = h["saved_steps"][-1]
    return (
        f"./scripts/start_train.sh {h['config']} {h['exp']} {target_gpu} "
        f"--resume{note}"
    )


def _gpu_idle(gpu_idx: int) -> bool:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu_idx), "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True
        )
        return int(float(out.strip())) < 500
    except Exception:
        return False


def auto_resume_exec(h: dict) -> tuple[bool, str]:
    """Actually launch the resume command (via start_train.sh with --resume).
    Returns (success, message). Requires config.resume support in train.py.
    """
    cmd = resume_command(h)
    if cmd is None:
        return False, f"{h['label']}: no saved ckpt, cannot resume"
    # Extract GPU from the suggested cmd to ensure we don't step on a running run
    try:
        exp = h["exp"]
        # strip trailing comment before executing
        shell_cmd = cmd.split("  #")[0].strip()
        # run detached — start_train.sh itself does nohup+disown
        proc = subprocess.run(shell_cmd, shell=True, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            return False, f"{h['label']} resume failed: {proc.stderr[:200]}"
        return True, f"{h['label']} resumed: {shell_cmd}"
    except Exception as e:
        return False, f"{h['label']} resume exception: {e}"


def gpu_ram_snapshot() -> tuple[list[dict], dict]:
    """Return per-GPU info list and a single RAM dict."""
    gpus = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"], text=True
        )
        for line in out.strip().splitlines():
            idx, util, used, total, power = [x.strip() for x in line.split(",")]
            gpus.append({
                "idx": int(idx), "util": int(util), "mem_used": int(float(used)),
                "mem_total": int(float(total)), "power": float(power),
            })
    except Exception:
        pass
    ram = {"total": 0, "used": 0, "avail": 0}
    try:
        out = subprocess.check_output(["free", "-g"], text=True)
        row = out.splitlines()[1].split()
        ram = {"total": int(row[1]), "used": int(row[2]), "avail": int(row[-1])}
    except Exception:
        pass
    return gpus, ram


# --- rendering ---

def fmt_loss_cell(v):
    if v is None:
        return "    -"
    return f"{v:.4f}"


STATUS_ICON = {"OK": "✅", "STALLED": "🟡", "HANG": "⚠️", "DEAD": "☠️",
               "DONE": "🏁", "NEVER-STARTED": "⚪"}


def render(args) -> str:
    lines = []
    gpus, ram = gpu_ram_snapshot()
    now = time.strftime("%H:%M:%S")

    # Health check pass for all experiments
    healths = [health_check(label, exp, gpu, pretty, cfg)
               for label, exp, gpu, pretty, cfg in EXPERIMENTS]

    lines.append(f"══════ Task E 4-experiment dashboard @ {now} ══════")
    # GPU / RAM
    gpu_str = "  ".join(
        f"GPU{g['idx']}:{g['util']:>3}% {g['mem_used']:>5}/{g['mem_total']:>5}MB {g['power']:>4.0f}W"
        for g in gpus
    )
    lines.append(gpu_str)
    lines.append(f"RAM: used={ram['used']}/{ram['total']}GB  avail={ram['avail']}GB")
    lines.append("")

    # Progress + health table
    lines.append(f"{'':2} {'exp':<4} {'GPU':>3} {'cfg':<14} {'step':>7}/total {'rate':>8} {'ETA':>10} {'status':<14} {'PID':>7}")
    lines.append("-" * 82)
    for h in healths:
        p = h["progress"]
        icon = STATUS_ICON.get(h["status"], "?")
        line_left = f"{icon:<2} {h['label']:<4} {h['gpu']:>3} {h['pretty']:<14}"
        if p:
            line_mid = f" {p['step']:>7}/{p['total']:<5} {p['rate']:>6.2f}it/s {p['eta']:>10}"
        else:
            line_mid = f" {'?':>7}/{'?':<5} {'?':>8} {'?':>10}"
        status_str = h["status"]
        pid_str = str(h["pid"]) if h["pid"] else "-"
        lines.append(line_left + line_mid + f" {status_str:<14} {pid_str:>7}")
    lines.append("")

    # Health diagnosis for any dead/hung runs
    abnormal = [h for h in healths if h["status"] in ("DEAD", "HANG", "STALLED")]
    if abnormal:
        lines.append("── ❗ abnormal runs — diagnosis + resume suggestion ──")
        for h in abnormal:
            last_step = h["saved_steps"][-1] if h["saved_steps"] else None
            first_step = h["saved_steps"][0] if h["saved_steps"] else None
            step_str = (f"[ckpts: {first_step}..{last_step} ({len(h['saved_steps'])} saved)]"
                        if h["saved_steps"] else "[no ckpts]")
            lines.append(f"  {h['label']} {h['pretty']} — status={h['status']}  {step_str}")
            if h["diag"]:
                lines.append(f"    cause: {h['diag']}")
            cmd = resume_command(h)
            if cmd:
                lines.append(f"    resume: {cmd}")
            else:
                lines.append("    resume: (no ckpt to resume from)")
        lines.append("")

    # Loss table
    lines.append("── loss ──")
    header = f"{'step':>6} | " + " | ".join(f"{lbl} {p:<11}" for lbl, _, _, p, _ in EXPERIMENTS)
    lines.append(header)
    lines.append("-" * len(header))
    tables = {}
    for label, exp, _, _, _ in EXPERIMENTS:
        d = latest_wandb_run(exp)
        if d is None:
            tables[label] = {}
            continue
        try:
            wf = next(d.glob("*.wandb"))
            tables[label] = scan_wandb_history(wf)
        except Exception as e:
            tables[label] = {}
    for s in LOSS_STEPS:
        row = [f"{s:>6}"]
        any_present = False
        for label, _, _, _, _ in EXPERIMENTS:
            rec = tables[label].get(s)
            v = rec.get("loss") if rec else None
            if v is not None:
                any_present = True
            row.append(f"   {fmt_loss_cell(v):<11}")
        if any_present:
            lines.append(" | ".join(row))
    lines.append("")

    # Inline eval table
    lines.append("── val/mae_1 (inline eval at save_interval) ──")
    evals = {label: inline_eval_records(LOGS_DIR / f"train_{exp}.log") for label, exp, _, _, _ in EXPERIMENTS}
    all_steps = sorted({e["step"] for rs in evals.values() for e in rs})
    if not all_steps:
        lines.append("  (no inline-eval results yet — first fires at step=save_interval)")
    else:
        header2 = f"{'step':>6} | " + " | ".join(f"{lbl} {p:<11}" for lbl, _, _, p, _ in EXPERIMENTS)
        lines.append(header2)
        lines.append("-" * len(header2))
        best = {label: min((e["mae_1"] for e in evals[label]), default=None) for label, _, _, _, _ in EXPERIMENTS}
        for s in all_steps:
            row = [f"{s:>6}"]
            for label, _, _, _, _ in EXPERIMENTS:
                hit = next((e for e in evals[label] if e["step"] == s), None)
                v = hit["mae_1"] if hit else None
                row.append(f"   {fmt_loss_cell(v):<11}")
            lines.append(" | ".join(row))
        # best row
        row_best = [f"{'best':>6}"]
        for label, _, _, _, _ in EXPERIMENTS:
            v = best[label]
            row_best.append(f"   {fmt_loss_cell(v):<11}")
        lines.append("-" * len(header2))
        lines.append(" | ".join(row_best))
        lines.append("")
        # full-metric summary for latest step across all runs
        latest = all_steps[-1]
        lines.append(f"── full MAE @ step={latest} (@1 / @10 / @25 / @50) ──")
        for label, exp, _, pretty, _ in EXPERIMENTS:
            hit = next((e for e in evals[label] if e["step"] == latest), None)
            if hit:
                lines.append(
                    f"  {label} {pretty:<14} @1={hit['mae_1']:.4f}  @10={hit['mae_10']:.4f}  "
                    f"@25={hit['mae_25']:.4f}  @50={hit['mae_50']:.4f}"
                )
            else:
                lines.append(f"  {label} {pretty:<14} (no eval at step {latest})")

    return "\n".join(lines)


def auto_resume_step(args) -> None:
    """Pass over health check results and launch resume for each DEAD run."""
    healths = [health_check(label, exp, gpu, pretty, cfg)
               for label, exp, gpu, pretty, cfg in EXPERIMENTS]
    dead = [h for h in healths if h["status"] == "DEAD"]
    if not dead:
        print("[auto-resume] no dead runs — nothing to do")
        return
    for h in dead:
        ok, msg = auto_resume_exec(h)
        mark = "✅" if ok else "❌"
        print(f"{mark} {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0,
                    help="refresh every N seconds; 0 = one-shot (default)")
    ap.add_argument("--auto-resume", action="store_true",
                    help="relaunch any DEAD experiment from its last ckpt on a healthy GPU")
    args = ap.parse_args()

    if args.auto_resume:
        # Print current dashboard once, then resume
        print(render(args))
        print("\n[auto-resume] scanning dead runs...")
        auto_resume_step(args)
        return

    if args.watch <= 0:
        print(render(args))
        return
    try:
        while True:
            os.system("clear")
            print(render(args))
            print(f"\n[watch every {args.watch}s — Ctrl-C to quit]")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
