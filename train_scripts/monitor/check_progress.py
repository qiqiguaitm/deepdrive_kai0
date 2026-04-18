#!/usr/bin/env python3
"""对比 gf0 (baseline) 和 gf1 (π0.7-style q5drop) 训练进度.

用法:
    python3 check_progress.py              # 单次对比
    python3 check_progress.py --watch      # 每 30 秒刷新
    python3 check_progress.py --plot       # 含 ASCII loss 曲线
    python3 check_progress.py --gpu        # 含 GPU 利用率采样
    python3 check_progress.py --tail N     # 只显示最近 N 个 Step 事件（默认全部）
"""
import argparse, glob, os, re, subprocess, sys, time
from pathlib import Path

LOG_DIR = "/vePFS/tim/workspace/deepdive_kai0/logs"
GF0_GLOB = "gf0_awbc_baseline_*.log"
GF1_GLOB = "gf1_awbc_q5drop_*.log"
GF0_SSH = ["ssh", "-p", "2222", "root@192.168.0.144"]
GF1_SSH = ["ssh", "-p", "2222", "root@192.168.0.144",
           "ssh", "-p", "2222", "-i", "/root/.ssh/ssh_worker_rsa_key",
           "-o", "StrictHostKeyChecking=no", "root@192.168.0.161"]

STEP_RE = re.compile(r"^Step (\d+): grad_norm=([\d.]+), loss=([\d.]+), param_norm=([\d.]+)")
TQDM_RE = re.compile(r"\[I\] Progress on:\s*(\S+)it/(\S+)kit rate:(\S+) remaining:(\S+) elapsed:(\S+)")
SKIP_RE = re.compile(r"Skipping check_timestamps_sync")
EVAL_RE = re.compile(r"Eval@(\d+):\s*(.+)$")


def latest_log(pattern):
    """Find the latest log matching pattern."""
    files = sorted(glob.glob(os.path.join(LOG_DIR, pattern)))
    if not files:
        return None
    return files[-1]


def parse_log(path):
    """Parse a training log. Returns dict with steps[], grad_norms[], losses[], param_norms[],
    and tqdm_latest (rate, elapsed, remaining, it)."""
    if path is None or not os.path.exists(path):
        return None
    data = {"path": path, "steps": [], "loss": [], "grad": [], "pnorm": [],
            "tqdm_latest": None, "skip_marker_hit": False,
            "initialized_data": False, "initialized_train": False,
            "eval_steps": [], "eval_metrics": []}
    with open(path) as f:
        for ln in f:
            m = STEP_RE.search(ln)
            if m:
                data["steps"].append(int(m.group(1)))
                data["grad"].append(float(m.group(2)))
                data["loss"].append(float(m.group(3)))
                data["pnorm"].append(float(m.group(4)))
                continue
            m = TQDM_RE.search(ln)
            if m:
                data["tqdm_latest"] = {
                    "it": m.group(1), "total_kit": m.group(2),
                    "rate": m.group(3), "remaining": m.group(4),
                    "elapsed": m.group(5),
                }
                continue
            if SKIP_RE.search(ln):
                data["skip_marker_hit"] = True
            if "Initialized data loader" in ln:
                data["initialized_data"] = True
            if "Initialized train state" in ln:
                data["initialized_train"] = True
            m = EVAL_RE.search(ln)
            if m:
                step = int(m.group(1))
                # 解析 "mae_joint_1=0.0100, mae_joint_10=..." 形式
                metrics_str = m.group(2)
                metrics = {}
                for kv in metrics_str.split(","):
                    kv = kv.strip()
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        try:
                            metrics[k.strip()] = float(v.strip())
                        except ValueError:
                            pass
                data["eval_steps"].append(step)
                data["eval_metrics"].append(metrics)
    return data


def process_alive(ssh_prefix, pattern="train.py"):
    """Check if training process is running on given machine."""
    try:
        cmd = ssh_prefix + [f"ps aux | grep -v grep | grep -c '{pattern}'"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        n = int(r.stdout.strip() or "0")
        return n > 0
    except Exception:
        return None


def get_gpu_util(ssh_prefix, duration=5):
    """Sample GPU utilization for `duration` seconds, return idle %."""
    cmd = ssh_prefix + [
        "python3 -c \"" + "\n".join([
            "import subprocess, time",
            "idle=0; tot=0",
            f"for i in range({duration}):",
            "    u = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split('\\n')",
            "    vals = [int(x.strip()) for x in u]",
            "    z = sum(1 for v in vals if v < 30)",
            "    idle += z; tot += len(vals)",
            "    time.sleep(1)",
            "print(f'{idle*100/tot:.1f}')",
        ]) + "\""
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 10)
        return float(r.stdout.strip())
    except Exception as e:
        return None


def plot_ascii(data0, data1, width=60, height=12):
    """Simple ASCII plot of loss curves."""
    if not data0 or not data0["loss"] or not data1 or not data1["loss"]:
        return "(no data to plot)"
    # Align by step
    steps0 = set(data0["steps"])
    steps1 = set(data1["steps"])
    common = sorted(steps0 & steps1)
    if not common:
        return "(no common steps between gf0 and gf1 yet)"
    idx0 = {s: i for i, s in enumerate(data0["steps"])}
    idx1 = {s: i for i, s in enumerate(data1["steps"])}
    l0 = [data0["loss"][idx0[s]] for s in common]
    l1 = [data1["loss"][idx1[s]] for s in common]
    ymax = max(max(l0), max(l1))
    ymin = min(min(l0), min(l1))
    if ymax == ymin:
        return "(loss values constant)"

    def scale(v):
        return int((height - 1) * (1 - (v - ymin) / (ymax - ymin)))

    canvas = [[" " for _ in range(width)] for _ in range(height)]
    pts = min(width, len(common))
    for i in range(pts):
        s_idx = int(i * len(common) / pts)
        x = i
        y0 = scale(l0[s_idx])
        y1 = scale(l1[s_idx])
        if 0 <= y0 < height:
            canvas[y0][x] = "0"
        if 0 <= y1 < height:
            canvas[y1][x] = "1" if canvas[y1][x] == " " else "X"

    out = []
    out.append(f"  loss    gf0={l0[-1]:.4f}, gf1={l1[-1]:.4f} (ymax={ymax:.3f}, ymin={ymin:.5f})")
    for r, row in enumerate(canvas):
        label = f"{ymax - (ymax-ymin)*r/(height-1):.3f}" if r % max(1, height//4) == 0 else "     "
        out.append(f"  {label:>6} |{''.join(row)}")
    out.append(f"         step {common[0]} {'-'*(width-14)} step {common[-1]}")
    return "\n".join(out)


def fmt_row(label, v0, v1, diff_fmt=None):
    """Format a comparison row."""
    if v0 is None:
        v0_s = "-"
    elif isinstance(v0, float):
        v0_s = f"{v0:.4f}"
    else:
        v0_s = str(v0)
    if v1 is None:
        v1_s = "-"
    elif isinstance(v1, float):
        v1_s = f"{v1:.4f}"
    else:
        v1_s = str(v1)
    diff_s = ""
    if diff_fmt and isinstance(v0, (int, float)) and isinstance(v1, (int, float)) and v0 != 0:
        diff = (v1 - v0) / abs(v0) * 100
        sign = "+" if diff >= 0 else ""
        diff_s = f"   ({sign}{diff:.1f}%)"
    return f"  {label:<24} {v0_s:>16}   {v1_s:>16}{diff_s}"


def show_once(tail_n=None, plot=False, gpu=False):
    log0 = latest_log(GF0_GLOB)
    log1 = latest_log(GF1_GLOB)
    data0 = parse_log(log0)
    data1 = parse_log(log1)

    print("=" * 80)
    print(f"AWBC 训练对比 — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"  gf0 log: {os.path.basename(log0) if log0 else 'NOT FOUND'}")
    print(f"  gf1 log: {os.path.basename(log1) if log1 else 'NOT FOUND'}")
    print()

    # 运行状态
    alive0 = process_alive(GF0_SSH)
    alive1 = process_alive(GF1_SSH)
    print(f"  process alive  gf0={'✅' if alive0 else '❌'}   gf1={'✅' if alive1 else '❌'}")
    print()

    # 里程碑
    print(f"  {'milestone':<24} {'gf0 (baseline)':>16}   {'gf1 (q5drop)':>16}")
    print(f"  {'-'*24} {'-'*16:>16}   {'-'*16:>16}")
    if data0 and data1:
        print(fmt_row("skip marker hit", data0["skip_marker_hit"], data1["skip_marker_hit"]))
        print(fmt_row("data loader ready", data0["initialized_data"], data1["initialized_data"]))
        print(fmt_row("train state ready", data0["initialized_train"], data1["initialized_train"]))
    print()

    # Step / rate
    tqdm0 = data0["tqdm_latest"] if data0 else None
    tqdm1 = data1["tqdm_latest"] if data1 else None
    last_step0 = data0["steps"][-1] if data0 and data0["steps"] else None
    last_step1 = data1["steps"][-1] if data1 and data1["steps"] else None
    print(fmt_row("latest Step (log)", last_step0, last_step1))
    print(fmt_row("latest iter (tqdm)", tqdm0["it"] if tqdm0 else None, tqdm1["it"] if tqdm1 else None))
    print(fmt_row("rate", tqdm0["rate"] if tqdm0 else None, tqdm1["rate"] if tqdm1 else None))
    print(fmt_row("elapsed", tqdm0["elapsed"] if tqdm0 else None, tqdm1["elapsed"] if tqdm1 else None))
    print(fmt_row("remaining", tqdm0["remaining"] if tqdm0 else None, tqdm1["remaining"] if tqdm1 else None))
    print()

    # 最新 loss / grad / pnorm
    last_loss0 = data0["loss"][-1] if data0 and data0["loss"] else None
    last_loss1 = data1["loss"][-1] if data1 and data1["loss"] else None
    last_grad0 = data0["grad"][-1] if data0 and data0["grad"] else None
    last_grad1 = data1["grad"][-1] if data1 and data1["grad"] else None
    last_pnorm0 = data0["pnorm"][-1] if data0 and data0["pnorm"] else None
    last_pnorm1 = data1["pnorm"][-1] if data1 and data1["pnorm"] else None
    print(fmt_row("latest loss", last_loss0, last_loss1, diff_fmt="%"))
    print(fmt_row("latest grad_norm", last_grad0, last_grad1, diff_fmt="%"))
    print(fmt_row("latest param_norm", last_pnorm0, last_pnorm1, diff_fmt="%"))
    print()

    # Step 对比表
    if data0 and data1 and data0["steps"] and data1["steps"]:
        s0 = set(data0["steps"])
        s1 = set(data1["steps"])
        common = sorted(s0 & s1)
        if tail_n:
            common = common[-tail_n:]
        if common:
            print(f"  Step-by-step 对比（每 100 步 log 一次）：")
            print(f"  {'step':>7}   {'gf0_loss':>10}  {'gf1_loss':>10}  {'Δ':>7}   {'gf0_grad':>10}  {'gf1_grad':>10}")
            idx0 = {s: i for i, s in enumerate(data0["steps"])}
            idx1 = {s: i for i, s in enumerate(data1["steps"])}
            for s in common:
                l0 = data0["loss"][idx0[s]]
                l1 = data1["loss"][idx1[s]]
                g0 = data0["grad"][idx0[s]]
                g1 = data1["grad"][idx1[s]]
                diff = (l1 - l0) / max(abs(l0), 1e-9) * 100
                sign = "+" if diff >= 0 else ""
                print(f"  {s:>7}   {l0:>10.4f}  {l1:>10.4f}  {sign}{diff:>5.1f}%   {g0:>10.4f}  {g1:>10.4f}")
    print()

    # Eval metrics comparison
    if data0 and data1 and (data0["eval_steps"] or data1["eval_steps"]):
        print(f"  📊 Eval MAE 对比（in-training eval）:")
        print(f"  {'step':>6}   {'metric':<15}   {'gf0':>10}  {'gf1':>10}  {'Δ':>8}  thresh")
        e0_map = dict(zip(data0["eval_steps"], data0["eval_metrics"]))
        e1_map = dict(zip(data1["eval_steps"], data1["eval_metrics"]))
        common = sorted(set(e0_map.keys()) & set(e1_map.keys()))
        thresh = {
            "mae_joint_1": 0.02, "mae_joint_10": 0.05, "mae_joint_50": 0.12,
            "mae_grip_1": 0.005, "mae_grip_10": 0.005, "mae_grip_50": 0.005,
        }
        for s in (common[-5:] if tail_n else common):
            m0, m1 = e0_map[s], e1_map[s]
            for k in ["mae_joint_1", "mae_joint_10", "mae_joint_50", "mae_grip_1", "mae_grip_10", "mae_grip_50"]:
                if k in m0 and k in m1:
                    v0, v1 = m0[k], m1[k]
                    diff = (v1 - v0) / max(abs(v0), 1e-9) * 100
                    sign = "+" if diff >= 0 else ""
                    th = thresh.get(k, 0)
                    p0 = "✅" if v0 < th else "❌"
                    p1 = "✅" if v1 < th else "❌"
                    print(f"  {s:>6}   {k:<15}   {v0:>10.4f}  {v1:>10.4f}  {sign}{diff:>5.1f}%  {th:.3f}  {p0}{p1}")
            print()
    # Checkpoint
    for tag, dir_ in [("gf0", "pi05_flatten_fold_awbc/gf0_awbc_baseline_v1"),
                      ("gf1", "pi05_flatten_fold_awbc_q5drop/gf1_awbc_q5drop_v1")]:
        ck = Path(f"/vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/{dir_}")
        steps = sorted([p.name for p in ck.iterdir() if p.is_dir() and p.name.isdigit()]) if ck.exists() else []
        latest_ck = steps[-1] if steps else "none"
        size_mb = sum(p.stat().st_size for p in ck.rglob('*') if p.is_file()) / 1e6 if ck.exists() else 0
        print(f"  checkpoint {tag}: latest=step_{latest_ck}, total_size={size_mb:.1f} MB")
    print()

    # GPU idle
    if gpu:
        print("  GPU 利用率采样中 (5秒)...")
        idle0 = get_gpu_util(GF0_SSH, duration=5)
        idle1 = get_gpu_util(GF1_SSH, duration=5)
        print(f"  {'GPU idle ratio':<24} {idle0 or '-':>15.1f}%   {idle1 or '-':>15.1f}%")
        print()

    # Plot
    if plot:
        print("=" * 80)
        print("Loss 曲线 (0=gf0, 1=gf1, X=overlap)")
        print("=" * 80)
        print(plot_ascii(data0, data1))
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", action="store_true", help="每 30 秒刷新")
    ap.add_argument("--interval", type=int, default=30, help="watch 间隔秒数（默认 30）")
    ap.add_argument("--plot", action="store_true", help="显示 ASCII loss 曲线")
    ap.add_argument("--gpu", action="store_true", help="采样 GPU 利用率（5 秒）")
    ap.add_argument("--tail", type=int, default=10, help="对比表最近 N 个 Step（默认 10，0 = 全部）")
    args = ap.parse_args()

    tail_n = args.tail if args.tail > 0 else None
    if args.watch:
        try:
            while True:
                os.system("clear" if os.name == "posix" else "cls")
                show_once(tail_n=tail_n, plot=args.plot, gpu=args.gpu)
                print(f"  [刷新中... Ctrl+C 退出，下次刷新 in {args.interval}s]")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n退出")
    else:
        show_once(tail_n=tail_n, plot=args.plot, gpu=args.gpu)


if __name__ == "__main__":
    main()
