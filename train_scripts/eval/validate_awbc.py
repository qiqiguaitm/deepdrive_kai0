"""
验证 AWBC 模型：训练 loss 曲线 + prompt 条件化差异 + action 合理性。

用法：
  # 1. 解析训练 loss 曲线
  python scripts/validate_awbc.py loss --log logs/gf2_awbc_v1.log

  # 2. 推理测试（需先启动 policy server）
  python scripts/validate_awbc.py infer \
    --host localhost --port 8000 \
    --dataset data/Task_A/advantage \
    --num-episodes 5

  # 3. Prompt 条件化对比（需先启动 policy server）
  python scripts/validate_awbc.py prompt-diff \
    --host localhost --port 8000 \
    --dataset data/Task_A/advantage \
    --num-episodes 10
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


# ─── Subcommand: loss ─────────────────────────────────────────────────

def cmd_loss(args):
    """Parse training log and extract loss curve."""
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"ERROR: Log file not found: {log_path}")
        sys.exit(1)

    # JAX train.py format: "Step 1000: loss=0.1234, grad_norm=1.23, param_norm=45.6"
    pattern_jax = re.compile(r"Step\s+(\d+):\s+loss=([\d.]+)")
    # PyTorch train_pytorch.py format: "step=1000 loss=0.1234 lr=1.00e-04"
    pattern_pt = re.compile(r"step=(\d+)\s+loss=([\d.]+)")

    steps = []
    losses = []

    for line in open(log_path):
        m = pattern_jax.search(line) or pattern_pt.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))

    if not steps:
        print("ERROR: No loss entries found in log. Check log format.")
        sys.exit(1)

    steps = np.array(steps)
    losses = np.array(losses)

    # Summary
    print(f"Loss curve: {len(steps)} entries, steps {steps[0]} → {steps[-1]}")
    print(f"  Initial loss (first 5):  {losses[:5].mean():.4f}")
    print(f"  Final loss (last 5):     {losses[-5:].mean():.4f}")
    print(f"  Min loss:                {losses.min():.4f} @ step {steps[losses.argmin()]}")

    # Check convergence: final 10% should be lower than first 10%
    n10 = max(1, len(losses) // 10)
    early_mean = losses[:n10].mean()
    late_mean = losses[-n10:].mean()
    converged = late_mean < early_mean * 0.95

    print(f"\n  Early mean (first 10%):  {early_mean:.4f}")
    print(f"  Late mean (last 10%):    {late_mean:.4f}")
    print(f"  Convergence:             {'PASS' if converged else 'WARN - loss did not decrease significantly'}")

    # Save curve data
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"steps": steps.tolist(), "losses": losses.tolist()}, f)
    print(f"\nLoss curve saved to {output}")


# ─── Subcommand: infer ────────────────────────────────────────────────

def cmd_infer(args):
    """Test AWBC model inference: action shape, range, and variability."""
    import cv2
    from openpi_client import WebsocketClientPolicy

    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    dataset_root = Path(args.dataset)
    info = json.load(open(dataset_root / "meta" / "info.json"))
    chunks_size = info["chunks_size"]

    prompt = "fold the cloth, Advantage: positive"
    results = []

    for ep in range(min(args.num_episodes, info["total_episodes"])):
        chunk = ep // chunks_size
        # Load first frame from each camera
        images = {}
        for cam in ["observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right"]:
            video_path = dataset_root / f"videos/chunk-{chunk:03d}/{cam}/episode_{ep:06d}.mp4"
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(f"  WARNING: Cannot read video for ep {ep} {cam}")
                break
            # BGR -> RGB, HWC uint8
            images[cam.split(".")[-1]] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if len(images) != 3:
            continue

        # Load state from parquet
        import pyarrow.parquet as pq
        parquet = pq.read_table(
            dataset_root / f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet",
            columns=["observation.state"]
        )
        state = np.array(parquet.column("observation.state").to_pylist()[0], dtype=np.float32)

        obs = {
            "images": images,
            "state": state,
            "prompt": prompt,
        }

        action = policy.infer(obs)

        # Extract action array
        if isinstance(action, dict) and "actions" in action:
            act = np.array(action["actions"])
        elif isinstance(action, dict) and "action" in action:
            act = np.array(action["action"])
        elif isinstance(action, np.ndarray):
            act = action
        else:
            act = np.array(list(action.values())[0]) if isinstance(action, dict) else np.array(action)

        results.append({
            "episode": ep,
            "shape": list(act.shape),
            "min": float(act.min()),
            "max": float(act.max()),
            "mean": float(act.mean()),
            "std": float(act.std()),
        })
        print(f"  ep{ep:4d}: shape={act.shape}, range=[{act.min():.3f}, {act.max():.3f}], std={act.std():.3f}")

    if not results:
        print("ERROR: No episodes processed.")
        return

    # Summary
    shapes = set(tuple(r["shape"]) for r in results)
    stds = [r["std"] for r in results]

    print(f"\n{'='*60}")
    print(f"Inference Test ({len(results)} episodes)")
    print(f"  Action shapes:    {shapes}")
    print(f"  Std across eps:   min={min(stds):.3f}, max={max(stds):.3f}, mean={np.mean(stds):.3f}")

    shape_ok = all(tuple(r["shape"]) in [(50, 14), (14,)] for r in results)
    nondegenerate = np.mean(stds) > 0.01
    print(f"  Shape check:      {'PASS' if shape_ok else 'FAIL'}")
    print(f"  Non-degenerate:   {'PASS' if nondegenerate else 'FAIL (actions may have collapsed)'}")
    print(f"{'='*60}")


# ─── Subcommand: prompt-diff ──────────────────────────────────────────

def cmd_prompt_diff(args):
    """Test AWBC prompt conditioning: positive vs negative prompt should produce different actions."""
    import cv2
    from openpi_client import WebsocketClientPolicy

    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    dataset_root = Path(args.dataset)
    info = json.load(open(dataset_root / "meta" / "info.json"))
    chunks_size = info["chunks_size"]

    prompt_pos = "fold the cloth, Advantage: positive"
    prompt_neg = "fold the cloth, Advantage: negative"

    diffs = []

    for ep in range(min(args.num_episodes, info["total_episodes"])):
        chunk = ep // chunks_size
        images = {}
        for cam in ["observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right"]:
            video_path = dataset_root / f"videos/chunk-{chunk:03d}/{cam}/episode_{ep:06d}.mp4"
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                break
            images[cam.split(".")[-1]] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if len(images) != 3:
            continue

        import pyarrow.parquet as pq
        parquet = pq.read_table(
            dataset_root / f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet",
            columns=["observation.state"]
        )
        state = np.array(parquet.column("observation.state").to_pylist()[0], dtype=np.float32)

        obs_pos = {"images": images, "state": state, "prompt": prompt_pos}
        obs_neg = {"images": images, "state": state, "prompt": prompt_neg}

        def extract_action(result):
            if isinstance(result, dict):
                for key in ["actions", "action"]:
                    if key in result:
                        return np.array(result[key])
                return np.array(list(result.values())[0])
            return np.array(result)

        act_pos = extract_action(policy.infer(obs_pos))
        act_neg = extract_action(policy.infer(obs_neg))

        l2_diff = np.sqrt(np.mean((act_pos - act_neg) ** 2))
        cos_sim = np.dot(act_pos.flatten(), act_neg.flatten()) / (
            np.linalg.norm(act_pos.flatten()) * np.linalg.norm(act_neg.flatten()) + 1e-8
        )

        diffs.append({"episode": ep, "l2_diff": float(l2_diff), "cos_sim": float(cos_sim)})
        print(f"  ep{ep:4d}: L2_diff={l2_diff:.4f}, cos_sim={cos_sim:.4f}")

    if not diffs:
        print("ERROR: No episodes processed.")
        return

    l2s = [d["l2_diff"] for d in diffs]
    coss = [d["cos_sim"] for d in diffs]

    print(f"\n{'='*60}")
    print(f"Prompt Conditioning Test ({len(diffs)} episodes)")
    print(f"  L2 diff (pos vs neg):  mean={np.mean(l2s):.4f}, min={min(l2s):.4f}, max={max(l2s):.4f}")
    print(f"  Cosine similarity:     mean={np.mean(coss):.4f}")

    conditioning_works = np.mean(l2s) > 0.01
    print(f"\n  Prompt conditioning:   {'PASS' if conditioning_works else 'FAIL (model ignores advantage prompt)'}")
    print(f"{'='*60}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(diffs, f, indent=2)
    print(f"Results saved to {output}")


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate AWBC model")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # loss subcommand
    p_loss = sub.add_parser("loss", help="Parse training log for loss curve")
    p_loss.add_argument("--log", type=str, required=True, help="Training log file")
    p_loss.add_argument("--output", type=str, default="logs/awbc_loss_curve.json")

    # infer subcommand
    p_infer = sub.add_parser("infer", help="Test inference action output")
    p_infer.add_argument("--host", type=str, default="localhost")
    p_infer.add_argument("--port", type=int, default=8000)
    p_infer.add_argument("--dataset", type=str, default="data/Task_A/advantage")
    p_infer.add_argument("--num-episodes", type=int, default=5)

    # prompt-diff subcommand
    p_diff = sub.add_parser("prompt-diff", help="Test prompt conditioning (positive vs negative)")
    p_diff.add_argument("--host", type=str, default="localhost")
    p_diff.add_argument("--port", type=int, default=8000)
    p_diff.add_argument("--dataset", type=str, default="data/Task_A/advantage")
    p_diff.add_argument("--num-episodes", type=int, default=10)
    p_diff.add_argument("--output", type=str, default="logs/awbc_prompt_diff.json")

    args = parser.parse_args()

    if args.cmd == "loss":
        cmd_loss(args)
    elif args.cmd == "infer":
        cmd_infer(args)
    elif args.cmd == "prompt-diff":
        cmd_prompt_diff(args)


if __name__ == "__main__":
    main()
