"""
验证 Advantage Estimator：自训模型预测 vs 官方预标注对比。

前置条件：
  1. 自训 Estimator 已完成 (experiment/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1/100000/)
  2. 已运行 eval.py 生成 data_KAI0_100000/ 目录

用法：
  python scripts/validate_advantage_estimator.py \
    --dataset data/Task_A/advantage \
    --pred-suffix KAI0_100000 \
    --num-episodes 200 \
    --output logs/adv_est_validation.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import stats


def load_episode(dataset_root: Path, chunk: int, ep: int, suffix: str | None = None):
    """Load a parquet file, optionally from the prediction subdirectory."""
    if suffix:
        path = dataset_root / f"data_{suffix}" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet"
    else:
        path = dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet"
    if not path.exists():
        return None
    return pq.read_table(path)


def compute_discretization_agreement(official: np.ndarray, predicted: np.ndarray, threshold_pct: float = 30.0):
    """Compute agreement rate after binary discretization (top threshold% = positive)."""
    off_thresh = np.percentile(official, 100 - threshold_pct)
    pred_thresh = np.percentile(predicted, 100 - threshold_pct)
    off_labels = (official >= off_thresh).astype(int)
    pred_labels = (predicted >= pred_thresh).astype(int)
    agreement = (off_labels == pred_labels).mean()
    return agreement, off_thresh, pred_thresh


def main():
    parser = argparse.ArgumentParser(description="Validate Advantage Estimator predictions vs official labels")
    parser.add_argument("--dataset", type=str, default="data/Task_A/advantage", help="Advantage dataset root")
    parser.add_argument("--pred-suffix", type=str, default="KAI0_100000", help="Prediction data dir suffix")
    parser.add_argument("--num-episodes", type=int, default=0, help="Number of episodes to compare (0=all)")
    parser.add_argument("--output", type=str, default="logs/adv_est_validation.json", help="Output JSON path")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    info = json.load(open(dataset_root / "meta" / "info.json"))
    total_episodes = info["total_episodes"]
    chunks_size = info["chunks_size"]

    num_episodes = args.num_episodes if args.num_episodes > 0 else total_episodes

    # Collect all frame-level values
    all_official_abs_adv = []
    all_predicted_abs_adv = []
    all_official_rel_adv = []
    all_predicted_rel_adv = []
    all_official_abs_val = []
    all_predicted_abs_val = []

    per_episode_corr = []
    skipped = 0

    for ep in range(min(num_episodes, total_episodes)):
        chunk = ep // chunks_size

        official = load_episode(dataset_root, chunk, ep, suffix=None)
        predicted = load_episode(dataset_root, chunk, ep, suffix=args.pred_suffix)

        if official is None or predicted is None:
            skipped += 1
            continue

        off_abs_adv = np.array(official.column("absolute_advantage").to_pylist(), dtype=np.float64)
        off_rel_adv = np.array(official.column("relative_advantage").to_pylist(), dtype=np.float64)
        off_abs_val = np.array(official.column("absolute_value").to_pylist(), dtype=np.float64)

        pred_abs_adv = np.array(predicted.column("absolute_advantage").to_pylist(), dtype=np.float64)
        pred_rel_adv = np.array(predicted.column("relative_advantage").to_pylist(), dtype=np.float64)
        pred_abs_val = np.array(predicted.column("absolute_value").to_pylist(), dtype=np.float64)

        n = min(len(off_abs_adv), len(pred_abs_adv))
        if n < 10:
            skipped += 1
            continue

        off_abs_adv, pred_abs_adv = off_abs_adv[:n], pred_abs_adv[:n]
        off_rel_adv, pred_rel_adv = off_rel_adv[:n], pred_rel_adv[:n]
        off_abs_val, pred_abs_val = off_abs_val[:n], pred_abs_val[:n]

        all_official_abs_adv.append(off_abs_adv)
        all_predicted_abs_adv.append(pred_abs_adv)
        all_official_rel_adv.append(off_rel_adv)
        all_predicted_rel_adv.append(pred_rel_adv)
        all_official_abs_val.append(off_abs_val)
        all_predicted_abs_val.append(pred_abs_val)

        # Per-episode Pearson on absolute_value (the primary prediction target)
        if np.std(off_abs_val) > 1e-6 and np.std(pred_abs_val) > 1e-6:
            r, _ = stats.pearsonr(off_abs_val, pred_abs_val)
            per_episode_corr.append(r)

    if not all_official_abs_adv:
        print("ERROR: No episodes could be compared. Check --pred-suffix and ensure eval.py has been run.")
        return

    # Concatenate all frames
    off_abs_adv = np.concatenate(all_official_abs_adv)
    pred_abs_adv = np.concatenate(all_predicted_abs_adv)
    off_rel_adv = np.concatenate(all_official_rel_adv)
    pred_rel_adv = np.concatenate(all_predicted_rel_adv)
    off_abs_val = np.concatenate(all_official_abs_val)
    pred_abs_val = np.concatenate(all_predicted_abs_val)

    n_frames = len(off_abs_adv)
    n_episodes = len(per_episode_corr)

    # --- Metrics ---
    results = {"n_frames": n_frames, "n_episodes": n_episodes, "skipped_episodes": skipped}

    # 1. Pearson correlation
    for name, off, pred in [
        ("absolute_advantage", off_abs_adv, pred_abs_adv),
        ("relative_advantage", off_rel_adv, pred_rel_adv),
        ("absolute_value", off_abs_val, pred_abs_val),
    ]:
        pearson_r, pearson_p = stats.pearsonr(off, pred)
        spearman_r, spearman_p = stats.spearmanr(off, pred)
        mae = np.mean(np.abs(off - pred))
        rmse = np.sqrt(np.mean((off - pred) ** 2))
        results[name] = {
            "pearson_r": round(float(pearson_r), 4),
            "spearman_r": round(float(spearman_r), 4),
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
        }

    # 2. Discretization agreement (binary, threshold=30%)
    for thresh in [30, 50]:
        agree, off_t, pred_t = compute_discretization_agreement(off_abs_adv, pred_abs_adv, thresh)
        results[f"discretize_agree_top{thresh}pct"] = {
            "agreement": round(float(agree), 4),
            "official_threshold": round(float(off_t), 4),
            "predicted_threshold": round(float(pred_t), 4),
        }

    # 3. Per-episode correlation stats
    corr_arr = np.array(per_episode_corr)
    results["per_episode_abs_val_pearson"] = {
        "mean": round(float(corr_arr.mean()), 4),
        "median": round(float(np.median(corr_arr)), 4),
        "min": round(float(corr_arr.min()), 4),
        "std": round(float(corr_arr.std()), 4),
    }

    # --- Output ---
    print("=" * 60)
    print(f"Advantage Estimator Validation ({n_episodes} episodes, {n_frames} frames)")
    print("=" * 60)

    for name in ["absolute_value", "absolute_advantage", "relative_advantage"]:
        m = results[name]
        print(f"\n  {name}:")
        print(f"    Pearson r  = {m['pearson_r']:.4f}")
        print(f"    Spearman r = {m['spearman_r']:.4f}")
        print(f"    MAE        = {m['mae']:.4f}")
        print(f"    RMSE       = {m['rmse']:.4f}")

    print(f"\n  Per-episode abs_val Pearson: mean={results['per_episode_abs_val_pearson']['mean']:.4f}, "
          f"median={results['per_episode_abs_val_pearson']['median']:.4f}, "
          f"min={results['per_episode_abs_val_pearson']['min']:.4f}")

    for thresh in [30, 50]:
        d = results[f"discretize_agree_top{thresh}pct"]
        print(f"\n  Discretization agreement (top {thresh}%): {d['agreement']:.1%}")

    print("\n" + "=" * 60)
    status = "PASS" if results["absolute_value"]["pearson_r"] > 0.85 else "CHECK"
    print(f"  Overall: {status} (abs_val pearson={results['absolute_value']['pearson_r']:.4f})")
    print("=" * 60)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
