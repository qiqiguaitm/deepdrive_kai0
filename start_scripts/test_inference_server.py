#!/usr/bin/env python3
"""
推理服务器测试 (统一入口)

模式:
  --check latency   延迟基准测试 (合成 payload)
  --check quality   输出质量验证 (数值范围、一致性、敏感��、平滑性)
  --check all       全部检查

合并自: bench_inference_latency.py + verify_inference_quality.py

前置: serve_policy.py 或 WebSocket policy server 在运行

Usage:
  python3 scripts/test_inference_server.py --check all [--host localhost] [--port 8000]
"""
import argparse
import time
import sys
import numpy as np

sys.path.insert(0, 'packages/openpi-client/src')
from openpi_client import websocket_client_policy

# Piper 关节限位 (radians) — 来源: 官方 URDF piper_description.urdf
JOINT_LIMITS = [
    (-2.618, 2.618),   # joint 0
    ( 0.000, 3.140),   # joint 1 (非对称)
    (-2.967, 0.000),   # joint 2 (非对称)
    (-1.745, 1.745),   # joint 3
    (-1.220, 1.220),   # joint 4
    (-2.0944, 2.0944), # joint 5
    ( 0.000, 0.035),   # gripper (rad, URDF 值)
]
ALL_LIMITS = JOINT_LIMITS + JOINT_LIMITS


def make_payload(img=None, state=None, prompt='Flatten and fold the cloth.'):
    if img is None:
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    if state is None:
        state = np.array([0.0, 0.3, -0.5, 0.0, 0.5, 0.0, 0.03,
                          0.0, 0.3, -0.5, 0.0, 0.5, 0.0, 0.03], dtype=np.float32)
    return {
        'images': {'top_head': img, 'hand_left': img, 'hand_right': img},
        'state': state,
        'prompt': prompt,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Latency benchmark
# ══════════════════════════════════════════════════════════════════════════════

def check_latency(policy, rounds=20, warmup=3):
    print('=' * 60)
    print(f'延迟基准测试 ({rounds} rounds, {warmup} warmup)')
    print('=' * 60)

    for i in range(warmup):
        t0 = time.monotonic()
        r = policy.infer(make_payload())
        dt = (time.monotonic() - t0) * 1000
        shape = r['actions'].shape if 'actions' in r else 'N/A'
        print(f'  warmup {i+1}: {dt:.0f}ms  shape={shape}')

    latencies = []
    for i in range(rounds):
        payload = make_payload(
            img=np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            state=np.random.randn(14).astype(np.float32),
        )
        t0 = time.monotonic()
        policy.infer(payload)
        latencies.append((time.monotonic() - t0) * 1000)
        print(f'  round {i+1:2d}: {latencies[-1]:.0f}ms')

    lat = np.array(latencies)
    print(f'\n  avg={lat.mean():.0f}ms  std={lat.std():.0f}ms  '
          f'p50={np.median(lat):.0f}ms  p95={np.percentile(lat, 95):.0f}ms  '
          f'p99={np.percentile(lat, 99):.0f}ms  max={lat.max():.0f}ms')
    print(f'  throughput: {1000/lat.mean():.1f} infer/s')

    ok = lat.mean() < 300
    tag = 'PASS' if ok else ('MARGINAL' if lat.mean() < 500 else 'FAIL')
    print(f'  结论: {tag} (要求 < 300ms, 实际 {lat.mean():.0f}ms)')
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# Quality checks
# ══════════════════════════════════════════════════════════════════════════════

def _check_range(actions):
    violations = 0
    for dim in range(14):
        lo, hi = ALL_LIMITS[dim]
        margin = 0.5
        below = (actions[:, dim] < lo - margin).sum()
        above = (actions[:, dim] > hi + margin).sum()
        if below or above:
            violations += below + above
            jn = f"{'L' if dim < 7 else 'R'}_j{dim % 7}"
            print(f'    [WARN] {jn}: {below} below {lo-margin:.2f}, {above} above {hi+margin:.2f}')
    return violations == 0


def check_quality(policy):
    print('=' * 60)
    print('推理质量验证')
    print('=' * 60)
    results = {}

    # warmup
    policy.infer(make_payload())

    # 1. 形状和范围
    print('\n--- Test 1: Action 形状和数值范围 ---')
    r = policy.infer(make_payload())
    actions = r['actions']
    print(f'  Shape: {actions.shape}  dtype: {actions.dtype}')
    print(f'  范围: [{actions.min():.4f}, {actions.max():.4f}]  均值: {actions.mean():.4f}')
    for dim in range(14):
        jn = f"{'L' if dim < 7 else 'R'}_j{dim % 7}"
        v = actions[:, dim]
        print(f'  {jn}: [{v.min():.3f}, {v.max():.3f}] mean={v.mean():.3f}')
    range_ok = _check_range(actions) and actions.std() > 0.001
    results['shape_range'] = range_ok
    print(f'  → {"PASS" if range_ok else "WARN"}')

    # 2. 一致性
    print('\n--- Test 2: 一致性 (同一输入 x5) ---')
    fixed = make_payload(
        img=np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        state=np.array([0.1, 0.5, -0.8, 0, 0.6, -0.1, 0.04,
                        -0.1, 0.5, -0.8, 0, 0.6, 0.1, 0.04], dtype=np.float32),
    )
    all_a = np.stack([policy.infer(fixed)['actions'] for _ in range(5)])
    std_across = all_a.std(axis=0).mean()
    print(f'  跨次 std: {std_across:.4f} rad  max偏差: {np.abs(all_a - all_a.mean(0)).max():.4f}')
    results['consistency'] = std_across < 0.3
    print(f'  → {"PASS" if results["consistency"] else "WARN"} (std < 0.3)')

    # 3. 敏感性
    print('\n--- Test 3: 敏感性 (不同状态) ---')
    s_a = np.zeros(14, dtype=np.float32)
    s_b = np.array([0.5, 1.0, -1.0, 0.3, 0.8, -0.5, 0.05,
                    -0.5, 1.0, -1.0, -0.3, 0.8, 0.5, 0.05], dtype=np.float32)
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    diff = np.abs(policy.infer(make_payload(img, s_a))['actions']
                  - policy.infer(make_payload(img, s_b))['actions']).mean()
    results['sensitivity'] = diff > 0.01
    print(f'  mean diff: {diff:.4f} rad → {"PASS" if results["sensitivity"] else "FAIL"}')

    # 4. 平滑性
    print('\n--- Test 4: 时序平滑性 ---')
    a = policy.infer(fixed)['actions']
    step_diffs = np.abs(np.diff(a, axis=0))
    results['smoothness'] = step_diffs.max() < 0.5
    print(f'  max jump: {step_diffs.max():.4f} rad  mean: {step_diffs.mean():.4f}')
    print(f'  → {"PASS" if results["smoothness"] else "WARN"} (max < 0.5)')

    # 5. Server timing
    print('\n--- Test 5: Server timing ---')
    r = policy.infer(make_payload())
    if 'server_timing' in r:
        st = r['server_timing']
        print(f'  infer_ms: {st.get("infer_ms", "N/A")}')
        results['timing'] = st.get('infer_ms', 999) < 200
    else:
        print('  No server_timing in response')
        results['timing'] = True

    # 总结
    print('\n' + '=' * 60)
    all_pass = all(results.values())
    for k, v in results.items():
        print(f'  {k}: {"PASS" if v else "FAIL"}')
    print(f'\n  总���: {"PASS" if all_pass else "NEEDS REVIEW"}')
    return all_pass


def main():
    parser = argparse.ArgumentParser(description='推理服务器测试')
    parser.add_argument('--check', choices=['latency', 'quality', 'all'], default='all')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--rounds', type=int, default=20, help='延迟测试轮数')
    args = parser.parse_args()

    print(f'Connecting to ws://{args.host}:{args.port}...')
    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print(f'Connected. metadata: {policy.get_server_metadata()}\n')

    ok = True
    if args.check in ('latency', 'all'):
        ok &= check_latency(policy, rounds=args.rounds)
        print()
    if args.check in ('quality', 'all'):
        ok &= check_quality(policy)

    policy.close()
    return 0 if ok else 1


if __name__ == '__main__':
    exit(main())
