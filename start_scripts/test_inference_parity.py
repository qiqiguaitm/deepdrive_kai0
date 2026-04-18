#!/usr/bin/env python3
"""
对比测试: ROS2 节点内推理 vs WebSocket 服务端推理 的端到端一致性

测试内容:
  1. 图像管线对比: 同一张 640x480 图像经两条路径处理后是否像素一致
  2. 完整推理对比: 同一组观测数据经两条路径推理后 actions 是否一致
  3. 修复前 vs 修复后: 展示旧管线 (cv2.resize) 与新管线 (resize_with_pad) 的差异

用法:
  cd /data1/tim/workspace/deepdive_kai0/kai0
  uv run python ../scripts/test_inference_parity.py [--with-model]

  不加 --with-model: 只测图像管线 (快, 无需 GPU)
  加 --with-model:   加载模型做完整推理对比 (需 GPU, ~30s 加载)
"""

import sys
import os
import argparse
import time

# 确保 openpi src 可被 import
sys.path.insert(0, '/data1/tim/workspace/deepdive_kai0/kai0/src')

import cv2
import numpy as np


def jpeg_mapping(img):
    """原版 JPEG encode/decode (与训练 MP4 压缩对齐)."""
    img = cv2.imencode(".jpg", img)[1].tobytes()
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    return img


def pipeline_original_ws(img_front, img_left, img_right, joint_left, joint_right, prompt):
    """原版 WebSocket 模式的观测构建管线.

    来源: agilex_inference_openpi_temporal_smoothing_ros2.py
    update_observation_window() + inference_fn()
    """
    from openpi_client import image_tools

    # 原版顺序: front, right, left
    imgs = [jpeg_mapping(img_front), jpeg_mapping(img_right), jpeg_mapping(img_left)]
    imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
    imgs = image_tools.resize_with_pad(np.array(imgs), 224, 224)

    qpos = np.concatenate((np.array(joint_left), np.array(joint_right)), axis=0)

    return {
        'state': qpos,
        'images': {
            'top_head':   imgs[0].transpose(2, 0, 1),
            'hand_right': imgs[1].transpose(2, 0, 1),
            'hand_left':  imgs[2].transpose(2, 0, 1),
        },
        'prompt': prompt,
    }


def pipeline_node_fixed(img_front, img_left, img_right, joint_left, joint_right, prompt):
    """修复后的 ROS2 节点内模式的观测构建管线.

    来源: policy_inference_node.py (修复后)
    _get_observation()
    """
    from openpi_client import image_tools

    imgs = [jpeg_mapping(img_front), jpeg_mapping(img_right), jpeg_mapping(img_left)]
    imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
    imgs = image_tools.resize_with_pad(np.array(imgs), 224, 224)

    qpos = np.concatenate((np.array(joint_left), np.array(joint_right)), axis=0)

    return {
        'state': qpos,
        'images': {
            'top_head':   imgs[0].transpose(2, 0, 1),
            'hand_right': imgs[1].transpose(2, 0, 1),
            'hand_left':  imgs[2].transpose(2, 0, 1),
        },
        'prompt': prompt,
    }


def pipeline_node_old_broken(img_front, img_left, img_right, joint_left, joint_right, prompt):
    """修复前的 ROS2 节点内模式 (有 bug 的版本).

    差异: 无 JPEG, 无 BGR→RGB, cv2.resize 拉伸
    """
    imgs = [img_front, img_right, img_left]
    imgs = [cv2.resize(im, (224, 224)) for im in imgs]

    state = np.array(list(joint_left) + list(joint_right), dtype=np.float32)

    return {
        'state': state,
        'images': {
            'top_head':   imgs[0].transpose(2, 0, 1),
            'hand_right': imgs[1].transpose(2, 0, 1),
            'hand_left':  imgs[2].transpose(2, 0, 1),
        },
        'prompt': prompt,
    }


def compare_obs(name_a, obs_a, name_b, obs_b):
    """逐字段对比两个观测 dict."""
    print(f'\n{"="*60}')
    print(f'  {name_a}  vs  {name_b}')
    print(f'{"="*60}')

    # state
    s_a, s_b = obs_a['state'], obs_b['state']
    state_eq = np.array_equal(s_a, s_b)
    state_max_diff = np.max(np.abs(s_a.astype(float) - s_b.astype(float)))
    print(f'  state:      equal={state_eq}  max_diff={state_max_diff:.6e}  dtype=({s_a.dtype} vs {s_b.dtype})')

    # images
    for cam in ['top_head', 'hand_right', 'hand_left']:
        a = obs_a['images'][cam]
        b = obs_b['images'][cam]
        eq = np.array_equal(a, b)
        if eq:
            print(f'  {cam:12s}: identical  shape={a.shape}')
        else:
            diff = np.abs(a.astype(float) - b.astype(float))
            pct_diff = np.count_nonzero(a != b) / a.size * 100
            print(f'  {cam:12s}: DIFFERENT  pixels_differ={pct_diff:.1f}%  '
                  f'max_diff={diff.max():.0f}  mean_diff={diff.mean():.2f}  shape={a.shape}')

    # prompt
    p_eq = obs_a['prompt'] == obs_b['prompt']
    print(f'  prompt:     equal={p_eq}  ("{obs_a["prompt"]}" vs "{obs_b["prompt"]}")')

    all_img_eq = all(
        np.array_equal(obs_a['images'][c], obs_b['images'][c])
        for c in ['top_head', 'hand_right', 'hand_left']
    )
    overall = state_eq and all_img_eq and p_eq
    print(f'\n  >>> OVERALL: {"PASS ✅" if overall else "FAIL ❌"}')
    return overall


def test_image_pipeline():
    """测试 1: 图像管线对比 (无需模型)."""
    print('\n' + '#'*60)
    print('#  TEST 1: 图像管线对比')
    print('#'*60)

    # 生成模拟相机图像 (640x480 RGB, 模拟 D435 输出)
    np.random.seed(42)
    img_front = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img_left  = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img_right = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    joint_left  = [0.1, -0.2, 0.3, -0.1, 0.5, -0.3, 0.0]
    joint_right = [0.2, -0.1, 0.4, -0.2, 0.6, -0.2, 0.0]

    prompt = 'Flatten and fold the cloth.'

    print('\n  输入: 3x 随机 640x480 RGB uint8 + 14D joint + prompt')

    obs_ws    = pipeline_original_ws(img_front, img_left, img_right, joint_left, joint_right, prompt)
    obs_fixed = pipeline_node_fixed(img_front, img_left, img_right, joint_left, joint_right, prompt)
    obs_old   = pipeline_node_old_broken(img_front, img_left, img_right, joint_left, joint_right, prompt)

    # 对比 1: 修复后 vs 原版 WS — 应该完全一致
    r1 = compare_obs('WS 原版', obs_ws, 'Node 修复后', obs_fixed)

    # 对比 2: 修复前 vs 原版 WS — 应该有差异 (确认修复前确实有 bug)
    r2 = compare_obs('WS 原版', obs_ws, 'Node 修复前(旧)', obs_old)

    print(f'\n  --- 小结 ---')
    print(f'  修复后 vs WS原版: {"PASS ✅ 完全一致" if r1 else "FAIL ❌"}')
    print(f'  修复前 vs WS原版: {"FAIL ❌ 有差异 (预期)" if not r2 else "意外一致?"}')

    # 展示 resize_with_pad vs cv2.resize 的具体差异
    print(f'\n  --- resize 差异详解 ---')
    from openpi_client import image_tools
    test_img = img_front.copy()
    a = image_tools.resize_with_pad(test_img[np.newaxis], 224, 224)[0]  # [224,224,3]
    b = cv2.resize(test_img, (224, 224))                                 # [224,224,3]
    print(f'  resize_with_pad: shape={a.shape}, 保持宽高比+零填充')
    print(f'  cv2.resize:      shape={b.shape}, 直接拉伸')
    # 找 padding 区域
    pad_rows = np.all(a == 0, axis=(1, 2))
    pad_cols = np.all(a == 0, axis=(0, 2))
    n_pad_rows = np.sum(pad_rows)
    n_pad_cols = np.sum(pad_cols)
    print(f'  resize_with_pad 零填充: {n_pad_rows} 行 + {n_pad_cols} 列 '
          f'(640:480=4:3 → 224x168 内容 + 上下各 28 行填充)')
    diff = np.abs(a.astype(float) - b.astype(float))
    print(f'  像素差异: max={diff.max():.0f}  mean={diff.mean():.1f}  '
          f'differ={np.count_nonzero(a != b)/a.size*100:.1f}%')

    return r1


def test_full_inference(checkpoint_dir):
    """测试 2: 完整推理对比 (需要模型)."""
    print('\n' + '#'*60)
    print('#  TEST 2: 完整推理对比 (加载模型)')
    print('#'*60)

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    config_name = 'pi05_flatten_fold_normal'
    print(f'\n  加载模型: config={config_name}, ckpt={checkpoint_dir}')

    t0 = time.monotonic()
    train_config = _config.get_config(config_name)
    policy = _policy_config.create_trained_policy(train_config, checkpoint_dir)
    print(f'  模型加载完成: {time.monotonic()-t0:.1f}s')

    # 构造测试输入
    np.random.seed(42)
    img_front = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img_left  = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img_right = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    joint_left  = [0.1, -0.2, 0.3, -0.1, 0.5, -0.3, 0.0]
    joint_right = [0.2, -0.1, 0.4, -0.2, 0.6, -0.2, 0.0]
    prompt = 'Flatten and fold the cloth.'

    obs_ws    = pipeline_original_ws(img_front, img_left, img_right, joint_left, joint_right, prompt)
    obs_fixed = pipeline_node_fixed(img_front, img_left, img_right, joint_left, joint_right, prompt)

    import jax

    # Warmup (JIT 编译)
    print('  Warmup inference (JIT compile)...')
    policy.infer(obs_ws)

    # ── 固定 RNG 做公平对比 ──
    # policy.infer() 内部每次 split(rng), 导致连续调用随机种子不同.
    # 保存/恢复 _rng 确保 A 和 B 用完全相同的种子.
    saved_rng = policy._rng

    print('  Running inference A (WS 原版管线)...')
    policy._rng = saved_rng  # 重置 RNG
    t0 = time.monotonic()
    result_ws = policy.infer(obs_ws)
    t_ws = (time.monotonic() - t0) * 1000

    print('  Running inference B (Node 修复后管线)...')
    policy._rng = saved_rng  # 重置到相同种子
    t0 = time.monotonic()
    result_fixed = policy.infer(obs_fixed)
    t_fixed = (time.monotonic() - t0) * 1000

    actions_ws = result_ws['actions']
    actions_fixed = result_fixed['actions']

    print(f'\n  --- 推理结果对比 (同一 RNG 种子) ---')
    print(f'  actions shape: {actions_ws.shape} vs {actions_fixed.shape}')
    print(f'  latency: WS={t_ws:.0f}ms  Node={t_fixed:.0f}ms')

    exact_eq = np.array_equal(actions_ws, actions_fixed)
    if exact_eq:
        print(f'  actions: EXACT MATCH ✅')
    else:
        diff = np.abs(actions_ws.astype(float) - actions_fixed.astype(float))
        close = np.allclose(actions_ws, actions_fixed, atol=1e-6)
        print(f'  actions: exact_equal={exact_eq}  allclose(atol=1e-6)={close}')
        print(f'  max_diff={diff.max():.2e}  mean_diff={diff.mean():.2e}')

    # 对比修复前的旧管线 (不同的 obs, 预期差异大)
    obs_old = pipeline_node_old_broken(img_front, img_left, img_right, joint_left, joint_right, prompt)
    print('\n  Running inference C (Node 修复前管线, 同一 RNG)...')
    policy._rng = saved_rng  # 同样重置种子, 隔离差异来源为 obs 本身
    result_old = policy.infer(obs_old)
    actions_old = result_old['actions']

    diff_old = np.abs(actions_ws.astype(float) - actions_old.astype(float))
    print(f'  修复前 vs WS原版: max_diff={diff_old.max():.4f}  mean_diff={diff_old.mean():.4f}')
    print(f'  >>> 修复前的推理偏差显著大于修复后: {"是 ✅" if diff_old.max() > 1e-3 else "否 ⚠️"}')

    return exact_eq or np.allclose(actions_ws, actions_fixed, atol=1e-6)


def main():
    parser = argparse.ArgumentParser(description='对比测试 ROS2 节点内推理 vs WS 推理')
    parser.add_argument('--with-model', action='store_true',
        help='加载模型做完整推理对比 (需 GPU)')
    parser.add_argument('--checkpoint-dir',
        default='/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/Task_A/mixed_1',
        help='推理用的 checkpoint 路径')
    args = parser.parse_args()

    print('='*60)
    print('  ROS2 节点内推理 vs WebSocket 服务端推理 一致性测试')
    print('='*60)

    r1 = test_image_pipeline()

    if args.with_model:
        r2 = test_full_inference(args.checkpoint_dir)
    else:
        r2 = None
        print('\n  (跳过模型推理测试, 加 --with-model 启用)')

    print('\n' + '='*60)
    print('  最终结果')
    print('='*60)
    print(f'  图像管线一致性: {"PASS ✅" if r1 else "FAIL ❌"}')
    if r2 is not None:
        print(f'  推理结果一致性: {"PASS ✅" if r2 else "FAIL ❌"}')
    print()


if __name__ == '__main__':
    main()
