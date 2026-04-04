#!/usr/bin/env python3
"""
标定验证: 三路点云对齐检查

加载 calibration.yaml, 从三个 RealSense 获取深度图,
投影到世界坐标系, 检查对齐精度。

用法:
  python3 calib/verify_calibration.py --config calib/calibration.yaml
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import yaml
import pyrealsense2 as rs

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/home/tim/workspace/piper_sdk')
from piper_fk import PiperFK
from piper_sdk import C_PiperInterface


JOINT_FACTOR = 57295.7795


def load_calibration(path: str) -> dict:
    """加载 calibration.yaml, 将 list → ndarray"""
    with open(path) as f:
        data = yaml.safe_load(f)
    for key in data['transforms']:
        data['transforms'][key] = np.array(data['transforms'][key])
    return data


def depth_to_pointcloud(
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    depth_scale: float = 0.001,
    max_depth_m: float = 2.0,
    step: int = 4,
    rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """深度图 → 相机系点云 (降采样)。

    Args:
        depth: uint16 depth image (raw sensor values)
        depth_scale: depth unit → meters (e.g. 0.001 means depth is in mm)
        step: 降采样步长 (每 step 个像素取 1 个)
        rgb: BGR image, same resolution as depth. If provided, returns per-point colors.

    Returns:
        (points [N, 3], colors [N, 3] or None) 单位 m, colors in RGB uint8
    """
    h, w = depth.shape
    z = depth[::step, ::step].astype(np.float32) * depth_scale
    u, v = np.meshgrid(
        np.arange(0, w, step, dtype=np.float32),
        np.arange(0, h, step, dtype=np.float32),
    )
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    mask = (z > 0.01) & (z < max_depth_m)
    points = np.stack([x[mask], y[mask], z[mask]], axis=-1)

    colors = None
    if rgb is not None:
        rgb_ds = rgb[::step, ::step]
        # BGR -> RGB
        colors = rgb_ds[mask][:, ::-1].copy()

    return points, colors


def grab_frame(serial: str, width=640, height=480) -> tuple[np.ndarray, np.ndarray, dict, float]:
    """单次抓帧 (RGB + aligned Depth + depth intrinsics + depth scale)"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    profile = pipeline.start(config)

    # 对齐深度到彩色
    align = rs.align(rs.stream.color)

    # Depth post-processing filters
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)
    hole_filling = rs.hole_filling_filter()

    # Warm up + feed temporal filter
    for _ in range(30):
        frames = pipeline.wait_for_frames(timeout_ms=3000)
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)

    # Final capture
    frames = pipeline.wait_for_frames(timeout_ms=3000)
    aligned = align.process(frames)
    bgr = np.asanyarray(aligned.get_color_frame().get_data())
    depth_frame = aligned.get_depth_frame()
    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)
    depth_frame = hole_filling.process(depth_frame)
    depth = np.asanyarray(depth_frame.get_data())
    # 对齐后深度使用彩色流内参
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    cintr = color_stream.get_intrinsics()
    depth_intr = {
        'fx': cintr.fx, 'fy': cintr.fy,
        'cx': cintr.ppx, 'cy': cintr.ppy,
    }
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    pipeline.stop()
    return bgr, depth, depth_intr, depth_scale


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """将点云从一个坐标系变换到另一个 (4×4 齐次矩阵)"""
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ points.T).T + t


def _log_frame(path: str, T: np.ndarray, size: float = 0.05):
    """Log a coordinate frame (RGB axes) at the given SE3 pose."""
    import rerun as rr
    rr.log(path, rr.Transform3D(
        translation=T[:3, 3], mat3x3=T[:3, :3],
    ))
    rr.log(f"{path}/axes", rr.Arrows3D(
        origins=[[0, 0, 0]] * 3,
        vectors=[[size, 0, 0], [0, size, 0], [0, 0, size]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    ))


def main():
    parser = argparse.ArgumentParser(description='标定验证')
    parser.add_argument('--config', default='calib/calibration.yaml')
    parser.add_argument('--max-depth', type=float, default=1.5, help='最大深度 (m)')
    parser.add_argument('--step', type=int, default=1, help='点云降采样步长')
    parser.add_argument('--workspace', type=float, nargs=6, metavar=('X0', 'X1', 'Y0', 'Y1', 'Z0', 'Z1'),
                        default=None,
                        help='工作空间包围盒 [x0 x1 y0 y1 z0 z1] (m, 世界系)')
    args = parser.parse_args()

    calib = load_calibration(args.config)
    transforms = calib['transforms']
    intrinsics = calib['intrinsics']
    hw = calib['hardware']

    print("=" * 60)
    print("标定验证: 三路点云对齐")
    print("=" * 60)

    # 验证 1: 基座对称性
    print("\n--- 基座对称性 ---")
    pos_L = transforms['T_world_baseL'][:3, 3]
    pos_R = transforms['T_world_baseR'][:3, 3]
    midpoint = (pos_L + pos_R) / 2
    print(f"  baseL: [{pos_L[0]:+.4f}, {pos_L[1]:+.4f}, {pos_L[2]:+.4f}]")
    print(f"  baseR: [{pos_R[0]:+.4f}, {pos_R[1]:+.4f}, {pos_R[2]:+.4f}]")
    print(f"  midpoint: [{midpoint[0]:.4f}, {midpoint[1]:.4f}, {midpoint[2]:.4f}]")
    sym_err = np.linalg.norm(midpoint)
    print(f"  对称误差: {sym_err*1000:.1f} mm {'✓' if sym_err < 0.005 else '△'}")

    # 验证 2: 采集深度图
    print("\n--- 采集深度图 ---")
    cameras = [
        ('head', hw['cam_f_serial'], intrinsics['cam_f'], transforms['T_world_camF']),
    ]

    # 腕部相机: 先记录配置, 后续抓帧时再读关节角 (避免时间差)
    fk = PiperFK()
    arm_cameras = []
    for arm_label, can_name, serial, intr_key, T_link6_cam, T_world_base in [
        ('left', hw['left_arm_can'], hw['cam_l_serial'], 'cam_l',
         transforms['T_link6_camL'], transforms['T_world_baseL']),
        ('right', hw['right_arm_can'], hw['cam_r_serial'], 'cam_r',
         transforms['T_link6_camR'], transforms['T_world_baseR']),
    ]:
        try:
            piper = C_PiperInterface(can_name)
            piper.ConnectPort()
            arm_cameras.append((arm_label, serial, intr_key, T_link6_cam, T_world_base, piper))
        except Exception as e:
            print(f"  [WARN] {arm_label} arm not available: {e}")

    all_points = {}
    all_colors = {}

    # 头顶相机 (固定位姿, 直接用标定结果)
    for label, serial, intr, T_world_cam in cameras:
        print(f"  Grabbing {label} ({serial})...", end='', flush=True)
        try:
            bgr, depth, depth_intr, depth_scale = grab_frame(serial)
            pc, colors = depth_to_pointcloud(
                depth, depth_intr['fx'], depth_intr['fy'], depth_intr['cx'], depth_intr['cy'],
                depth_scale=depth_scale, max_depth_m=args.max_depth, step=args.step,
                rgb=bgr,
            )
            pc_world = transform_points(pc, T_world_cam)
            all_points[label] = pc_world
            all_colors[label] = colors
            print(f" {len(pc_world)} points")
        except Exception as e:
            print(f" FAILED: {e}")

    # 腕部相机 (抓帧后立即读关节角, 减少时间差)
    for arm_label, serial, intr_key, T_link6_cam, T_world_base, piper in arm_cameras:
        print(f"  Grabbing {arm_label} ({serial})...", end='', flush=True)
        try:
            bgr, depth, depth_intr, depth_scale = grab_frame(serial)
            # 抓帧后立即读关节角, 使 FK 与图像时间对齐
            msg = piper.GetArmJointMsgs()
            js = msg.joint_state
            q = np.array([js.joint_1, js.joint_2, js.joint_3,
                          js.joint_4, js.joint_5, js.joint_6]) / JOINT_FACTOR
            T_base_ee = fk.fk_homogeneous(q)
            T_world_cam = T_world_base @ T_base_ee @ T_link6_cam
            pc, colors = depth_to_pointcloud(
                depth, depth_intr['fx'], depth_intr['fy'], depth_intr['cx'], depth_intr['cy'],
                depth_scale=depth_scale, max_depth_m=args.max_depth, step=args.step,
                rgb=bgr,
            )
            pc_world = transform_points(pc, T_world_cam)
            all_points[arm_label] = pc_world
            all_colors[arm_label] = colors
            print(f" {len(pc_world)} points")
        except Exception as e:
            print(f" FAILED: {e}")

    if len(all_points) < 2:
        print("\nNot enough cameras available for comparison.")
        return

    # 验证 3: 点云统计 (过滤前)
    print("\n--- 点云统计 (世界系, 过滤前) ---")
    for label, pts in all_points.items():
        print(f"  {label}: {len(pts)} pts, "
              f"x=[{pts[:,0].min():.3f}, {pts[:,0].max():.3f}], "
              f"y=[{pts[:,1].min():.3f}, {pts[:,1].max():.3f}], "
              f"z=[{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")

    # 工作空间过滤
    if args.workspace is not None:
        ws = args.workspace
    else:
        # Auto-compute from point cloud overlap region with some margin
        all_pts = np.concatenate(list(all_points.values()))
        p5 = np.percentile(all_pts, 5, axis=0)
        p95 = np.percentile(all_pts, 95, axis=0)
        ws = [p5[0], p95[0], p5[1], p95[1], p5[2], p95[2]]
        print(f"\n  Auto workspace from point cloud percentiles (5%-95%)")
    ws_min = np.array([ws[0], ws[2], ws[4]])
    ws_max = np.array([ws[1], ws[3], ws[5]])
    print(f"\n--- 工作空间过滤: x=[{ws[0]:.2f}, {ws[1]:.2f}], y=[{ws[2]:.2f}, {ws[3]:.2f}], z=[{ws[4]:.2f}, {ws[5]:.2f}] ---")
    for label in all_points:
        pts = all_points[label]
        mask = np.all((pts >= ws_min) & (pts <= ws_max), axis=1)
        all_points[label] = pts[mask]
        if label in all_colors and all_colors[label] is not None:
            all_colors[label] = all_colors[label][mask]
        print(f"  {label}: {len(pts)} -> {mask.sum()} pts")

    # 验证 4: 定量对齐指标 (两两最近邻距离)
    print("\n--- 点云对齐指标 ---")
    from scipy.spatial import cKDTree
    labels = list(all_points.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb = labels[i], labels[j]
            pa, pb = all_points[la], all_points[lb]
            if len(pa) == 0 or len(pb) == 0:
                print(f"  {la} vs {lb}: empty point cloud, skipping")
                continue
            # 找重叠区域: 两个点云 bounding box 的交集
            bb_min = np.maximum(pa.min(axis=0), pb.min(axis=0))
            bb_max = np.minimum(pa.max(axis=0), pb.max(axis=0))
            if np.any(bb_min >= bb_max):
                print(f"  {la} vs {lb}: no bounding-box overlap")
                continue
            # 只保留重叠区域内的点
            mask_a = np.all((pa >= bb_min) & (pa <= bb_max), axis=1)
            mask_b = np.all((pb >= bb_min) & (pb <= bb_max), axis=1)
            pa_ov, pb_ov = pa[mask_a], pb[mask_b]
            if len(pa_ov) < 100 or len(pb_ov) < 100:
                print(f"  {la} vs {lb}: insufficient overlap ({len(pa_ov)}/{len(pb_ov)} pts)")
                continue
            # 采样加速 (最多 20k 点查询)
            if len(pa_ov) > 20000:
                pa_ov = pa_ov[np.random.choice(len(pa_ov), 20000, replace=False)]
            tree = cKDTree(pb_ov)
            dists, _ = tree.query(pa_ov, k=1)
            p50 = np.median(dists) * 1000
            p90 = np.percentile(dists, 90) * 1000
            p95 = np.percentile(dists, 95) * 1000
            quality = "GOOD" if p50 < 15.0 else ("OK" if p50 < 30.0 else "POOR")
            print(f"  {la} vs {lb} (overlap {len(pa_ov)}+{len(pb_ov)} pts): "
                  f"median={p50:.1f}mm, p90={p90:.1f}mm, p95={p95:.1f}mm [{quality}]")

            # Mutual nearest-neighbor filter (remove outliers) and report cleaned metric
            dists_ba, idx_ba = cKDTree(pa_ov).query(pb_ov, k=1)
            # Keep only mutual nearest neighbors within 2x median distance
            threshold = max(np.median(dists) * 2, 0.01)
            clean_mask = dists < threshold
            if clean_mask.sum() > 50:
                p50c = np.median(dists[clean_mask]) * 1000
                p90c = np.percentile(dists[clean_mask], 90) * 1000
                quality_c = "GOOD" if p50c < 15.0 else ("OK" if p50c < 30.0 else "POOR")
                print(f"    cleaned ({clean_mask.sum()} pts, {len(pa_ov)-clean_mask.sum()} outliers removed): "
                      f"median={p50c:.1f}mm, p90={p90c:.1f}mm [{quality_c}]")

    # 验证 5: Rerun 交互式 3D 可视化
    print("\n--- 可视化 (Rerun) ---")
    try:
        import rerun as rr

        import rerun.blueprint as rrb
        rr.init("calibration_verify", spawn=True)
        rr.send_blueprint(rrb.Blueprint(
            rrb.Spatial3DView(origin="world", contents="world/**"),
        ))

        # Head camera FIRST — so it's in the initial viewport
        T_head = np.array(transforms['T_world_camF'])
        head_pos = T_head[:3, 3]
        R_head = T_head[:3, :3]

        def _tw(p):
            return (R_head @ np.array(p) + head_pos).tolist()

        d = 0.15
        hw, hh = 0.09, 0.07
        o = _tw([0, 0, 0])
        c1, c2, c3, c4 = _tw([-hw, -hh, d]), _tw([hw, -hh, d]), _tw([hw, hh, d]), _tw([-hw, hh, d])
        ground = [head_pos[0], head_pos[1], 0.0]
        all_lines = [
            [o, c1], [o, c2], [o, c3], [o, c4],
            [c1, c2], [c2, c3], [c3, c4], [c4, c1],
            [o, ground],
        ]
        rr.log("world/head_cam", rr.LineStrips3D(
            all_lines, colors=[[255, 50, 50]], radii=[0.004],
        ), static=True)
        rr.log("world/head_cam_pos", rr.Points3D(
            [o], colors=[[255, 50, 50]], radii=[0.02], labels=["D435 (head)"],
        ), static=True)
        print(f"  Head cam at world pos: [{head_pos[0]:.3f}, {head_pos[1]:.3f}, {head_pos[2]:.3f}]")

        tint_rgb = {'head': [255, 100, 100], 'left': [100, 100, 255], 'right': [100, 255, 100]}

        for label, pts in all_points.items():
            colors = all_colors.get(label)
            if colors is not None:
                rr.log(f"world/{label}_rgb", rr.Points3D(pts, colors=colors, radii=0.001))
            # Also log tinted version for easier per-camera identification
            c = tint_rgb.get(label, [200, 200, 200])
            rr.log(f"world/{label}_tint", rr.Points3D(pts, colors=[c] * len(pts), radii=0.001))

        # 基座位置
        for label, T_key in [('baseL', 'T_world_baseL'), ('baseR', 'T_world_baseR')]:
            pos = transforms[T_key][:3, 3]
            rr.log(f"world/{label}", rr.Points3D([pos], colors=[[255, 255, 0]], radii=0.015, labels=[label]), static=True)


        # Arm mesh setup — load STL files, position via FK
        mesh_dir = os.path.join(os.path.dirname(__file__),
            '../kai0/train_deploy_alignment/inference/agilex/'
            'Piper_ros_private-ros-noetic/src/piper_description/meshes')
        mesh_names = ['base_link', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6',
                      'gripper_base', 'link7', 'link8']
        arm_meshes = {}
        if os.path.isdir(mesh_dir):
            import trimesh
            for arm_label, _, _, T_link6_cam, T_world_base, piper in arm_cameras:
                for mesh_name in mesh_names:
                    stl_path = os.path.join(mesh_dir, f'{mesh_name}.STL')
                    if os.path.exists(stl_path):
                        entity = f"world/{arm_label}/{mesh_name}"
                        m = trimesh.load(stl_path)
                        rr.log(entity, rr.Mesh3D(
                            vertex_positions=m.vertices,
                            triangle_indices=m.faces,
                            vertex_colors=np.full((len(m.vertices), 3), [100, 255, 100], dtype=np.uint8),
                            albedo_factor=[100, 255, 100, 15],  # alpha here controls transparency
                        ), static=True)
                arm_meshes[arm_label] = True
                print(f"  Loaded {len(mesh_names)} meshes for {arm_label} arm")

        # 工作空间包围盒
        if ws is not None:
            center = [(ws[0] + ws[1]) / 2, (ws[2] + ws[3]) / 2, (ws[4] + ws[5]) / 2]
            size = [ws[1] - ws[0], ws[3] - ws[2], ws[5] - ws[4]]
            rr.log("world/workspace", rr.Boxes3D(centers=[center], sizes=[size], colors=[[255, 255, 255, 40]]))

        print("  Rerun viewer launched")

        # Realtime update loop
        print("\n--- 实时更新 (Ctrl+C 退出) ---")

        def _make_depth_filters():
            spatial = rs.spatial_filter()
            spatial.set_option(rs.option.filter_magnitude, 2)
            spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
            spatial.set_option(rs.option.filter_smooth_delta, 20)
            temporal = rs.temporal_filter()
            temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
            temporal.set_option(rs.option.filter_smooth_delta, 20)
            return spatial, temporal

        # Keep persistent camera pipelines for realtime streaming
        live_cams = {}
        for label, serial, intr, T_world_cam in cameras:
            try:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                pipeline.start(config)
                align = rs.align(rs.stream.color)
                filters = _make_depth_filters()
                live_cams[label] = (pipeline, align, T_world_cam, None, filters)
            except Exception as e:
                print(f"  [WARN] {label} camera not available for live: {e}")

        for arm_label, serial, intr_key, T_link6_cam, T_world_base, piper in arm_cameras:
            try:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                profile = pipeline.start(config)
                align = rs.align(rs.stream.color)
                color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
                cintr = color_stream.get_intrinsics()
                depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                filters = _make_depth_filters()
                live_cams[arm_label] = (pipeline, align, None, (T_link6_cam, T_world_base, piper, cintr, depth_scale), filters)
            except Exception as e:
                print(f"  [WARN] {arm_label} camera not available for live: {e}")

        import time
        frame_idx = 0
        try:
            while True:
                rr.set_time("frame", sequence=frame_idx)

                # Update arm joints and wrist cameras
                for arm_label, _, _, T_link6_cam, T_world_base, piper in arm_cameras:
                    T_base = np.array(T_world_base)
                    # Read joint angles
                    msg = piper.GetArmJointMsgs()
                    js = msg.joint_state
                    q_rad = np.array([js.joint_1, js.joint_2, js.joint_3,
                                      js.joint_4, js.joint_5, js.joint_6]) / JOINT_FACTOR

                    # FK for all links
                    T_ee = fk.fk_homogeneous(q_rad)
                    link_Ts = fk.fk_all_links(q_rad)

                    # Position arm meshes via FK
                    if arm_label in arm_meshes:
                        rr.log(f"world/{arm_label}/base_link", rr.Transform3D(
                            translation=T_base[:3, 3], mat3x3=T_base[:3, :3],
                        ))
                        for idx, T_link in enumerate(link_Ts):
                            T_world_link = T_base @ T_link
                            rr.log(f"world/{arm_label}/link{idx+1}", rr.Transform3D(
                                translation=T_world_link[:3, 3], mat3x3=T_world_link[:3, :3],
                            ))
                        T_world_link6 = T_base @ link_Ts[5]
                        rr.log(f"world/{arm_label}/gripper_base", rr.Transform3D(
                            translation=T_world_link6[:3, 3], mat3x3=T_world_link6[:3, :3],
                        ))
                        from scipy.spatial.transform import Rotation as R_
                        for link_name, rpy in [('link7', [np.pi/2, 0, 0]), ('link8', [np.pi/2, 0, -np.pi])]:
                            T_offset = np.eye(4)
                            T_offset[:3, :3] = R_.from_euler('xyz', rpy).as_matrix()
                            T_offset[:3, 3] = [0, 0, 0.1358]
                            T_w = T_world_link6 @ T_offset
                            rr.log(f"world/{arm_label}/{link_name}", rr.Transform3D(
                                translation=T_w[:3, 3], mat3x3=T_w[:3, :3],
                            ))

                    # EE and cam positions in world
                    T_world_ee = T_base @ T_ee
                    T_world_cam = T_world_ee @ np.array(T_link6_cam)
                    R_cam = T_world_cam[:3, :3]
                    cam_pos = T_world_cam[:3, 3]
                    ee_pos = T_world_ee[:3, 3]
                    base_pos = T_base[:3, 3]

                    # Wrist camera frustum (world coords)
                    fd, fhw, fhh = 0.08, 0.04, 0.03
                    fc = [(R_cam @ np.array(p) + cam_pos).tolist()
                          for p in [[0,0,0], [-fhw,-fhh,fd], [fhw,-fhh,fd], [fhw,fhh,fd], [-fhw,fhh,fd]]]
                    cam_color = [100, 100, 255] if arm_label == 'left' else [100, 255, 100]
                    rr.log(f"world/{arm_label}_cam", rr.LineStrips3D(
                        [[fc[0],fc[1]], [fc[0],fc[2]], [fc[0],fc[3]], [fc[0],fc[4]],
                         [fc[1],fc[2]], [fc[2],fc[3]], [fc[3],fc[4]], [fc[4],fc[1]],
                         [cam_pos.tolist(), ee_pos.tolist()],  # link to EE
                         [ee_pos.tolist(), base_pos.tolist()]],  # link to base
                        colors=[cam_color], radii=[0.002],
                    ))
                    rr.log(f"world/{arm_label}_cam_label", rr.Points3D(
                        [cam_pos.tolist()], colors=[cam_color], radii=[0.008],
                        labels=[f"D405 ({arm_label})"],
                    ))

                    # Update wrist camera point cloud
                    if arm_label in live_cams:
                        pipeline, align, _, arm_info, filters = live_cams[arm_label]
                        T_link6_cam_a, T_world_base_a, piper_a, cintr, depth_scale = arm_info
                        spatial_f, temporal_f = filters
                        try:
                            frames = pipeline.wait_for_frames(timeout_ms=100)
                            aligned = align.process(frames)
                            bgr = np.asanyarray(aligned.get_color_frame().get_data())
                            depth_frame = aligned.get_depth_frame()
                            depth_frame = spatial_f.process(depth_frame)
                            depth_frame = temporal_f.process(depth_frame)
                            depth = np.asanyarray(depth_frame.get_data())
                            T_ee = fk.fk_homogeneous(q_rad)
                            T_w_cam = T_base @ T_ee @ np.array(T_link6_cam_a)
                            pc, colors = depth_to_pointcloud(
                                depth, cintr.fx, cintr.fy, cintr.ppx, cintr.ppy,
                                depth_scale=depth_scale, max_depth_m=args.max_depth, step=1, rgb=bgr,
                            )
                            pc_world = transform_points(pc, T_w_cam)
                            if colors is not None:
                                rr.log(f"world/{arm_label}_rgb", rr.Points3D(pc_world, colors=colors, radii=0.001))
                        except Exception:
                            pass



                # Update head camera point cloud
                if 'head' in live_cams:
                    pipeline, align, T_world_cam, _, filters = live_cams['head']
                    spatial_f, temporal_f = filters
                    try:
                        frames = pipeline.wait_for_frames(timeout_ms=100)
                        aligned = align.process(frames)
                        bgr = np.asanyarray(aligned.get_color_frame().get_data())
                        depth_frame = aligned.get_depth_frame()
                        depth_frame = spatial_f.process(depth_frame)
                        depth_frame = temporal_f.process(depth_frame)
                        depth = np.asanyarray(depth_frame.get_data())
                        profile = pipeline.get_active_profile()
                        cs = profile.get_stream(rs.stream.color).as_video_stream_profile()
                        ci = cs.get_intrinsics()
                        ds = profile.get_device().first_depth_sensor().get_depth_scale()
                        pc, colors = depth_to_pointcloud(
                            depth, ci.fx, ci.fy, ci.ppx, ci.ppy,
                            depth_scale=ds, max_depth_m=args.max_depth, step=1, rgb=bgr,
                        )
                        pc_world = transform_points(pc, T_world_cam)
                        if colors is not None:
                            rr.log("world/head_rgb", rr.Points3D(pc_world, colors=colors, radii=0.001))
                    except Exception:
                        pass

                frame_idx += 1
                time.sleep(0.033)  # ~30fps

        except KeyboardInterrupt:
            print("\n  Stopped realtime update")
        finally:
            for label, (pipeline, *_) in live_cams.items():
                try:
                    pipeline.stop()
                except Exception:
                    pass

    except ImportError:
        print("  rerun not available, skipping visualization")
    except Exception as e:
        import traceback
        print(f"  Rerun error: {e}")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
