"""Static background mesh builder using Open3D tensor TSDF (CUDA backend).

See docs/deployment/inference_visualization_mesh.md §5. This module is optional — the
viz node handles ImportError gracefully and simply skips the background
mesh when open3d is not installed.

Install on sim01 (inside the kai0 uv venv):

    GIT_LFS_SKIP_SMUDGE=1 uv pip install 'open3d>=0.18'

Runtime contract:

    frames        : list of (depth_u16 (H,W), rgb_hwc_uint8 (H,W,3))
    K             : (3,3) float64 pinhole intrinsics
    T_world_cam   : (4,4) float64 camera pose in world frame
    bbox_exclude_* : optional (3,) arrays — vertices inside this AABB are
                    dropped from the final mesh (this is the foreground
                    workspace, meshed per-tick by _tick_fg_mesh)

Returns (verts, tris, colors, normals) as numpy arrays ready for rr.Mesh3D,
or None if the TSDF volume produced nothing usable.
"""

import os
import numpy as np

try:
    import open3d as o3d
    import open3d.core as o3c
    _OPEN3D_OK = True
    _OPEN3D_IMPORT_ERROR = ''
except Exception as _e:  # pragma: no cover — optional dependency
    o3d = None
    o3c = None
    _OPEN3D_OK = False
    _OPEN3D_IMPORT_ERROR = f'{type(_e).__name__}: {_e}'


def open3d_available() -> bool:
    return _OPEN3D_OK


def open3d_error() -> str:
    return _OPEN3D_IMPORT_ERROR


def build_bg_mesh_gpu(frames, K, T_world_cam,
                      bbox_exclude_min=None, bbox_exclude_max=None,
                      voxel_size=0.005, sdf_trunc=0.04,
                      depth_scale=1000.0, depth_trunc=2.0,
                      block_count=50000, device_str='CUDA:0'):
    """Fuse a short burst of RGB-D frames from one fixed camera into a
    background triangle mesh. See module docstring for shapes and semantics.
    """
    if not _OPEN3D_OK:
        raise RuntimeError(
            f'open3d not available ({_OPEN3D_IMPORT_ERROR}). '
            f'Run: uv pip install open3d>=0.18  (or set bg_enable:=false)')

    if len(frames) == 0:
        return None

    device = o3c.Device(device_str)

    # Intrinsics and extrinsics as float64 tensors on the same device.
    intrinsic_t = o3c.Tensor(np.asarray(K, dtype=np.float64))
    extrinsic_t = o3c.Tensor(
        np.linalg.inv(np.asarray(T_world_cam, dtype=np.float64)))

    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel_size,
        block_resolution=16,
        block_count=block_count,
        device=device,
    )

    for depth_u16, rgb in frames:
        # Open3D tensor images want C-contiguous arrays. The explicit dtype
        # casts are defensive: callers usually pass uint16 / uint8 already.
        depth_np = np.ascontiguousarray(depth_u16, dtype=np.uint16)
        rgb_np = np.ascontiguousarray(rgb, dtype=np.uint8)
        depth_t = o3d.t.geometry.Image(o3c.Tensor(depth_np, device=device))
        color_t = o3d.t.geometry.Image(o3c.Tensor(rgb_np, device=device))

        frustum = vbg.compute_unique_block_coordinates(
            depth_t, intrinsic_t, extrinsic_t,
            depth_scale=depth_scale, depth_max=depth_trunc)
        vbg.integrate(frustum, depth_t, color_t,
                      intrinsic_t, extrinsic_t,
                      depth_scale=depth_scale, depth_max=depth_trunc)

    mesh_t = vbg.extract_triangle_mesh()
    # .to_legacy() transfers everything back to CPU float64 numpy-friendly
    # arrays. This is the point where VoxelBlockGrid GPU memory can be freed.
    mesh_legacy = mesh_t.to_legacy()
    mesh_legacy.compute_vertex_normals()

    verts = np.asarray(mesh_legacy.vertices, dtype=np.float32)
    tris = np.asarray(mesh_legacy.triangles, dtype=np.int32)

    if len(verts) == 0 or len(tris) == 0:
        _release_cache(device_str)
        return None

    colors_f = np.asarray(mesh_legacy.vertex_colors)
    colors = None
    if colors_f.size > 0 and colors_f.shape[0] == len(verts):
        colors = (colors_f * 255.0).clip(0, 255).astype(np.uint8)

    normals_f = np.asarray(mesh_legacy.vertex_normals, dtype=np.float32)
    normals = normals_f if normals_f.shape[0] == len(verts) else None

    # Crop the foreground workspace AABB — those points are the job of the
    # per-tick dynamic mesh. Keep any triangle that has at least one vertex
    # OUTSIDE the excluded bbox, otherwise the background and foreground
    # meshes would overlap inside the workspace.
    if bbox_exclude_min is not None and bbox_exclude_max is not None:
        bmin = np.asarray(bbox_exclude_min, dtype=np.float32)
        bmax = np.asarray(bbox_exclude_max, dtype=np.float32)
        v_in = ((verts[:, 0] >= bmin[0]) & (verts[:, 0] <= bmax[0]) &
                (verts[:, 1] >= bmin[1]) & (verts[:, 1] <= bmax[1]) &
                (verts[:, 2] >= bmin[2]) & (verts[:, 2] <= bmax[2]))
        keep = ~v_in[tris].all(axis=1)
        tris = tris[keep]
        if len(tris) == 0:
            _release_cache(device_str)
            return None
        # Remap to only the vertices still referenced.
        used = np.unique(tris)
        remap = np.full(len(verts), -1, dtype=np.int64)
        remap[used] = np.arange(len(used), dtype=np.int64)
        tris = remap[tris].astype(np.int32)
        verts = verts[used]
        if colors is not None:
            colors = colors[used]
        if normals is not None:
            normals = normals[used]

    # Release Open3D's CUDA caching allocator so the JAX projection path
    # isn't starved for memory after the rebuild. This is why we preferred
    # `del vbg` + `release_cache` over leaving the volume alive.
    del vbg
    _release_cache(device_str)

    return verts, tris, colors, normals


def _release_cache(device_str: str) -> None:
    if not device_str.startswith('CUDA'):
        return
    try:
        o3c.cuda.release_cache()
    except Exception:
        # Older open3d builds do not expose release_cache; harmless.
        pass


# ── Persistence (survives node restarts so bg mesh is instant on reload) ──

def save_mesh_npz(path: str, verts, tris, colors, normals) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez_compressed(
        path,
        verts=verts.astype(np.float32),
        tris=tris.astype(np.int32),
        colors=(colors if colors is not None
                else np.empty((0, 3), dtype=np.uint8)),
        normals=(normals if normals is not None
                 else np.empty((0, 3), dtype=np.float32)),
    )


def load_mesh_npz(path: str):
    if not os.path.isfile(path):
        return None
    try:
        z = np.load(path)
    except Exception:
        return None
    colors = z['colors'] if z['colors'].size > 0 else None
    normals = z['normals'] if z['normals'].size > 0 else None
    return z['verts'], z['tris'], colors, normals
