"""V1 Triton inference WebSocket serve for deepdive_kai0 (B4 Phase 1).

Wraps optimize/v1_triton/Pi05InferenceTuned as BasePolicy that
WebsocketPolicyServer can host. Protocol identical to serve_policy.py
(:8000 JAX backend), just different inference path.

Scope (Phase 1):
  ✓ WebSocket server on :8002, msgpack protocol identical to JAX serve
  ✓ V1 Triton inference (Pi05InferenceTuned, P50=32 ms on 5090)
  ✓ Image preprocess (resize 224×224, bfloat16, cuda)
  ✓ Action denormalize via norm_stats.json
  ✓ Action chunk return (50, 14)
  ⚠ TODO Phase 2: per-inference state encoding via sentencepiece
                  (now uses prebaked language_embeds → NOT state-conditioned)

Usage (sim01, after `convert_kai0_to_v1.py` 出 .pkl):
    .venv_5090_trt/bin/python kai0/scripts/serve_policy_v1.py \\
        --pkl optimize/results/task_a_mix_b6000_p1200_v1.pkl \\
        --norm-stats kai0/assets/<asset_id>/<repo_id>/norm_stats.json \\
        --port 8002 \\
        --num-views 3 --chunk-size 50 --action-dim 14

Health check: curl http://<host>:8002/healthz → "OK"

See docs/deployment/realtime_vla_optimization_analysis.md §7.2 for context.
"""
import argparse
import json
import logging
import os
import pickle
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Repo paths (lazy imports in main() to keep --help working without all deps)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_V1_TRITON_DIR = _REPO_ROOT / "optimize" / "v1_triton"
_OPENPI_SRC = _REPO_ROOT / "kai0" / "src"
_OPENPI_CLIENT_SRC = _REPO_ROOT / "kai0" / "packages" / "openpi-client" / "src"


def _ensure_imports():
    """Lazy import V1 + openpi modules (kept out of module top so --help works).

    Required deps in serving venv (kai0/.venv_5090_trt missing some by default):
      - torch, triton (have)
      - sentencepiece (have)
      - websockets (pip install websockets)
      - msgpack-numpy (pip install msgpack-numpy)
      - openpi_client (PYTHONPATH includes kai0/packages/openpi-client/src)
      - openpi (PYTHONPATH includes kai0/src)
    """
    sys.path.insert(0, str(_V1_TRITON_DIR))
    sys.path.insert(0, str(_OPENPI_SRC))
    sys.path.insert(0, str(_OPENPI_CLIENT_SRC))

    global Pi05InferenceTuned, websocket_policy_server, _base_policy
    from pi05_infer_tuned import Pi05InferenceTuned as _Pi05  # noqa
    from openpi.serving import websocket_policy_server as _wsps  # noqa
    from openpi_client import base_policy as _bp  # noqa
    Pi05InferenceTuned = _Pi05
    websocket_policy_server = _wsps
    _base_policy = _bp


logger = logging.getLogger(__name__)

# Populated by _ensure_imports() in main(). Type annotations use string forms below.
Pi05InferenceTuned = None  # type: ignore[assignment]
websocket_policy_server = None  # type: ignore[assignment]
_base_policy = None  # type: ignore[assignment]


def _resize_image_to_224(img_uint8: np.ndarray) -> np.ndarray:
    """Resize HxWx3 uint8 → 224x224x3 uint8.

    Uses PIL (avoids opencv dependency). Bilinear interpolation.
    """
    from PIL import Image as _PIL_Image

    if img_uint8.shape[:2] == (224, 224):
        return img_uint8
    pil = _PIL_Image.fromarray(img_uint8)
    pil = pil.resize((224, 224), _PIL_Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def _normalize_image_uint8_to_bf16(img_uint8: np.ndarray) -> torch.Tensor:
    """uint8 [0,255] HxWx3 → bf16 [-1,1] HxWx3 cuda.

    pi05 image normalization: x / 127.5 - 1.0 (the openpi/big_vision standard).
    """
    arr = torch.from_numpy(img_uint8).cuda()  # uint8
    arr = arr.to(torch.float32) / 127.5 - 1.0
    return arr.to(torch.bfloat16)


class V1Policy:
    """Adapt V1 Triton Pi05InferenceTuned to BasePolicy protocol.

    The obs dict (msgpack from ROS2 client) contains camera images + joint
    state + prompt. Phase 1 only consumes images + builds noise; state and
    prompt are NOT re-encoded per-inference (uses prebaked language_embeds
    from convert_kai0_to_v1.py). See Phase 2 TODO at module docstring.
    """

    def __init__(
        self,
        v1_infer,  # Pi05InferenceTuned
        action_norm_mean: np.ndarray,
        action_norm_std: np.ndarray,
        action_dim: int,
        image_keys: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        metadata: dict[str, Any] | None = None,
    ):
        self._v1 = v1_infer
        self._action_dim = action_dim
        # Save denorm stats as torch bf16 cuda for fast apply
        self._a_mean = torch.from_numpy(action_norm_mean[:action_dim].astype(np.float32)).cuda()
        self._a_std = torch.from_numpy(action_norm_std[:action_dim].astype(np.float32)).cuda()
        # Guard against zero std
        self._a_std = torch.where(self._a_std < 1e-6, torch.ones_like(self._a_std), self._a_std)
        self._image_keys = image_keys
        self._metadata = metadata or {"backend": "v1_triton", "version": 1}
        self._chunk_size = v1_infer.chunk_size
        self._num_views = v1_infer.num_views
        # Stable noise per-call would defeat diversity; sample fresh each infer.
        self._noise_gen = torch.Generator(device="cuda")
        self._noise_gen.manual_seed(0)

    def infer(self, obs: dict) -> dict:
        """Run one V1 inference cycle.

        Args:
            obs: dict from ROS2 client. Expected keys (subset of openpi protocol):
              - images: dict[str, HxWx3 uint8 numpy] — one entry per camera
              - state: (state_dim,) float numpy (joint state, kai0 = 14)
              - prompt: str (task instruction) — Phase 1 ignored (prebaked)

        Returns:
            dict with:
              - actions: (chunk_size, action_dim) float32 numpy (denormalized)
              - policy_timing: {"infer_ms": float}
              - server_backend: "v1_triton"
        """
        t_start = time.monotonic()

        # 1. Image preprocess: pick views in fixed order, resize, normalize, stack
        images_dict = obs.get("images") or obs.get("image") or {}
        if not isinstance(images_dict, dict):
            raise ValueError(f"obs['images'] must be dict, got {type(images_dict)}")
        view_tensors = []
        for key in self._image_keys[: self._num_views]:
            if key not in images_dict:
                # Fall back: take first num_views images in insertion order
                view_tensors = [
                    _normalize_image_uint8_to_bf16(_resize_image_to_224(np.asarray(v)))
                    for v in list(images_dict.values())[: self._num_views]
                ]
                break
            view_tensors.append(
                _normalize_image_uint8_to_bf16(_resize_image_to_224(np.asarray(images_dict[key])))
            )
        if len(view_tensors) != self._num_views:
            raise ValueError(
                f"Need {self._num_views} views, got {len(view_tensors)} (keys: {list(images_dict.keys())})"
            )
        image_input = torch.stack(view_tensors, dim=0).contiguous()  # (num_views, 224, 224, 3) bf16 cuda

        # 2. Sample fresh noise (chunk_size, 32) bf16 cuda
        noise = torch.randn(
            self._chunk_size, 32,
            dtype=torch.bfloat16, device="cuda",
            generator=self._noise_gen,
        )

        # 3. V1 inference (Phase 1: discrete_state_input=False, uses prebaked language_embeds)
        # TODO Phase 2: re-tokenize prompt + state via sentencepiece, pass via build_prompt_embeds
        t_infer = time.monotonic()
        with torch.inference_mode():
            action_chunk = self._v1.forward(image_input, noise)  # (chunk_size, 32) bf16 cuda
        torch.cuda.synchronize()
        infer_ms = (time.monotonic() - t_infer) * 1000

        # 4. Take first action_dim, denormalize
        a = action_chunk[:, : self._action_dim].to(torch.float32)  # (chunk_size, action_dim)
        a = a * self._a_std[None, :] + self._a_mean[None, :]
        actions_np = a.detach().cpu().numpy()  # (chunk_size, action_dim) float32

        total_ms = (time.monotonic() - t_start) * 1000

        return {
            "actions": actions_np,
            "policy_timing": {
                "infer_ms": float(infer_ms),
                "total_ms": float(total_ms),
            },
            "server_backend": "v1_triton",
            "action_kind": "joint",
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def load_v1_inference(pkl_path: str, num_views: int, chunk_size: int):
    logger.info(f"Loading V1 ckpt from {pkl_path} ...")
    t0 = time.perf_counter()
    with open(pkl_path, "rb") as f:
        ckpt = pickle.load(f)
    logger.info(
        f"  loaded {sum(v.numel()*v.element_size() for v in ckpt.values())/1e9:.2f} GB tensors "
        f"in {time.perf_counter()-t0:.1f}s"
    )
    logger.info(
        f"Building Pi05InferenceTuned(num_views={num_views}, chunk_size={chunk_size}) "
        f"+ CUDA Graph capture ..."
    )
    t0 = time.perf_counter()
    infer = Pi05InferenceTuned(
        ckpt, num_views=num_views, chunk_size=chunk_size,
        discrete_state_input=False,  # Phase 1: uses prebaked language_embeds
    )
    logger.info(f"  build + capture in {time.perf_counter()-t0:.1f}s")
    return infer


def load_action_norm_stats(norm_stats_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load action mean/std from openpi-format norm_stats.json.

    Format (per kai0/assets/<asset>/<repo>/norm_stats.json):
      {"norm_stats": {"state": {"mean": [...], "std": [...]}, "actions": {"mean": [...], "std": [...]}}}
    """
    with open(norm_stats_path) as f:
        data = json.load(f)
    norm = data["norm_stats"]
    if "actions" not in norm:
        raise ValueError(f"norm_stats.json missing 'actions' key (have: {list(norm.keys())})")
    a = norm["actions"]
    if "mean" not in a or "std" not in a:
        raise ValueError(f"norm_stats['actions'] needs mean+std (have: {list(a.keys())})")
    return np.asarray(a["mean"], dtype=np.float32), np.asarray(a["std"], dtype=np.float32)


def warmup(policy: V1Policy, n: int = 3) -> None:
    """Warm-up to ensure CUDA Graph + caches are hot before serving."""
    logger.info(f"Warming up V1 inference ({n} dummy iters) ...")
    H, W = 480, 640  # arbitrary; resize to 224 happens inside
    dummy_obs = {
        "images": {
            f"view_{i}": np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
            for i in range(policy._num_views)
        },
        "state": np.zeros(policy._action_dim, dtype=np.float32),
        "prompt": "warmup",
    }
    for i in range(n):
        out = policy.infer(dummy_obs)
        logger.info(f"  warmup {i+1}/{n}: infer_ms={out['policy_timing']['infer_ms']:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="V1 Triton WebSocket serve for deepdive_kai0")
    parser.add_argument("--pkl", required=True, help="V1 pickle from convert_kai0_to_v1.py")
    parser.add_argument("--norm-stats", required=True,
                        help="kai0 norm_stats.json (for action denormalize)")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=14,
                        help="output action dim to take from V1's (chunk, 32) output")
    parser.add_argument("--image-keys", nargs="+",
                        default=["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
                        help="Camera keys in obs['images'] dict, in stack order")
    parser.add_argument("--warmup-iters", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    # Lazy import heavy deps now that --help has succeeded.
    _ensure_imports()

    if not torch.cuda.is_available():
        logger.error("CUDA not available — V1 requires a CUDA-capable GPU.")
        sys.exit(1)
    logger.info(f"Device: {torch.cuda.get_device_name(0)} (sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}0)")
    logger.info(f"torch {torch.__version__}, cuda {torch.version.cuda}")

    # 1. Load V1 inference
    v1_infer = load_v1_inference(args.pkl, args.num_views, args.chunk_size)

    # 2. Load action denorm stats
    a_mean, a_std = load_action_norm_stats(args.norm_stats)
    logger.info(f"Action norm: dim={len(a_mean)} (taking first {args.action_dim})")

    # 3. Build V1Policy
    policy = V1Policy(
        v1_infer,
        action_norm_mean=a_mean,
        action_norm_std=a_std,
        action_dim=args.action_dim,
        image_keys=tuple(args.image_keys),
        metadata={
            "backend": "v1_triton",
            "version": 1,
            "ckpt_pkl": str(Path(args.pkl).resolve()),
            "norm_stats": str(Path(args.norm_stats).resolve()),
            "num_views": args.num_views,
            "chunk_size": args.chunk_size,
            "action_dim": args.action_dim,
            "phase": "1_no_state_encoding",  # ← flag for client-side awareness
        },
    )

    # 4. Warm-up
    warmup(policy, n=args.warmup_iters)

    # 5. Start WebSocket server
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "?"
    logger.info(f"Serving V1 Triton policy on {args.host}:{args.port} (hostname: {hostname}, ip: {local_ip})")
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
