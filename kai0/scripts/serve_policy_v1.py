"""V1 Triton inference WebSocket serve for deepdive_kai0 (B4 Phase 2).

Wraps optimize/v1_triton/Pi05InferenceTuned as BasePolicy that
WebsocketPolicyServer can host. Protocol identical to serve_policy.py
(:8000 JAX backend), just different inference path.

Scope:
  ✓ WebSocket server on :8002, msgpack protocol identical to JAX serve
  ✓ V1 Triton inference (Pi05InferenceTuned, P50=32 ms on 5090)
  ✓ Image preprocess (resize 224×224, bfloat16, cuda)
  ✓ Action denormalize via norm_stats.json
  ✓ Action chunk return (50, action_dim)
  ✓ **Phase 2**: per-inference state encoding via kai0 sentencepiece
                 (256-bin discretize + prefix "Task: {p}, State: {s};\n")
                 + PaliGemma embedding lookup → encoder_x buffer write
                 (绕开 V1 prebaked language_embeds)
  ✓ **B1 server-side profile**: preproc_ms / infer_ms / postproc_ms /
                 state_encode_ms / total_ms in policy_timing dict

Usage (sim01, after `convert_kai0_to_v1.py` 出 .pkl):
    .venv_5090_trt/bin/python kai0/scripts/serve_policy_v1.py \\
        --pkl optimize/results/task_a_mix_b6000_p1200_v1.pkl \\
        --norm-stats kai0/assets/<asset_id>/<repo_id>/norm_stats.json \\
        --tokenizer openpi_cache/big_vision/paligemma_tokenizer.model \\
        --port 8002 \\
        --num-views 3 --chunk-size 50 --action-dim 14 --state-dim 14

Health check: curl http://<host>:8002/healthz → "OK"

See docs/deployment/realtime_vla_optimization_analysis.md §7.2 + §6 for context.
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


class SentencepieceStateEncoder:
    """kai0-compatible per-inference prompt+state encoding (B4 Phase 2).

    Matches kai0/src/openpi/models/tokenizer.py:64-117 (training):
      prefix = f"Task: {cleaned_prompt}, State: {state_str};\\n"
      where state_str = " ".join(map(str, np.digitize(state, 257-edge bins) - 1))
    Then sentencepiece encode (add_bos=True) → PaliGemma embedding lookup
    → scale by sqrt(2048).

    Bypasses V1's HF AutoTokenizer path (kai0 uses sentencepiece .model).
    """

    PG_SCALE = 2048 ** 0.5  # PaliGemma embed scale (sqrt of d_model)
    _BIN_EDGES = np.linspace(-1, 1, 257)[:-1]  # 256 bins in [-1, 1]

    def __init__(
        self,
        v1_infer,  # Pi05InferenceTuned
        tokenizer_model_path: str,
        embedding_weight: torch.Tensor,
        state_norm_mean: np.ndarray,
        state_norm_std: np.ndarray,
    ):
        import sentencepiece

        self.v1 = v1_infer
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=tokenizer_model_path)

        # PaliGemma embedding table (from V1 pkl 'embedding_weight').
        # V1 doesn't keep it in self.weights (only baked language_embeds), so we
        # accept it separately and load to CUDA bf16.
        # Shape (vocab=257152, d=2048).
        if embedding_weight is None:
            raise RuntimeError(
                "embedding_weight is None. convert_kai0_to_v1.py must have stashed it "
                "in the pkl as 'embedding_weight' (does as of v0.11). Pass it explicitly."
            )
        if embedding_weight.device.type != "cuda":
            embedding_weight = embedding_weight.cuda()
        if embedding_weight.dtype != torch.bfloat16:
            embedding_weight = embedding_weight.to(torch.bfloat16)
        self._embed_w = embedding_weight  # (vocab, 2048) bf16 cuda

        # State norm (used to bring raw joint state into [-1, 1] before discretization)
        self._s_mean = torch.from_numpy(state_norm_mean.astype(np.float32)).cuda()
        self._s_std = torch.from_numpy(state_norm_std.astype(np.float32)).cuda()
        # Guard zero std
        self._s_std = torch.where(self._s_std < 1e-6, torch.ones_like(self._s_std), self._s_std)
        self._state_dim = len(state_norm_mean)
        self.max_prompt_len = v1_infer.max_prompt_len

    def encode(self, task_prompt: str, state_raw: np.ndarray) -> tuple[torch.Tensor, int]:
        """Encode (prompt + state) → embeds for V1 encoder.

        Args:
            task_prompt: task instruction string
            state_raw: (state_dim,) float, raw joint state (NOT normalized)

        Returns:
            (embeds, prompt_len) — embeds shape (prompt_len, 2048) bf16 cuda
        """
        # 1. Normalize state → [-1, 1] approx using norm_stats
        s = torch.from_numpy(np.asarray(state_raw, dtype=np.float32)).cuda()
        if s.numel() != self._state_dim:
            # truncate/pad to declared state_dim
            if s.numel() < self._state_dim:
                pad = torch.zeros(self._state_dim - s.numel(), device="cuda")
                s = torch.cat([s, pad])
            else:
                s = s[: self._state_dim]
        s_norm = (s - self._s_mean) / self._s_std  # ~ [-1, 1]
        s_norm_np = s_norm.detach().cpu().numpy()

        # 2. Discretize to 256 bins (kai0 convention)
        discretized = np.digitize(s_norm_np, bins=self._BIN_EDGES) - 1
        discretized = np.clip(discretized, 0, 255)
        state_str = " ".join(map(str, discretized.astype(int).tolist()))

        # 3. Build prefix (kai0 training format)
        cleaned = task_prompt.lower().strip().replace("_", " ")
        prefix = f"Task: {cleaned}, State: {state_str};\n"

        # 4. Sentencepiece tokenize (add_bos=True matches kai0 training line 75)
        token_ids = self.tokenizer.encode(prefix, add_bos=True)
        # Truncate to max_prompt_len (CUDA Graph buffer ceiling)
        if len(token_ids) > self.max_prompt_len:
            logger.warning(
                f"Prompt+state tokenized to {len(token_ids)} tokens > "
                f"max_prompt_len {self.max_prompt_len}; truncating."
            )
            token_ids = token_ids[: self.max_prompt_len]
        plen = len(token_ids)
        token_ids_t = torch.tensor(token_ids, dtype=torch.long, device="cuda")

        # 5. Lookup embedding + scale (matches kai0 PaliGemma forward + V1 line 53)
        embeds = self._embed_w[token_ids_t]  # (plen, 2048) bf16
        embeds = embeds * self.PG_SCALE

        return embeds, plen

    def write_to_v1_buffer(self, embeds: torch.Tensor, plen: int) -> None:
        """Write embeds into V1 encoder_x buffer + set valid_encoder_len + RoPE.

        Replicates V1.forward()'s prompt-handling lines (816-822 in
        discrete_state_input=False branch) but with our own (prompt+state)
        embeds instead of prebaked language_embeds. Must be called RIGHT
        BEFORE the CUDA Graph replay (v1.infer_graph.replay()), but the
        replay path also re-copies prebaked language_embeds into encoder_x,
        so we use forward_with_state() below to bypass replay's overwrite.
        """
        start = self.v1.num_views * 256
        self.v1.buffers["encoder_x"][start : start + plen].copy_(embeds)
        self.v1.buffers["valid_encoder_len"].fill_(start + plen)
        self.v1.buffers["decoder_rope_weights"].copy_(self.v1.get_decoder_rope_weights(plen))


def v1_forward_with_state(
    v1,  # Pi05InferenceTuned
    image: torch.Tensor,
    noise: torch.Tensor,
    embeds: torch.Tensor,
    plen: int,
) -> torch.Tensor:
    """V1 forward with externally-supplied (prompt+state) embeds.

    Replicates Pi05Inference.forward() body but skips the prebaked-
    language_embeds copy (which would overwrite our state-conditioned
    embeds). Must be called inside torch.inference_mode().
    """
    start = v1.num_views * 256
    v1.buffers["encoder_x"][start : start + plen].copy_(embeds)
    v1.buffers["valid_encoder_len"].fill_(start + plen)
    v1.buffers["decoder_rope_weights"].copy_(v1.get_decoder_rope_weights(plen))
    v1.buffers["observation_images_normalized"].copy_(image)
    v1.buffers["diffusion_noise"].copy_(noise)
    v1.infer_graph.replay()
    return v1.buffers["diffusion_noise"]


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
        state_encoder: "SentencepieceStateEncoder | None" = None,
        default_prompt: str = "Flatten and fold the cloth",
        image_keys: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        metadata: dict[str, Any] | None = None,
    ):
        self._v1 = v1_infer
        self._action_dim = action_dim
        # Save denorm stats as torch fp32 cuda for fast apply
        self._a_mean = torch.from_numpy(action_norm_mean[:action_dim].astype(np.float32)).cuda()
        self._a_std = torch.from_numpy(action_norm_std[:action_dim].astype(np.float32)).cuda()
        # Guard against zero std
        self._a_std = torch.where(self._a_std < 1e-6, torch.ones_like(self._a_std), self._a_std)
        self._image_keys = image_keys
        self._metadata = metadata or {"backend": "v1_triton", "version": 2}
        self._chunk_size = v1_infer.chunk_size
        self._num_views = v1_infer.num_views
        # Optional Phase 2 state encoder. If None, falls back to prebaked
        # language_embeds (Phase 1 behavior; NOT state-conditioned).
        self._state_encoder = state_encoder
        self._default_prompt = default_prompt
        # Stable noise per-call would defeat diversity; sample fresh each infer.
        self._noise_gen = torch.Generator(device="cuda")
        self._noise_gen.manual_seed(0)

    def infer(self, obs: dict) -> dict:
        """Run one V1 inference cycle.

        Args:
            obs: dict from ROS2 client. Expected keys (subset of openpi protocol):
              - images: dict[str, HxWx3 uint8 numpy] — one entry per camera
              - state: (state_dim,) float numpy (joint state, kai0 = 14)
              - prompt: str (task instruction) — Phase 2 used

        Returns:
            dict with:
              - actions: (chunk_size, action_dim) float32 numpy (denormalized)
              - policy_timing: {preproc_ms, state_encode_ms, infer_ms,
                                postproc_ms, total_ms} — B1 server-side profile
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
        torch.cuda.synchronize()
        preproc_ms = (time.monotonic() - t_start) * 1000

        # 2. Sample fresh noise (chunk_size, 32) bf16 cuda
        noise = torch.randn(
            self._chunk_size, 32,
            dtype=torch.bfloat16, device="cuda",
            generator=self._noise_gen,
        )

        # 3. State encoding (Phase 2 if state_encoder available; else Phase 1 fallback)
        t_state = time.monotonic()
        state_embeds = None
        plen = 0
        prompt_used = obs.get("prompt", self._default_prompt) or self._default_prompt
        if self._state_encoder is not None:
            state_raw = obs.get("state")
            if state_raw is None:
                raise ValueError(
                    "obs['state'] required when state_encoder is configured (Phase 2). "
                    "Disable state encoder to use prebaked-prompt Phase 1 fallback."
                )
            state_embeds, plen = self._state_encoder.encode(prompt_used, state_raw)
        torch.cuda.synchronize()
        state_encode_ms = (time.monotonic() - t_state) * 1000

        # 4. V1 inference
        t_infer = time.monotonic()
        with torch.inference_mode():
            if self._state_encoder is not None:
                # Phase 2: bypass V1's prebaked language_embeds, write state-conditioned embeds
                action_chunk = v1_forward_with_state(
                    self._v1, image_input, noise, state_embeds, plen,
                )
            else:
                # Phase 1 fallback: V1 forward uses prebaked language_embeds
                action_chunk = self._v1.forward(image_input, noise)
        torch.cuda.synchronize()
        infer_ms = (time.monotonic() - t_infer) * 1000

        # 5. Take first action_dim, denormalize, to numpy
        t_post = time.monotonic()
        a = action_chunk[:, : self._action_dim].to(torch.float32)  # (chunk_size, action_dim)
        a = a * self._a_std[None, :] + self._a_mean[None, :]
        actions_np = a.detach().cpu().numpy()  # (chunk_size, action_dim) float32
        postproc_ms = (time.monotonic() - t_post) * 1000

        total_ms = (time.monotonic() - t_start) * 1000

        return {
            "actions": actions_np,
            # B1 server-side profile (each step's wall-clock, sum ≈ total_ms)
            "policy_timing": {
                "preproc_ms": float(preproc_ms),
                "state_encode_ms": float(state_encode_ms),
                "infer_ms": float(infer_ms),  # V1 forward only
                "postproc_ms": float(postproc_ms),
                "total_ms": float(total_ms),
            },
            "server_backend": "v1_triton",
            "phase": 2 if self._state_encoder is not None else 1,
            "action_kind": "joint",
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def load_v1_inference(pkl_path: str, num_views: int, chunk_size: int):
    """Load V1 pkl + build Pi05InferenceTuned.

    Returns (infer, embedding_weight). embedding_weight is extracted from
    pkl 'embedding_weight' field (full PaliGemma table 257152×2048, baked by
    convert_kai0_to_v1.py for Phase 2 re-tokenization); None if absent.
    """
    logger.info(f"Loading V1 ckpt from {pkl_path} ...")
    t0 = time.perf_counter()
    with open(pkl_path, "rb") as f:
        ckpt = pickle.load(f)
    logger.info(
        f"  loaded {sum(v.numel()*v.element_size() for v in ckpt.values())/1e9:.2f} GB tensors "
        f"in {time.perf_counter()-t0:.1f}s"
    )
    embedding_weight = ckpt.get("embedding_weight")  # (vocab, 2048) bf16 cpu; for Phase 2
    if embedding_weight is None:
        logger.warning("pkl missing 'embedding_weight'; Phase 2 state encoding unavailable")

    logger.info(
        f"Building Pi05InferenceTuned(num_views={num_views}, chunk_size={chunk_size}) "
        f"+ CUDA Graph capture ..."
    )
    t0 = time.perf_counter()
    infer = Pi05InferenceTuned(
        ckpt, num_views=num_views, chunk_size=chunk_size,
        discrete_state_input=False,  # Phase 2 overrides via v1_forward_with_state
    )
    logger.info(f"  build + capture in {time.perf_counter()-t0:.1f}s")
    return infer, embedding_weight


def load_norm_stats(norm_stats_path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load state + action mean/std from openpi-format norm_stats.json.

    Format (per kai0/assets/<asset>/<repo>/norm_stats.json):
      {"norm_stats": {"state": {"mean": [...], "std": [...]},
                      "actions": {"mean": [...], "std": [...]}}}

    Returns: {"state": (mean, std), "actions": (mean, std)}
    """
    with open(norm_stats_path) as f:
        data = json.load(f)
    norm = data["norm_stats"]
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in ("state", "actions"):
        if key not in norm:
            if key == "actions":
                raise ValueError(f"norm_stats.json missing 'actions' (have: {list(norm.keys())})")
            continue  # state is optional (Phase 1 doesn't need it)
        entry = norm[key]
        if "mean" not in entry or "std" not in entry:
            raise ValueError(f"norm_stats['{key}'] needs mean+std (have: {list(entry.keys())})")
        out[key] = (
            np.asarray(entry["mean"], dtype=np.float32),
            np.asarray(entry["std"], dtype=np.float32),
        )
    return out


def warmup(policy: V1Policy, n: int = 3, state_dim: int = 14) -> None:
    """Warm-up to ensure CUDA Graph + caches are hot before serving."""
    logger.info(f"Warming up V1 inference ({n} dummy iters) ...")
    H, W = 480, 640  # arbitrary; resize to 224 happens inside
    dummy_obs = {
        "images": {
            f"view_{i}": np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
            for i in range(policy._num_views)
        },
        "state": np.zeros(state_dim, dtype=np.float32),
        "prompt": "warmup test",
    }
    for i in range(n):
        out = policy.infer(dummy_obs)
        pt = out["policy_timing"]
        logger.info(
            f"  warmup {i+1}/{n}: total={pt['total_ms']:.1f}ms "
            f"(preproc={pt['preproc_ms']:.1f} + state={pt['state_encode_ms']:.1f} "
            f"+ infer={pt['infer_ms']:.1f} + post={pt['postproc_ms']:.1f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="V1 Triton WebSocket serve for deepdive_kai0")
    parser.add_argument("--pkl", required=True, help="V1 pickle from convert_kai0_to_v1.py")
    parser.add_argument("--norm-stats", required=True,
                        help="kai0 norm_stats.json (for state+action normalize)")
    parser.add_argument("--tokenizer", default=None,
                        help="sentencepiece .model path (e.g. openpi_cache/big_vision/"
                             "paligemma_tokenizer.model). REQUIRED for Phase 2 "
                             "state encoding; omit for Phase 1 prebaked-prompt mode.")
    parser.add_argument("--default-prompt", default="Flatten and fold the cloth",
                        help="Used when obs lacks 'prompt' field (matches V1 pkl bake prompt)")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=14,
                        help="output action dim to take from V1's (chunk, 32) output")
    parser.add_argument("--state-dim", type=int, default=14,
                        help="state dim used for sentencepiece discretization (Phase 2)")
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

    # 1. Load V1 inference + extract embedding_weight (for Phase 2)
    v1_infer, embedding_weight = load_v1_inference(args.pkl, args.num_views, args.chunk_size)

    # 2. Load state + action norm stats
    norm = load_norm_stats(args.norm_stats)
    a_mean, a_std = norm["actions"]
    logger.info(f"Action norm: dim={len(a_mean)} (taking first {args.action_dim})")

    # 3. Build sentencepiece state encoder (Phase 2) if tokenizer provided
    state_encoder = None
    if args.tokenizer:
        if "state" not in norm:
            raise ValueError(
                "Phase 2 needs norm_stats['state'] for state normalization, "
                "but norm_stats.json has no 'state' entry."
            )
        if embedding_weight is None:
            raise ValueError(
                "Phase 2 needs 'embedding_weight' in V1 pkl; re-run convert_kai0_to_v1.py "
                "(or use expand_v1_pkl_for_phase2.py if pkl already has embedding_weight "
                "but small language_embeds)."
            )
        s_mean, s_std = norm["state"]
        if len(s_mean) < args.state_dim:
            raise ValueError(
                f"norm_stats['state'] has {len(s_mean)} dims but --state-dim={args.state_dim}"
            )
        logger.info(f"Loading sentencepiece tokenizer: {args.tokenizer}")
        state_encoder = SentencepieceStateEncoder(
            v1_infer,
            tokenizer_model_path=args.tokenizer,
            embedding_weight=embedding_weight,
            state_norm_mean=s_mean[: args.state_dim],
            state_norm_std=s_std[: args.state_dim],
        )
        logger.info(
            f"  Phase 2 state encoding enabled: state_dim={args.state_dim}, "
            f"max_prompt_len={state_encoder.max_prompt_len}, "
            f"default_prompt={args.default_prompt!r}"
        )
    else:
        logger.warning(
            "No --tokenizer; running Phase 1 (prebaked language_embeds, "
            "NOT state-conditioned). Inference will not react to changing state."
        )

    # 4. Build V1Policy
    phase = 2 if state_encoder is not None else 1
    policy = V1Policy(
        v1_infer,
        action_norm_mean=a_mean,
        action_norm_std=a_std,
        action_dim=args.action_dim,
        state_encoder=state_encoder,
        default_prompt=args.default_prompt,
        image_keys=tuple(args.image_keys),
        metadata={
            "backend": "v1_triton",
            "version": 2,
            "ckpt_pkl": str(Path(args.pkl).resolve()),
            "norm_stats": str(Path(args.norm_stats).resolve()),
            "tokenizer": str(Path(args.tokenizer).resolve()) if args.tokenizer else None,
            "num_views": args.num_views,
            "chunk_size": args.chunk_size,
            "action_dim": args.action_dim,
            "state_dim": args.state_dim,
            "default_prompt": args.default_prompt,
            "phase": phase,
        },
    )

    # 5. Warm-up
    warmup(policy, n=args.warmup_iters, state_dim=args.state_dim)

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
