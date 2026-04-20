"""Video stage classifier for Task A flat→fold boundary detection.

Architecture (per docs/training/stage_classifier_plan.md):
  Frozen V-JEPA 2 backbone → 8 tube features (T=16, tubelet=2)
  → 2× cross-attention with 16 learnable frame queries
  → MLP head → per-frame 2-class logits

Inference: sliding-window + overlap average + DP boundary detection
  for hard [0..0 1..1] monotonic output.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ─── Backbone loading (V-JEPA 2 / VideoMAE v2) ───────────────────────────────

BACKBONE_CHOICES = {
    # === V-JEPA 2.1 (via torch.hub, released 2026-03-16, all 384 resolution) ===
    # Primary: largest V-JEPA 2.1 variant
    "vjepa2_1_gigantic":   {"loader": "torch_hub", "repo": "facebookresearch/vjepa2",
                             "entry": "vjepa2_1_vit_gigantic_384", "image_size": 384,
                             "num_frames": 16, "tubelet_size": 2,
                             "patch_size": 16, "hidden_size": 1280, "params_B": 2.0},
    "vjepa2_1_giant":      {"loader": "torch_hub", "repo": "facebookresearch/vjepa2",
                             "entry": "vjepa2_1_vit_giant_384", "image_size": 384,
                             "num_frames": 16, "tubelet_size": 2,
                             "patch_size": 16, "hidden_size": 1408, "params_B": 1.0},
    "vjepa2_1_large":      {"loader": "torch_hub", "repo": "facebookresearch/vjepa2",
                             "entry": "vjepa2_1_vit_large_384", "image_size": 384,
                             "num_frames": 16, "tubelet_size": 2,
                             "patch_size": 16, "hidden_size": 1024, "params_B": 0.3},

    # === V-JEPA 2 (via torch.hub, 2025 release) ===
    "vjepa2_giant_384":    {"loader": "torch_hub", "repo": "facebookresearch/vjepa2",
                             "entry": "vjepa2_vit_giant_384", "image_size": 384,
                             "num_frames": 16, "tubelet_size": 2,
                             "patch_size": 16, "hidden_size": 1408, "params_B": 1.0},
    "vjepa2_giant":        {"loader": "torch_hub", "repo": "facebookresearch/vjepa2",
                             "entry": "vjepa2_vit_giant", "image_size": 256,
                             "num_frames": 64, "tubelet_size": 2,
                             "patch_size": 16, "hidden_size": 1408, "params_B": 1.0},
    "vjepa2_huge":         {"loader": "torch_hub", "repo": "facebookresearch/vjepa2",
                             "entry": "vjepa2_vit_huge", "image_size": 256,
                             "num_frames": 64, "tubelet_size": 2,
                             "patch_size": 16, "hidden_size": 1280, "params_B": 0.6},

    # === VideoMAE v2 (via HF) ===
    # OpenGVLab VideoMAEv2-giant: patch_size=14 (not 16), so 224/14=16 → 16×16=256 spatial
    "videomae_v2_giant":   {"loader": "hf", "repo": "OpenGVLab/VideoMAEv2-giant",
                             "image_size": 224, "num_frames": 16, "tubelet_size": 2,
                             "patch_size": 14, "hidden_size": 1408, "params_B": 1.0},
    "videomae_v2_huge":    {"loader": "hf", "repo": "OpenGVLab/VideoMAEv2-Huge",
                             "image_size": 224, "num_frames": 16, "tubelet_size": 2,
                             "patch_size": 16, "hidden_size": 1280, "params_B": 0.6},
    "videomae_v2_base":    {"loader": "hf", "repo": "OpenGVLab/VideoMAEv2-Base",
                             "image_size": 224, "num_frames": 16, "tubelet_size": 2,
                             "patch_size": 16, "hidden_size": 768, "params_B": 0.09},
}


@dataclass
class BackboneInfo:
    """Metadata about a loaded video backbone."""

    hf_repo: str
    hidden_size: int
    num_frames: int       # clip length T
    tubelet_size: int     # temporal patch size
    image_size: int       # spatial resolution
    num_spatial: int      # S*S where S = image_size // patch_size
    num_tubes: int        # T / tubelet_size


def load_backbone(
    name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Module, BackboneInfo]:
    """Load V-JEPA 2 / V-JEPA 2.1 (torch.hub) or VideoMAE v2 (HF) backbone frozen.

    Args:
        name: short name from BACKBONE_CHOICES (e.g. 'vjepa2_1_gigantic')
        device: 'cuda' or 'cpu'
        dtype: torch dtype for backbone weights

    Returns (model, BackboneInfo). Model already eval() and requires_grad=False.
    """
    if name not in BACKBONE_CHOICES:
        raise ValueError(f"Unknown backbone '{name}'. Choices: {list(BACKBONE_CHOICES)}")
    spec = BACKBONE_CHOICES[name]
    loader = spec["loader"]

    logger.info(f"Loading backbone '{name}' via {loader}: {spec['repo']}")

    if loader == "torch_hub":
        # V-JEPA 2 / 2.1 via torch.hub
        # Try local cache first (for offline gf1 env), fall back to GitHub
        import os
        torch_home = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
        # facebookresearch/vjepa2 → facebookresearch_vjepa2_main (torch.hub naming)
        local_repo = os.path.join(
            torch_home, "hub",
            spec["repo"].replace("/", "_") + "_main"
        )
        if os.path.isdir(local_repo):
            logger.info(f"Using local cached repo: {local_repo}")
            model = torch.hub.load(
                local_repo, spec["entry"],
                source="local", trust_repo=True, skip_validation=True,
            )
        else:
            model = torch.hub.load(
                spec["repo"], spec["entry"],
                trust_repo=True, skip_validation=True,
            )
        # V-JEPA 2 hub entries return (encoder, predictor) tuple — we only need encoder
        if isinstance(model, tuple):
            logger.info(f"Hub entry returned tuple of {len(model)}; taking [0] (encoder)")
            model = model[0]
        hidden_size = spec["hidden_size"]
        num_frames = spec["num_frames"]
        tubelet_size = spec["tubelet_size"]
        image_size = spec["image_size"]
        patch_size = spec["patch_size"]
        hf_repo = f"torch_hub:{spec['repo']}:{spec['entry']}"
    elif loader == "hf":
        # VideoMAE v2 via HuggingFace (OpenGVLab port)
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            spec["repo"], trust_remote_code=True, torch_dtype=dtype
        )
        # OpenGVLab VideoMAEv2.forward pools to (B, D); we need per-token (B, N, D).
        # Monkey-patch: override outer forward to bypass VisionTransformer.forward's pool.
        inner = model.model  # VisionTransformer

        def _per_token_forward(self_inner, x):
            import torch.utils.checkpoint as cp  # noqa: F401 (pattern used in blocks below)
            B = x.size(0)
            x = self_inner.patch_embed(x)
            if self_inner.pos_embed is not None:
                x = x + self_inner.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
            x = self_inner.pos_drop(x)
            for blk in self_inner.blocks:
                x = blk(x)
            # Keep (B, N, D) — skip x.mean(1) / x[:, 0]
            if self_inner.norm is not None:
                x = self_inner.norm(x)
            return x

        import types
        inner.forward_per_token = types.MethodType(_per_token_forward, inner)

        # Replace outer VideoMAEv2.forward to call forward_per_token
        def _outer_forward(self_outer, pixel_values):
            return self_outer.model.forward_per_token(pixel_values)

        model.forward = types.MethodType(_outer_forward, model)

        hidden_size = spec["hidden_size"]
        num_frames = spec["num_frames"]
        tubelet_size = spec["tubelet_size"]
        image_size = spec["image_size"]
        patch_size = spec["patch_size"]
        hf_repo = spec["repo"]
    else:
        raise ValueError(f"Unknown loader '{loader}'")

    num_spatial = (image_size // patch_size) ** 2
    num_tubes = num_frames // tubelet_size

    info = BackboneInfo(
        hf_repo=hf_repo,
        hidden_size=hidden_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        image_size=image_size,
        num_spatial=num_spatial,
        num_tubes=num_tubes,
    )

    # Convert dtype
    if dtype != torch.float32:
        model = model.to(dtype)

    # Freeze + move
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Backbone loaded: {n_params/1e6:.0f}M params, hidden={hidden_size}, "
        f"T={num_frames}, tubelet={tubelet_size}, tubes={num_tubes}, "
        f"img={image_size}, spatial={num_spatial}"
    )
    return model, info


def extract_tube_features(
    backbone: nn.Module,
    info: BackboneInfo,
    clips: Tensor,
) -> Tensor:
    """Run frozen backbone and extract tube-level features.

    clips: (B, T, 3, H, W) preprocessed tensor (will be permuted if needed)
           Where T = info.num_frames, H = W = info.image_size
    returns: (B, num_tubes, hidden_size) tube features

    Handles both:
      - V-JEPA 2 (torch.hub): expects (B, C=3, T, H, W); returns (B, num_tokens, D) tensor
      - VideoMAE v2 (HF):     expects (B, T, 3, H, W); returns BaseModelOutput with .last_hidden_state
    """
    B = clips.shape[0]
    with torch.no_grad():
        # Detect output format by trying V-JEPA 2 format first (C, T, H, W)
        # V-JEPA 2 expects clips shape (B, C, T, H, W); user passes (B, T, C, H, W) → permute
        # Both V-JEPA 2 (torch.hub) and OpenGVLab VideoMAEv2 use Conv3d patch embed
        # with shape [out_ch, C=3, kT, kH, kW] → expect (B, C, T, H, W).
        # User passes (B, T, C, H, W) → always permute.
        x = clips.permute(0, 2, 1, 3, 4).contiguous()

        outputs = backbone(x)

        if hasattr(outputs, "last_hidden_state"):
            tokens = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            tokens = outputs[0]
        elif torch.is_tensor(outputs):
            tokens = outputs
        else:
            raise ValueError(f"Unknown backbone output type: {type(outputs)}")

        # tokens shape: (B, num_tokens, hidden_size)
        NT = tokens.shape[1]
        expected_no_cls = info.num_tubes * info.num_spatial
        if NT == expected_no_cls + 1:
            # Drop CLS token
            tokens = tokens[:, 1:]
        elif NT != expected_no_cls:
            logger.warning(
                f"Unexpected token count {NT}, expected {expected_no_cls} "
                f"(num_tubes={info.num_tubes} × num_spatial={info.num_spatial})."
            )
            # Adaptive pool: assume tokens are in (t, s, d) order and reshape accordingly
            chunk = NT // info.num_tubes
            tokens = tokens[:, : chunk * info.num_tubes]
            tokens = tokens.reshape(B, info.num_tubes, chunk, tokens.shape[-1])
            return tokens.mean(dim=2).float()

        # Reshape (B, num_tubes * num_spatial, D) → (B, num_tubes, num_spatial, D)
        tokens = tokens.reshape(B, info.num_tubes, info.num_spatial, tokens.shape[-1])
        # Spatial mean pool
        tube_feats = tokens.mean(dim=2)  # (B, num_tubes, hidden)
    return tube_feats.float()


# ─── Main model: Cross-Attention + MLP on cached tube features ─────────────

class VideoStageClassifier(nn.Module):
    """Per-frame binary stage classifier on top of frozen video backbone.

    Two entry points:
      forward(clips, backbone, info): end-to-end (runs backbone inside)
      forward_from_tubes(tube_feats):  cached features (skip backbone)
    """

    def __init__(
        self,
        backbone_hidden: int = 1024,
        num_tubes: int = 8,
        num_frames: int = 16,
        hidden_dim: int = 384,
        n_cross_attn_layers: int = 2,
        n_heads: int = 8,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_tubes = num_tubes
        self.num_frames = num_frames

        # Project tube features to model hidden_dim
        self.tube_proj = nn.Linear(backbone_hidden, hidden_dim)
        self.tube_norm_in = nn.LayerNorm(hidden_dim)

        # Learnable positional embeddings
        self.tube_pos_emb = nn.Parameter(torch.randn(num_tubes, hidden_dim) * 0.02)
        self.frame_queries = nn.Parameter(torch.randn(num_frames, hidden_dim) * 0.02)

        # Cross-attention stack
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_cross_attn_layers)
        ])
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(n_cross_attn_layers)
        ])
        self.norms_q = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_cross_attn_layers)])
        self.norms_ff = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_cross_attn_layers)])

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward_from_tubes(self, tube_feats: Tensor) -> Tensor:
        """
        tube_feats: (B, num_tubes, backbone_hidden)  — from cached precompute
        returns: (B, num_frames, num_classes) logits
        """
        B = tube_feats.shape[0]
        # Project tubes to hidden_dim
        tubes = self.tube_proj(tube_feats)                # (B, num_tubes, hidden)
        tubes = self.tube_norm_in(tubes)
        tubes = tubes + self.tube_pos_emb.unsqueeze(0)    # add position

        # Cross-attention: frame queries attend to tubes
        x = self.frame_queries.unsqueeze(0).expand(B, -1, -1)  # (B, num_frames, hidden)

        for attn, ff, n_q, n_ff in zip(
            self.cross_attn_layers, self.ff_layers, self.norms_q, self.norms_ff
        ):
            attn_out, _ = attn(n_q(x), tubes, tubes, need_weights=False)
            x = x + attn_out
            x = x + ff(n_ff(x))

        logits = self.head(x)                             # (B, num_frames, num_classes)
        return logits

    def forward(
        self,
        clips: Tensor,
        backbone: nn.Module,
        info: BackboneInfo,
    ) -> Tensor:
        """End-to-end forward: clips → backbone → tubes → logits.

        clips: (B, T, 3, H, W)
        returns: (B, T, 2)
        """
        tube_feats = extract_tube_features(backbone, info, clips)
        return self.forward_from_tubes(tube_feats)


# ─── Loss ─────────────────────────────────────────────────────────────────

def compute_stage_loss(
    logits: Tensor,
    labels: Tensor,
    class_weights: tuple[float, float] = (1.0, 3.0),
    smooth_weight: float = 0.1,
    mono_weight: float = 0.2,
) -> tuple[Tensor, dict]:
    """Combined loss for stage classifier.

    logits: (B, T, 2)
    labels: (B, T) ∈ {0, 1}
    returns: (total_loss, info_dict)
    """
    B, T, _ = logits.shape
    device = logits.device

    # (a) Weighted CE
    w = torch.tensor(class_weights, device=device, dtype=logits.dtype)
    loss_ce = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1), weight=w)

    # (b) Smoothness: adjacent logit diffs small
    delta = logits[:, 1:] - logits[:, :-1]           # (B, T-1, 2)
    loss_smooth = delta.pow(2).mean()

    # (c) Monotonicity: P(fold) should not decrease
    probs = F.softmax(logits, dim=-1)[:, :, 1]        # (B, T)
    delta_p1 = probs[:, 1:] - probs[:, :-1]           # (B, T-1)
    loss_mono = F.relu(-delta_p1).mean()

    total = loss_ce + smooth_weight * loss_smooth + mono_weight * loss_mono

    return total, {
        "loss_total": total.detach(),
        "loss_ce": loss_ce.detach(),
        "loss_smooth": loss_smooth.detach(),
        "loss_mono": loss_mono.detach(),
    }


# ─── Inference: DP boundary detection ────────────────────────────────────

@torch.no_grad()
def best_boundary_dp(logits: Tensor) -> tuple[Tensor, int, float]:
    """Find optimal single boundary for arbitrary-length sequence.

    logits: (N, 2) per-frame logits for a whole episode
    returns:
        labels: (N,) ∈ {0, 1}  guaranteed monotonic [0..0 1..1]
        t_star: int           boundary frame index (last 0 index)
        confidence: float     DP score margin (in log-space, higher = more confident)
    """
    assert logits.ndim == 2 and logits.shape[1] == 2
    N = logits.shape[0]
    device = logits.device

    log_p = F.log_softmax(logits, dim=-1)            # (N, 2)
    log_p0 = log_p[:, 0]                              # (N,)
    log_p1 = log_p[:, 1]                              # (N,)

    # Prefix sums
    cum_log_p0 = torch.cumsum(log_p0, dim=0)          # (N,) sum of log_p0[0..i]
    total_log_p1 = log_p1.sum()
    cum_log_p1 = torch.cumsum(log_p1, dim=0)          # (N,) sum of log_p1[0..i]

    # score(t*) = log P(frames 0..t* = 0) + log P(frames t*+1..N-1 = 1)
    #           = cum_log_p0[t*] + (total_log_p1 - cum_log_p1[t*])
    scores = cum_log_p0 + (total_log_p1 - cum_log_p1)  # (N,)

    # Handle "all class 1" edge case:
    # if all frames should be 1, t_star = -1 (no prefix of zeros)
    # Score for this = 0 (empty prefix sum) + total_log_p1
    # We include this by considering i=-1 virtual position with score = total_log_p1
    all_one_score = total_log_p1
    t_star = int(scores.argmax().item())
    best_score = float(scores[t_star].item())

    if all_one_score > best_score:
        t_star = -1  # no frames are 0
        best_score = float(all_one_score.item())

    labels = torch.zeros(N, dtype=torch.long, device=device)
    if t_star >= 0:
        labels[t_star + 1:] = 1
    else:
        labels[:] = 1

    # Confidence: margin between best score and mean score
    confidence = float((best_score - scores.mean()).item())

    return labels, t_star, confidence


# ─── Utility: count trainable params ────────────────────────────────────

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check (CPU, no backbone load)
    model = VideoStageClassifier(
        backbone_hidden=1024, num_tubes=8, num_frames=16, hidden_dim=384
    )
    print(f"Trainable params: {count_trainable_params(model):,}")

    # Dummy tube features
    tube_feats = torch.randn(2, 8, 1024)
    logits = model.forward_from_tubes(tube_feats)
    print(f"Logits shape: {logits.shape}")  # (2, 16, 2)

    # Loss
    labels = torch.zeros(2, 16, dtype=torch.long)
    labels[:, 8:] = 1
    loss, info = compute_stage_loss(logits, labels)
    print(f"Loss: {loss.item():.4f}, info: {k: v.item() for k, v in info.items()}")

    # DP
    logits_ep = torch.randn(300, 2)  # fake episode
    lbl, t, conf = best_boundary_dp(logits_ep)
    print(f"DP boundary: t*={t}, conf={conf:.3f}, labels_sum={lbl.sum().item()}")
