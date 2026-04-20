"""Video-backbone stage classifier for Task A flat→fold boundary detection."""

from .stage_classifier import (
    VideoStageClassifier,
    best_boundary_dp,
    compute_stage_loss,
    load_backbone,
)

__all__ = [
    "VideoStageClassifier",
    "best_boundary_dp",
    "compute_stage_loss",
    "load_backbone",
]
