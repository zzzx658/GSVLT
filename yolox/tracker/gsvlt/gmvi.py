from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..basetrack import TrackState
from .config import GSVLTConfig
from .geometry import velocity_to_kalman
from .gva import DetectionEncoder


class GMVI(nn.Module):
    """Group Motion-Guided Velocity Inference residual head."""

    def __init__(
        self, config: GSVLTConfig, detection_encoder: DetectionEncoder | None = None
    ) -> None:
        super().__init__()
        dimension = config.hidden_dim
        self.detection_encoder = detection_encoder or DetectionEncoder(dimension)
        self.velocity_head = nn.Sequential(
            nn.Linear(dimension * 2, dimension),
            nn.GELU(),
            nn.LayerNorm(dimension),
            nn.Dropout(config.dropout),
            nn.Linear(dimension, 4),
        )

    def forward(self, pgms: torch.Tensor, detections: torch.Tensor) -> torch.Tensor:
        detection_features = self.detection_encoder(detections)
        return self.velocity_head(torch.cat([pgms, detection_features], dim=-1))

    @staticmethod
    def infer_velocity(pre_velocity: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return pre_velocity + residual

    @staticmethod
    def rectify_track(
        track: object,
        detection: object,
        normalized_velocity: np.ndarray,
        image_size: tuple[int, int],
        frame_id: int,
    ) -> None:
        measurement = track.tlwh_to_xyah(detection.tlwh)
        mean, covariance = track.kalman_filter.initiate(measurement)
        mean[4:] = velocity_to_kalman(normalized_velocity, detection.tlwh, image_size)
        track.mean = mean
        track.covariance = covariance
        track.tracklet_len = 0
        track.state = TrackState.Tracked
        track.is_activated = True
        track.frame_id = frame_id
        track.score = detection.score
        if hasattr(track, "record_observation"):
            track.record_observation(frame_id, detection.tlwh, detection.score)
