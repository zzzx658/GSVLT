from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import GSVLTConfig
from .geometry import motion_order_parameter
from .types import LongTermTrackContext


@dataclass(frozen=True)
class GDIComponents:
    motion: float
    structure: float
    temporal: float
    total: float
    confidence: float
    smoothed_confidence: float


class GroupStabilityManager:
    """Group Stability-Aware Trajectory Management (GAM)."""

    def __init__(self, config: GSVLTConfig) -> None:
        config.validate()
        self.config = config

    def update(
        self,
        context: LongTermTrackContext,
        survivor_velocities: Iterable[np.ndarray],
        survivor_count: int,
        elapsed_frames: int,
    ) -> tuple[GDIComponents, bool]:
        external_count = len(context.external_member_ids)
        if external_count:
            current_phi = motion_order_parameter(survivor_velocities, self.config.epsilon)
            motion = max(
                0.0,
                1.0 - current_phi / (context.initial_phi + self.config.epsilon),
            )
            structure = 1.0 - min(survivor_count / external_count, 1.0)
        else:
            # The paper's no-neighbor branch uses target history without group cues.
            motion = 0.0
            structure = 0.0

        temporal = min(
            self.config.temporal_alpha
            * elapsed_frames
            / max(self.config.temporal_horizon_frames, 1),
            1.0,
        )
        total = np.clip(
            self.config.gdi_motion_weight * motion
            + self.config.gdi_structure_weight * structure
            + self.config.gdi_temporal_weight * temporal,
            0.0,
            1.0,
        )
        confidence = self.config.confidence_scale * (1.0 - total)
        context.confidence_history.append(float(confidence))
        context.confidence_history = context.confidence_history[-self.config.confidence_window :]
        smoothed = float(np.mean(context.confidence_history))

        if smoothed < self.config.termination_threshold:
            context.low_confidence_count += 1
        else:
            context.low_confidence_count = 0
        terminate = context.low_confidence_count >= self.config.termination_patience
        components = GDIComponents(
            motion=float(motion),
            structure=float(structure),
            temporal=float(temporal),
            total=float(total),
            confidence=float(confidence),
            smoothed_confidence=smoothed,
        )
        return components, terminate
