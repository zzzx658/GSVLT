from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class MotionObservation:
    frame_id: int
    tlwh: np.ndarray
    score: float = 1.0

    def __post_init__(self) -> None:
        box = np.asarray(self.tlwh, dtype=np.float32)
        if box.shape != (4,):
            raise ValueError("tlwh must have shape (4,)")
        object.__setattr__(self, "tlwh", box.copy())


@dataclass(frozen=True)
class GroupMemberMemory:
    track_id: int
    history: Sequence[MotionObservation]
    is_target: bool = False


@dataclass
class LongTermTrackContext:
    track: Any
    lost_frame: int
    image_size: tuple[int, int]
    members: List[GroupMemberMemory]
    initial_phi: float
    pre_velocity: np.ndarray
    confidence_history: List[float] = field(default_factory=list)
    low_confidence_count: int = 0

    @property
    def track_id(self) -> int:
        return int(self.track.track_id)

    @property
    def external_member_ids(self) -> List[int]:
        return [member.track_id for member in self.members if not member.is_target]


@dataclass
class GVAOutput:
    logits: torch.Tensor
    pgms: torch.Tensor
    attention: torch.Tensor
    core_mask: torch.Tensor


@dataclass(frozen=True)
class Recovery:
    track: Any
    detection_index: int
    score: float
    inferred_velocity: np.ndarray


@dataclass(frozen=True)
class ManagementResult:
    recoveries: Sequence[Recovery]
    terminated_track_ids: Sequence[int]
    scores: Optional[np.ndarray] = None


TensorBatch = Dict[str, torch.Tensor]
