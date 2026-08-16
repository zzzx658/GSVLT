from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


@dataclass(frozen=True)
class GSVLTConfig:
    """Configuration for the paper modules and ByteTrack integration.

    Values explicitly reported by the paper are the defaults. Values not
    reported by the paper remain configurable and are listed in
    ``docs/IMPLEMENTATION_NOTES.md``.
    """

    history_size: int = 30
    state_dim: int = 8
    hidden_dim: int = 128
    positional_dim: int = 128
    attention_heads: int = 4
    attention_layers: int = 2
    dropout: float = 0.1
    max_group_size: int = 16
    candidate_count: int = 16

    group_eiou_threshold: float = 1.7
    attention_threshold: float = 0.15
    affinity_threshold: float = 0.5
    neighbor_mask_probability: float = 0.15
    trajectory_crop_probability: float = 0.2
    identity_swap_probability: float = 0.1

    short_term_lost: int = 25
    hard_max_lost: int = 600

    gdi_motion_weight: float = 0.2
    gdi_structure_weight: float = 0.5
    gdi_temporal_weight: float = 0.3
    temporal_horizon_frames: int = 90
    temporal_alpha: float = 1.0
    confidence_scale: float = 1.0
    confidence_window: int = 5
    termination_threshold: float = 0.68
    termination_patience: int = 3
    epsilon: float = 1e-6

    gva_learning_rate: float = 1e-4
    gmvi_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    epochs: int = 80

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "GSVLTConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError("Unknown GSVLT configuration keys: {}".format(", ".join(unknown)))
        config = cls(**dict(values))
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GSVLTConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, dict):
            raise ValueError("GSVLT configuration must be a YAML mapping")
        return cls.from_mapping(values)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.state_dim != 8:
            raise ValueError("The paper motion representation has exactly 8 dimensions")
        if self.hidden_dim != self.positional_dim:
            raise ValueError("Hidden and positional dimensions must match")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if self.history_size < 2:
            raise ValueError("history_size must be at least 2")
        if self.max_group_size < 1:
            raise ValueError("max_group_size must include the target anchor")
        weights = self.gdi_motion_weight + self.gdi_structure_weight + self.gdi_temporal_weight
        if abs(weights - 1.0) > 1e-6:
            raise ValueError("GDI weights must sum to one")
        for name in (
            "attention_threshold",
            "affinity_threshold",
            "neighbor_mask_probability",
            "trajectory_crop_probability",
            "identity_swap_probability",
            "termination_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError("{} must be in [0, 1]".format(name))


DATASET_EPOCHS = {"mot17": 80, "mot20": 120, "dancetrack": 150}
