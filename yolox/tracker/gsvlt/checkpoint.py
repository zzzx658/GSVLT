from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import torch

from .config import GSVLTConfig
from .gmvi import GMVI
from .gva import GVA

CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    config: GSVLTConfig,
    gva: GVA,
    gmvi: GMVI,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_VERSION,
            "config": config.to_dict(),
            "gva": gva.state_dict(),
            "gmvi": gmvi.state_dict(),
            "metadata": dict(metadata or {}),
        },
        destination,
    )


def load_checkpoint(
    path: str | Path,
    device: torch.device,
    expected_config: GSVLTConfig | None = None,
) -> tuple[GSVLTConfig, GVA, GMVI, Dict[str, Any]]:
    try:
        checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        checkpoint = torch.load(Path(path), map_location=device)
    if checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported GSVLT checkpoint format")
    checkpoint_config = GSVLTConfig.from_mapping(checkpoint["config"])
    if expected_config is not None:
        architecture_keys = (
            "state_dim",
            "hidden_dim",
            "positional_dim",
            "attention_heads",
            "attention_layers",
        )
        mismatched = [
            key
            for key in architecture_keys
            if getattr(checkpoint_config, key) != getattr(expected_config, key)
        ]
        if mismatched:
            raise ValueError(
                "Checkpoint architecture does not match configuration: {}".format(
                    ", ".join(mismatched)
                )
            )
        config = expected_config
    else:
        config = checkpoint_config
    gva = GVA(config).to(device)
    gmvi = GMVI(config).to(device)
    gva.load_state_dict(checkpoint["gva"])
    gmvi.load_state_dict(checkpoint["gmvi"])
    return config, gva, gmvi, dict(checkpoint.get("metadata", {}))
