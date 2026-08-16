"""Group Structure Variation for Long-Term Tracking (GSVLT)."""

from .config import GSVLTConfig
from .gam import GroupStabilityManager
from .gmvi import GMVI
from .gva import GVA
from .manager import GSVLTManager

__all__ = [
    "GAM",
    "GMVI",
    "GSVLTConfig",
    "GSVLTManager",
    "GVA",
    "GroupStabilityManager",
]

# Paper-facing alias.
GAM = GroupStabilityManager
