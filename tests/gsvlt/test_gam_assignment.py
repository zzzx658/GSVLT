from dataclasses import replace

import numpy as np

from yolox.tracker.gsvlt.assignment import bidirectional_assignment
from yolox.tracker.gsvlt.config import GSVLTConfig
from yolox.tracker.gsvlt.gam import GroupStabilityManager
from yolox.tracker.gsvlt.types import GroupMemberMemory, LongTermTrackContext, MotionObservation


def test_bidirectional_assignment_requires_mutual_best():
    scores = np.array([[0.9, 0.8], [0.85, 0.2]], dtype=np.float32)
    matches, unmatched_rows, unmatched_columns = bidirectional_assignment(scores, 0.5)
    assert matches == [(0, 0)]
    assert unmatched_rows == (1,)
    assert unmatched_columns == (1,)


def test_gam_terminates_after_persistent_structural_dissolution():
    config = replace(
        GSVLTConfig(),
        gdi_motion_weight=0.0,
        gdi_structure_weight=1.0,
        gdi_temporal_weight=0.0,
        confidence_window=1,
        termination_threshold=0.5,
        termination_patience=2,
    )
    observation = MotionObservation(1, np.array([0, 0, 10, 10], dtype=np.float32))
    context = LongTermTrackContext(
        track=object(),
        lost_frame=1,
        image_size=(100, 100),
        members=[
            GroupMemberMemory(1, [observation], is_target=True),
            GroupMemberMemory(2, [observation]),
        ],
        initial_phi=1.0,
        pre_velocity=np.zeros(4, dtype=np.float32),
    )
    gam = GroupStabilityManager(config)
    first, terminate = gam.update(context, [], survivor_count=0, elapsed_frames=30)
    assert first.structure == 1.0
    assert not terminate
    _, terminate = gam.update(context, [], survivor_count=0, elapsed_frames=31)
    assert terminate
