import numpy as np

from yolox.tracker.gsvlt.geometry import (
    eiou_distance,
    motion_order_parameter,
    normalized_motion_sequence,
    normalized_velocity,
)
from yolox.tracker.gsvlt.types import MotionObservation


def observations():
    return [
        MotionObservation(1, np.array([10, 20, 8, 12], dtype=np.float32)),
        MotionObservation(2, np.array([12, 23, 8, 12], dtype=np.float32)),
    ]


def test_eiou_distance_is_zero_for_identical_boxes():
    box = np.array([10, 20, 8, 12], dtype=np.float32)
    assert eiou_distance(box, box) == 0.0
    assert eiou_distance(box, box + np.array([100, 0, 0, 0])) > 1.0


def test_paper_motion_representation_and_padding():
    states = normalized_motion_sequence(observations(), (100, 200), history_size=4)
    assert states.shape == (4, 8)
    assert states.dtype == np.float32
    np.testing.assert_allclose(states[-1, 2:4], [2 / 200, 3 / 100])
    np.testing.assert_allclose(states[-1, 4:6], [0.0, 0.0])


def test_velocity_and_order_parameter():
    velocity = normalized_velocity(observations(), (100, 200))
    np.testing.assert_allclose(velocity[:2], [2 / 200, 3 / 100])
    assert motion_order_parameter([velocity, velocity]) == 1.0
