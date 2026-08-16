from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.gsvlt.config import GSVLTConfig
from yolox.tracker.gsvlt.manager import GSVLTManager
from yolox.tracker.gsvlt.types import GVAOutput
from yolox.tracker.gsvlt_tracker import GSVLTTracker


class FakeTrack:
    def __init__(self, track_id, history):
        self.track_id = track_id
        self.motion_history = history
        self.frame_id = history[-1][0]


class FakeDetection:
    def __init__(self, tlwh):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = 0.9


class FakeGVA(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def affinity(self, **inputs):
        batch = inputs["past_sequences"].shape[0]
        candidates = inputs["detections"].shape[1]
        scores = torch.full((batch, candidates), 0.1, device=inputs["detections"].device)
        scores[:, 0] = 0.9
        group = inputs["initial_mask"].shape[1]
        return GVAOutput(
            logits=scores,
            pgms=torch.zeros(batch, self.hidden_dim, device=scores.device),
            attention=torch.zeros(batch, group, group, device=scores.device),
            core_mask=inputs["initial_mask"],
        )


class FakeGMVI(nn.Module):
    def forward(self, pgms, detections):
        return torch.zeros(pgms.shape[0], 4, device=pgms.device)

    @staticmethod
    def infer_velocity(pre_velocity, residual):
        return pre_velocity + residual


def history(x_offset=0):
    return [
        (frame, np.array([x_offset + frame, 20, 10, 20], dtype=np.float32), 0.9)
        for frame in range(1, 11)
    ]


def test_manager_recovers_eligible_long_term_track():
    config = replace(
        GSVLTConfig(),
        history_size=5,
        hidden_dim=16,
        positional_dim=16,
        attention_heads=4,
        attention_layers=1,
        max_group_size=4,
        short_term_lost=2,
    )
    target = FakeTrack(1, history())
    neighbor = FakeTrack(2, history(5))
    manager = GSVLTManager(
        config,
        device="cpu",
        gva=FakeGVA(config.hidden_dim),
        gmvi=FakeGMVI(),
    )
    manager.register_lost(target, [neighbor], (100, 200))
    neighbor.motion_history.append((12, np.array([17, 20, 10, 20], dtype=np.float32), 0.9))
    result = manager.update(
        12,
        [neighbor],
        [FakeDetection([30, 20, 10, 20]), FakeDetection([150, 20, 10, 20])],
        (100, 200),
    )
    assert len(result.recoveries) == 1
    assert result.recoveries[0].track is target
    assert result.recoveries[0].detection_index == 0
    assert target.track_id not in manager.contexts


def test_bytetrack_integration_recovers_after_short_term_window():
    BaseTrack._count = 0
    config = replace(
        GSVLTConfig(),
        history_size=2,
        hidden_dim=16,
        positional_dim=16,
        attention_heads=4,
        attention_layers=1,
        max_group_size=3,
        short_term_lost=2,
    )
    manager = GSVLTManager(
        config,
        device="cpu",
        gva=FakeGVA(config.hidden_dim),
        gmvi=FakeGMVI(),
    )
    args = SimpleNamespace(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        mot20=False,
    )
    tracker = GSVLTTracker(args, config=config, manager=manager)
    image_info = [100, 200]
    image_size = (100, 200)
    first = tracker.update(
        np.array([[10, 20, 20, 40, 0.9]], dtype=np.float32), image_info, image_size
    )
    assert [track.track_id for track in first] == [1]
    tracker.update(np.array([[11, 20, 21, 40, 0.9]], dtype=np.float32), image_info, image_size)
    empty = np.empty((0, 5), dtype=np.float32)
    tracker.update(empty, image_info, image_size)
    tracker.update(empty, image_info, image_size)
    recovered = tracker.update(
        np.array([[100, 20, 110, 40, 0.9]], dtype=np.float32), image_info, image_size
    )
    assert [track.track_id for track in recovered] == [1]
    assert recovered[0].frame_id == 5
