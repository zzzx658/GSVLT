from __future__ import annotations

import configparser
import csv
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import GSVLTConfig
from .geometry import (
    eiou_distance,
    normalized_detection,
    normalized_motion_sequence,
    normalized_velocity,
)
from .types import MotionObservation


@dataclass
class MOTSequence:
    name: str
    image_size: tuple[int, int]
    frame_rate: int
    frames: Dict[int, Dict[int, MotionObservation]]
    tracks: Dict[int, list[MotionObservation]]

    @classmethod
    def load(cls, root: str | Path) -> "MOTSequence":
        root = Path(root)
        parser = configparser.ConfigParser()
        if not parser.read(root / "seqinfo.ini"):
            raise FileNotFoundError("Missing {}".format(root / "seqinfo.ini"))
        sequence = parser["Sequence"]
        image_size = (int(sequence["imHeight"]), int(sequence["imWidth"]))
        frame_rate = int(sequence.get("frameRate", 30))
        frames: Dict[int, Dict[int, MotionObservation]] = {}
        tracks: Dict[int, list[MotionObservation]] = {}
        ground_truth = root / "gt" / "gt.txt"
        with ground_truth.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) < 6:
                    continue
                frame_id, track_id = int(float(row[0])), int(float(row[1]))
                confidence = float(row[6]) if len(row) > 6 else 1.0
                category = int(float(row[7])) if len(row) > 7 else 1
                if confidence <= 0 or category != 1:
                    continue
                observation = MotionObservation(
                    frame_id,
                    np.asarray([float(value) for value in row[2:6]], dtype=np.float32),
                    confidence,
                )
                frames.setdefault(frame_id, {})[track_id] = observation
                tracks.setdefault(track_id, []).append(observation)
        for history in tracks.values():
            history.sort(key=lambda item: item.frame_id)
        return cls(root.name, image_size, frame_rate, frames, tracks)


def discover_sequences(root: str | Path) -> list[MOTSequence]:
    root = Path(root)
    candidates = [root] if (root / "seqinfo.ini").exists() else sorted(root.iterdir())
    sequences = [MOTSequence.load(path) for path in candidates if (path / "seqinfo.ini").exists()]
    if not sequences:
        raise FileNotFoundError("No MOTChallenge sequences found below {}".format(root))
    return sequences


class GSVLTMOTDataset(Dataset):
    """Dynamic paper-style GVA/GMVI samples from MOTChallenge ground truth."""

    def __init__(
        self,
        roots: str | Path | Sequence[str | Path],
        config: GSVLTConfig,
        seed: int = 42,
        max_gap: int = 180,
    ) -> None:
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.sequences = [sequence for root in roots for sequence in discover_sequences(root)]
        self.config = config
        self.seed = seed
        self.epoch = 0
        self.max_gap = max_gap
        self.index = []
        for sequence_index, sequence in enumerate(self.sequences):
            for track_id, history in sequence.tracks.items():
                frames = [item.frame_id for item in history]
                for frame in frames:
                    if frame - frames[0] + 1 < config.history_size:
                        continue
                    start = bisect_left(frames, frame + config.short_term_lost)
                    end = bisect_right(frames, frame + max_gap)
                    if start < end:
                        self.index.append((sequence_index, track_id, frame, start, end))
        if not self.index:
            raise ValueError("No trajectories are long enough to construct GSVLT samples")

    def __len__(self) -> int:
        return len(self.index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _history_until(
        observations: Iterable[MotionObservation], frame_id: int
    ) -> list[MotionObservation]:
        return [item for item in observations if item.frame_id <= frame_id]

    def _synthetic_candidate(
        self, positive: MotionObservation, rng: np.random.Generator
    ) -> MotionObservation:
        box = positive.tlwh.copy()
        box[:2] += rng.normal(0.0, np.maximum(box[2:] * 0.75, 2.0))
        box[2:] *= np.exp(rng.normal(0.0, 0.2, size=2))
        box[2:] = np.maximum(box[2:], 2.0)
        return MotionObservation(positive.frame_id, box, 1.0)

    def _maybe_crop_history(
        self, history: list[MotionObservation], rng: np.random.Generator
    ) -> list[MotionObservation]:
        if len(history) > 2 and rng.random() < self.config.trajectory_crop_probability:
            retained = int(rng.integers(2, len(history) + 1))
            return history[-retained:]
        return history

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sequence_index, target_id, lost_frame, future_start, future_end = self.index[index]
        sequence = self.sequences[sequence_index]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        future_index = int(rng.integers(future_start, future_end))
        current_frame = sequence.tracks[target_id][future_index].frame_id
        target_history = self._maybe_crop_history(
            self._history_until(sequence.tracks[target_id], lost_frame), rng
        )
        positive = sequence.frames[current_frame][target_id]
        target_at_lost = sequence.frames[lost_frame][target_id]

        neighbors = []
        for neighbor_id, observation in sequence.frames[lost_frame].items():
            if neighbor_id == target_id:
                continue
            distance = eiou_distance(target_at_lost.tlwh, observation.tlwh)
            if distance < self.config.group_eiou_threshold:
                neighbors.append((distance, neighbor_id))
        neighbors.sort()
        neighbors = neighbors[: self.config.max_group_size - 1]

        group_size = self.config.max_group_size
        past = np.zeros(
            (group_size, self.config.history_size, self.config.state_dim), dtype=np.float32
        )
        current = np.zeros_like(past)
        initial_mask = np.zeros(group_size, dtype=bool)
        survivor_mask = np.zeros(group_size, dtype=bool)
        past[0] = normalized_motion_sequence(
            target_history, sequence.image_size, self.config.history_size
        )
        current[0] = past[0]
        initial_mask[0] = True
        survivor_mask[0] = True
        for group_index, (_, neighbor_id) in enumerate(neighbors, start=1):
            past_history = self._history_until(sequence.tracks[neighbor_id], lost_frame)
            past_history = self._maybe_crop_history(past_history, rng)
            past[group_index] = normalized_motion_sequence(
                past_history, sequence.image_size, self.config.history_size
            )
            initial_mask[group_index] = True
            if (
                neighbor_id in sequence.frames.get(current_frame, {})
                and rng.random() >= self.config.neighbor_mask_probability
            ):
                current_identity = neighbor_id
                swap_candidates = [
                    identity
                    for identity in sequence.frames[current_frame]
                    if identity not in (target_id, neighbor_id)
                ]
                if swap_candidates and rng.random() < self.config.identity_swap_probability:
                    current_identity = int(rng.choice(swap_candidates))
                current_history = self._maybe_crop_history(
                    self._history_until(sequence.tracks[current_identity], current_frame),
                    rng,
                )
                current[group_index] = normalized_motion_sequence(
                    current_history, sequence.image_size, self.config.history_size
                )
                survivor_mask[group_index] = True

        frame_candidates = [
            observation
            for identity, observation in sequence.frames[current_frame].items()
            if identity != target_id
        ]
        positive_center = positive.tlwh[:2] + positive.tlwh[2:] / 2.0
        frame_candidates.sort(
            key=lambda item: float(
                np.linalg.norm(item.tlwh[:2] + item.tlwh[2:] / 2.0 - positive_center)
            )
        )
        candidates = [positive] + frame_candidates[: self.config.candidate_count - 1]
        while len(candidates) < self.config.candidate_count:
            candidates.append(self._synthetic_candidate(positive, rng))
        order = rng.permutation(len(candidates))
        candidates = [candidates[item] for item in order]
        positive_index = int(np.flatnonzero(order == 0)[0])
        labels = np.zeros(len(order), dtype=np.float32)
        labels[positive_index] = 1.0
        detections = np.stack(
            [normalized_detection(item.tlwh, sequence.image_size) for item in candidates]
        )

        pre_velocity = normalized_velocity(target_history, sequence.image_size)
        current_velocity = normalized_velocity(
            self._history_until(sequence.tracks[target_id], current_frame),
            sequence.image_size,
        )
        return {
            "past_sequences": torch.from_numpy(past),
            "current_sequences": torch.from_numpy(current),
            "initial_mask": torch.from_numpy(initial_mask),
            "survivor_mask": torch.from_numpy(survivor_mask),
            "time_offsets": torch.tensor(current_frame - lost_frame, dtype=torch.float32),
            "detections": torch.from_numpy(detections),
            "detection_mask": torch.ones(len(candidates), dtype=torch.bool),
            "labels": torch.from_numpy(labels),
            "positive_index": torch.tensor(positive_index, dtype=torch.long),
            "pre_velocity": torch.from_numpy(pre_velocity),
            "velocity_residual": torch.from_numpy(current_velocity - pre_velocity),
        }
