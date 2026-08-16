from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .types import MotionObservation


def tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
    result = np.asarray(tlwh, dtype=np.float32).copy()
    result[:2] += result[2:] / 2.0
    return result


def eiou_distance(first_tlwh: np.ndarray, second_tlwh: np.ndarray, epsilon: float = 1e-6) -> float:
    """Efficient-IoU loss used as the paper's group distance.

    This is ``1 - IoU`` plus normalized center, width, and height penalties.
    It is zero for identical boxes and can exceed one for separated boxes,
    which is compatible with the paper's M1=1.7 threshold.
    """

    first = np.asarray(first_tlwh, dtype=np.float32)
    second = np.asarray(second_tlwh, dtype=np.float32)
    first_br = first[:2] + first[2:]
    second_br = second[:2] + second[2:]

    intersection_tl = np.maximum(first[:2], second[:2])
    intersection_br = np.minimum(first_br, second_br)
    intersection_wh = np.maximum(intersection_br - intersection_tl, 0.0)
    intersection = float(np.prod(intersection_wh))
    union = float(np.prod(first[2:]) + np.prod(second[2:]) - intersection)
    iou = intersection / max(union, epsilon)

    first_center = first[:2] + first[2:] / 2.0
    second_center = second[:2] + second[2:] / 2.0
    enclosing_tl = np.minimum(first[:2], second[:2])
    enclosing_br = np.maximum(first_br, second_br)
    enclosing_wh = np.maximum(enclosing_br - enclosing_tl, epsilon)

    center_penalty = float(np.sum((first_center - second_center) ** 2)) / float(
        np.sum(enclosing_wh**2) + epsilon
    )
    width_penalty = float((first[2] - second[2]) ** 2) / float(enclosing_wh[0] ** 2 + epsilon)
    height_penalty = float((first[3] - second[3]) ** 2) / float(enclosing_wh[1] ** 2 + epsilon)
    return float(1.0 - iou + center_penalty + width_penalty + height_penalty)


def pad_history(
    observations: Sequence[MotionObservation], history_size: int
) -> list[MotionObservation]:
    if not observations:
        raise ValueError("At least one observation is required")
    ordered = sorted(observations, key=lambda observation: observation.frame_id)
    ordered = ordered[-history_size:]
    if len(ordered) < history_size:
        ordered = [ordered[0]] * (history_size - len(ordered)) + ordered
    return ordered


def normalized_motion_sequence(
    observations: Sequence[MotionObservation],
    image_size: tuple[int, int],
    history_size: int,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Build the paper's 8-D normalized state sequence."""

    image_height, image_width = image_size
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_size must contain positive height and width")
    history = pad_history(observations, history_size)
    boxes = np.stack([tlwh_to_xywh(item.tlwh) for item in history])

    centers = boxes[:, :2]
    sizes = np.maximum(boxes[:, 2:], epsilon)
    center_delta = np.diff(centers, axis=0, prepend=centers[:1])
    size_ratio = np.log(sizes / np.concatenate([sizes[:1], sizes[:-1]], axis=0))

    states = np.column_stack(
        [
            centers[:, 0] / image_width,
            centers[:, 1] / image_height,
            center_delta[:, 0] / image_width,
            center_delta[:, 1] / image_height,
            size_ratio[:, 0],
            size_ratio[:, 1],
            sizes[:, 0] / image_width,
            sizes[:, 1] / image_height,
        ]
    )
    return states.astype(np.float32)


def normalized_detection(tlwh: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    image_height, image_width = image_size
    box = tlwh_to_xywh(tlwh)
    return np.asarray(
        [box[0] / image_width, box[1] / image_height, box[2] / image_width, box[3] / image_height],
        dtype=np.float32,
    )


def normalized_velocity(
    observations: Sequence[MotionObservation], image_size: tuple[int, int], epsilon: float = 1e-6
) -> np.ndarray:
    if len(observations) < 2:
        return np.zeros(4, dtype=np.float32)
    previous, current = sorted(observations, key=lambda item: item.frame_id)[-2:]
    frame_gap = max(current.frame_id - previous.frame_id, 1)
    previous_box = tlwh_to_xywh(previous.tlwh)
    current_box = tlwh_to_xywh(current.tlwh)
    height, width = image_size
    return np.asarray(
        [
            (current_box[0] - previous_box[0]) / (width * frame_gap),
            (current_box[1] - previous_box[1]) / (height * frame_gap),
            np.log(max(current_box[2], epsilon) / max(previous_box[2], epsilon)) / frame_gap,
            np.log(max(current_box[3], epsilon) / max(previous_box[3], epsilon)) / frame_gap,
        ],
        dtype=np.float32,
    )


def motion_order_parameter(velocities: Iterable[np.ndarray], epsilon: float = 1e-6) -> float:
    unit_vectors = []
    for velocity in velocities:
        vector = np.asarray(velocity, dtype=np.float32)[:2]
        norm = float(np.linalg.norm(vector))
        if norm > epsilon:
            unit_vectors.append(vector / norm)
    if not unit_vectors:
        return 0.0
    return float(np.clip(np.linalg.norm(np.mean(unit_vectors, axis=0)), 0.0, 1.0))


def velocity_to_kalman(
    normalized: np.ndarray, current_tlwh: np.ndarray, image_size: tuple[int, int]
) -> np.ndarray:
    """Convert [dx/W, dy/H, dlog(w), dlog(h)] to ByteTrack velocity."""

    height, width = image_size
    box = tlwh_to_xywh(current_tlwh)
    new_width = box[2] * float(np.exp(np.clip(normalized[2], -2.0, 2.0)))
    new_height = box[3] * float(np.exp(np.clip(normalized[3], -2.0, 2.0)))
    current_aspect = box[2] / max(box[3], 1e-6)
    new_aspect = new_width / max(new_height, 1e-6)
    return np.asarray(
        [
            normalized[0] * width,
            normalized[1] * height,
            new_aspect - current_aspect,
            new_height - box[3],
        ],
        dtype=np.float32,
    )
