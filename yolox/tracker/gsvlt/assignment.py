from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def bidirectional_assignment(
    scores: np.ndarray, threshold: float
) -> Tuple[list[tuple[int, int]], Sequence[int], Sequence[int]]:
    """Mutual-best matching described as bi-directional matching in the paper."""

    matrix = np.asarray(scores, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("scores must be a two-dimensional matrix")
    rows, columns = matrix.shape
    if rows == 0 or columns == 0:
        return [], tuple(range(rows)), tuple(range(columns))

    best_column = matrix.argmax(axis=1)
    best_row = matrix.argmax(axis=0)
    matches = []
    for row, column in enumerate(best_column.tolist()):
        if best_row[column] == row and matrix[row, column] >= threshold:
            matches.append((row, column))
    matched_rows = {row for row, _ in matches}
    matched_columns = {column for _, column in matches}
    unmatched_rows = tuple(index for index in range(rows) if index not in matched_rows)
    unmatched_columns = tuple(index for index in range(columns) if index not in matched_columns)
    return matches, unmatched_rows, unmatched_columns
