"""Small, deterministic helpers for inspecting segmentation augmentations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def semantic_boundary_mask(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    """Mark both sides of 4-neighbor transitions between valid semantic classes.

    Ignore labels and invalid pixels do not create class boundaries. The result
    measures semantic class edges; it cannot separate adjacent objects that share
    the same class because the ISPRS labels do not contain instance IDs.
    """
    labels = np.asarray(labels)
    valid = np.asarray(valid_mask, dtype=bool)
    if labels.ndim != 2 or valid.shape != labels.shape:
        raise ValueError("Labels and validity mask must be aligned 2D arrays")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    semantic = valid & (labels >= 0) & (labels < num_classes)
    boundaries = np.zeros(labels.shape, dtype=bool)

    vertical_transition = (
        semantic[1:, :] & semantic[:-1, :] & (labels[1:, :] != labels[:-1, :])
    )
    boundaries[1:, :] |= vertical_transition
    boundaries[:-1, :] |= vertical_transition

    horizontal_transition = (
        semantic[:, 1:] & semantic[:, :-1] & (labels[:, 1:] != labels[:, :-1])
    )
    boundaries[:, 1:] |= horizontal_transition
    boundaries[:, :-1] |= horizontal_transition
    return boundaries


def window_starts(length: int, window_size: int, stride: int) -> tuple[int, ...]:
    """Return in-bounds starts at a fixed stride, including the far scene edge."""
    if length <= 0 or window_size <= 0 or stride <= 0:
        raise ValueError("length, window_size, and stride must be positive")
    if window_size > length:
        raise ValueError("window_size must not exceed the scene dimension")

    final_start = length - window_size
    starts = list(range(0, final_start + 1, stride))
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def window_sum_grid(
    mask: np.ndarray,
    row_starts: Sequence[int],
    column_starts: Sequence[int],
    window_shape: tuple[int, int],
) -> np.ndarray:
    """Count true pixels in every rectangular window using an integral image."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    window_height, window_width = window_shape
    if window_height <= 0 or window_width <= 0:
        raise ValueError("window dimensions must be positive")
    rows = np.asarray(row_starts, dtype=np.int64)
    columns = np.asarray(column_starts, dtype=np.int64)
    if rows.ndim != 1 or columns.ndim != 1:
        raise ValueError("row_starts and column_starts must be one-dimensional")
    if rows.size == 0 or columns.size == 0:
        return np.zeros((rows.size, columns.size), dtype=np.int64)
    if (
        np.any(rows < 0)
        or np.any(columns < 0)
        or np.any(rows + window_height > mask.shape[0])
        or np.any(columns + window_width > mask.shape[1])
    ):
        raise ValueError("A requested window falls outside the mask")

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required for efficient integral-image window counts"
        ) from error

    integral = cv2.integral(
        np.ascontiguousarray(mask, dtype=np.uint8),
        sdepth=cv2.CV_32S,
    )
    row0 = rows[:, None]
    row1 = row0 + window_height
    column0 = columns[None, :]
    column1 = column0 + window_width
    sums = (
        integral[row1, column1]
        - integral[row0, column1]
        - integral[row1, column0]
        + integral[row0, column0]
    )
    return np.asarray(sums, dtype=np.int64)
