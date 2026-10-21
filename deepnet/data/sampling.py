"""Deterministic training-window samplers and full-coverage grid windows."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence

import numpy as np

from .contracts import SplitManifest, WindowSpec


def _window_starts(length: int, window_size: int, stride: int) -> tuple[int, ...]:
    if length <= 0 or window_size <= 0:
        raise ValueError("Image and window dimensions must be positive")
    if stride <= 0 or stride > window_size:
        raise ValueError("Grid stride must be positive and no larger than the window")
    return tuple(range(0, length, stride))


def grid_windows(
    height: int,
    width: int,
    window_size: tuple[int, int],
    stride: tuple[int, int] | None = None,
) -> tuple[WindowSpec, ...]:
    """Return deterministic fixed-size windows; edge windows use constant padding."""
    window_height, window_width = window_size
    stride_height, stride_width = stride or window_size
    rows = _window_starts(height, window_height, stride_height)
    columns = _window_starts(width, window_width, stride_width)
    return tuple(
        WindowSpec(row, col, window_height, window_width, padding="constant")
        for row in rows
        for col in columns
    )


def coverage_counts(
    height: int,
    width: int,
    windows: Sequence[WindowSpec],
) -> np.ndarray:
    """Count how many windows cover each real pixel, excluding padded pixels."""
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")
    coverage = np.zeros((height, width), dtype=np.uint32)
    for window in windows:
        row_end = min(height, window.row + window.height)
        col_end = min(width, window.col + window.width)
        if window.row >= height or window.col >= width:
            continue
        coverage[window.row : row_end, window.col : col_end] += 1
    return coverage


def uniform_window(
    rng: random.Random,
    height: int,
    width: int,
    window_size: tuple[int, int],
) -> WindowSpec:
    window_height, window_width = window_size
    if height < window_height or width < window_width:
        raise ValueError("Window must fit in the image for random training sampling")
    row = rng.randint(0, height - window_height)
    col = rng.randint(0, width - window_width)
    return WindowSpec(row, col, window_height, window_width)


def window_containing_point(
    rng: random.Random,
    height: int,
    width: int,
    window_size: tuple[int, int],
    point: tuple[int, int],
) -> WindowSpec:
    """Sample a valid window that contains a selected class pixel."""
    window_height, window_width = window_size
    if height < window_height or width < window_width:
        raise ValueError("Window must fit in the image for class-aware sampling")
    row, col = point
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError("Target pixel is outside the scene")
    min_row = max(0, row - window_height + 1)
    max_row = min(row, height - window_height)
    min_col = max(0, col - window_width + 1)
    max_col = min(col, width - window_width)
    return WindowSpec(
        rng.randint(min_row, max_row),
        rng.randint(min_col, max_col),
        window_height,
        window_width,
    )


class UniformSampler:
    """Sample a scene and crop uniformly from the selected split."""

    def __init__(
        self,
        scene_shapes: Mapping[str, tuple[int, int]],
        scene_ids: Sequence[str],
        window_size: tuple[int, int],
    ) -> None:
        self.scene_shapes = dict(scene_shapes)
        self.scene_ids = tuple(scene_ids)
        self.window_size = window_size
        if not self.scene_ids or set(self.scene_ids) - self.scene_shapes.keys():
            raise ValueError("Uniform sampler requires shapes for every selected scene")
        if len(window_size) != 2 or any(size <= 0 for size in window_size):
            raise ValueError("window_size must contain two positive dimensions")
        if any(
            self.scene_shapes[scene_id][0] < window_size[0]
            or self.scene_shapes[scene_id][1] < window_size[1]
            for scene_id in self.scene_ids
        ):
            raise ValueError(
                "Every selected scene must fit the requested training window"
            )

    def sample(self, rng: random.Random) -> tuple[str, WindowSpec]:
        scene_id = rng.choice(self.scene_ids)
        height, width = self.scene_shapes[scene_id]
        return scene_id, uniform_window(rng, height, width, self.window_size)


class ClassAwareSampler:
    """Sample from a precomputed train-only index of target-class pixel locations."""

    def __init__(
        self,
        scene_shapes: Mapping[str, tuple[int, int]],
        candidates: Mapping[int, Mapping[str, Sequence[int]]],
        split: SplitManifest,
        window_size: tuple[int, int],
        *,
        class_weights: Mapping[int, float] | None = None,
    ) -> None:
        self.scene_shapes = dict(scene_shapes)
        self.candidates = {
            int(class_id): {
                scene_id: np.asarray(indices, dtype=np.int64)
                for scene_id, indices in scene_candidates.items()
                if len(indices)
            }
            for class_id, scene_candidates in candidates.items()
        }
        for by_scene in self.candidates.values():
            for scene_id, indices in by_scene.items():
                if scene_id not in self.scene_shapes:
                    raise ValueError(
                        f"No shape is declared for candidate scene {scene_id!r}"
                    )
                if indices.ndim != 1 or np.any(indices < 0):
                    raise ValueError(
                        "Class-aware indexes must be one-dimensional and non-negative"
                    )
                if np.any(
                    indices
                    >= self.scene_shapes[scene_id][0] * self.scene_shapes[scene_id][1]
                ):
                    raise ValueError(f"Class-aware index is outside scene {scene_id!r}")
        train_ids = set(split.train_scene_ids)
        leaking = {
            scene_id
            for scene_candidates in self.candidates.values()
            for scene_id in scene_candidates
            if scene_id not in train_ids
        }
        if leaking:
            raise ValueError(
                "Class-aware sampling index includes non-training scenes: "
                + ", ".join(sorted(leaking))
            )
        self.window_size = window_size
        self.class_ids = tuple(
            class_id for class_id, by_scene in self.candidates.items() if by_scene
        )
        if not self.class_ids:
            raise ValueError("Class-aware sampler has no target pixels")
        self.class_weights = {
            class_id: float((class_weights or {}).get(class_id, 1.0))
            for class_id in self.class_ids
        }
        if any(
            not math.isfinite(weight) or weight <= 0
            for weight in self.class_weights.values()
        ):
            raise ValueError("Class sampling weights must be finite and positive")

    def sample(self, rng: random.Random) -> tuple[str, WindowSpec]:
        class_id = rng.choices(
            self.class_ids,
            weights=[self.class_weights[class_id] for class_id in self.class_ids],
            k=1,
        )[0]
        by_scene = self.candidates[class_id]
        scene_id = rng.choice(tuple(by_scene))
        indices = by_scene[scene_id]
        height, width = self.scene_shapes[scene_id]
        flat_index = int(indices[rng.randrange(indices.size)])
        point = divmod(flat_index, width)
        return scene_id, window_containing_point(
            rng, height, width, self.window_size, point
        )


class MixedSampler:
    """Mix uniform windows with class-aware windows using a fixed probability."""

    def __init__(
        self,
        uniform: UniformSampler,
        alpha: float,
        aware: ClassAwareSampler | None = None,
    ) -> None:
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if alpha and aware is None:
            raise ValueError("Class-aware sampler is required when alpha is non-zero")
        self.uniform = uniform
        self.alpha = alpha
        self.aware = aware

    def sample(self, rng: random.Random) -> tuple[str, WindowSpec]:
        if self.aware is not None and rng.random() < self.alpha:
            return self.aware.sample(rng)
        return self.uniform.sample(rng)


def sample_rng(seed: int, epoch: int, index: int) -> random.Random:
    """Create a schedule-independent RNG for a sample index and epoch."""
    payload = f"{seed}:{epoch}:{index}".encode("ascii")
    derived_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return random.Random(derived_seed)
