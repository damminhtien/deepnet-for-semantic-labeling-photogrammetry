"""Deterministic training-window samplers and full-coverage grid windows."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .contracts import PreprocessingSpec, SplitManifest, WindowSpec


class ClassPixelHistogram:
    """Accumulate class pixel and patch-presence counts from sampled labels."""

    def __init__(
        self,
        class_ids: Sequence[int],
        ignore_label: int,
        *,
        class_names: Mapping[int, str] | None = None,
    ) -> None:
        self.class_ids = tuple(class_ids)
        if not self.class_ids or any(
            isinstance(class_id, (bool, np.bool_))
            or not isinstance(class_id, (int, np.integer))
            or class_id < 0
            for class_id in self.class_ids
        ):
            raise ValueError("class_ids must contain non-negative integers")
        if len(set(self.class_ids)) != len(self.class_ids):
            raise ValueError("class_ids must be unique")
        if (
            isinstance(ignore_label, (bool, np.bool_))
            or not isinstance(ignore_label, (int, np.integer))
            or ignore_label < 0
            or ignore_label in self.class_ids
        ):
            raise ValueError("ignore_label must be non-negative and outside class_ids")
        self.ignore_label = int(ignore_label)
        self.class_names = self._validate_class_names(class_names)
        self.reset()

    @classmethod
    def from_preprocessing(
        cls,
        preprocessing: PreprocessingSpec,
        *,
        class_names: Mapping[int, str] | None = None,
    ) -> ClassPixelHistogram:
        """Create a histogram using the canonical class IDs and ignore label."""
        class_ids = tuple(
            sorted(
                {
                    class_id
                    for _, _, _, class_id in preprocessing.label_palette
                    if class_id != preprocessing.ignore_label
                }
            )
        )
        if not class_ids:
            raise ValueError("Preprocessing spec does not declare a label palette")
        return cls(class_ids, preprocessing.ignore_label, class_names=class_names)

    def _validate_class_names(
        self,
        class_names: Mapping[int, str] | None,
    ) -> dict[int, str]:
        if class_names is None:
            return {class_id: str(class_id) for class_id in self.class_ids}
        names = dict(class_names)
        if set(names) != set(self.class_ids) or any(
            not isinstance(name, str) or not name.strip() for name in names.values()
        ):
            raise ValueError("class_names must name every configured class ID")
        if len(set(names.values())) != len(names):
            raise ValueError("class_names must be unique")
        return names

    def reset(self) -> None:
        """Clear accumulated counts before the next training epoch."""
        self.num_patches = 0
        self.total_pixels = 0
        self.valid_pixels = 0
        self.ignored_label_pixels = 0
        self.invalid_pixels = 0
        self.class_pixel_counts = dict.fromkeys(self.class_ids, 0)
        self.class_patch_counts = dict.fromkeys(self.class_ids, 0)

    def update(
        self,
        labels: Any,
        valid_mask: Any | None = None,
    ) -> None:
        """Count labels from one patch or a batch after augmentation.

        Labels must have shape ``(height, width)`` or
        ``(batch, height, width)``. Invalid pixels are excluded from class counts.
        """
        label_array = np.asarray(labels)
        if label_array.ndim not in (2, 3) or any(
            size <= 0 for size in label_array.shape
        ):
            raise ValueError("labels must be a non-empty HW or BHW array")
        if not np.issubdtype(label_array.dtype, np.integer):
            raise ValueError("labels must contain integer class IDs")
        if label_array.ndim == 2:
            label_array = label_array[np.newaxis, ...]

        if valid_mask is None:
            valid = np.ones(label_array.shape, dtype=bool)
        else:
            valid = np.asarray(valid_mask, dtype=bool)
            if valid.shape != label_array.shape:
                if label_array.shape[0] == 1 and valid.shape == label_array.shape[1:]:
                    valid = valid[np.newaxis, ...]
                else:
                    raise ValueError("valid_mask must have the same shape as labels")

        recognized = np.isin(label_array, (*self.class_ids, self.ignore_label))
        unknown = np.unique(label_array[valid & ~recognized])
        if unknown.size:
            raise ValueError(
                f"Unknown valid class IDs in sampled labels: {unknown.tolist()}"
            )

        self.num_patches += label_array.shape[0]
        self.total_pixels += label_array.size
        self.valid_pixels += int(np.count_nonzero(valid))
        self.ignored_label_pixels += int(
            np.count_nonzero(valid & (label_array == self.ignore_label))
        )
        self.invalid_pixels += int(np.count_nonzero(~valid))

        for class_id in self.class_ids:
            class_pixels = valid & (label_array == class_id)
            self.class_pixel_counts[class_id] += int(np.count_nonzero(class_pixels))
            self.class_patch_counts[class_id] += int(
                np.count_nonzero(np.any(class_pixels, axis=(1, 2)))
            )

    def merge(self, other: ClassPixelHistogram) -> None:
        """Add counts from another compatible accumulator, such as another rank."""
        if (self.class_ids, self.ignore_label) != (
            other.class_ids,
            other.ignore_label,
        ):
            raise ValueError("Only histograms with matching class contracts can merge")
        self.num_patches += other.num_patches
        self.total_pixels += other.total_pixels
        self.valid_pixels += other.valid_pixels
        self.ignored_label_pixels += other.ignored_label_pixels
        self.invalid_pixels += other.invalid_pixels
        for class_id in self.class_ids:
            self.class_pixel_counts[class_id] += other.class_pixel_counts[class_id]
            self.class_patch_counts[class_id] += other.class_patch_counts[class_id]

    def summary(self, *, epoch: int) -> dict[str, Any]:
        """Return JSON-compatible epoch counts and normalized frequencies."""
        if (
            isinstance(epoch, (bool, np.bool_))
            or not isinstance(epoch, (int, np.integer))
            or epoch < 0
        ):
            raise ValueError("epoch must be non-negative")
        epoch = int(epoch)
        semantic_pixels = sum(self.class_pixel_counts.values())
        classes = {}
        for class_id in self.class_ids:
            pixel_count = self.class_pixel_counts[class_id]
            patch_count = self.class_patch_counts[class_id]
            classes[str(class_id)] = {
                "name": self.class_names[class_id],
                "pixel_count": pixel_count,
                "pixel_percent": round(100 * pixel_count / semantic_pixels, 6)
                if semantic_pixels
                else 0.0,
                "patch_presence_count": patch_count,
                "patch_presence_percent": round(100 * patch_count / self.num_patches, 6)
                if self.num_patches
                else 0.0,
            }
        return {
            "epoch": epoch,
            "patch_count": self.num_patches,
            "total_pixels": self.total_pixels,
            "valid_pixels": self.valid_pixels,
            "semantic_pixels": semantic_pixels,
            "ignored_label_pixels": self.ignored_label_pixels,
            "invalid_pixels": self.invalid_pixels,
            "classes": classes,
        }

    def log(self, logger: logging.Logger, *, epoch: int) -> None:
        """Write one structured JSON log record for the completed epoch."""
        logger.info(
            "sampled_training_label_histogram %s",
            json.dumps(self.summary(epoch=epoch), ensure_ascii=False, sort_keys=True),
        )


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
