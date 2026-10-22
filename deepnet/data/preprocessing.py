"""Deterministic label and model-view transformations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np

from .contracts import (
    ModelInputSpec,
    NormalizationRule,
    PreprocessingSpec,
    SceneRecord,
    SplitManifest,
)
from .raster import RasterioSceneReader
from .sampling import grid_windows


def decode_rgb_labels(
    image: np.ndarray,
    spec: PreprocessingSpec,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Map the declared RGB palette to IDs, rejecting unsupported valid colors.

    Optional tolerance uses Euclidean distance in RGB space to accommodate a
    verified source color transform. Equal-distance matches and colors outside the
    configured tolerance remain errors rather than being guessed.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB label array, got shape {image.shape}")
    valid = (
        np.ones(image.shape[:2], dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != image.shape[:2]:
        raise ValueError("Label validity mask does not match the label raster")
    result = np.full(image.shape[:2], spec.ignore_label, dtype=np.uint8)
    recognized = np.zeros(image.shape[:2], dtype=bool)
    for red, green, blue, class_id in spec.label_palette:
        match = np.all(image == (red, green, blue), axis=-1)
        result[match] = class_id
        recognized |= match
    unknown = valid & ~recognized
    if np.any(unknown):
        palette = np.asarray(
            [entry[:3] for entry in spec.label_palette], dtype=np.int32
        )
        class_ids = np.asarray(
            [entry[3] for entry in spec.label_palette], dtype=np.uint8
        )
        pixel_indices = np.flatnonzero(unknown)
        pixels = image.reshape(-1, 3)[pixel_indices].astype(np.int32, copy=False)
        nearest_squared = np.full(len(pixels), np.iinfo(np.int32).max, dtype=np.int32)
        second_squared = nearest_squared.copy()
        nearest_indices = np.zeros(len(pixels), dtype=np.intp)

        for palette_index, color in enumerate(palette):
            delta = pixels - color
            squared = np.sum(delta * delta, axis=1)
            closer = squared < nearest_squared
            second_squared = np.where(
                closer, nearest_squared, np.minimum(second_squared, squared)
            )
            nearest_squared = np.minimum(nearest_squared, squared)
            nearest_indices[closer] = palette_index

        tolerance_squared = spec.label_color_tolerance**2
        resolved = (nearest_squared <= tolerance_squared) & (
            nearest_squared < second_squared
        )
        result.reshape(-1)[pixel_indices[resolved]] = class_ids[
            nearest_indices[resolved]
        ]
        unresolved = ~resolved
        if np.any(unresolved):
            colors = pixels[unresolved]
            color_keys = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
            samples = [
                [int(key >> 16), int((key >> 8) & 255), int(key & 255)]
                for key in np.unique(color_keys)[:10]
            ]
            max_distance = float(np.sqrt(nearest_squared[unresolved].max()))
            ambiguous_count = int(
                np.count_nonzero((nearest_squared == second_squared) & unresolved)
            )
            raise ValueError(
                "Label raster contains unresolved RGB colors: "
                f"{samples}; maximum nearest-palette distance={max_distance:.2f}, "
                f"tolerance={spec.label_color_tolerance:g}, "
                f"ambiguous_pixels={ambiguous_count}"
            )
    return result


def _rule_map(view: ModelInputSpec) -> dict[str, NormalizationRule]:
    return {rule.channel: rule for rule in view.normalization}


def apply_model_view(
    source_arrays: Mapping[tuple[str, str], np.ndarray],
    view: ModelInputSpec,
    preprocessing: PreprocessingSpec,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build a model tensor from named asset bands without modifying source arrays."""
    stats = {
        channel: (mean, std) for channel, mean, std in preprocessing.fitted_statistics
    }
    rules = _rule_map(view)
    output_channels = []
    spatial_shape = None
    valid = None
    for channel, asset_name, band_name in view.channel_sources:
        key = (asset_name, band_name)
        if key not in source_arrays:
            raise KeyError(
                f"Model input {channel!r} requires missing source band {key}"
            )
        array = np.asarray(source_arrays[key])
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim != 2:
            raise ValueError(f"Source band {key} must be 2D, got shape {array.shape}")
        if spatial_shape is None:
            spatial_shape = array.shape
            valid = (
                np.ones(spatial_shape, dtype=bool)
                if valid_mask is None
                else np.asarray(valid_mask, dtype=bool)
            )
            if valid.shape != spatial_shape:
                raise ValueError(
                    "Model-input validity mask does not match source bands"
                )
        elif array.shape != spatial_shape:
            raise ValueError(f"Source band {key} is not on the view's reference window")
        rule = rules[channel]
        values = array.astype(np.float32, copy=True)
        if rule.method == "scale":
            assert rule.scale is not None
            values /= rule.scale
        elif rule.method == "fixed_standardize":
            assert rule.mean is not None and rule.std is not None
            values -= rule.mean
            values /= rule.std
        elif rule.method == "zscore":
            if channel not in stats:
                raise ValueError(
                    "No training-fitted statistics are recorded for "
                    f"channel {channel!r}"
                )
            mean, std = stats[channel]
            values -= mean
            values /= std
        elif rule.method != "identity":
            raise ValueError(f"Unsupported normalization rule: {rule.method}")
        assert valid is not None
        values[~valid] = 0.0
        output_channels.append(values)
    result = np.stack(output_channels, axis=0 if view.layout == "CHW" else -1)
    return np.ascontiguousarray(result, dtype=np.float32)


def fit_train_statistics(
    spec: PreprocessingSpec,
    split: SplitManifest,
    samples: Iterable[tuple[str, Mapping[str, np.ndarray], np.ndarray]],
    *,
    channels: tuple[str, ...],
) -> PreprocessingSpec:
    """Fit population mean/std from valid pixels in supplied train scenes."""
    train_ids = set(split.train_scene_ids)
    state = {channel: [0, 0.0, 0.0] for channel in channels}
    seen_scenes: set[str] = set()
    for scene_id, arrays, valid_mask in samples:
        if scene_id not in train_ids:
            raise ValueError(
                "Cannot fit preprocessing statistics from "
                f"non-training scene {scene_id!r}"
            )
        seen_scenes.add(scene_id)
        valid = np.asarray(valid_mask, dtype=bool)
        for channel in channels:
            if channel not in arrays:
                raise KeyError(f"Statistics sample is missing channel {channel!r}")
            values = np.asarray(arrays[channel], dtype=np.float64)
            if values.shape != valid.shape:
                raise ValueError(
                    f"Statistics mask shape does not match channel {channel!r}"
                )
            values = values[valid & np.isfinite(values)]
            if values.size == 0:
                continue
            count, mean, m2 = state[channel]
            batch_count = int(values.size)
            batch_mean = float(values.mean())
            batch_m2 = float(np.square(values - batch_mean).sum())
            if count == 0:
                state[channel] = [batch_count, batch_mean, batch_m2]
                continue
            total = count + batch_count
            delta = batch_mean - mean
            combined_mean = mean + delta * batch_count / total
            combined_m2 = m2 + batch_m2 + delta * delta * count * batch_count / total
            state[channel] = [total, combined_mean, combined_m2]

    if not seen_scenes:
        raise ValueError("No training scenes were supplied for statistics fitting")
    fitted = dict(
        (channel, (mean, std)) for channel, mean, std in spec.fitted_statistics
    )
    for channel, (count, mean, m2) in state.items():
        if count <= 1:
            raise ValueError(f"Not enough valid values to fit channel {channel!r}")
        std = float(np.sqrt(m2 / count))
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"Channel {channel!r} has zero or invalid variance")
        fitted[channel] = (float(mean), std)
    result = replace(
        spec,
        version=spec.version + 1,
        fitted_statistics=tuple(
            (channel, mean, std) for channel, (mean, std) in sorted(fitted.items())
        ),
        fit_scene_ids=tuple(sorted(seen_scenes)),
        fit_split_hash=split.content_hash,
    )
    result.validate_fit_scope(split)
    return result


def fit_model_statistics(
    scenes: Iterable[SceneRecord],
    data_root: Path,
    split: SplitManifest,
    preprocessing: PreprocessingSpec,
    model_input: ModelInputSpec,
    *,
    tile_size: int = 1024,
) -> PreprocessingSpec:
    """Fit model-view z-score channels by streaming only training-scene windows."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    preprocessing.validate_fit_scope(split)
    statistic_channels = tuple(
        rule.channel for rule in model_input.normalization if rule.method == "zscore"
    )
    if not statistic_channels:
        return preprocessing

    records = tuple(scenes)
    scene_map = {scene.scene_id: scene for scene in records}
    if len(scene_map) != len(records):
        raise ValueError("Scene catalog contains duplicate scene IDs")
    if set(split.scene_ids) - scene_map.keys():
        raise ValueError("Split manifest references scenes missing from the catalog")
    if {scene.dataset_id for scene in records} != {split.dataset_id}:
        raise ValueError("Scene catalog and split manifest dataset IDs differ")

    source_by_channel = {
        channel: (asset_name, band_name)
        for channel, asset_name, band_name in model_input.channel_sources
    }
    missing_sources = set(statistic_channels) - source_by_channel.keys()
    if missing_sources:
        raise ValueError(
            "Model input has no source for channels: "
            + ", ".join(sorted(missing_sources))
        )

    def training_samples() -> Iterable[
        tuple[str, Mapping[str, np.ndarray], np.ndarray]
    ]:
        """Yield bounded raster windows while closing each scene reader promptly."""
        for scene_id in split.train_scene_ids:
            scene = scene_map[scene_id]
            if scene.label_ontology != preprocessing.label_ontology:
                raise ValueError(
                    f"Scene {scene_id} uses ontology {scene.label_ontology!r}; "
                    f"preprocessing expects {preprocessing.label_ontology!r}"
                )
            required_assets = tuple(
                dict.fromkeys(
                    source_by_channel[channel][0] for channel in statistic_channels
                )
            )
            scene.require_aligned(required_assets)
            reader = RasterioSceneReader(scene, data_root)
            grid = scene.reference_grid
            try:
                for window in grid_windows(
                    grid.height,
                    grid.width,
                    (tile_size, tile_size),
                ):
                    arrays: dict[str, np.ndarray] = {}
                    valid = np.ones((window.height, window.width), dtype=bool)
                    asset_arrays: dict[str, np.ndarray] = {}
                    for channel in statistic_channels:
                        asset_name, band_name = source_by_channel[channel]
                        if asset_name not in asset_arrays:
                            array, asset_valid = reader.read_window_with_mask(
                                asset_name,
                                window,
                                resampling=preprocessing.continuous_resampling,
                            )
                            asset_arrays[asset_name] = array
                            valid &= asset_valid
                        asset = scene.asset_map()[asset_name]
                        if band_name not in asset.bands:
                            raise ValueError(
                                f"Scene {scene_id} asset {asset_name!r} has no band "
                                f"{band_name!r}"
                            )
                        values = asset_arrays[asset_name][
                            ..., asset.bands.index(band_name)
                        ]
                        arrays[channel] = values
                        valid &= np.isfinite(values)
                    yield scene_id, arrays, valid
            finally:
                reader.close()

    return fit_train_statistics(
        preprocessing,
        split,
        training_samples(),
        channels=statistic_channels,
    )
