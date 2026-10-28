"""Windowed data-quality checks and pixel statistics for immutable catalogs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import (
    PreprocessingSpec,
    SceneRecord,
    SplitManifest,
    active_split_names,
)
from .preprocessing import decode_rgb_labels
from .raster import RasterioSceneReader
from .sampling import grid_windows


def _class_counts(labels: np.ndarray, mask: np.ndarray) -> Counter[int]:
    values, counts = np.unique(labels[mask], return_counts=True)
    return Counter(
        {int(value): int(count) for value, count in zip(values, counts, strict=True)}
    )


def _coverage(
    counts: Counter[int], valid: int, total: int, ignore_label: int
) -> dict[str, Any]:
    return {
        "valid_pixels": valid,
        "total_pixels": total,
        "valid_pixel_fraction": valid / total if total else 0.0,
        "scored_pixels": valid - counts[ignore_label],
        "class_counts": {str(key): value for key, value in sorted(counts.items())},
    }


def validate_data_quality(
    scenes: tuple[SceneRecord, ...],
    data_root: Path,
    split: SplitManifest,
    preprocessing: PreprocessingSpec,
    required_assets_by_view: Mapping[str, tuple[str, ...]],
    *,
    tile_size: int = 1024,
    include_splits: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Check asset alignment, TIFF metadata, label palette, and per-split coverage."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not required_assets_by_view or any(
        not assets for assets in required_assets_by_view.values()
    ):
        raise ValueError("At least one non-empty view asset requirement is needed")
    by_id = {scene.scene_id: scene for scene in scenes}
    if len(by_id) != len(scenes):
        raise ValueError("Scene catalog contains duplicate scene IDs")
    if {scene.dataset_id for scene in scenes} != {split.dataset_id}:
        raise ValueError("Scene catalog and split manifest dataset IDs differ")
    missing = set(split.scene_ids) - by_id.keys()
    if missing:
        raise ValueError(f"Split references unknown scenes: {sorted(missing)}")
    included_splits = active_split_names(split, include_splits)

    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": split.dataset_id,
        "split_id": split.split_id,
        "split_version": split.version,
        "split_hash": split.content_hash,
        "preprocessing_id": preprocessing.preprocessing_id,
        "preprocessing_version": preprocessing.version,
        "preprocessing_hash": preprocessing.content_hash,
        "tile_size": tile_size,
        "included_splits": list(included_splits),
        "validity_scope": "label mask intersected with all required view asset masks "
        "and finite image values; class_counts include the ignore label",
        "view_asset_requirements": {
            name: list(asset_names)
            for name, asset_names in required_assets_by_view.items()
        },
        "splits": {},
    }
    for split_name in included_splits:
        class_counts: Counter[int] = Counter()
        valid_pixels = 0
        total_pixels = 0
        scene_reports = []
        label_asset = split.label_asset_for(split_name)
        split_view_counts = {name: Counter() for name in required_assets_by_view}
        split_view_valid = dict.fromkeys(required_assets_by_view, 0)
        for scene_id in split.scenes_for(split_name):
            scene = by_id[scene_id]
            view_alignment = {}
            for view_name, view_assets in required_assets_by_view.items():
                scene.require_aligned(tuple(dict.fromkeys((*view_assets, label_asset))))
                view_alignment[view_name] = "passed"
            reader = RasterioSceneReader(scene, data_root)
            counts = Counter()
            view_counts = {name: Counter() for name in required_assets_by_view}
            view_valid = dict.fromkeys(required_assets_by_view, 0)
            image_assets = sorted(
                {
                    asset
                    for assets in required_assets_by_view.values()
                    for asset in assets
                }
                - {label_asset}
            )
            valid_count = 0
            total_count = 0
            try:
                for window in grid_windows(
                    scene.reference_grid.height,
                    scene.reference_grid.width,
                    (tile_size, tile_size),
                ):
                    image, label_mask = reader.read_window_with_mask(
                        label_asset,
                        window,
                        resampling=preprocessing.categorical_resampling,
                    )
                    labels = decode_rgb_labels(
                        image, preprocessing, valid_mask=label_mask
                    )
                    asset_masks = {label_asset: label_mask}
                    for asset_name in image_assets:
                        array, mask = reader.read_window_with_mask(
                            asset_name,
                            window,
                            resampling=preprocessing.continuous_resampling,
                        )
                        asset_masks[asset_name] = mask & np.isfinite(array).all(axis=-1)
                    common_mask = label_mask.copy()
                    for mask in asset_masks.values():
                        common_mask &= mask
                    for view_name, view_assets in required_assets_by_view.items():
                        view_mask = label_mask.copy()
                        for asset_name in view_assets:
                            view_mask &= asset_masks[asset_name]
                        view_valid[view_name] += int(view_mask.sum())
                        view_counts[view_name].update(_class_counts(labels, view_mask))
                    total_count += min(
                        tile_size, scene.reference_grid.height - window.row
                    ) * min(tile_size, scene.reference_grid.width - window.col)
                    valid_count += int(common_mask.sum())
                    counts.update(_class_counts(labels, common_mask))
            finally:
                reader.close()
            class_counts.update(counts)
            valid_pixels += valid_count
            total_pixels += total_count
            for name in required_assets_by_view:
                split_view_counts[name].update(view_counts[name])
                split_view_valid[name] += view_valid[name]
            scene_reports.append(
                {
                    "scene_id": scene_id,
                    "shape": [scene.reference_grid.height, scene.reference_grid.width],
                    "crs": scene.reference_grid.crs,
                    "affine": (
                        list(scene.reference_grid.affine)
                        if scene.reference_grid.affine is not None
                        else None
                    ),
                    "assets": {
                        name: {
                            "uri": asset.uri,
                            "dtype": asset.dtype,
                            "sha256_present": asset.sha256 is not None,
                            "alignment_evidence": asset.alignment_evidence,
                        }
                        for name, asset in scene.assets
                    },
                    "view_alignment": view_alignment,
                    **_coverage(
                        counts, valid_count, total_count, preprocessing.ignore_label
                    ),
                    "views": {
                        name: _coverage(
                            view_counts[name],
                            view_valid[name],
                            total_count,
                            preprocessing.ignore_label,
                        )
                        for name in required_assets_by_view
                    },
                }
            )
        report["splits"][split_name] = {
            "scene_ids": list(split.scenes_for(split_name)),
            **_coverage(
                class_counts, valid_pixels, total_pixels, preprocessing.ignore_label
            ),
            "views": {
                name: _coverage(
                    split_view_counts[name],
                    split_view_valid[name],
                    total_pixels,
                    preprocessing.ignore_label,
                )
                for name in required_assets_by_view
            },
            "scenes": scene_reports,
        }
    return report
