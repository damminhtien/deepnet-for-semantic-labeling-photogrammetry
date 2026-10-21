"""Immutable, serializable contracts for reproducible raster pipelines."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any


class AlignmentError(ValueError):
    """Raised when assets cannot be proven to share a spatial grid."""


def canonical_hash(value: Any) -> str:
    """Hash a JSON-compatible value using a stable UTF-8 representation."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class GridSpec:
    """A raster grid, either georeferenced or explicitly pixel-local."""

    grid_id: str
    width: int
    height: int
    crs: str | None = None
    affine: tuple[float, float, float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.grid_id.strip():
            raise ValueError("grid_id must not be empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Grid dimensions must be positive")
        if self.crs is not None and self.affine is None:
            raise ValueError("A CRS requires an affine transform")
        if self.crs is not None and not self.crs.strip():
            raise ValueError("crs must not be empty")
        if self.affine is not None and (
            len(self.affine) != 6 or not all(math.isfinite(v) for v in self.affine)
        ):
            raise ValueError("affine must contain six finite coefficients")

    @property
    def georeferenced(self) -> bool:
        return self.crs is not None and self.affine is not None

    @property
    def resolution(self) -> tuple[float, float] | None:
        if self.affine is None:
            return None
        a, b, _, d, e, _ = self.affine
        return math.hypot(a, d), math.hypot(b, e)

    def same_grid(self, other: GridSpec, tolerance: float = 1e-9) -> bool:
        if (self.width, self.height) != (other.width, other.height):
            return False
        if self.georeferenced != other.georeferenced:
            return False
        if self.georeferenced and self.crs != other.crs:
            return False
        if not self.georeferenced and self.grid_id != other.grid_id:
            return False
        if self.affine is None or other.affine is None:
            return self.affine is None and other.affine is None
        return all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)
            for left, right in zip(self.affine, other.affine, strict=True)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_id": self.grid_id,
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "affine": list(self.affine) if self.affine is not None else None,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GridSpec:
        affine = value.get("affine")
        return cls(
            grid_id=value["grid_id"],
            width=int(value["width"]),
            height=int(value["height"]),
            crs=value.get("crs"),
            affine=tuple(float(item) for item in affine)
            if affine is not None
            else None,
        )


@dataclass(frozen=True, slots=True)
class RasterAsset:
    """A source or derived raster and the metadata verified at ingestion."""

    asset_id: str
    uri: str
    bands: tuple[str, ...]
    dtype: str
    grid: GridSpec
    nodata: float | int | str | None = None
    sha256: str | None = None
    alignment_evidence: str | None = None

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise ValueError("asset_id must not be empty")
        path = PurePosixPath(self.uri)
        if path.is_absolute() or ".." in path.parts or not self.uri.strip():
            raise ValueError(
                f"uri must be a safe path relative to the data root: {self.uri}"
            )
        if not self.bands or any(not band.strip() for band in self.bands):
            raise ValueError("Raster assets must declare at least one named band")
        if len(set(self.bands)) != len(self.bands):
            raise ValueError("Band names must be unique within a raster asset")
        if not self.dtype.strip():
            raise ValueError("dtype must not be empty")
        if self.sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        if self.alignment_evidence is not None and not self.alignment_evidence.strip():
            raise ValueError("alignment_evidence must not be an empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "uri": self.uri,
            "bands": list(self.bands),
            "dtype": self.dtype,
            "grid": self.grid.to_dict(),
            "nodata": self.nodata,
            "sha256": self.sha256,
            "alignment_evidence": self.alignment_evidence,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RasterAsset:
        return cls(
            asset_id=value["asset_id"],
            uri=value["uri"],
            bands=tuple(value["bands"]),
            dtype=value["dtype"],
            grid=GridSpec.from_dict(value["grid"]),
            nodata=value.get("nodata"),
            sha256=value.get("sha256"),
            alignment_evidence=value.get("alignment_evidence"),
        )


@dataclass(frozen=True, slots=True)
class SceneRecord:
    """A scene's assets, reference grid, and source provenance."""

    dataset_id: str
    scene_id: str
    schema_version: int
    reference_grid: GridSpec
    label_ontology: str
    assets: tuple[tuple[str, RasterAsset], ...]
    provenance: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.dataset_id.strip() or not self.scene_id.strip():
            raise ValueError("dataset_id and scene_id must not be empty")
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if not self.label_ontology.strip():
            raise ValueError("label_ontology must not be empty")
        names = [name for name, _ in self.assets]
        if len(names) != len(set(names)):
            raise ValueError("Scene asset names must be unique")
        if any(not name.strip() for name in names):
            raise ValueError("Scene asset names must not be empty")
        if len({key for key, _ in self.provenance}) != len(self.provenance):
            raise ValueError("Provenance keys must be unique")
        if any(not key.strip() or not value.strip() for key, value in self.provenance):
            raise ValueError("Provenance keys and values must not be empty")
        if any(name != asset.asset_id for name, asset in self.assets):
            raise ValueError("Scene asset keys must match each RasterAsset asset_id")

    def asset_map(self) -> dict[str, RasterAsset]:
        return dict(self.assets)

    def require_aligned(
        self,
        asset_names: tuple[str, ...],
        *,
        allow_reprojection: bool = False,
    ) -> None:
        """Check assets against the reference without shape-only inference."""
        assets = self.asset_map()
        missing = sorted(set(asset_names) - assets.keys())
        if missing:
            raise AlignmentError(f"Scene {self.scene_id} is missing assets: {missing}")
        for name in asset_names:
            asset = assets[name]
            grid = asset.grid
            if grid.same_grid(self.reference_grid):
                continue
            if (
                allow_reprojection
                and grid.georeferenced
                and self.reference_grid.georeferenced
            ):
                continue
            if (
                (not grid.georeferenced or not self.reference_grid.georeferenced)
                and (grid.width, grid.height)
                == (self.reference_grid.width, self.reference_grid.height)
                and asset.alignment_evidence
            ):
                continue
            raise AlignmentError(
                f"Scene {self.scene_id} asset {name!r} is not proven aligned with "
                f"reference grid {self.reference_grid.grid_id!r}; matching dimensions "
                "alone are insufficient"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "scene_id": self.scene_id,
            "schema_version": self.schema_version,
            "reference_grid": self.reference_grid.to_dict(),
            "label_ontology": self.label_ontology,
            "assets": {name: asset.to_dict() for name, asset in self.assets},
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SceneRecord:
        return cls(
            dataset_id=value["dataset_id"],
            scene_id=value["scene_id"],
            schema_version=int(value["schema_version"]),
            reference_grid=GridSpec.from_dict(value["reference_grid"]),
            label_ontology=value["label_ontology"],
            assets=tuple(
                (name, RasterAsset.from_dict(asset))
                for name, asset in value["assets"].items()
            ),
            provenance=tuple(sorted(value.get("provenance", {}).items())),
        )


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """A crop expressed in pixel coordinates and an explicit edge policy."""

    row: int
    col: int
    height: int
    width: int
    padding: str = "error"

    def __post_init__(self) -> None:
        if self.row < 0 or self.col < 0:
            raise ValueError("Window row and column must be non-negative")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("Window dimensions must be positive")
        if self.padding not in {"error", "constant"}:
            raise ValueError("padding must be 'error' or 'constant'")

    def validate_for(self, grid: GridSpec) -> None:
        if self.row >= grid.height or self.col >= grid.width:
            raise ValueError(f"Window {self} does not intersect grid bounds")
        if self.padding == "error" and (
            self.row + self.height > grid.height or self.col + self.width > grid.width
        ):
            raise ValueError(
                f"Window {self} exceeds grid bounds {(grid.height, grid.width)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Immutable scene assignment with an explicit group-leakage check."""

    dataset_id: str
    split_id: str
    version: int
    train_scene_ids: tuple[str, ...]
    validation_scene_ids: tuple[str, ...]
    test_scene_ids: tuple[str, ...]
    spatial_groups: tuple[tuple[str, str], ...]
    label_assets: tuple[tuple[str, str], ...]
    spatial_scope: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.dataset_id.strip() or not self.split_id.strip():
            raise ValueError("dataset_id and split_id must not be empty")
        if self.version <= 0 or self.schema_version <= 0:
            raise ValueError("Split and schema versions must be positive")
        splits = {
            "train": self.train_scene_ids,
            "validation": self.validation_scene_ids,
            "test": self.test_scene_ids,
        }
        for name, scene_ids in splits.items():
            if not scene_ids:
                raise ValueError(f"Split {name!r} must not be empty")
            if any(not scene_id.strip() for scene_id in scene_ids):
                raise ValueError(f"Split {name!r} contains an empty scene ID")
            if len(scene_ids) != len(set(scene_ids)):
                raise ValueError(f"Split {name!r} contains duplicate scene IDs")
        owners: dict[str, str] = {}
        for split_name, scene_ids in splits.items():
            for scene_id in scene_ids:
                previous = owners.setdefault(scene_id, split_name)
                if previous != split_name:
                    raise ValueError(
                        f"Scene {scene_id!r} occurs in both {previous} and {split_name}"
                    )
        groups = dict(self.spatial_groups)
        if len(groups) != len(self.spatial_groups):
            raise ValueError("A scene has multiple spatial-group assignments")
        if any(
            not scene_id.strip() or not group.strip()
            for scene_id, group in groups.items()
        ):
            raise ValueError("Spatial-group scene IDs and group IDs must not be empty")
        all_scenes = set(owners)
        if groups.keys() != all_scenes:
            missing = sorted(all_scenes - groups.keys())
            extra = sorted(groups.keys() - all_scenes)
            raise ValueError(
                f"Spatial-group coverage mismatch; missing={missing}, extra={extra}"
            )
        group_owners: dict[str, str] = {}
        for split_name, scene_ids in splits.items():
            for scene_id in scene_ids:
                group = groups[scene_id]
                previous = group_owners.setdefault(group, split_name)
                if previous != split_name:
                    raise ValueError(
                        f"Spatial group {group!r} leaks across "
                        f"{previous} and {split_name}"
                    )
        labels = dict(self.label_assets)
        if set(labels) != set(splits) or any(
            not value.strip() for value in labels.values()
        ):
            raise ValueError(
                "Each split must declare exactly one non-empty label asset role"
            )
        if not self.spatial_scope.strip():
            raise ValueError(
                "spatial_scope must state the unit used to prevent leakage"
            )

    @property
    def scene_ids(self) -> tuple[str, ...]:
        return self.train_scene_ids + self.validation_scene_ids + self.test_scene_ids

    def scenes_for(self, split_name: str) -> tuple[str, ...]:
        split_values = {
            "train": self.train_scene_ids,
            "validation": self.validation_scene_ids,
            "test": self.test_scene_ids,
        }
        try:
            return split_values[split_name]
        except KeyError as error:
            raise ValueError(f"Unknown split: {split_name}") from error

    def label_asset_for(self, split_name: str) -> str:
        try:
            return dict(self.label_assets)[split_name]
        except KeyError as error:
            raise ValueError(f"Unknown split: {split_name}") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "split_id": self.split_id,
            "version": self.version,
            "train_scene_ids": list(self.train_scene_ids),
            "validation_scene_ids": list(self.validation_scene_ids),
            "test_scene_ids": list(self.test_scene_ids),
            "spatial_groups": dict(self.spatial_groups),
            "label_assets": dict(self.label_assets),
            "spatial_scope": self.spatial_scope,
        }

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SplitManifest:
        return cls(
            schema_version=int(value.get("schema_version", 1)),
            dataset_id=value["dataset_id"],
            split_id=value["split_id"],
            version=int(value["version"]),
            train_scene_ids=tuple(value["train_scene_ids"]),
            validation_scene_ids=tuple(value["validation_scene_ids"]),
            test_scene_ids=tuple(value["test_scene_ids"]),
            spatial_groups=tuple(sorted(value["spatial_groups"].items())),
            label_assets=tuple(sorted(value["label_assets"].items())),
            spatial_scope=value["spatial_scope"],
        )


@dataclass(frozen=True, slots=True)
class PreprocessingSpec:
    """Versioned deterministic operations and train-fitted statistics."""

    preprocessing_id: str
    version: int
    label_ontology: str
    alignment_policy: str
    continuous_resampling: str
    categorical_resampling: str
    nodata_policy: str
    label_palette: tuple[tuple[int, int, int, int], ...] = ()
    ignore_label: int = 6
    fitted_statistics: tuple[tuple[str, float, float], ...] = ()
    fit_scene_ids: tuple[str, ...] = ()
    fit_split_hash: str | None = None

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.preprocessing_id,
                self.label_ontology,
                self.alignment_policy,
                self.continuous_resampling,
                self.categorical_resampling,
                self.nodata_policy,
            )
        ):
            raise ValueError("Preprocessing spec identifiers and policies are required")
        if self.version <= 0:
            raise ValueError("Preprocessing version must be positive")
        if self.ignore_label < 0:
            raise ValueError("ignore_label must be non-negative")
        colors = [(red, green, blue) for red, green, blue, _ in self.label_palette]
        class_ids = [class_id for _, _, _, class_id in self.label_palette]
        if len(colors) != len(set(colors)) or len(class_ids) != len(set(class_ids)):
            raise ValueError("Label palette colors and class IDs must be unique")
        if any(value < 0 or value > 255 for color in colors for value in color):
            raise ValueError("Palette RGB values must be in [0, 255]")
        if any(class_id < 0 for class_id in class_ids):
            raise ValueError("Palette class IDs must be non-negative")
        channels = [item[0] for item in self.fitted_statistics]
        if len(channels) != len(set(channels)):
            raise ValueError("Fitted statistics must have unique channel names")
        for channel, mean, std in self.fitted_statistics:
            if (
                not channel.strip()
                or not math.isfinite(mean)
                or not math.isfinite(std)
                or std <= 0
            ):
                raise ValueError(
                    "Fitted channel means must be finite and std values positive"
                )
        if len(self.fit_scene_ids) != len(set(self.fit_scene_ids)):
            raise ValueError("Fitted statistics contain duplicate scene IDs")
        if self.fitted_statistics:
            if not self.fit_scene_ids or not self.fit_split_hash:
                raise ValueError(
                    "Fitted statistics require training scene IDs and split hash"
                )
        elif self.fit_scene_ids or self.fit_split_hash:
            raise ValueError("Fit provenance cannot be set without fitted statistics")

    def validate_fit_scope(self, split: SplitManifest) -> None:
        if not self.fitted_statistics:
            return
        if self.fit_split_hash != split.content_hash:
            raise ValueError(
                "Fitted statistics were computed for a different split manifest"
            )
        if not set(self.fit_scene_ids) <= set(split.train_scene_ids):
            raise ValueError("Fitted statistics include non-training scenes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "preprocessing_id": self.preprocessing_id,
            "version": self.version,
            "label_ontology": self.label_ontology,
            "alignment_policy": self.alignment_policy,
            "continuous_resampling": self.continuous_resampling,
            "categorical_resampling": self.categorical_resampling,
            "nodata_policy": self.nodata_policy,
            "label_palette": [list(item) for item in self.label_palette],
            "ignore_label": self.ignore_label,
            "fitted_statistics": {
                channel: {"mean": mean, "std": std}
                for channel, mean, std in self.fitted_statistics
            },
            "fit_scene_ids": list(self.fit_scene_ids),
            "fit_split_hash": self.fit_split_hash,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PreprocessingSpec:
        statistics = value.get("fitted_statistics", {})
        return cls(
            preprocessing_id=value["preprocessing_id"],
            version=int(value["version"]),
            label_ontology=value["label_ontology"],
            alignment_policy=value["alignment_policy"],
            continuous_resampling=value["continuous_resampling"],
            categorical_resampling=value["categorical_resampling"],
            nodata_policy=value["nodata_policy"],
            label_palette=tuple(
                tuple(int(channel) for channel in row)
                for row in value.get("label_palette", ())
            ),
            ignore_label=int(value.get("ignore_label", 6)),
            fitted_statistics=tuple(
                sorted(
                    (
                        channel,
                        float(statistic["mean"]),
                        float(statistic["std"]),
                    )
                    for channel, statistic in statistics.items()
                )
            ),
            fit_scene_ids=tuple(value.get("fit_scene_ids", ())),
            fit_split_hash=value.get("fit_split_hash"),
        )

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class NormalizationRule:
    """Normalization for one named model input channel."""

    channel: str
    method: str
    scale: float | None = None
    mean: float | None = None
    std: float | None = None

    def __post_init__(self) -> None:
        if not self.channel.strip():
            raise ValueError("Normalization channel must not be empty")
        if self.method not in {"identity", "scale", "zscore", "fixed_standardize"}:
            raise ValueError(f"Unsupported normalization method: {self.method}")
        if self.method == "scale" and (self.scale is None or self.scale <= 0):
            raise ValueError("scale normalization requires a positive scale")
        if self.scale is not None and not math.isfinite(self.scale):
            raise ValueError("Normalization scale must be finite")
        if self.method == "fixed_standardize" and (
            self.mean is None
            or self.std is None
            or not math.isfinite(self.std)
            or self.std <= 0
        ):
            raise ValueError(
                "fixed_standardize requires a finite mean and positive std"
            )
        if self.mean is not None and not math.isfinite(self.mean):
            raise ValueError("Normalization mean must be finite")
        if self.method != "scale" and self.scale is not None:
            raise ValueError("scale is only valid for scale normalization")
        if self.method != "fixed_standardize" and (
            self.mean is not None or self.std is not None
        ):
            raise ValueError("mean/std are only valid for fixed_standardize")

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "method": self.method,
            "scale": self.scale,
            "mean": self.mean,
            "std": self.std,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> NormalizationRule:
        return cls(
            channel=value["channel"],
            method=value["method"],
            scale=float(value["scale"]) if value.get("scale") is not None else None,
            mean=float(value["mean"]) if value.get("mean") is not None else None,
            std=float(value["std"]) if value.get("std") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class ModelInputSpec:
    """Named source channels and model-specific tensor semantics."""

    model_id: str
    version: int
    channels: tuple[str, ...]
    channel_sources: tuple[tuple[str, str, str], ...]
    normalization: tuple[NormalizationRule, ...]
    layout: str = "CHW"
    dtype: str = "float32"
    pixel_semantics: str = ""

    def __post_init__(self) -> None:
        if not self.model_id.strip() or self.version <= 0:
            raise ValueError("Model input ID and positive version are required")
        if not self.channels or any(not name.strip() for name in self.channels):
            raise ValueError("Model input channels must be named")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("Model input channels must be unique")
        sources = {
            channel: (asset, band) for channel, asset, band in self.channel_sources
        }
        if len(sources) != len(self.channel_sources) or sources.keys() != set(
            self.channels
        ):
            raise ValueError(
                "Every input channel must map to exactly one asset and band"
            )
        if any(
            not asset.strip() or not band.strip() for asset, band in sources.values()
        ):
            raise ValueError("Channel source asset and band names must not be empty")
        rules = {rule.channel: rule for rule in self.normalization}
        if len(rules) != len(self.normalization) or rules.keys() != set(self.channels):
            raise ValueError(
                "Every input channel must have exactly one normalization rule"
            )
        if self.layout not in {"CHW", "HWC"}:
            raise ValueError("layout must be 'CHW' or 'HWC'")
        if self.dtype != "float32":
            raise ValueError(
                "Only float32 model inputs are supported by this data layer"
            )
        if not self.pixel_semantics.strip():
            raise ValueError("Model input pixel semantics must be documented")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "channels": list(self.channels),
            "channel_sources": [list(source) for source in self.channel_sources],
            "normalization": [rule.to_dict() for rule in self.normalization],
            "layout": self.layout,
            "dtype": self.dtype,
            "pixel_semantics": self.pixel_semantics,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelInputSpec:
        return cls(
            model_id=value["model_id"],
            version=int(value["version"]),
            channels=tuple(value["channels"]),
            channel_sources=tuple(tuple(source) for source in value["channel_sources"]),
            normalization=tuple(
                NormalizationRule.from_dict(item) for item in value["normalization"]
            ),
            layout=value.get("layout", "CHW"),
            dtype=value.get("dtype", "float32"),
            pixel_semantics=value.get("pixel_semantics", ""),
        )


@dataclass(frozen=True, slots=True)
class Sample:
    """One model-ready image/label pair with explicit window provenance."""

    image: Any
    labels: Any
    valid_mask: Any
    scene_id: str
    window: WindowSpec
    layout: str = "CHW"

    def __post_init__(self) -> None:
        import numpy as np

        image = np.asarray(self.image)
        labels = np.asarray(self.labels)
        valid = np.asarray(self.valid_mask)
        if image.ndim != 3:
            raise ValueError(
                f"Sample image must be a 3D channel-first/last array, got {image.shape}"
            )
        if self.layout not in {"CHW", "HWC"}:
            raise ValueError("Sample layout must be 'CHW' or 'HWC'")
        spatial_shape = image.shape[-2:] if self.layout == "CHW" else image.shape[:2]
        if labels.ndim != 2 or valid.shape != labels.shape:
            raise ValueError("Sample labels and valid_mask must be aligned 2D arrays")
        if labels.shape != spatial_shape:
            raise ValueError("Sample image and label spatial dimensions do not match")
        if not self.scene_id.strip():
            raise ValueError("Sample scene_id must not be empty")
