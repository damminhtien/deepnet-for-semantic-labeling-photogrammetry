"""Dataset descriptors and immutable, per-scene source catalogs."""

from __future__ import annotations

import json
import re
import sysconfig
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from .contracts import (
    ModelInputSpec,
    PreprocessingSpec,
    RasterAsset,
    SceneRecord,
    SplitManifest,
    canonical_hash,
)
from .raster import inspect_raster


def _config_dir(name: str) -> Path:
    repository_path = Path(__file__).resolve().parents[2] / "configs" / name
    if repository_path.is_dir():
        return repository_path
    return (
        Path(sysconfig.get_path("data"))
        / "share"
        / "deepnet-satellite-labeling"
        / "configs"
        / name
    )


DEFAULT_DATASET_CONFIG_DIR = _config_dir("datasets")
DEFAULT_SPLIT_CONFIG_DIR = _config_dir("splits")
DEFAULT_PREPROCESSING_CONFIG_DIR = _config_dir("preprocessing")
DEFAULT_MODEL_INPUT_CONFIG_DIR = _config_dir("model-inputs")


@dataclass(frozen=True, slots=True)
class DatasetDescriptor:
    dataset_id: str
    provider: str
    source_uri: str | None
    license_id: str | None
    discovery_asset: str
    discovery_glob: str
    discovery_regex: str
    reference_asset: str
    assets: tuple[tuple[str, dict[str, Any]], ...]
    label_ontology: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.dataset_id.strip() or not self.provider.strip():
            raise ValueError("Dataset ID and provider must not be empty")
        if self.schema_version <= 0:
            raise ValueError("Dataset schema version must be positive")
        configs = self.asset_configs()
        if not configs or self.reference_asset not in configs:
            raise ValueError("Dataset descriptor must configure its reference asset")
        if self.discovery_asset not in configs:
            raise ValueError("Dataset descriptor discovery asset is not configured")
        if not self.discovery_glob.strip() or not self.label_ontology.strip():
            raise ValueError(
                "Dataset discovery pattern and label ontology are required"
            )
        try:
            expression = re.compile(self.discovery_regex)
        except re.error as error:
            raise ValueError("Dataset discovery regex is invalid") from error
        if "scene_id" not in expression.groupindex:
            raise ValueError(
                "Dataset discovery regex must define a named scene_id group"
            )
        for name, config in configs.items():
            if not name.strip() or not isinstance(config.get("path"), str):
                raise ValueError(f"Dataset asset {name!r} must declare a relative path")
            if not config.get("bands") or not all(
                isinstance(band, str) and band.strip() for band in config["bands"]
            ):
                raise ValueError(f"Dataset asset {name!r} must declare named bands")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DatasetDescriptor:
        return cls(
            dataset_id=value["dataset_id"],
            provider=value["provider"],
            source_uri=value.get("source_uri"),
            license_id=value.get("license_id"),
            discovery_asset=value["discovery_asset"],
            discovery_glob=value["discovery_glob"],
            discovery_regex=value["discovery_regex"],
            reference_asset=value["reference_asset"],
            assets=tuple(sorted(value["assets"].items())),
            label_ontology=value["label_ontology"],
            schema_version=int(value.get("schema_version", 1)),
        )

    def asset_configs(self) -> dict[str, dict[str, Any]]:
        return dict(self.assets)


class DatasetCatalog:
    """Load stable dataset IDs from JSON descriptors, never trainer-local paths."""

    def __init__(self, descriptors: dict[str, DatasetDescriptor]) -> None:
        if not descriptors:
            raise ValueError("Dataset catalog must contain at least one descriptor")
        if set(descriptors) != {value.dataset_id for value in descriptors.values()}:
            raise ValueError("Dataset catalog keys must match descriptor dataset IDs")
        self._descriptors = dict(descriptors)

    @classmethod
    def load(cls, config_dir: Path = DEFAULT_DATASET_CONFIG_DIR) -> DatasetCatalog:
        descriptors = {}
        for path in sorted(Path(config_dir).glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            descriptor = DatasetDescriptor.from_dict(value)
            if descriptor.dataset_id in descriptors:
                raise ValueError(f"Duplicate dataset ID: {descriptor.dataset_id}")
            descriptors[descriptor.dataset_id] = descriptor
        return cls(descriptors)

    def get(self, dataset_id: str) -> DatasetDescriptor:
        try:
            return self._descriptors[dataset_id]
        except KeyError as error:
            raise KeyError(
                f"Unknown dataset ID {dataset_id!r}; "
                f"available: {sorted(self._descriptors)}"
            ) from error


def load_split_manifest(path: Path) -> SplitManifest:
    return SplitManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_preprocessing_spec(
    preprocessing_id: str,
    config_dir: Path = DEFAULT_PREPROCESSING_CONFIG_DIR,
) -> PreprocessingSpec:

    path = Path(config_dir) / f"{preprocessing_id}.json"
    return PreprocessingSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_model_input_spec(
    model_input_id: str,
    config_dir: Path = DEFAULT_MODEL_INPUT_CONFIG_DIR,
) -> ModelInputSpec:

    path = Path(config_dir) / f"{model_input_id}.json"
    return ModelInputSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_scene_catalog(
    path: Path, *, expected_dataset_id: str | None = None
) -> tuple[SceneRecord, ...]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    dataset_id = value["dataset_id"]
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported scene catalog schema version")
    if expected_dataset_id is not None and dataset_id != expected_dataset_id:
        raise ValueError(
            f"Catalog has dataset_id={dataset_id!r}; expected {expected_dataset_id!r}"
        )
    records = tuple(SceneRecord.from_dict(record) for record in value["scenes"])
    scene_ids = [record.scene_id for record in records]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("Scene catalog contains duplicate scene IDs")
    if any(record.dataset_id != dataset_id for record in records):
        raise ValueError("Scene catalog mixes dataset IDs")
    expected_hash = value.get("catalog_content_hash")
    actual_hash = canonical_hash([record.to_dict() for record in records])
    if expected_hash is not None and expected_hash != actual_hash:
        raise ValueError("Scene catalog content hash does not match its records")
    return records


def _scene_format_values(scene_id: str) -> dict[str, Any]:
    parts = scene_id.split("_")
    if len(parts) == 1:
        return {"scene_id": scene_id, "area": int(parts[0])}
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return {
            "scene_id": scene_id,
            "row": int(parts[0]),
            "column": int(parts[1]),
        }
    raise ValueError(f"Unsupported scene ID format: {scene_id}")


def _safe_source_path(data_root: Path, relative_uri: str) -> Path:
    relative = PurePosixPath(relative_uri)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Dataset descriptor has an unsafe asset URI: {relative_uri}")
    path = (data_root / Path(*relative.parts)).resolve()
    root = data_root.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Dataset path escapes its configured root: {relative_uri}")
    return path


def build_scene_catalog(
    dataset_id: str,
    data_root: Path,
    *,
    dataset_catalog: DatasetCatalog | None = None,
    hash_assets: bool = False,
) -> dict[str, Any]:
    """Inspect sources without modifying them; optionally hash at ingestion time."""
    datasets = dataset_catalog or DatasetCatalog.load()
    descriptor = datasets.get(dataset_id)
    data_root = Path(data_root).expanduser().resolve()
    discovery_pattern = _safe_source_path(data_root, descriptor.discovery_glob)
    scene_pattern = re.compile(descriptor.discovery_regex)
    discovered: dict[str, Path] = {}
    for path in sorted(data_root.glob(str(discovery_pattern.relative_to(data_root)))):
        match = scene_pattern.fullmatch(path.name)
        if match is None:
            continue
        scene_id = match.group("scene_id")
        if scene_id in discovered:
            raise ValueError(f"Duplicate source files for scene {scene_id}: {path}")
        discovered[scene_id] = path
    if not discovered:
        raise FileNotFoundError(
            f"No scenes found for {dataset_id} under {data_root} using "
            f"{descriptor.discovery_glob}"
        )

    asset_configs = descriptor.asset_configs()
    if descriptor.reference_asset not in asset_configs:
        raise ValueError("Dataset descriptor's reference asset is not configured")
    records: list[SceneRecord] = []
    missing: dict[str, list[str]] = {}
    for scene_id, reference_path in discovered.items():
        values = _scene_format_values(scene_id)
        reference_config = asset_configs[descriptor.reference_asset]
        reference_uri = reference_config["path"].format(**values)
        if _safe_source_path(data_root, reference_uri) != reference_path.resolve():
            raise ValueError(
                "Discovery path differs from the configured reference path "
                f"for {scene_id}"
            )
        pixel_grid_id = f"{dataset_id}:{scene_id}:pixel-local"
        reference_asset = inspect_raster(
            reference_path,
            asset_id=descriptor.reference_asset,
            uri=reference_uri,
            bands=tuple(reference_config["bands"]),
            grid_id=pixel_grid_id,
            alignment_evidence=reference_config.get("alignment_evidence"),
            hash_file=hash_assets,
        )
        assets: list[tuple[str, RasterAsset]] = [
            (descriptor.reference_asset, reference_asset)
        ]
        absent = []
        for asset_name, config in asset_configs.items():
            if asset_name == descriptor.reference_asset:
                continue
            uri = config["path"].format(**values)
            path = _safe_source_path(data_root, uri)
            if not path.is_file():
                absent.append(asset_name)
                continue
            asset = inspect_raster(
                path,
                asset_id=asset_name,
                uri=uri,
                bands=tuple(config["bands"]),
                grid_id=f"{dataset_id}:{scene_id}:{asset_name}:pixel-local",
                alignment_evidence=config.get("alignment_evidence"),
                hash_file=hash_assets,
            )
            if (
                asset.alignment_evidence is None
                and (asset.grid.width, asset.grid.height)
                == (reference_asset.grid.width, reference_asset.grid.height)
                and asset.grid.affine is not None
                and reference_asset.grid.affine is not None
                and asset.grid.affine == reference_asset.grid.affine
            ):
                asset = replace(
                    asset,
                    alignment_evidence=(
                        "Asset world-file affine exactly matches the scene "
                        "reference raster; the affine was read from the source "
                        "files during catalog creation."
                    ),
                )
            assets.append((asset_name, asset))
        if absent:
            missing[scene_id] = sorted(absent)
        records.append(
            SceneRecord(
                dataset_id=dataset_id,
                scene_id=scene_id,
                schema_version=1,
                reference_grid=reference_asset.grid,
                label_ontology=descriptor.label_ontology,
                assets=tuple(sorted(assets)),
                provenance=tuple(
                    sorted(
                        {
                            "provider": descriptor.provider,
                            "source_uri": descriptor.source_uri or "",
                            "license_id": descriptor.license_id or "unverified",
                        }.items()
                    )
                ),
            )
        )

    records.sort(key=lambda record: record.scene_id)
    record_dicts = [record.to_dict() for record in records]
    content_hash = canonical_hash(record_dicts)
    return {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "provider": descriptor.provider,
        "source_uri": descriptor.source_uri,
        "license_id": descriptor.license_id,
        "label_ontology": descriptor.label_ontology,
        "hash_status": "complete" if hash_assets else "not_computed",
        "source_version": content_hash if hash_assets else None,
        "catalog_content_hash": content_hash,
        "scene_count": len(records),
        "missing_assets_by_scene": missing,
        "scenes": record_dicts,
    }


def write_scene_catalog(value: dict[str, Any], path: Path) -> None:
    """Atomically write a metadata catalog; the raw source tree remains untouched."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_scene_catalog(
    scenes: tuple[SceneRecord, ...],
    *,
    split: SplitManifest | None = None,
    required_asset_names: tuple[str, ...] = (),
    require_hashes: bool = False,
) -> None:
    if not scenes:
        raise ValueError("Scene catalog is empty")
    by_id = {record.scene_id: record for record in scenes}
    if len(by_id) != len(scenes):
        raise ValueError("Scene catalog contains duplicate scene IDs")
    if split is not None:
        if {record.dataset_id for record in scenes} != {split.dataset_id}:
            raise ValueError("Split and scene catalog dataset IDs differ")
        missing_scenes = set(split.scene_ids) - by_id.keys()
        if missing_scenes:
            raise ValueError(
                f"Split references unknown scenes: {sorted(missing_scenes)}"
            )
    for record in scenes:
        for name in required_asset_names:
            if name not in record.asset_map():
                raise ValueError(
                    f"Scene {record.scene_id} is missing required asset {name!r}"
                )
        if require_hashes:
            missing_hashes = [
                name for name, asset in record.assets if asset.sha256 is None
            ]
            if missing_hashes:
                raise ValueError(
                    f"Scene {record.scene_id} has unhashed assets: {missing_hashes}"
                )
