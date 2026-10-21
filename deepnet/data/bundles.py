"""Portable, checksummed exports of one model view over a split manifest."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
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
from .raster import sha256_file


def _safe_file(root: Path, relative_uri: str) -> Path:
    relative = PurePosixPath(relative_uri)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe bundle path: {relative_uri}")
    resolved_root = root.resolve()
    path = (resolved_root / Path(*relative.parts)).resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise ValueError(f"Bundle path escapes its root: {relative_uri}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _source_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def export_model_bundle(
    scenes: Iterable[SceneRecord],
    data_root: Path,
    split: SplitManifest,
    preprocessing: PreprocessingSpec,
    model_input: ModelInputSpec,
    output_dir: Path,
    *,
    source_revision: str | None = None,
) -> dict[str, Any]:
    """Copy only the view's assets and split-specific labels into an atomic bundle.

    The source tree is read-only from this operation. Files are copied, not linked,
    so editing a bundle cannot mutate the original dataset through a hard link.
    """
    scene_records = tuple(scenes)
    by_id = {scene.scene_id: scene for scene in scene_records}
    if len(by_id) != len(scene_records):
        raise ValueError("Scene catalog contains duplicate scene IDs")
    if not scene_records or {scene.dataset_id for scene in scene_records} != {
        split.dataset_id
    }:
        raise ValueError("Scene catalog and split manifest dataset IDs differ")
    if set(split.scene_ids) - by_id.keys():
        raise ValueError("Split manifest references scenes missing from the catalog")
    preprocessing.validate_fit_scope(split)

    channel_assets = {asset_name for _, asset_name, _ in model_input.channel_sources}
    output_dir = Path(output_dir).expanduser().resolve()
    staging_dir = output_dir.with_name(output_dir.name + ".tmp")
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError(
            f"Bundle output or staging directory already exists: {output_dir}"
        )
    root = Path(data_root).expanduser().resolve()
    if (
        output_dir == root
        or output_dir.is_relative_to(root)
        or root.is_relative_to(output_dir)
    ):
        raise ValueError("Bundle output and source data root must not overlap")
    staging_dir.mkdir(parents=True)
    try:
        selected_records = []
        copied_paths: set[str] = set()
        for split_name in ("train", "validation", "test"):
            label_asset_name = split.label_asset_for(split_name)
            for scene_id in split.scenes_for(split_name):
                scene = by_id[scene_id]
                required_assets = tuple(sorted(channel_assets | {label_asset_name}))
                scene.require_aligned(required_assets)
                selected_assets = []
                for asset_name in required_assets:
                    asset = scene.asset_map()[asset_name]
                    source_path = _safe_file(root, asset.uri)
                    relative_path = PurePosixPath(asset.uri)
                    bundle_path = staging_dir.joinpath(*relative_path.parts)
                    bundle_path.parent.mkdir(parents=True, exist_ok=True)
                    key = relative_path.as_posix()
                    if key not in copied_paths:
                        shutil.copy2(source_path, bundle_path)
                        copied_paths.add(key)
                    actual_hash = sha256_file(bundle_path)
                    if asset.sha256 is not None and actual_hash != asset.sha256:
                        raise RuntimeError(f"Source checksum mismatch for {asset.uri}")
                    selected_assets.append(
                        (
                            asset_name,
                            RasterAsset(
                                asset_id=asset.asset_id,
                                uri=key,
                                bands=asset.bands,
                                dtype=asset.dtype,
                                grid=asset.grid,
                                nodata=asset.nodata,
                                sha256=actual_hash,
                                alignment_evidence=asset.alignment_evidence,
                            ),
                        )
                    )
                selected_records.append(
                    SceneRecord(
                        dataset_id=scene.dataset_id,
                        scene_id=scene.scene_id,
                        schema_version=scene.schema_version,
                        reference_grid=scene.reference_grid,
                        label_ontology=scene.label_ontology,
                        assets=tuple(selected_assets),
                        provenance=scene.provenance,
                    )
                )

        selected_records.sort(key=lambda item: item.scene_id)
        catalog = {
            "schema_version": 1,
            "dataset_id": split.dataset_id,
            "scene_count": len(selected_records),
            "catalog_content_hash": canonical_hash(
                [record.to_dict() for record in selected_records]
            ),
            "scenes": [record.to_dict() for record in selected_records],
        }
        _write_json(staging_dir / "catalog.json", catalog)
        _write_json(staging_dir / "split.json", split.to_dict())
        _write_json(staging_dir / "preprocessing.json", preprocessing.to_dict())
        _write_json(staging_dir / "model_input.json", model_input.to_dict())

        file_paths = sorted(
            (path for path in staging_dir.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(staging_dir).as_posix(),
        )
        files = [_file_record(staging_dir, path) for path in file_paths]
        split_counts = {
            name: len(split.scenes_for(name))
            for name in ("train", "validation", "test")
        }
        manifest = {
            "schema_version": 1,
            "dataset_id": split.dataset_id,
            "split_id": split.split_id,
            "split_version": split.version,
            "split_hash": split.content_hash,
            "preprocessing_id": preprocessing.preprocessing_id,
            "preprocessing_version": preprocessing.version,
            "preprocessing_hash": preprocessing.content_hash,
            "model_input_id": model_input.model_id,
            "model_input_version": model_input.version,
            "model_input_hash": canonical_hash(model_input.to_dict()),
            "catalog_hash": catalog["catalog_content_hash"],
            "source_revision": source_revision or _source_revision(),
            "scene_ids": list(split.scene_ids),
            "scene_counts": split_counts,
            "files": files,
            "file_count": len(files),
            "total_bytes": sum(record["bytes"] for record in files),
            "bundle_sha256": canonical_hash(files),
        }
        _write_json(staging_dir / "bundle_manifest.json", manifest)
        os.replace(staging_dir, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def verify_bundle(bundle_root: Path) -> dict[str, Any]:
    """Verify every inventory path, byte count, digest, and bundle fingerprint."""
    root = Path(bundle_root).expanduser().resolve()
    manifest_path = _safe_file(root, "bundle_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Bundle has no file inventory")
    seen = set()
    for record in records:
        relative_path = record.get("path")
        if not isinstance(relative_path, str) or relative_path in seen:
            raise ValueError("Bundle inventory contains a missing or duplicate path")
        seen.add(relative_path)
        path = _safe_file(root, relative_path)
        if path.stat().st_size != record.get("bytes"):
            raise RuntimeError(f"Bundle file size mismatch: {relative_path}")
        if sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"Bundle checksum mismatch: {relative_path}")
    if canonical_hash(records) != manifest.get("bundle_sha256"):
        raise RuntimeError("Bundle fingerprint does not match its inventory")
    if manifest.get("file_count") != len(records):
        raise RuntimeError("Bundle file count does not match its inventory")
    return manifest


@dataclass(frozen=True, slots=True)
class BundleData:
    """Loaded catalog contracts and root for model-specific data access."""

    root: Path
    manifest: dict[str, Any]
    scenes: tuple[SceneRecord, ...]
    split: SplitManifest
    preprocessing: PreprocessingSpec
    model_input: ModelInputSpec

    def scene_access(self, split_name: str, *, verify_metadata: bool = True) -> Any:
        from .dataset import SceneAccess

        return SceneAccess(
            self.scenes,
            self.root,
            self.split,
            split_name,
            self.preprocessing,
            self.model_input,
            verify_metadata=verify_metadata,
        )


def load_bundle(bundle_root: Path, *, verify: bool = True) -> BundleData:
    """Load verified pipeline contracts from an exported portable bundle."""
    root = Path(bundle_root).expanduser().resolve()
    manifest = (
        verify_bundle(root)
        if verify
        else json.loads(
            _safe_file(root, "bundle_manifest.json").read_text(encoding="utf-8")
        )
    )
    catalog = json.loads(_safe_file(root, "catalog.json").read_text(encoding="utf-8"))
    split = SplitManifest.from_dict(
        json.loads(_safe_file(root, "split.json").read_text(encoding="utf-8"))
    )
    preprocessing = PreprocessingSpec.from_dict(
        json.loads(_safe_file(root, "preprocessing.json").read_text(encoding="utf-8"))
    )
    model_input = ModelInputSpec.from_dict(
        json.loads(_safe_file(root, "model_input.json").read_text(encoding="utf-8"))
    )
    scenes = tuple(SceneRecord.from_dict(value) for value in catalog["scenes"])
    calculated_catalog_hash = canonical_hash([scene.to_dict() for scene in scenes])
    if calculated_catalog_hash != catalog.get("catalog_content_hash"):
        raise RuntimeError(
            "Bundle catalog fingerprint does not match its scene records"
        )
    if calculated_catalog_hash != manifest.get("catalog_hash"):
        raise RuntimeError("Bundle catalog and top-level manifest fingerprints differ")
    contract_pairs = (
        ("dataset_id", split.dataset_id),
        ("split_id", split.split_id),
        ("split_version", split.version),
        ("split_hash", split.content_hash),
        ("preprocessing_id", preprocessing.preprocessing_id),
        ("preprocessing_version", preprocessing.version),
        ("preprocessing_hash", preprocessing.content_hash),
        ("model_input_id", model_input.model_id),
        ("model_input_version", model_input.version),
        ("model_input_hash", canonical_hash(model_input.to_dict())),
    )
    for key, expected in contract_pairs:
        if manifest.get(key) != expected:
            raise RuntimeError(f"Bundle {key} does not match its embedded contract")
    return BundleData(root, manifest, scenes, split, preprocessing, model_input)
