"""Prepare immutable RGBIR sources and scoped, checksummed scene caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import zipfile
import zlib
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile

from deepnet.data.bundles import load_bundle
from deepnet.data.contracts import WindowSpec, canonical_hash
from deepnet.data.raster import sha256_file
from deepnet.data.sampling import grid_windows
from deepnet.potsdam_data import verify_dsm
from deepnet.training_utils import write_json

LOGGER = logging.getLogger(__name__)
SOURCE = "https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx"


def snapshot_rgbir(archive: Path, irrg: Path, output: Path) -> dict:
    """Extract the official four-band archive into a new, audited snapshot."""
    if (output / "rgbir_manifest.json").exists():
        raise FileExistsError("RGBIR snapshot is already finalized")
    output.mkdir(parents=True, exist_ok=True)
    outer_hash = sha256_file(archive)
    nested = output / "4_Ortho_RGBIR.zip"
    with zipfile.ZipFile(archive) as source:
        names = [n for n in source.namelist() if n.endswith("/4_Ortho_RGBIR.zip")]
        if len(names) != 1:
            raise ValueError("Expected one official RGBIR archive")
        if nested.exists():
            with source.open(names[0]) as stream:
                expected = hashlib.file_digest(stream, "sha256").hexdigest()
            if sha256_file(nested) != expected:
                raise ValueError("Existing nested RGBIR archive differs")
        else:
            with source.open(names[0]) as stream, nested.open("xb") as target:
                shutil.copyfileobj(stream, target, length=8 * 1024 * 1024)
    manifest = {
        "schema_version": 1,
        "source": SOURCE,
        "outer_sha256": outer_hash,
        "archive_sha256": sha256_file(nested),
        "bands": ["red", "green", "blue", "ir"],
        "scenes": {},
    }
    with zipfile.ZipFile(nested) as source:
        for item in source.infolist():
            relative = Path(item.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe RGBIR archive member")
            if item.is_dir():
                continue
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                crc = 0
                with target.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        crc = zlib.crc32(chunk, crc)
                if target.stat().st_size != item.file_size or crc != item.CRC:
                    raise ValueError(f"Existing extracted source differs: {relative}")
                continue
            with source.open(item) as stream, target.open("xb") as dst:
                shutil.copyfileobj(stream, dst, length=8 * 1024 * 1024)
    paths = sorted(output.rglob("top_potsdam_*_RGBIR.tif"))
    if len(paths) != 38:
        raise ValueError(f"Expected 38 RGBIR rasters, found {len(paths)}")
    with (
        zipfile.ZipFile(archive) as outer,
        outer.open("Potsdam/2_Ortho_RGB.zip") as stream,
    ):
        manifest["rgb_reference_archive_sha256"] = hashlib.file_digest(
            stream, "sha256"
        ).hexdigest()
    with (
        zipfile.ZipFile(archive) as outer,
        outer.open("Potsdam/2_Ortho_RGB.zip") as stream,
        zipfile.ZipFile(stream) as visible,
    ):
        _verify_rgbir(paths, irrg, output, manifest, visible)
    manifest["snapshot_hash"] = canonical_hash(manifest)
    write_json(output / "rgbir_manifest.json", manifest)
    return manifest


def _verify_rgbir(paths, irrg, output, manifest, visible):
    for path in paths:
        scene = path.stem.removeprefix("top_potsdam_").removesuffix("_RGBIR")
        reference = irrg / f"top_potsdam_{scene}_IRRG.tif"
        rgb_bytes = visible.read(f"2_Ortho_RGB/top_potsdam_{scene}_RGB.tif")
        with (
            MemoryFile(rgb_bytes) as memory,
            memory.open() as rgb,
            rasterio.open(path) as image,
            rasterio.open(reference) as original,
        ):
            if (
                image.count != 4
                or image.dtypes != ("uint8",) * 4
                or image.shape != original.shape
                or image.crs != original.crs
                or rgb.crs != image.crs
                or rgb.shape != image.shape
                or not np.allclose(rgb.transform, image.transform, rtol=0, atol=1e-8)
                or not np.allclose(
                    image.transform, original.transform, rtol=0, atol=1e-8
                )
            ):
                raise ValueError(f"RGBIR grid/band mismatch: {scene}")
            compared = different = maximum = 0
            total_difference = 0
            coverage = 0
            mapped_difference = np.zeros(2, dtype=np.int64)
            for _, window in image.block_windows(1):
                all_bands = image.read(window=window)
                np.testing.assert_array_equal(all_bands[:3], rgb.read(window=window))
                actual = all_bands[[3, 0, 1]]
                expected = original.read(window=window)
                valid = original.read_masks(window=window).all(axis=0)
                available = image.read_masks(window=window).all(axis=0)
                if np.any(valid & ~available):
                    raise ValueError(f"RGBIR loses existing optical coverage: {scene}")
                delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
                delta = delta[:, valid]
                compared += delta.size
                different += int(np.count_nonzero(delta))
                total_difference += int(delta.sum())
                maximum = max(maximum, int(delta.max(initial=0)))
                for idx, mapping in enumerate(([3, 0, 1], [3, 1, 2])):
                    mapped_difference[idx] += np.count_nonzero(
                        all_bands[mapping][:, valid] != expected[:, valid]
                    )
                coverage += int(available.sum())
            matches = np.flatnonzero(mapped_difference == 0)
            if len(matches) != 1:
                raise ValueError(
                    f"Unexplained historical IRRG channel mismatch: {scene}"
                )
            mapping = ([4, 1, 2], [4, 2, 3])[int(matches[0])]
            manifest["scenes"][scene] = {
                "path": path.relative_to(output).as_posix(),
                "sha256": sha256_file(path),
                "irrg_sha256": sha256_file(reference),
                "rgb_reference_sha256": hashlib.sha256(rgb_bytes).hexdigest(),
                "rgb_reference_exact": True,
                "historical_irrg_rgbir_band_numbers": mapping,
                "dtype": image.dtypes[0],
                "shape": list(image.shape),
                "crs": str(image.crs) if image.crs else None,
                "affine": list(image.transform)[:6],
                "extent": list(image.bounds),
                "coverage": coverage / (image.height * image.width),
                "agreement": {
                    "values": compared,
                    "different": different,
                    "max_abs_dn": maximum,
                    "mean_abs_dn": total_difference / max(compared, 1),
                },
            }
        LOGGER.info("RGBIR verified %s (%d/38)", scene, len(manifest["scenes"]))


def _array(root: Path, name: str, dtype: str, shape: tuple):
    return np.lib.format.open_memmap(
        root / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
    )


def build_cache(bundle_path: Path, rgbir: Path, dsm: Path, output: Path) -> dict:
    """Create a new role-scoped cache without changing a source bundle or mask."""
    bundle = load_bundle(bundle_path)
    rgb = json.loads((rgbir / "rgbir_manifest.json").read_text())
    rgb_payload = {k: v for k, v in rgb.items() if k != "snapshot_hash"}
    if canonical_hash(rgb_payload) != rgb["snapshot_hash"]:
        raise ValueError("RGBIR snapshot hash mismatch")
    selected = bundle.manifest["included_scene_ids"]
    from deepnet.potsdam_study import _split

    parent = _split()
    is_refit = bundle.split.refit_parent_hash is not None
    if is_refit:
        if (
            bundle.split.refit_parent_hash != parent.content_hash
            or bundle.split.train_scene_ids
            != parent.train_scene_ids + parent.validation_scene_ids
            or bundle.manifest["included_splits"] != ["train"]
        ):
            raise ValueError("Refit cache must be the exact locked 24-scene union")
    elif bundle.split.content_hash != parent.content_hash:
        raise ValueError("Cache requires the historical Potsdam selection split")
    dsm_manifest = verify_dsm(dsm, parent, selected)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "split.json", bundle.split.to_dict())
    write_json(output / "preprocessing.json", bundle.preprocessing.to_dict())
    write_json(output / "catalog.json", [scene.to_dict() for scene in bundle.scenes])
    records = {}
    for role in bundle.manifest["included_splits"]:
        with bundle.scene_access(role) as access:
            for scene in bundle.split.scenes_for(role):
                LOGGER.info("Caching %s/%s", role, scene)
                entry = rgb["scenes"][scene]
                source_path = rgbir / entry["path"]
                if sha256_file(source_path) != entry["sha256"]:
                    raise ValueError(f"RGBIR checksum mismatch: {scene}")
                root = output / scene
                root.mkdir()
                height, width = access.scene_shapes[scene]
                shape = (height, width)
                arrays = {
                    "optical": _array(root, "optical", "uint8", (4, *shape)),
                    "relief": _array(root, "relief", "float32", shape),
                    "dsm_available": _array(root, "dsm_available", "uint8", shape),
                    "labels": _array(root, "labels", "uint8", shape),
                    "valid": _array(root, "valid", "uint8", shape),
                }
                counts = np.zeros(6, dtype=np.int64)
                valid_pixels = 0
                with (
                    rasterio.open(source_path) as image,
                    rasterio.open(dsm / f"{scene}.tif") as elevation,
                ):
                    for block in grid_windows(height, width, (512, 512)):
                        h, w = min(512, height - block.row), min(512, width - block.col)
                        window = WindowSpec(block.row, block.col, h, w)
                        original = access.read_sample(scene, window)
                        roi = rasterio.windows.Window(block.col, block.row, w, h)
                        optical = image.read(window=roi)
                        available = image.read_masks(window=roi).all(axis=0)
                        if np.any(original.valid_mask & ~available):
                            raise ValueError(f"RGBIR changes scoring mask: {scene}")
                        mapping = (
                            np.array(entry["historical_irrg_rgbir_band_numbers"]) - 1
                        )
                        np.testing.assert_array_equal(
                            optical[mapping][:, original.valid_mask],
                            np.rint(
                                original.image[:3, original.valid_mask] * 255
                            ).astype(np.uint8),
                        )
                        extra = elevation.read(window=roi)
                        if (
                            not np.isfinite(extra).all()
                            or not np.isin(extra[1], (0, 1)).all()
                        ):
                            raise ValueError(f"Invalid DSM sidecar: {scene}")
                        region = np.s_[
                            block.row : block.row + h, block.col : block.col + w
                        ]
                        arrays["optical"][(slice(None), *region)] = optical
                        arrays["relief"][region] = np.where(
                            extra[1] > 0,
                            np.clip(extra[0] / dsm_manifest["scale"], -1, 1),
                            0,
                        )
                        arrays["dsm_available"][region] = extra[1].astype(np.uint8)
                        arrays["labels"][region] = original.labels.astype(np.uint8)
                        arrays["valid"][region] = original.valid_mask.astype(np.uint8)
                        scored = original.valid_mask & (original.labels < 6)
                        counts += np.bincount(original.labels[scored], minlength=6)
                        valid_pixels += int(scored.sum())
                for array in arrays.values():
                    array.flush()
                arrays.clear()
                records[scene] = {
                    "role": role,
                    "shape": list(shape),
                    "label_asset": bundle.split.label_asset_for(role),
                    "source_rgbir_sha256": entry["sha256"],
                    "source_dsm_sha256": dsm_manifest["scenes"][scene]["sha256"],
                    "class_support": counts.tolist(),
                    "valid_pixels": valid_pixels,
                }
    files = [
        {
            "path": p.relative_to(output).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in sorted(output.rglob("*"))
        if p.is_file()
    ]
    manifest = {
        "schema_version": 1,
        "kind": "potsdam-rgbir-memmap-v2",
        "dataset_id": "isprs-potsdam",
        "source_bundle_sha256": bundle.manifest["bundle_sha256"],
        "selection_split_hash": parent.content_hash,
        "split_hash": bundle.split.content_hash,
        "included_splits": bundle.manifest["included_splits"],
        "included_scene_ids": selected,
        "rgbir_snapshot_hash": rgb["snapshot_hash"],
        "dsm_manifest_sha256": sha256_file(dsm / "manifest.json"),
        "feature_recipe": dsm_manifest["recipe"],
        "relief_scale": dsm_manifest["scale"],
        "scenes": records,
        "files": files,
    }
    manifest["cache_hash"] = canonical_hash(manifest)
    write_json(output / "cache_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--archive", type=Path, required=True)
    snapshot.add_argument("--irrg", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    cache = commands.add_parser("cache")
    cache.add_argument("--bundle", type=Path, required=True)
    cache.add_argument("--rgbir", type=Path, required=True)
    cache.add_argument("--dsm", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.command == "snapshot":
        snapshot_rgbir(args.archive, args.irrg, args.output)
    else:
        build_cache(args.bundle, args.rgbir, args.dsm, args.output)


if __name__ == "__main__":
    main()
