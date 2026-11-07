"""Data and deterministic patch recipes for the Potsdam DINOv2 study."""

from __future__ import annotations

import functools
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np

from deepnet.data.bundles import BundleData, export_model_bundle
from deepnet.data.catalog import DatasetCatalog, load_domain_spec, scene_format_values
from deepnet.data.contracts import BandSpec, NormalizationRule, SplitManifest
from deepnet.data.raster import inspect_raster, sha256_file
from deepnet.data.sampling import grid_windows
from deepnet.training_utils import write_json

WINDOW = (504, 504)
STRIDE = (252, 252)
DSM_CHANNELS = ("relief", "dsm_available")


def refit_split(parent: SplitManifest) -> SplitManifest:
    scene_ids = parent.train_scene_ids + parent.validation_scene_ids
    if parent.dataset_id != "isprs-potsdam" or (
        len(parent.train_scene_ids),
        len(parent.validation_scene_ids),
    ) != (18, 6):
        raise ValueError("Refit requires the locked 18/6 Potsdam selection split")
    groups = dict(parent.spatial_groups)
    return SplitManifest(
        dataset_id=parent.dataset_id,
        split_id="potsdam-dinov2-refit",
        version=1,
        train_scene_ids=scene_ids,
        validation_scene_ids=(),
        test_scene_ids=(),
        spatial_groups=tuple((scene, groups[scene]) for scene in sorted(scene_ids)),
        label_assets=(("train", "label_participant"),),
        spatial_scope=parent.spatial_scope,
        refit_parent_hash=parent.content_hash,
    )


def export_refit(selection: BundleData, raw: Path, output: Path):
    split = refit_split(selection.split)
    config = (
        DatasetCatalog.load().get(split.dataset_id).asset_configs()["label_participant"]
    )
    scenes = []
    for scene in selection.scenes:
        assets = scene.asset_map()
        if "label_participant" not in assets:
            uri = config["path"].format(**scene_format_values(scene.scene_id))
            assets["label_participant"] = inspect_raster(
                raw / uri,
                asset_id="label_participant",
                uri=uri,
                bands=tuple(config["bands"]),
                grid_id=f"{scene.scene_id}:participant",
                alignment_evidence=config["alignment_evidence"],
                hash_file=True,
            )
        scenes.append(replace(scene, assets=tuple(sorted(assets.items()))))
    return export_model_bundle(
        scenes,
        raw,
        split,
        selection.preprocessing,
        selection.model_input,
        output,
        include_splits=("train",),
        include_world_files=True,
    )


def verify_scope(bundle: BundleData, split: SplitManifest, roles: tuple[str, ...]):
    expected = {scene: role for role in roles for scene in split.scenes_for(role)}
    if (
        bundle.split.content_hash != split.content_hash
        or set(bundle.manifest["included_splits"]) != set(roles)
        or set(bundle.manifest["included_scene_ids"]) != set(expected)
        or set(scene.scene_id for scene in bundle.scenes) != set(expected)
    ):
        raise ValueError("Bundle does not match the locked split and requested roles")
    assets = {asset for _, asset, _ in bundle.model_input.channel_sources}
    for scene in bundle.scenes:
        label = split.label_asset_for(expected[scene.scene_id])
        if set(scene.asset_map()) != assets | {label}:
            raise ValueError(f"Unexpected assets in scene {scene.scene_id}")


def dsm_model_view(model_input, enabled: bool):
    domain = load_domain_spec("isprs-irrg-aerial-v1")
    if enabled:
        domain = replace(
            domain,
            bands=domain.bands + tuple(BandSpec(c, "auxiliary") for c in DSM_CHANNELS),
        )
        model_input = replace(
            model_input,
            model_id="potsdam-irrg-relief",
            channels=model_input.channels + DSM_CHANNELS,
            channel_sources=model_input.channel_sources
            + tuple((c, "dsm_features", c) for c in DSM_CHANNELS),
            normalization=model_input.normalization
            + tuple(NormalizationRule(c, "identity") for c in DSM_CHANNELS),
        )
    return domain, model_input


class DsmAccess:
    """Append finite, aligned features while preserving the label mask."""

    def __init__(self, access, root: Path, manifest: dict):
        self.access = access
        self.root = root
        self.manifest = manifest
        _, self.model_input = dsm_model_view(access.model_input, True)

    def __getattr__(self, name):
        return getattr(self.__dict__["access"], name)

    def read_sample(self, scene, window):
        import rasterio

        sample = self.access.read_sample(scene, window)
        with rasterio.open(self.root / f"{scene}.tif") as source:
            extra = source.read(
                window=rasterio.windows.Window(
                    window.col, window.row, window.width, window.height
                ),
                boundless=True,
                fill_value=0,
            )
        extra[0] = np.clip(extra[0] / self.manifest["scale"], -1, 1)
        extra[:, ~sample.valid_mask] = 0
        return replace(sample, image=np.concatenate((sample.image, extra), axis=0))


def verify_dsm(root: Path, split: SplitManifest, scene_ids):
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest["selection_split_hash"] != split.content_hash
        or manifest["fit_scene_ids"] != list(split.train_scene_ids)
        or manifest["recipe"] != "valid-box-101-p99abs-stride16-v1"
        or not np.isfinite(manifest["scale"])
        or manifest["scale"] <= 0
        or set(manifest["scenes"]) != set(split.scene_ids)
    ):
        raise ValueError("DSM provenance does not match the locked Potsdam study")
    for scene_id in scene_ids:
        record = manifest["scenes"][scene_id]
        if (
            record["coverage"] < 0.99
            or sha256_file(root / f"{scene_id}.tif") != record["sha256"]
        ):
            raise ValueError(f"DSM feature verification failed: {scene_id}")
    return manifest


def prepare_dsm(split, scenes, raw_root: Path, inventory: Path, output: Path):
    """Derive relief from GeoTIFFs verified by SHA-256."""
    import cv2
    import rasterio

    source = json.loads(inventory.read_text())
    if (
        source.get("source") != "ISPRS Potsdam 1_DSM"
        or set(source["sha256"]) != set(split.scene_ids)
        or not source.get("archive_sha256")
    ):
        raise ValueError(
            "Require a source archive checksum and all official DSM hashes"
        )
    by_id = {scene.scene_id: scene for scene in scenes}
    manifest = {
        "selection_split_hash": split.content_hash,
        "fit_scene_ids": list(split.train_scene_ids),
        "source_inventory_sha256": sha256_file(inventory),
        "recipe": "valid-box-101-p99abs-stride16-v1",
        "scenes": {},
    }
    output.mkdir(parents=True, exist_ok=False)
    fitted = []
    for scene_id in split.scene_ids:
        scene = by_id[scene_id]
        grid = scene.reference_grid
        area, tile = map(int, scene_id.split("_"))
        path = raw_root / f"dsm_potsdam_{area:02d}_{tile:02d}.tif"
        digest = sha256_file(path)
        if digest != source["sha256"][scene_id]:
            raise ValueError(f"Raw DSM checksum mismatch: {scene_id}")
        target = output / f"{scene_id}.tif"
        with rasterio.open(path) as raster:
            expected_width = grid.width - int(scene_id == "3_13")
            if (
                raster.count != 1
                or raster.width not in {expected_width, grid.width}
                or raster.height != grid.height
                or raster.crs != rasterio.crs.CRS.from_user_input(grid.crs)
                or not np.allclose(
                    tuple(raster.transform)[:6], grid.affine, rtol=0, atol=1e-8
                )
            ):
                raise ValueError(f"Unexpected DSM grid: {scene_id}")
            profile = raster.profile.copy()
            profile.update(
                width=grid.width,
                count=2,
                dtype="float32",
                nodata=None,
                tiled=True,
                blockxsize=256,
                blockysize=256,
                compress="deflate",
            )
            coverage = 0
            minimum, maximum = np.inf, -np.inf
            with rasterio.open(target, "w", **profile) as dst:
                for _, block in dst.block_windows(1):
                    col, row, width, height = map(int, block.flatten())
                    halo = rasterio.windows.Window(
                        col - 50, row - 50, width + 100, height + 100
                    )
                    raw = raster.read(1, window=halo, masked=True, boundless=True)
                    valid = ~np.ma.getmaskarray(raw) & np.isfinite(raw.data)
                    # GeoTIFFs without nodata still require explicit geometric padding.
                    valid &= (
                        (np.arange(row - 50, row + height + 50) >= 0)
                        & (np.arange(row - 50, row + height + 50) < raster.height)
                    )[:, None] & (
                        (np.arange(col - 50, col + width + 50) >= 0)
                        & (np.arange(col - 50, col + width + 50) < raster.width)
                    )[None, :]
                    values = np.where(valid, raw.data, 0).astype(np.float64)
                    support = cv2.boxFilter(
                        valid.astype(np.float64), -1, (101, 101), normalize=False
                    )
                    sums = cv2.boxFilter(values, -1, (101, 101), normalize=False)
                    center = (slice(50, 50 + height), slice(50, 50 + width))
                    available = valid[center]
                    center_values = values[center]
                    relief = np.where(
                        available,
                        center_values - sums[center] / np.maximum(support[center], 1),
                        0,
                    ).astype(np.float32)
                    coverage += int(available.sum())
                    if available.any():
                        minimum = min(minimum, float(center_values[available].min()))
                        maximum = max(maximum, float(center_values[available].max()))
                    if scene_id in split.train_scene_ids:
                        sample = np.s_[(-row) % 16 :: 16, (-col) % 16 :: 16]
                        fitted.append(np.abs(relief[sample][available[sample]]))
                    dst.write(
                        np.stack((relief, available)).astype(np.float32), window=block
                    )
            fraction = coverage / (grid.width * grid.height)
            if fraction < 0.99:
                raise ValueError(f"DSM coverage below 99%: {scene_id}")
            manifest["scenes"][scene_id] = {
                "source_sha256": digest,
                "sha256": sha256_file(target),
                "coverage": fraction,
                "minimum": minimum,
                "maximum": maximum,
                "crs": str(raster.crs),
                "transform": list(raster.transform)[:6],
                "source_width": raster.width,
                "source_height": raster.height,
                "extent": list(raster.bounds),
            }
    scale = float(np.quantile(np.concatenate(fitted), 0.99))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Degenerate train-only relief scale")
    manifest["scale"] = scale
    write_json(output / "manifest.json", manifest)
    return manifest


def augment_patch(sample, seed: int, strong: bool, rare: bool):
    """Apply D4, then optional mask-aware affine and optical-only gain."""
    replay = random.Random(seed)
    turns, flip = replay.randrange(4), replay.choice((False, True))
    image = np.rot90(sample.image, turns, axes=(-2, -1))
    labels = np.rot90(sample.labels, turns)
    valid = np.rot90(sample.valid_mask, turns)
    if flip:
        image, labels, valid = image[..., ::-1], labels[:, ::-1], valid[:, ::-1]
    image = image.copy()
    if strong and not rare and replay.random() < 0.25:
        import cv2

        height, width = labels.shape
        matrix = cv2.getRotationMatrix2D(
            ((width - 1) / 2, (height - 1) / 2),
            replay.uniform(-6, 6),
            replay.uniform(0.95, 1.05),
        )
        matrix[:, 2] += (
            replay.uniform(-0.02, 0.02) * width,
            replay.uniform(-0.02, 0.02) * height,
        )
        warp = functools.partial(cv2.warpAffine, M=matrix, dsize=(width, height))
        valid_float = valid.astype(np.float32)
        support = warp(valid_float, flags=cv2.INTER_LINEAR)
        optical = [
            warp(channel * valid_float, flags=cv2.INTER_LINEAR)
            / np.maximum(support, 1e-6)
            for channel in image[:3]
        ]
        if len(image) == 5:
            available = image[4] * valid_float
            dsm_support = warp(available, flags=cv2.INTER_LINEAR)
            dsm_valid = warp(available, flags=cv2.INTER_NEAREST) > 0.5
            relief = warp(image[3] * available, flags=cv2.INTER_LINEAR)
            optical += [
                np.where(dsm_valid, relief / np.maximum(dsm_support, 1e-6), 0),
                dsm_valid,
            ]
        image = np.stack(optical).astype(np.float32)
        labels = warp(labels.astype(np.uint8), flags=cv2.INTER_NEAREST, borderValue=6)
        valid = warp(valid_float, flags=cv2.INTER_NEAREST) > 0.5
    if strong and replay.random() < 0.25:
        image[:3] = np.clip(image[:3] * replay.uniform(0.95, 1.05), 0, 1)
    image[:, ~valid] = 0
    labels = np.where(valid, labels, 6).astype(np.int64)
    return replace(sample, image=image, labels=labels, valid_mask=valid.copy())


def grid_for(access):
    return [
        (scene, window)
        for scene in access.split.scenes_for(access.split_name)
        for window in grid_windows(*access.scene_shapes[scene], WINDOW)
    ]


def index_training(access):
    """Count each pixel once and index windows with real car or clutter pixels."""
    grid = grid_for(access)
    counts = np.zeros(6, dtype=np.int64)
    indexed = []
    for scene, window in grid:
        labels, valid = access.read_labels(scene, window)
        histogram = np.bincount(labels[valid & (labels < 6)], minlength=6)
        counts += histogram
        indexed.append((scene, window, histogram))
    if np.any(counts <= 0):
        raise ValueError("Training grid must contain all six classes")
    return indexed, counts


def patch_schedule(indexed, seed: int, pass_number: int):
    rng = random.Random(f"{seed}/{pass_number}")
    base = [(scene, window, False) for scene, window, _ in indexed]
    quota, remainder = divmod(len(base), 6)
    if remainder:
        raise ValueError("Expected grid size divisible by six for rare sampling")
    for class_id in (4, 5):
        pool = [
            (scene, window, True) for scene, window, hist in indexed if hist[class_id]
        ]
        base.extend(rng.choices(pool, k=quota))
    rng.shuffle(base)
    return base


def fixed_panel(access, per_scene: int, seed: int):
    rng = random.Random(seed)
    by_scene = {}
    for scene, window in grid_for(access):
        by_scene.setdefault(scene, []).append(window)
    return [
        (scene, window)
        for scene, windows in by_scene.items()
        for window in rng.sample(windows, per_scene)
    ]


def fit_panel(indexed):
    # Greedily include patches carrying substantial support for rare classes.
    chosen = []
    for class_id in (4, 5, 0, 1, 2, 3, 4, 5):
        scene, window, _ = max(
            (entry for entry in indexed if entry[:2] not in chosen),
            key=lambda entry: entry[2][class_id],
        )
        chosen.append((scene, window))
    return chosen
