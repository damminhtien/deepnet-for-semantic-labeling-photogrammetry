"""A single validation scene's Rasterio reference for cache equivalence/profiling."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import rasterio

from deepnet.data.bundles import load_bundle
from deepnet.data.contracts import Sample
from deepnet.data.preprocessing import decode_rgb_labels
from deepnet.data.raster import sha256_file
from deepnet.training_utils import write_json

from .data import CacheAccess, CacheBundle


def prepare_calibration(bundle_path, cache_path, rgbir, dsm, output, scene="4_12"):
    bundle = load_bundle(bundle_path)
    cache = CacheBundle.load(cache_path)
    if scene not in bundle.split.validation_scene_ids:
        raise ValueError("Calibration must use a validation scene")
    rgb = json.loads((rgbir / "rgbir_manifest.json").read_text())
    record = next(s for s in bundle.scenes if s.scene_id == scene)
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "rgbir.tif": rgbir / rgb["scenes"][scene]["path"],
        "dsm.tif": dsm / f"{scene}.tif",
        "labels.tif": bundle.root
        / record.asset_map()[bundle.split.label_asset_for("validation")].uri,
    }
    files = {}
    for name, source in paths.items():
        shutil.copy2(source, output / name)
        files[name] = sha256_file(output / name)
    write_json(
        output / "calibration.json",
        {
            "scene": scene,
            "cache_hash": cache.manifest["cache_hash"],
            "selection_split_hash": cache.manifest["selection_split_hash"],
            "relief_scale": cache.manifest["relief_scale"],
            "files": files,
        },
    )


class RasterioReferenceAccess(CacheAccess):
    def __init__(self, bundle, root, *, auxiliary):
        super().__init__(bundle, "validation", auxiliary=auxiliary)
        self.root = Path(root)
        self.calibration = json.loads((self.root / "calibration.json").read_text())
        if self.calibration["cache_hash"] != bundle.manifest["cache_hash"]:
            raise ValueError("Calibration belongs to a different cache")
        for name, digest in self.calibration["files"].items():
            if (
                name not in {"rgbir.tif", "dsm.tif", "labels.tif"}
                or sha256_file(self.root / name) != digest
            ):
                raise ValueError("Calibration raster hash mismatch")
        self.readers = {
            name: rasterio.open(self.root / name) for name in self.calibration["files"]
        }

    def close(self):
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()
        super().close()

    def read_sample(self, scene, window):
        if scene != self.calibration["scene"]:
            raise ValueError("Only the calibration validation scene is available")
        window.validate_for(self.scenes[scene].reference_grid)
        roi = rasterio.windows.Window(
            window.col, window.row, window.width, window.height
        )
        rgb = self.readers["rgbir.tif"]
        label = self.readers["labels.tif"]
        started = time.perf_counter() if self.profile_times is not None else 0
        image = rgb.read(window=roi, boundless=True, fill_value=0)
        valid = rgb.read_masks(window=roi, boundless=True).all(axis=0)
        valid &= label.read_masks(window=roi, boundless=True).all(axis=0)
        raw_labels = label.read(window=roi, boundless=True, fill_value=0).transpose(
            1, 2, 0
        )
        if self.auxiliary:
            extra = self.readers["dsm.tif"].read(
                window=roi, boundless=True, fill_value=0
            )
        ready = time.perf_counter() if self.profile_times is not None else 0
        image = image.astype(np.float32) / 255
        labels = decode_rgb_labels(
            raw_labels, self.preprocessing, valid_mask=valid
        ).astype(np.int64)
        if self.auxiliary:
            relief = np.where(
                extra[1] > 0,
                np.clip(extra[0] / self.calibration["relief_scale"], -1, 1),
                0,
            )
            image = np.concatenate((image, relief[None], extra[1:2]))
        else:
            image = image[:3]
        image[:, ~valid] = 0
        labels[~valid] = 6
        self.record_profile(started, ready)
        return Sample(image, labels, valid, scene, window)
