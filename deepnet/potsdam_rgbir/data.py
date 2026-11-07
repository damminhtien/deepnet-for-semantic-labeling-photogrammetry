"""Read-only memory-mapped scene access with independent DSM availability."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deepnet.data.contracts import (
    BandSpec,
    DomainSpec,
    ModelInputSpec,
    NormalizationRule,
    PreprocessingSpec,
    Sample,
    SceneRecord,
    SplitManifest,
    WindowSpec,
    canonical_hash,
)
from deepnet.data.raster import sha256_file

RGB = ("red", "green", "blue")
EXTRA = ("ir", "relief", "dsm_available")
ARRAYS = ("optical", "relief", "dsm_available", "labels", "valid")


def model_view(auxiliary: bool) -> tuple[DomainSpec, ModelInputSpec]:
    names = RGB + EXTRA if auxiliary else RGB
    domain = DomainSpec(
        "isprs-potsdam-rgbir",
        2,
        "Official ISPRS RGBIR and verified DSM relief",
        tuple(
            BandSpec(name, "auxiliary" if name in EXTRA[1:] else "optical")
            for name in names
        ),
    )
    view = ModelInputSpec(
        "potsdam-rgbir-relief" if auxiliary else "potsdam-rgb",
        2,
        names,
        tuple(
            (name, "rgbir" if name in (*RGB, "ir") else "dsm_features", name)
            for name in names
        ),
        tuple(NormalizationRule(name, "identity") for name in names),
        pixel_semantics=(
            "RGB/IR scaled by 255; train-fitted relief in [-1,1]; independent DSM mask."
        ),
    )
    return domain, view


@dataclass(frozen=True)
class CacheBundle:
    root: Path
    manifest: dict
    split: SplitManifest
    preprocessing: PreprocessingSpec
    scenes: tuple[SceneRecord, ...]

    @classmethod
    def load(cls, root: Path, *, verify: bool = True):
        root = Path(root).resolve()
        manifest = json.loads((root / "cache_manifest.json").read_text())
        payload = {k: v for k, v in manifest.items() if k != "cache_hash"}
        if (
            manifest["kind"] != "potsdam-rgbir-memmap-v2"
            or manifest["schema_version"] != 1
            or canonical_hash(payload) != manifest["cache_hash"]
        ):
            raise ValueError("Cache manifest hash/schema mismatch")
        seen = set()
        for record in manifest["files"]:
            relative = Path(record["path"])
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() in seen
            ):
                raise ValueError("Unsafe or duplicate cache file")
            seen.add(relative.as_posix())
            path = root / relative
            if not path.resolve().is_relative_to(root):
                raise ValueError("Cache file escapes its root")
            if path.stat().st_size != record["bytes"] or (
                verify and sha256_file(path) != record["sha256"]
            ):
                raise ValueError(f"Cache file mismatch: {relative}")
        split = SplitManifest.from_dict(json.loads((root / "split.json").read_text()))
        scenes = tuple(
            SceneRecord.from_dict(s)
            for s in json.loads((root / "catalog.json").read_text())
        )
        expected = {
            scene: role
            for role in manifest["included_splits"]
            for scene in split.scenes_for(role)
        }
        if (
            split.content_hash != manifest["split_hash"]
            or set(manifest["included_scene_ids"]) != set(expected)
            or set(manifest["scenes"]) != set(expected)
            or {s.scene_id for s in scenes} != set(expected)
        ):
            raise ValueError("Cache does not match its split roles")
        required = {"split.json", "preprocessing.json", "catalog.json"}
        for scene, role in expected.items():
            entry = manifest["scenes"][scene]
            grid = next(s.reference_grid for s in scenes if s.scene_id == scene)
            if entry["shape"] != [grid.height, grid.width]:
                raise ValueError("Cache scene shape differs from its source grid")
            if entry["role"] != role or entry["label_asset"] != split.label_asset_for(
                role
            ):
                raise ValueError("Cache label role mismatch")
            required.update(f"{scene}/{name}.npy" for name in ARRAYS)
        if seen != required:
            raise ValueError("Cache inventory contains missing or unexpected assets")
        parent_hash = split.refit_parent_hash or split.content_hash
        if parent_hash != manifest["selection_split_hash"]:
            raise ValueError("Cache parent selection hash mismatch")
        from deepnet.potsdam_study import _split

        parent = _split()
        if parent_hash != parent.content_hash:
            raise ValueError("Cache must derive from the locked historical split")
        if split.refit_parent_hash and (
            manifest["included_splits"] != ["train"]
            or split.train_scene_ids
            != parent.train_scene_ids + parent.validation_scene_ids
            or split.validation_scene_ids
            or split.test_scene_ids
            or dict(split.label_assets) != {"train": "label_participant"}
        ):
            raise ValueError("Refit cache must be train-only with exactly 24 scenes")
        return cls(
            root,
            manifest,
            split,
            PreprocessingSpec.from_dict(
                json.loads((root / "preprocessing.json").read_text())
            ),
            scenes,
        )

    def access(self, role: str, *, auxiliary: bool):
        if role not in self.manifest["included_splits"]:
            raise ValueError(f"Role is not included in cache: {role}")
        return CacheAccess(self, role, auxiliary=auxiliary)


class CacheAccess:
    """Expose SceneAccess methods and open arrays lazily per worker."""

    def __init__(self, bundle: CacheBundle, role: str, *, auxiliary: bool):
        self.bundle = bundle
        self.split = bundle.split
        self.split_name = role
        self.preprocessing = bundle.preprocessing
        self.scenes = {scene.scene_id: scene for scene in bundle.scenes}
        self.scene_shapes = {
            scene: tuple(bundle.manifest["scenes"][scene]["shape"])
            for scene in self.split.scenes_for(role)
        }
        self.auxiliary = auxiliary
        _, self.model_input = model_view(auxiliary)
        self._maps = {}
        self.profile_times = None

    def __getstate__(self):
        return {**self.__dict__, "_maps": {}}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._maps.clear()

    def _arrays(self, scene: str):
        if scene not in self.scene_shapes:
            raise ValueError(f"Scene outside selected {self.split_name} role: {scene}")
        if scene not in self._maps:
            self._maps[scene] = {
                name: np.load(
                    self.bundle.root / scene / f"{name}.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                )
                for name in ARRAYS
            }
            arrays = self._maps[scene]
            shape = self.scene_shapes[scene]
            for name, array in arrays.items():
                expected = (4, *shape) if name == "optical" else shape
                dtype = np.dtype("float32" if name == "relief" else "uint8")
                if array.shape != expected or array.dtype != dtype:
                    raise ValueError(
                        f"Cache array shape/dtype mismatch: {scene}/{name}"
                    )
        return self._maps[scene]

    def read_labels(self, scene: str, window: WindowSpec):
        arrays = self._arrays(scene)
        window.validate_for(self.scenes[scene].reference_grid)
        labels = self._read(arrays["labels"], window, 6).astype(np.int64)
        valid = self._read(arrays["valid"], window, 0).astype(bool)
        return labels, valid

    @staticmethod
    def _read(array, window, fill):
        row, col = window.row, window.col
        h = min(window.height, max(0, array.shape[-2] - row))
        w = min(window.width, max(0, array.shape[-1] - col))
        target = np.full(
            (*array.shape[:-2], window.height, window.width), fill, dtype=array.dtype
        )
        target[..., :h, :w] = array[..., row : row + h, col : col + w]
        return target

    def read_sample(self, scene: str, window: WindowSpec):
        started = time.perf_counter() if self.profile_times is not None else 0
        arrays = self._arrays(scene)
        window.validate_for(self.scenes[scene].reference_grid)
        count = 4 if self.auxiliary else 3
        optical = self._read(arrays["optical"][:count], window, 0)
        if self.auxiliary:
            relief = self._read(arrays["relief"], window, 0)
            mask = self._read(arrays["dsm_available"], window, 0)
        labels = self._read(arrays["labels"], window, 6)
        valid = self._read(arrays["valid"], window, 0)
        ready = time.perf_counter() if self.profile_times is not None else 0
        optical = optical.astype(np.float32) / 255
        if self.auxiliary:
            optical = np.concatenate(
                (optical, relief[None], mask[None].astype(np.float32))
            )
        labels, valid = labels.astype(np.int64), valid.astype(bool)
        optical[:, ~valid] = 0
        labels[~valid] = 6
        self.record_profile(started, ready)
        return Sample(optical, labels, valid, scene, window)

    def record_profile(self, started, ready):
        if self.profile_times is not None:
            for key, seconds in (
                ("source_io", ready - started),
                ("preprocessing", time.perf_counter() - ready),
            ):
                self.profile_times[key] = self.profile_times.get(key, 0) + seconds

    def read_context(self, scene: str, window: WindowSpec, halo: int):
        context = WindowSpec(
            max(0, window.row - halo),
            max(0, window.col - halo),
            window.height + 2 * halo,
            window.width + 2 * halo,
            padding="constant",
        )
        return self.read_sample(scene, context)
