"""Shared scene/window access used by training, inference, and validation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from typing_extensions import Self

from .contracts import (
    ModelInputSpec,
    PreprocessingSpec,
    Sample,
    SceneRecord,
    SplitManifest,
    WindowSpec,
)
from .preprocessing import apply_model_view, decode_rgb_labels
from .raster import RasterioSceneReader
from .sampling import (
    UniformSampler,
    augment_aligned,
    grid_windows,
    sample_rng,
)


class SceneAccess:
    """Load a model view and matching labels from one verified scene record."""

    def __init__(
        self,
        scenes: Iterable[SceneRecord],
        data_root: Path,
        split: SplitManifest,
        split_name: str,
        preprocessing: PreprocessingSpec,
        model_input: ModelInputSpec,
        *,
        allow_reprojection: bool = False,
        verify_metadata: bool = True,
    ) -> None:
        scene_records = tuple(scenes)
        self.scenes = {scene.scene_id: scene for scene in scene_records}
        if len(self.scenes) != len(scene_records):
            raise ValueError("Scene records contain duplicate scene IDs")
        if not self.scenes:
            raise ValueError("At least one scene record is required")
        if {scene.dataset_id for scene in self.scenes.values()} != {split.dataset_id}:
            raise ValueError(
                "Scene catalog and split manifest use different dataset IDs"
            )
        if split_name not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split_name}")
        self.split = split
        self.split_name = split_name
        self.preprocessing = preprocessing
        self.preprocessing.validate_fit_scope(split)
        self.model_input = model_input
        self.data_root = Path(data_root).expanduser().resolve()
        self.allow_reprojection = allow_reprojection
        self.verify_metadata = verify_metadata
        self.label_asset_name = split.label_asset_for(split_name)
        selected_scene_ids = split.scenes_for(split_name)
        self.selected_scene_ids = frozenset(selected_scene_ids)
        self.channel_sources = tuple(model_input.channel_sources)
        self.required_assets = tuple(
            dict.fromkeys(
                [asset_name for _, asset_name, _ in self.channel_sources]
                + [self.label_asset_name]
            )
        )
        self._readers: dict[str, RasterioSceneReader] = {}
        missing = set(selected_scene_ids) - self.scenes.keys()
        if missing:
            raise ValueError(
                f"Split references missing scene records: {sorted(missing)}"
            )
        for scene_id in selected_scene_ids:
            scene = self.scenes[scene_id]
            if scene.label_ontology != preprocessing.label_ontology:
                raise ValueError(
                    f"Scene {scene_id} uses ontology {scene.label_ontology!r}; "
                    f"preprocessing expects {preprocessing.label_ontology!r}"
                )
            scene.require_aligned(
                self.required_assets, allow_reprojection=allow_reprojection
            )
            asset_map = scene.asset_map()
            for _, asset_name, band_name in self.channel_sources:
                if (
                    asset_name not in asset_map
                    or band_name not in asset_map[asset_name].bands
                ):
                    raise ValueError(
                        f"Scene {scene_id} has no channel source "
                        f"{asset_name}:{band_name}"
                    )
            if self.label_asset_name not in asset_map:
                raise ValueError(
                    f"Scene {scene_id} has no label asset {self.label_asset_name!r}"
                )
        fitted_channels = {channel for channel, _, _ in preprocessing.fitted_statistics}
        required_statistics = {
            rule.channel
            for rule in model_input.normalization
            if rule.method == "zscore"
        }
        missing_statistics = required_statistics - fitted_channels
        if missing_statistics:
            raise ValueError(
                "Model input requires train-fitted statistics for channels: "
                + ", ".join(sorted(missing_statistics))
            )

    @property
    def scene_shapes(self) -> dict[str, tuple[int, int]]:
        return {
            scene_id: (
                self.scenes[scene_id].reference_grid.height,
                self.scenes[scene_id].reference_grid.width,
            )
            for scene_id in self.split.scenes_for(self.split_name)
        }

    def _reader(self, scene_id: str) -> RasterioSceneReader:
        if scene_id not in self._readers:
            self._readers[scene_id] = RasterioSceneReader(
                self.scenes[scene_id],
                self.data_root,
                allow_reprojection=self.allow_reprojection,
                verify_metadata=self.verify_metadata,
            )
        return self._readers[scene_id]

    def read_sample(
        self,
        scene_id: str,
        window: WindowSpec,
        *,
        augment: bool = False,
        rng: Any | None = None,
    ) -> Sample:
        if scene_id not in self.selected_scene_ids:
            raise ValueError(
                f"Scene {scene_id!r} is outside the selected {self.split_name} split"
            )
        scene = self.scenes[scene_id]
        reader = self._reader(scene_id)
        source_arrays: dict[tuple[str, str], np.ndarray] = {}
        valid = np.ones((window.height, window.width), dtype=bool)
        assets: dict[str, np.ndarray] = {}
        for _, asset_name, band_name in self.channel_sources:
            if asset_name not in assets:
                array, asset_valid = reader.read_window_with_mask(
                    asset_name,
                    window,
                    resampling=self.preprocessing.continuous_resampling,
                )
                assets[asset_name] = array
                valid &= asset_valid
            asset = scene.asset_map()[asset_name]
            band_index = asset.bands.index(band_name)
            source_arrays[(asset_name, band_name)] = assets[asset_name][..., band_index]
        for array in source_arrays.values():
            valid &= np.isfinite(array)
        image_chw = apply_model_view(
            source_arrays,
            self.model_input,
            self.preprocessing,
            valid_mask=valid,
        )
        image_hwc = np.ascontiguousarray(image_chw.transpose(1, 2, 0))

        labels, label_valid = self.read_labels(scene_id, window)
        valid &= label_valid
        labels[~valid] = self.preprocessing.ignore_label
        if augment:
            if rng is None:
                raise ValueError("A deterministic RNG is required for augmentation")
            image_hwc, labels, valid = augment_aligned(image_hwc, labels, valid, rng)
        image_chw = np.ascontiguousarray(image_hwc.transpose(2, 0, 1), dtype=np.float32)
        labels = np.ascontiguousarray(labels.astype(np.int64, copy=False))
        valid = np.ascontiguousarray(valid)
        return Sample(
            image=image_chw,
            labels=labels,
            valid_mask=valid,
            scene_id=scene_id,
            window=window,
            layout="CHW",
        )

    def read_labels(
        self,
        scene_id: str,
        window: WindowSpec,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read canonical class IDs and the raster validity mask for this split."""
        if scene_id not in self.selected_scene_ids:
            raise ValueError(
                f"Scene {scene_id!r} is outside the selected {self.split_name} split"
            )
        label_rgb, valid = self._reader(scene_id).read_window_with_mask(
            self.label_asset_name,
            window,
            resampling=self.preprocessing.categorical_resampling,
        )
        labels = decode_rgb_labels(label_rgb, self.preprocessing, valid_mask=valid)
        return labels, valid

    def iter_grid_samples(
        self,
        scene_ids: Sequence[str],
        window_size: tuple[int, int],
        stride: tuple[int, int] | None = None,
    ) -> Iterable[Sample]:
        """Yield reproducible windows that cover selected valid scene pixels."""
        for scene_id in scene_ids:
            if scene_id not in self.selected_scene_ids:
                raise ValueError(
                    f"Scene {scene_id!r} is outside the selected "
                    f"{self.split_name} split"
                )
            grid = self.scenes[scene_id].reference_grid
            windows = grid_windows(grid.height, grid.width, window_size, stride)
            for window in windows:
                yield self.read_sample(scene_id, window)

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_readers"] = {}
        return state

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ScenePatchDataset:
    """Random training samples over scene records; compatible with Torch DataLoader."""

    def __init__(
        self,
        access: SceneAccess,
        window_size: tuple[int, int],
        epoch_size: int,
        *,
        seed: int,
        sampler: Any | None = None,
        augmentation: bool = True,
    ) -> None:
        if access.split_name != "train":
            raise ValueError(
                "Random patch sampling is restricted to the training split"
            )
        if epoch_size <= 0:
            raise ValueError("epoch_size must be positive")
        if window_size[0] != window_size[1]:
            raise ValueError(
                "Synchronized right-angle augmentation requires square windows"
            )
        self.access = access
        self.window_size = window_size
        self.epoch_size = epoch_size
        self.seed = seed
        self.augmentation = augmentation
        self.epoch = 0
        self.sampler = sampler or UniformSampler(
            access.scene_shapes,
            access.split.train_scene_ids,
            window_size,
        )

    def __len__(self) -> int:
        return self.epoch_size

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __getitem__(self, index: int) -> Sample:
        if not 0 <= index < self.epoch_size:
            raise IndexError(index)
        rng = sample_rng(self.seed, self.epoch, index)
        scene_id, window = self.sampler.sample(rng)
        sample = self.access.read_sample(
            scene_id,
            window,
            augment=self.augmentation,
            rng=rng,
        )
        return sample


def collate_samples(samples: Sequence[Sample]) -> dict[str, Any]:
    """Stack fixed-size samples for a model DataLoader without torch coupling."""
    if not samples:
        raise ValueError("Cannot collate an empty sample batch")
    shape = samples[0].image.shape
    label_shape = samples[0].labels.shape
    if any(sample.image.shape != shape for sample in samples):
        raise ValueError("A batch must contain image tensors with a common shape")
    if any(sample.labels.shape != label_shape for sample in samples):
        raise ValueError("A batch must contain label maps with a common shape")
    return {
        "image": np.stack([sample.image for sample in samples]),
        "labels": np.stack([sample.labels for sample in samples]),
        "valid_mask": np.stack([sample.valid_mask for sample in samples]),
        "scene_ids": tuple(sample.scene_id for sample in samples),
        "windows": tuple(sample.window for sample in samples),
    }


def collate_torch_samples(samples: Sequence[Sample]) -> dict[str, Any]:
    """Convert collated NumPy arrays to Torch tensors at the training boundary."""
    import torch

    batch = collate_samples(samples)
    for key in ("image", "labels", "valid_mask"):
        batch[key] = torch.from_numpy(batch[key])
    return batch


def _reset_worker_raster_handles(_: int) -> None:
    """Close inherited GDAL handles after worker processes start."""
    import torch

    worker = torch.utils.data.get_worker_info()
    if worker is not None and isinstance(worker.dataset, ScenePatchDataset):
        worker.dataset.access.close()


def make_torch_dataloader(
    dataset: ScenePatchDataset,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Any:
    """Build a deterministic map-style loader safe for local and CUDA workers.

    Persistent workers are disabled so dataset.set_epoch reaches worker copies
    at the start of each epoch. Every sample's random stream is derived from its
    seed, epoch, and index rather than worker scheduling.
    """
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    import torch

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        worker_init_fn=_reset_worker_raster_handles if num_workers else None,
        collate_fn=collate_torch_samples,
    )


def build_class_pixel_index(
    access: SceneAccess,
    class_ids: Sequence[int],
    *,
    tile_size: int = 1024,
    max_pixels_per_scene: int = 8192,
    seed: int = 0,
) -> dict[int, dict[str, np.ndarray]]:
    """Build a bounded, deterministic target-pixel index from train labels only."""
    if access.split_name != "train":
        raise ValueError("Class-aware indexes may be built only from training access")
    if tile_size <= 0 or max_pixels_per_scene <= 0:
        raise ValueError("tile_size and max_pixels_per_scene must be positive")
    requested = tuple(dict.fromkeys(int(class_id) for class_id in class_ids))
    if not requested or any(
        class_id < 0 or class_id >= access.preprocessing.ignore_label
        for class_id in requested
    ):
        raise ValueError("class_ids must contain valid non-ignore class IDs")

    rng = np.random.default_rng(seed)
    indexes: dict[int, dict[str, np.ndarray]] = {class_id: {} for class_id in requested}
    for scene_id in access.split.train_scene_ids:
        grid = access.scenes[scene_id].reference_grid
        reservoirs = {
            class_id: (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64))
            for class_id in requested
        }
        for window in grid_windows(grid.height, grid.width, (tile_size, tile_size)):
            labels, valid = access.read_labels(scene_id, window)
            for class_id in requested:
                local_indices = np.flatnonzero(valid & (labels == class_id))
                if not local_indices.size:
                    continue
                local_rows, local_cols = np.divmod(local_indices, window.width)
                global_indices = (
                    (window.row + local_rows) * grid.width + window.col + local_cols
                ).astype(np.int64, copy=False)
                priorities = rng.random(local_indices.size)
                previous_priorities, previous_indices = reservoirs[class_id]
                priorities = np.concatenate((previous_priorities, priorities))
                global_indices = np.concatenate((previous_indices, global_indices))
                if priorities.size > max_pixels_per_scene:
                    keep = np.argpartition(priorities, max_pixels_per_scene - 1)[
                        :max_pixels_per_scene
                    ]
                    priorities = priorities[keep]
                    global_indices = global_indices[keep]
                reservoirs[class_id] = priorities, global_indices
        for class_id, (_, reservoir) in reservoirs.items():
            if reservoir.size:
                indexes[class_id][scene_id] = np.sort(reservoir)
    return {class_id: by_scene for class_id, by_scene in indexes.items() if by_scene}
