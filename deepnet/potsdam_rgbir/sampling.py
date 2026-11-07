"""Full-grid coverage plus deterministic, pixel-centered rare-class draws."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from deepnet.data.contracts import WindowSpec
from deepnet.data.dataset import collate_torch_samples
from deepnet.data.sampling import grid_windows

from .types import TrainingIndex

WINDOW = (512, 512)


def draw_seed(seed: int, pass_number: int, index: int) -> int:
    value = hashlib.sha256(f"{seed}/{pass_number}/{index}".encode()).digest()
    return int.from_bytes(value[:8], "big")


@dataclass(frozen=True)
class Draw:
    scene: str
    window: WindowSpec
    rare_class: int | None = None


def training_index(access: Any) -> TrainingIndex:
    """Count training pixels and index rare-class counts by tile."""
    base, support, rare = [], np.zeros(6, dtype=np.int64), {}
    for scene, shape in access.scene_shapes.items():
        tiles = {str(c): [] for c in (4, 5)}
        for window in grid_windows(shape[0], shape[1], WINDOW):
            labels, valid = access.read_labels(scene, window)
            valid &= labels < 6
            counts = np.bincount(labels[valid], minlength=6)
            support += counts
            for class_id in (4, 5):
                if counts[class_id]:
                    tiles[str(class_id)].append([len(base), int(counts[class_id])])
            base.append([scene, window.to_dict(), counts.tolist()])
        rare[scene] = tiles
    if np.any(support <= 0):
        raise ValueError("Train index must represent all six classes")
    return {"base": base, "class_support": support.tolist(), "rare_tiles": rare}


def schedule(
    index: TrainingIndex,
    access: Any,
    seed: int,
    pass_number: int,
    *,
    jitter: int = 96,
) -> list[Draw]:
    rng = random.Random(f"{seed}/{pass_number}")
    draws = [Draw(scene, WindowSpec(**window)) for scene, window, _ in index["base"]]
    quota, remainder = divmod(len(draws), 6)
    if remainder:
        raise ValueError("Coverage grid must be divisible by six for rare quotas")
    for class_id in (4, 5):
        eligible = sorted(s for s, p in index["rare_tiles"].items() if p[str(class_id)])
        if not eligible:
            raise ValueError(f"No train pixels for prioritized class {class_id}")
        for _ in range(quota):
            scene = rng.choice(eligible)
            tiles = index["rare_tiles"][scene][str(class_id)]
            rank = rng.randrange(sum(count for _, count in tiles))
            for tile, count in tiles:
                if rank < count:
                    source = WindowSpec(**index["base"][tile][1])
                    break
                rank -= count
            labels, valid = access.read_labels(scene, source)
            points = np.argwhere(valid & (labels == class_id))
            if len(points) != count:
                raise ValueError("Rare-class index differs from train labels")
            row, col = points[rank] + (source.row, source.col)
            h, w = access.scene_shapes[scene]
            row = max(0, min(max(0, h - 512), row - 256 + rng.randint(-jitter, jitter)))
            col = max(0, min(max(0, w - 512), col - 256 + rng.randint(-jitter, jitter)))
            draws.append(
                Draw(scene, WindowSpec(row, col, 512, 512, "constant"), class_id)
            )
    rng.shuffle(draws)
    return draws


def fixed_panel(access: Any, per_scene: int, seed: int) -> list[Draw]:
    rng = random.Random(seed)
    return [
        Draw(scene, window)
        for scene, shape in access.scene_shapes.items()
        for window in rng.sample(grid_windows(shape[0], shape[1], WINDOW), per_scene)
    ]


def fit_panel(index: TrainingIndex) -> list[Draw]:
    chosen = []
    for class_id in (4, 5, 0, 1, 2, 3, 4, 5):
        entry = max(
            (e for e in index["base"] if e[:2] not in [c[:2] for c in chosen]),
            key=lambda e: e[2][class_id],
        )
        chosen.append(entry)
    return [Draw(scene, WindowSpec(**window)) for scene, window, _ in chosen]


class PatchDataset(Dataset):
    def __init__(
        self,
        access: Any,
        draws: list[Draw],
        seed: int,
        pass_number: int,
        augmentation: Mapping[str, Any],
        start: int = 0,
    ) -> None:
        self.access, self.draws = access, draws
        self.seed, self.pass_number = seed, pass_number
        self.augmentation, self.start = augmentation, start

    def __len__(self) -> int:
        return len(self.draws) - self.start

    def __getitem__(self, index: int) -> dict[str, Any]:
        from .augmentation import augment

        ordinal = self.start + index
        return augment(
            self.access,
            self.draws[ordinal],
            seed=draw_seed(self.seed, self.pass_number, ordinal),
            recipe=dict(self.augmentation),
        )


def loader(
    access: Any,
    draws: list[Draw],
    seed: int,
    pass_number: int,
    augmentation: Mapping[str, Any],
    start: int = 0,
    *,
    workers: int = 2,
    cuda: bool = False,
) -> DataLoader:
    dataset = PatchDataset(access, draws, seed, pass_number, augmentation, start)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_torch_samples,
        pin_memory=cuda,
        generator=torch.Generator().manual_seed(draw_seed(seed, pass_number, 0)),
        persistent_workers=workers > 0,
        worker_init_fn=_worker_init,
        prefetch_factor=2 if workers else None,
    )


def _worker_init(_worker: int) -> None:
    import cv2  # pyright: ignore[reportMissingImports]

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
