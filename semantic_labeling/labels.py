"""Conversions between ISPRS RGB labels and integer class IDs."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from skimage import io

from .constants import IGNORE_LABEL, INVERT_PALETTE, PALETTE


def convert_to_color(labels, palette: Mapping[int, tuple[int, int, int]] = PALETTE):
    """Convert a 2D label array to RGB, rejecting unknown class IDs."""
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2D label array, got shape {labels.shape}")

    valid_labels = np.isin(labels, tuple(palette))
    if not np.all(valid_labels):
        unknown = np.unique(labels[~valid_labels]).tolist()
        raise ValueError(f"Unknown label IDs: {unknown}")

    result = np.empty((*labels.shape, 3), dtype=np.uint8)
    for label, color in palette.items():
        result[labels == label] = color
    return result


def convert_from_color(
    image,
    palette: Mapping[tuple[int, int, int], int] = INVERT_PALETTE,
):
    """Convert an RGB label image to IDs, rejecting unrecognized colors."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB label image, got shape {image.shape}")

    result = np.empty(image.shape[:2], dtype=np.uint8)
    recognized = np.zeros(image.shape[:2], dtype=bool)
    for color, label in palette.items():
        mask = np.all(image == color, axis=-1)
        result[mask] = label
        recognized |= mask

    if not np.all(recognized):
        unknown = np.unique(image[~recognized].reshape(-1, 3), axis=0).tolist()
        raise ValueError(f"Unknown RGB label colors: {unknown[:10]}")
    return result


def convert_from_color_near_palette(image, max_channel_error=64):
    """Decode a color map with bounded per-channel encoding error."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB label image, got shape {image.shape}")
    if max_channel_error < 0:
        raise ValueError(
            f"Maximum channel error must be non-negative, got {max_channel_error}"
        )

    colors = np.asarray(tuple(INVERT_PALETTE), dtype=np.int16)
    label_ids = np.asarray(tuple(INVERT_PALETTE.values()), dtype=np.uint8)
    pixels = image.reshape(-1, 3)
    result = np.empty(pixels.shape[0], dtype=np.uint8)
    maximum_error = 0
    remapped_pixels = 0
    chunk_size = 1_000_000
    for start in range(0, pixels.shape[0], chunk_size):
        stop = min(start + chunk_size, pixels.shape[0])
        values = pixels[start:stop].astype(np.int16, copy=False)
        distances = np.abs(values[:, np.newaxis, :] - colors[np.newaxis, :, :])
        distances = distances.max(axis=-1)
        nearest = distances.argmin(axis=1)
        nearest_error = distances[np.arange(stop - start), nearest]
        nearest_two = np.partition(distances, kth=1, axis=1)[:, :2]
        if np.any(nearest_two[:, 0] == nearest_two[:, 1]):
            raise ValueError("Some label colors are equally close to multiple classes")
        chunk_maximum_error = int(nearest_error.max(initial=0))
        if chunk_maximum_error > max_channel_error:
            raise ValueError(
                "Label colors exceed the permitted palette error: "
                f"{chunk_maximum_error} > {max_channel_error}"
            )
        maximum_error = max(maximum_error, chunk_maximum_error)
        remapped_pixels += int(np.count_nonzero(nearest_error))
        result[start:stop] = label_ids[nearest]

    return result.reshape(image.shape[:2]), {
        "method": "nearest_palette_with_bounded_channel_error",
        "maximum_channel_error": maximum_error,
        "remapped_pixels": remapped_pixels,
        "max_channel_error_limit": max_channel_error,
    }


def load_label(path):
    """Load a color TIFF or a compressed class-ID ``.npz`` label map."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "labels" not in archive:
                raise ValueError(f"Prepared label archive has no 'labels' array: {path}")
            labels = archive["labels"]
        if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(
                f"Prepared labels must be a 2D integer map, got {labels.shape} {labels.dtype}"
            )
        if np.any((labels < 0) | (labels > IGNORE_LABEL)):
            unknown = np.unique(labels[(labels < 0) | (labels > IGNORE_LABEL)]).tolist()
            raise ValueError(f"Prepared labels contain unknown IDs: {unknown[:10]}")
        return labels.astype(np.uint8, copy=False)
    return convert_from_color(io.imread(path))
