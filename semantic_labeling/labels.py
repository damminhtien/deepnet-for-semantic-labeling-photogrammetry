"""Conversions between ISPRS RGB labels and integer class IDs."""

from collections.abc import Mapping

import numpy as np

from .constants import INVERT_PALETTE, PALETTE


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
