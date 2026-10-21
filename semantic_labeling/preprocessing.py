"""Historical input assembly used by pre-pipeline experiments.

New model views are declared with ``ModelInputSpec`` and transformed by
``deepnet.data.preprocessing``.
"""

import numpy as np

from .constants import INPUT_CHANNELS, INPUT_SCALE


def _as_single_channel(image, name):
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[..., np.newaxis]
    if image.ndim != 3 or image.shape[-1] != 1:
        raise ValueError(f"{name} must have one channel, got shape {image.shape}")
    return image


def assemble_input(top, dsm, ndsm):
    """Join IR-R-G, DSM, and nDSM rasters into a five-channel HWC image."""
    top = np.asarray(top)
    if top.ndim != 3 or top.shape[-1] != 3:
        raise ValueError(f"TOP input must have three channels, got shape {top.shape}")

    dsm = _as_single_channel(dsm, "DSM")
    ndsm = _as_single_channel(ndsm, "nDSM")
    if dsm.shape[:2] != top.shape[:2] or ndsm.shape[:2] != top.shape[:2]:
        raise ValueError(
            "TOP, DSM, and nDSM rasters must share spatial dimensions; "
            f"got {top.shape[:2]}, {dsm.shape[:2]}, and {ndsm.shape[:2]}"
        )
    return np.concatenate((top, dsm, ndsm), axis=-1)


def normalize_input(image, scale=INPUT_SCALE):
    """Apply the repository's historical uniform input scale in float32."""
    if scale <= 0:
        raise ValueError(f"Input scale must be positive, got {scale}")
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != INPUT_CHANNELS:
        raise ValueError(
            f"Expected an HWC image with {INPUT_CHANNELS} channels, got {image.shape}"
        )
    return image / scale
