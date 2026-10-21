"""Historical patch coordinates used by pre-pipeline experiments.

New patch and grid strategies belong in ``deepnet.data.sampling``.
"""

import random
from itertools import islice


def get_random_pos(img, window_shape, rng=None):
    """Return a valid ``(top, bottom, left, right)`` crop for an array."""
    if len(img.shape) < 2:
        raise ValueError(
            f"Expected an image with at least two dimensions, got {img.shape}"
        )

    height, width = img.shape[-2:]
    window_height, window_width = window_shape
    if window_height <= 0 or window_width <= 0:
        raise ValueError(f"Window dimensions must be positive, got {window_shape}")
    if window_height > height or window_width > width:
        raise ValueError(
            f"Window {window_shape} exceeds image dimensions {(height, width)}"
        )

    rng = rng if rng is not None else random
    top = rng.randint(0, height - window_height)
    left = rng.randint(0, width - window_width)
    return top, top + window_height, left, left + window_width


def _window_starts(length, window_size, step):
    if step <= 0:
        raise ValueError(f"Step must be positive, got {step}")
    if window_size <= 0 or window_size > length:
        raise ValueError(
            f"Window size {window_size} must be between 1 and image length {length}"
        )
    if step > window_size:
        raise ValueError(
            f"Step {step} exceeds window size {window_size} and would leave gaps"
        )

    starts = list(range(0, length - window_size + 1, step))
    final_start = length - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def sliding_window(top, step=10, window_size=(20, 20)):
    """Yield each covering window once as ``(top, left, height, width)``."""
    if len(top.shape) < 2:
        raise ValueError(
            f"Expected an image with at least two dimensions, got {top.shape}"
        )

    window_height, window_width = window_size
    height, width = top.shape[:2]
    rows = _window_starts(height, window_height, step)
    columns = _window_starts(width, window_width, step)
    for y in rows:
        for x in columns:
            yield y, x, window_height, window_width


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """Count windows using the same coordinate generator as inference."""
    return sum(1 for _ in sliding_window(top, step, window_size))


def grouper(size, iterable):
    """Yield consecutive groups of at most ``size`` items."""
    if size <= 0:
        raise ValueError(f"Group size must be positive, got {size}")

    iterator = iter(iterable)
    while chunk := tuple(islice(iterator, size)):
        yield chunk
