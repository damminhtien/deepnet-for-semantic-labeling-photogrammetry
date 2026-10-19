"""Stable label and input contracts used by the model notebooks."""

LABELS = (
    "impervious surfaces",
    "buildings",
    "low vegetation",
    "trees",
    "cars",
    "clutter/background",
)
N_CLASSES = len(LABELS)

# Black pixels in ISPRS no-boundary references are excluded from scoring.
IGNORE_LABEL = 6

PALETTE = {
    0: (255, 255, 255),
    1: (0, 0, 255),
    2: (0, 255, 255),
    3: (0, 255, 0),
    4: (255, 255, 0),
    5: (255, 0, 0),
    IGNORE_LABEL: (0, 0, 0),
}
INVERT_PALETTE = {color: label for label, color in PALETTE.items()}

INPUT_CHANNELS = 5
INPUT_SCALE = 255.0
