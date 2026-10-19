"""Compatibility entry point for the Potsdam evaluation pipeline."""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semantic_labeling.constants import (
    INPUT_CHANNELS,
    INVERT_PALETTE,
    LABELS,
    N_CLASSES,
    PALETTE,
)
from semantic_labeling.dataset import ISPRS_dataset
from semantic_labeling.inference import run_inference
from semantic_labeling.labels import convert_from_color, convert_to_color
from semantic_labeling.metrics import accuracy, metrics
from semantic_labeling.preprocessing import assemble_input, normalize_input
from semantic_labeling.sampling import (
    count_sliding_window,
    get_random_pos,
    grouper,
    sliding_window,
)

WINDOW_SIZE = (256, 256)
STRIDE = 32
IN_CHANNELS = INPUT_CHANNELS
BATCH_SIZE = 6
WEIGHTS = torch.ones(N_CLASSES)
CACHE = True

FOLDER = "../input/potsdamvaihingen/"
MAIN_FOLDER = FOLDER
DATA_FOLDER = MAIN_FOLDER + "3_ortho_irrg/3_Ortho_IRRG/top_potsdam_{}_IRRG.tif"
DSM_FOLDER = MAIN_FOLDER + "1_dsm/1_DSM/dsm_potsdam_0{}.tif"
NDSM_FOLDER = MAIN_FOLDER + "1_dsm_normalisation/1_DSM_normalisation/dsm_potsdam_0{}_normalized_lastools.jpg"
LABEL_FOLDER = "../input/label-all-potsdam-vaihingen/5_labels_all/top_potsdam_{}_label.tif"
ERODED_FOLDER = "../input/label-all-potsdam-vaihingen/5_labels_all_noboundary/top_potsdam_{}_label_noBoundary.tif"
palette = PALETTE
invert_palette = INVERT_PALETTE
list_acc = []


def test(
    net,
    test_ids,
    all=False,
    stride=WINDOW_SIZE[0],
    batch_size=BATCH_SIZE,
    window_size=WINDOW_SIZE,
):
    """Evaluate ``net`` and retain the original optional prediction return."""
    result = run_inference(
        net,
        test_ids,
        DATA_FOLDER,
        DSM_FOLDER,
        NDSM_FOLDER,
        ERODED_FOLDER,
        stride=stride,
        batch_size=batch_size,
        window_size=window_size,
        return_predictions=all,
        return_tile_scores=all,
    )
    if all:
        score, predictions, ground_truths, tile_scores = result
        list_acc.extend(tile_scores)
        return score, predictions, ground_truths
    return result


# Keep pytest from mistaking this notebook-compatible helper for a test case.
test.__test__ = False

__all__ = (
    "BATCH_SIZE",
    "CACHE",
    "DATA_FOLDER",
    "DSM_FOLDER",
    "ERODED_FOLDER",
    "FOLDER",
    "INPUT_CHANNELS",
    "INVERT_PALETTE",
    "IN_CHANNELS",
    "LABELS",
    "LABEL_FOLDER",
    "MAIN_FOLDER",
    "NDSM_FOLDER",
    "N_CLASSES",
    "PALETTE",
    "STRIDE",
    "WEIGHTS",
    "WINDOW_SIZE",
    "ISPRS_dataset",
    "accuracy",
    "assemble_input",
    "convert_from_color",
    "convert_to_color",
    "count_sliding_window",
    "get_random_pos",
    "grouper",
    "invert_palette",
    "list_acc",
    "metrics",
    "normalize_input",
    "palette",
    "run_inference",
    "sliding_window",
    "test",
)
