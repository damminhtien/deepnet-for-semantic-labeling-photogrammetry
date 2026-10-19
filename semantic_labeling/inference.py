"""Shared tiled inference for the Potsdam evaluation notebooks."""

import math

import numpy as np
import torch
from skimage import io
from tqdm import tqdm

from .constants import N_CLASSES
from .labels import load_label
from .metrics import metrics
from .preprocessing import assemble_input, normalize_input
from .sampling import count_sliding_window, grouper, sliding_window


def run_inference(
    net,
    tile_ids,
    data_files,
    dsm_files,
    ndsm_files,
    eroded_label_files,
    stride=256,
    batch_size=6,
    window_size=(256, 256),
    return_predictions=False,
    return_tile_scores=False,
):
    """Run tiled inference and score against no-boundary reference labels."""
    if not tile_ids:
        raise ValueError("At least one test tile is required")
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {batch_size}")

    try:
        device = next(net.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    net.eval()
    predictions = []
    ground_truths = []
    tile_scores = []

    for tile_id in tqdm(tile_ids, total=len(tile_ids), leave=False):
        image = normalize_input(
            assemble_input(
                io.imread(data_files.format(tile_id)),
                io.imread(dsm_files.format(tile_id)),
                io.imread(ndsm_files.format(tile_id)),
            )
        )
        ground_truth = load_label(eroded_label_files.format(tile_id))
        if ground_truth.shape != image.shape[:2]:
            raise ValueError(
                f"Tile {tile_id} input/reference dimensions do not match: "
                f"{image.shape[:2]} and {ground_truth.shape}"
            )

        score_map = np.zeros((*image.shape[:2], N_CLASSES), dtype=np.float32)
        windows = sliding_window(image, step=stride, window_size=window_size)
        window_count = count_sliding_window(image, step=stride, window_size=window_size)
        batch_count = math.ceil(window_count / batch_size)
        batches = grouper(batch_size, windows)

        for coordinates in tqdm(batches, total=batch_count, leave=False):
            patches = np.asarray(
                [
                    image[top : top + height, left : left + width].transpose(2, 0, 1)
                    for top, left, height, width in coordinates
                ],
                dtype=np.float32,
            )
            inputs = torch.from_numpy(patches).to(device)
            with torch.no_grad():
                outputs = net(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            outputs = outputs.detach().cpu().numpy()

            for output, (top, left, height, width) in zip(outputs, coordinates):
                score_map[top : top + height, left : left + width] += output.transpose(
                    1, 2, 0
                )

        prediction = np.argmax(score_map, axis=-1)
        tile_scores.append(metrics(prediction.ravel(), ground_truth.ravel()))
        predictions.append(prediction)
        ground_truths.append(ground_truth)

    score = metrics(
        np.concatenate([prediction.ravel() for prediction in predictions]),
        np.concatenate([ground_truth.ravel() for ground_truth in ground_truths]),
    )
    if return_predictions:
        if return_tile_scores:
            return score, predictions, ground_truths, tile_scores
        return score, predictions, ground_truths
    return score
