"""Full-scene tiled evaluation with overlap weighting and data provenance."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from .contracts import canonical_hash
from .dataset import SceneAccess
from .sampling import grid_windows


def confusion_matrix(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    num_classes: int,
    ignore_label: int,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    if truth.shape != prediction.shape:
        raise ValueError("Truth and prediction arrays must have equal shapes")
    if not np.issubdtype(truth.dtype, np.integer) or not np.issubdtype(
        prediction.dtype, np.integer
    ):
        raise ValueError("Truth and prediction arrays must contain integer class IDs")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if ignore_label < num_classes or ignore_label < 0:
        raise ValueError("ignore_label must be non-negative and outside class range")
    if valid_mask is not None:
        supplied_mask = np.asarray(valid_mask, dtype=bool)
        if supplied_mask.shape != truth.shape:
            raise ValueError("Validity mask must match truth and prediction shapes")
    else:
        supplied_mask = np.ones(truth.shape, dtype=bool)
    valid_targets = (truth >= 0) & (truth < num_classes)
    ignored_targets = truth == ignore_label
    invalid_targets = supplied_mask & ~(valid_targets | ignored_targets)
    if np.any(invalid_targets):
        values = np.unique(truth[invalid_targets]).tolist()
        raise ValueError(f"Truth contains undeclared class IDs: {values[:10]}")
    valid = supplied_mask & valid_targets
    if np.any((prediction[valid] < 0) | (prediction[valid] >= num_classes)):
        raise ValueError("Predictions contain class IDs outside the configured range")
    encoded = truth[valid].astype(np.int64) * num_classes + prediction[valid].astype(
        np.int64
    )
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def summarize_confusion(matrix: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Confusion matrix must be square")
    true_positive = np.diag(matrix)
    ground_truth = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    union = ground_truth + predicted - true_positive
    iou = np.divide(
        true_positive,
        union,
        out=np.full(matrix.shape[0], np.nan, dtype=np.float64),
        where=union != 0,
    )
    total = int(matrix.sum())
    return {
        "overall_accuracy": float(true_positive.sum() / total) if total else 0.0,
        "mean_iou": float(np.nanmean(iou)) if np.any(np.isfinite(iou)) else 0.0,
        "per_class_iou": [
            None if not np.isfinite(value) else float(value) for value in iou
        ],
        "pixels_scored": total,
        "confusion_matrix": matrix.tolist(),
    }


def _weight_map(height: int, width: int) -> np.ndarray:
    rows = np.linspace(-math.pi, math.pi, height, dtype=np.float32)
    cols = np.linspace(-math.pi, math.pi, width, dtype=np.float32)
    row_weights = 0.5 * (1.0 + np.cos(rows))
    col_weights = 0.5 * (1.0 + np.cos(cols))
    return np.maximum(np.outer(row_weights, col_weights), 1e-3)


def _model_device(model: Any, torch: Any, requested: Any | None) -> Any:
    if requested is not None:
        return torch.device(requested)
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def _evaluate_scene(
    model: Any,
    access: SceneAccess,
    scene_id: str,
    *,
    number_of_classes: int,
    window_size: tuple[int, int],
    stride: tuple[int, int] | None,
    batch_size: int,
    device: Any,
    scratch_root: Path,
    torch: Any,
) -> dict[str, Any]:
    grid = access.scenes[scene_id].reference_grid
    scene_height, scene_width = grid.height, grid.width
    windows = grid_windows(scene_height, scene_width, window_size, stride)
    weight_template = _weight_map(*window_size)
    shape = (number_of_classes, scene_height, scene_width)
    scores = np.memmap(
        scratch_root / "scores.bin", mode="w+", dtype=np.float32, shape=shape
    )
    weight_sum = np.memmap(
        scratch_root / "weights.bin",
        mode="w+",
        dtype=np.float32,
        shape=(scene_height, scene_width),
    )
    truth_map = np.memmap(
        scratch_root / "truth.bin",
        mode="w+",
        dtype=np.int64,
        shape=(scene_height, scene_width),
    )
    valid_scene = np.memmap(
        scratch_root / "valid.bin",
        mode="w+",
        dtype=np.bool_,
        shape=(scene_height, scene_width),
    )
    scores[:] = 0
    weight_sum[:] = 0
    truth_map[:] = access.preprocessing.ignore_label
    valid_scene[:] = False

    for start in range(0, len(windows), batch_size):
        batch_windows = windows[start : start + batch_size]
        samples = [access.read_sample(scene_id, window) for window in batch_windows]
        inputs = torch.from_numpy(np.stack([sample.image for sample in samples])).to(
            device=device, dtype=torch.float32
        )
        outputs = model(inputs)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        logits = outputs.detach().float().cpu().numpy()
        if logits.shape[1] != number_of_classes:
            raise ValueError(
                f"Model returned {logits.shape[1]} classes; "
                f"expected {number_of_classes}"
            )
        if logits.shape[-2:] != window_size:
            raise ValueError(
                f"Model output shape {logits.shape[-2:]} "
                f"differs from window {window_size}"
            )
        if not np.isfinite(logits).all():
            raise ValueError("Model returned non-finite logits")

        for sample, window, patch_logits in zip(
            samples, batch_windows, logits, strict=True
        ):
            real_height = min(window.height, scene_height - window.row)
            real_width = min(window.width, scene_width - window.col)
            row_slice = slice(window.row, window.row + real_height)
            col_slice = slice(window.col, window.col + real_width)
            valid = sample.valid_mask[:real_height, :real_width]
            label = sample.labels[:real_height, :real_width]
            prior_valid = valid_scene[row_slice, col_slice]
            prior_labels = truth_map[row_slice, col_slice]
            inconsistent = prior_valid & valid & (prior_labels != label)
            if np.any(inconsistent):
                raise ValueError(
                    f"Overlapping label windows disagree in scene {scene_id}"
                )
            prior_labels[valid] = label[valid]
            valid_scene[row_slice, col_slice] |= valid
            weights = weight_template[:real_height, :real_width] * valid
            scores[:, row_slice, col_slice] += (
                patch_logits[:, :real_height, :real_width] * weights[None, ...]
            )
            weight_sum[row_slice, col_slice] += weights

    matrix = np.zeros((number_of_classes, number_of_classes), dtype=np.int64)
    scored_pixels = 0
    rows_per_chunk = 256
    for row in range(0, scene_height, rows_per_chunk):
        row_end = min(scene_height, row + rows_per_chunk)
        valid = np.asarray(valid_scene[row:row_end]) & (
            np.asarray(truth_map[row:row_end]) != access.preprocessing.ignore_label
        )
        if np.any(valid & (np.asarray(weight_sum[row:row_end]) <= 0)):
            raise RuntimeError(
                f"Some valid pixels were not covered in scene {scene_id}"
            )
        if not np.any(valid):
            continue
        score_block = np.asarray(scores[:, row:row_end])
        weights = np.asarray(weight_sum[row:row_end])
        predictions = np.argmax(
            score_block / np.maximum(weights, 1e-3)[None, ...], axis=0
        )
        labels = np.asarray(truth_map[row:row_end])
        matrix += confusion_matrix(
            labels,
            predictions,
            num_classes=number_of_classes,
            ignore_label=access.preprocessing.ignore_label,
            valid_mask=valid,
        )
        scored_pixels += int(valid.sum())

    del scores, weight_sum, truth_map, valid_scene
    result = summarize_confusion(matrix)
    result.update(
        {
            "scene_id": scene_id,
            "window_count": len(windows),
            "valid_pixels": scored_pixels,
        }
    )
    return result


def evaluate_scenes(
    model: Any,
    access: SceneAccess,
    scene_ids: Sequence[str],
    *,
    window_size: tuple[int, int],
    stride: tuple[int, int] | None = None,
    batch_size: int = 1,
    device: Any | None = None,
    class_names: Sequence[str] | None = None,
    scratch_dir: Path | None = None,
) -> dict[str, Any]:
    """Score scenes; labels are read only through the selected split contract."""
    if not scene_ids:
        raise ValueError("At least one scene is required for evaluation")
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("Evaluation scene IDs must not contain duplicates")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if access.split_name not in {"validation", "test"}:
        raise ValueError(
            "Full-scene evaluation requires validation or test data access"
        )
    allowed = set(access.split.scenes_for(access.split_name))
    if not set(scene_ids) <= allowed:
        raise ValueError(
            "Evaluation scene IDs must come from the selected manifest split"
        )
    number_of_classes = sum(
        class_id != access.preprocessing.ignore_label
        for _, _, _, class_id in access.preprocessing.label_palette
    )
    if class_names is not None and len(class_names) != number_of_classes:
        raise ValueError("class_names must match the model's class count")
    if scratch_dir is not None:
        Path(scratch_dir).expanduser().mkdir(parents=True, exist_ok=True)

    import torch

    class_ids = sorted(
        class_id
        for _, _, _, class_id in access.preprocessing.label_palette
        if class_id != access.preprocessing.ignore_label
    )
    if class_ids != list(range(len(class_ids))):
        raise ValueError("Model classes must use contiguous IDs starting from zero")
    number_of_classes = len(class_ids)
    if access.preprocessing.ignore_label < number_of_classes:
        raise ValueError("The ignore label must be outside the model class range")
    selected_device = _model_device(model, torch, device)
    model.eval()
    total_confusion = np.zeros((number_of_classes, number_of_classes), dtype=np.int64)
    scene_results = []
    with TemporaryDirectory(dir=scratch_dir, prefix="deepnet-evaluation-") as scratch:
        for index, scene_id in enumerate(scene_ids):
            scene_scratch = Path(scratch) / f"scene-{index:04d}"
            scene_scratch.mkdir()
            with torch.no_grad():
                scene_result = _evaluate_scene(
                    model,
                    access,
                    scene_id,
                    number_of_classes=number_of_classes,
                    window_size=window_size,
                    stride=stride,
                    batch_size=batch_size,
                    device=selected_device,
                    scratch_root=scene_scratch,
                    torch=torch,
                )
            matrix = np.asarray(scene_result.pop("confusion_matrix"), dtype=np.int64)
            total_confusion += matrix
            scene_results.append(scene_result)

    result = summarize_confusion(total_confusion)
    result.update(
        {
            "dataset_id": access.split.dataset_id,
            "split_id": access.split.split_id,
            "split_version": access.split.version,
            "split_hash": access.split.content_hash,
            "preprocessing_id": access.preprocessing.preprocessing_id,
            "preprocessing_version": access.preprocessing.version,
            "preprocessing_hash": access.preprocessing.content_hash,
            "model_input_id": access.model_input.model_id,
            "model_input_version": access.model_input.version,
            "model_input_hash": canonical_hash(access.model_input.to_dict()),
            "scene_ids": list(scene_ids),
            "window_size": list(window_size),
            "stride": list(stride or window_size),
            "scenes": scene_results,
        }
    )
    if class_names is not None:
        result["class_names"] = list(class_names)
    return result
