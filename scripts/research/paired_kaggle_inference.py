"""Inference-only paired prediction export for the archived Potsdam cohort."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deepnet.data.evaluation import _weight_map, confusion_matrix, summarize_confusion
from deepnet.data.sampling import grid_windows
from deepnet.potsdam_rgbir.data import CacheBundle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _predict_scene(
    model: torch.nn.Module,
    access: Any,
    scene_id: str,
    *,
    model_name: str,
    output_dir: Path,
    device: torch.device,
    window_size: tuple[int, int],
    stride: tuple[int, int],
    batch_size: int,
    mixed_precision: str,
) -> dict[str, Any]:
    height, width = access.scene_shapes[scene_id]
    windows = grid_windows(height, width, window_size, stride)
    weight_template = _weight_map(*window_size)
    scratch = output_dir / ".scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    prefix = f"{scene_id}-{time.time_ns()}"
    scores = np.memmap(
        scratch / f"{prefix}-scores.bin",
        mode="w+",
        dtype=np.float32,
        shape=(6, height, width),
    )
    weight_sum = np.memmap(
        scratch / f"{prefix}-weights.bin",
        mode="w+",
        dtype=np.float32,
        shape=(height, width),
    )
    truth = np.memmap(
        scratch / f"{prefix}-truth.bin",
        mode="w+",
        dtype=np.int64,
        shape=(height, width),
    )
    valid_scene = np.memmap(
        scratch / f"{prefix}-valid.bin",
        mode="w+",
        dtype=np.bool_,
        shape=(height, width),
    )
    scores[:] = 0
    weight_sum[:] = 0
    truth[:] = access.preprocessing.ignore_label
    valid_scene[:] = False

    for start in range(0, len(windows), batch_size):
        batch_windows = windows[start : start + batch_size]
        samples = [access.read_sample(scene_id, window) for window in batch_windows]
        inputs = torch.from_numpy(np.stack([sample.image for sample in samples])).to(
            device=device, dtype=torch.float32
        )
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda" and mixed_precision == "float16",
            ),
        ):
            logits = model(inputs)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = logits.detach().float().cpu().numpy()
        if logits.shape != (len(samples), 6, *window_size):
            raise ValueError(f"Unexpected logits shape for {scene_id}: {logits.shape}")
        if not np.isfinite(logits).all():
            raise ValueError(f"Non-finite logits for {scene_id}")

        for sample, window, patch_logits in zip(
            samples, batch_windows, logits, strict=True
        ):
            real_height = min(window.height, height - window.row)
            real_width = min(window.width, width - window.col)
            row_slice = slice(window.row, window.row + real_height)
            col_slice = slice(window.col, window.col + real_width)
            valid = sample.valid_mask[:real_height, :real_width]
            labels = sample.labels[:real_height, :real_width]
            prior_valid = valid_scene[row_slice, col_slice]
            prior_labels = truth[row_slice, col_slice]
            if np.any(prior_valid & valid & (prior_labels != labels)):
                raise ValueError(f"Overlapping labels disagree in {scene_id}")
            prior_labels[valid] = labels[valid]
            valid_scene[row_slice, col_slice] |= valid
            weights = weight_template[:real_height, :real_width] * valid
            scores[:, row_slice, col_slice] += (
                patch_logits[:, :real_height, :real_width] * weights[None, ...]
            )
            weight_sum[row_slice, col_slice] += weights

    prediction = np.lib.format.open_memmap(
        scratch / f"{prefix}-prediction.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(height, width),
    )
    prediction[:] = access.preprocessing.ignore_label
    matrix = np.zeros((6, 6), dtype=np.int64)
    scored_pixels = 0
    for row in range(0, height, 256):
        row_end = min(height, row + 256)
        block_truth = np.asarray(truth[row:row_end])
        block_valid = np.asarray(valid_scene[row:row_end]) & (
            block_truth != access.preprocessing.ignore_label
        )
        if np.any(block_valid & (np.asarray(weight_sum[row:row_end]) <= 0)):
            raise RuntimeError(f"Uncovered valid pixels in {scene_id}")
        if not np.any(block_valid):
            continue
        block_scores = np.asarray(scores[:, row:row_end])
        block_weights = np.asarray(weight_sum[row:row_end])
        block_prediction = np.argmax(
            block_scores / np.maximum(block_weights, 1e-3)[None, ...], axis=0
        ).astype(np.uint8)
        prediction[row:row_end] = np.where(
            block_valid, block_prediction, access.preprocessing.ignore_label
        )
        matrix += confusion_matrix(
            block_truth,
            block_prediction,
            num_classes=6,
            ignore_label=access.preprocessing.ignore_label,
            valid_mask=block_valid,
        )
        scored_pixels += int(block_valid.sum())

    output_path = output_dir / "predictions" / model_name / f"{scene_id}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.flush()
    np.savez_compressed(output_path, prediction=prediction)
    prediction_hash = _sha256(output_path)
    result = {
        **summarize_confusion(matrix),
        "scene_id": scene_id,
        "window_count": len(windows),
        "valid_pixels": scored_pixels,
        "prediction_path": output_path.relative_to(output_dir).as_posix(),
        "prediction_sha256": prediction_hash,
    }
    del scores, weight_sum, truth, valid_scene, prediction
    for path in scratch.glob(f"{prefix}-*"):
        path.unlink()
    return result


def _verify_expected(
    model_name: str, actual_scenes: list[dict[str, Any]], expected: dict[str, Any]
) -> None:
    actual_matrix = np.asarray(
        summarize_confusion(
            sum(
                (
                    np.asarray(row["confusion_matrix"], dtype=np.int64)
                    for row in actual_scenes
                ),
                np.zeros((6, 6), dtype=np.int64),
            )
        )["confusion_matrix"],
        dtype=np.int64,
    )
    if not np.array_equal(actual_matrix, np.asarray(expected["confusion_matrix"])):
        raise ValueError(f"Supplemental {model_name} matrix differs from archived run")
    if [row["scene_id"] for row in actual_scenes] != expected["scene_ids"]:
        raise ValueError(f"Supplemental {model_name} scene order differs")
    for actual, reference in zip(actual_scenes, expected["scenes"], strict=True):
        if not np.array_equal(
            np.asarray(actual["confusion_matrix"]),
            np.asarray(reference["confusion_matrix"]),
        ):
            raise ValueError(
                f"Supplemental {model_name} matrix differs for {actual['scene_id']}"
            )


def _index_outputs(root: Path) -> None:
    records = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.name == "output-index.json"
            or ".scratch" in path.parts
        ):
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    (root / "output-index.json").write_text(
        json.dumps({"files": records}, indent=2) + "\n"
    )


def run_inference(
    *,
    cache: CacheBundle,
    model_factories: dict[str, Any],
    references: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if cache.manifest["included_splits"] != ["test"]:
        raise ValueError("Paired analysis requires the locked held-out cache only")
    protocol = references["protocol"]
    if cache.manifest["cache_hash"] != protocol["cache_hash"]:
        raise ValueError("Paired-analysis cache hash differs from archived benchmark")
    if cache.split.content_hash != protocol["split_hash"]:
        raise ValueError("Paired-analysis split differs from archived benchmark")
    scene_ids = protocol["scene_ids"]
    if list(cache.split.test_scene_ids) != scene_ids:
        raise ValueError("Held-out scene order differs from archived benchmark")
    if protocol["window_size"] != [512, 512] or protocol["stride"] != [256, 256]:
        raise ValueError("Paired-analysis inference policy is not the archived policy")
    if (
        protocol["inference"] != "identity"
        or protocol["batch_size"] != 4
        or protocol["mixed_precision"] not in {"float16", "none"}
    ):
        raise ValueError("Paired-analysis inference settings differ from benchmark")

    started = time.monotonic()
    scene_results = {}
    for model_name in ("dinov3", "pspnet"):
        if model_name not in model_factories:
            raise ValueError(f"Missing inference model factory: {model_name}")
        if device.type == "cuda":
            # Reset/read the current device without passing an ordinal. Kaggle's
            # PyTorch 2.11 build rejects an integer here despite accepting it
            # for tensor placement.
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats()
        model = model_factories[model_name]()
        model_started = time.monotonic()
        model.eval()
        with cache.access("test", auxiliary=False) as access:
            records = [
                _predict_scene(
                    model,
                    access,
                    scene_id,
                    model_name=model_name,
                    output_dir=output_dir,
                    device=device,
                    window_size=(512, 512),
                    stride=(256, 256),
                    batch_size=4,
                    mixed_precision=protocol["mixed_precision"],
                )
                for scene_id in scene_ids
            ]
        _verify_expected(model_name, records, references[model_name])
        scene_results[model_name] = {
            "scenes": records,
            "inference_seconds": time.monotonic() - model_started,
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated() if device.type == "cuda" else None
            ),
            "peak_cuda_reserved_bytes": (
                torch.cuda.max_memory_reserved() if device.type == "cuda" else None
            ),
        }
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest = {
        "status": "verified-exact-confusion-reproduction",
        "analysis_kind": "supplemental-paired-prediction-inference",
        "training_performed": False,
        "independent_benchmark_claim": False,
        "original_benchmark_artifacts_changed": False,
        "cache_hash": protocol["cache_hash"],
        "split_hash": protocol["split_hash"],
        "scene_ids": scene_ids,
        "pixels_scored": protocol["pixels_scored"],
        "window_size": protocol["window_size"],
        "stride": protocol["stride"],
        "batch_size": protocol["batch_size"],
        "inference": protocol["inference"],
        "models": {
            name: {
                "checkpoint_sha256": references[name]["checkpoint_sha256"],
                "archived_metrics_sha256": references[name]["metrics_sha256"],
                **result,
            }
            for name, result in scene_results.items()
        },
        "wall_seconds": time.monotonic() - started,
    }
    (output_dir / "paired-predictions.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    _index_outputs(output_dir)
    return manifest
