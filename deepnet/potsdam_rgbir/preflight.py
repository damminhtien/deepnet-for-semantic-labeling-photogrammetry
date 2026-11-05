"""CUDA quality/throughput pilot that freezes the model roster and schedule."""

from __future__ import annotations

import gc
import logging
import time

import numpy as np
import torch

from deepnet.data.contracts import WindowSpec, canonical_hash
from deepnet.runtime import configure_torch_runtime, seed_everything
from deepnet.training_utils import artifact_index, write_json
from semantic_labeling.losses import inverse_sqrt_class_weights

from .data import CacheBundle
from .engine import (
    describe,
    distribution,
    fit_small,
    optimizer_and_scaler,
    score_patches,
    score_scenes,
    sync,
    timed,
    train_step,
)
from .model import MODEL_SPECS, load_model, real_smoke
from .protocol import (
    Budget,
    choose_schedule,
    device_binding,
    freeze,
    freeze_protocol,
    read_json,
    source_hash,
)
from .reference import RasterioReferenceAccess
from .sampling import fixed_panel, loader, schedule, training_index

LOGGER = logging.getLogger(__name__)


def load_config(path):
    config = read_json(path)
    if (
        config["schema_version"] != 2
        or config["window_size"] != [512, 512]
        or config["stride"] != [256, 256]
        or config["global_batch"] != 4
        or config["models"] != MODEL_SPECS
        or sum(v for k, v in config["gpu_hours"].items() if k != "total") != 24
    ):
        raise ValueError("Unsupported RGBIR study configuration")
    return config


def _profile(
    model,
    train,
    val,
    index,
    panel,
    runtime,
    config,
    budget,
    scratch,
    calibration,
    setup_seconds,
):
    weights = torch.as_tensor(
        inverse_sqrt_class_weights(index["class_support"]), dtype=torch.float32
    )
    optimizer, scaler = optimizer_and_scaler(model, runtime, config)
    draws = schedule(index, train, 42, 0, jitter=config["sampling"]["center_jitter"])
    data = loader(
        train,
        draws,
        42,
        0,
        config["augmentation"]["enhanced"],
        workers=runtime.num_workers,
        cuda=True,
    )
    iterator = iter(data)
    durations = []
    model.train()
    for step in range(48):
        budget.check()
        sync(runtime)
        started = time.monotonic()
        batches = [next(iterator) for _ in range(4)]
        train_step(
            model, batches, optimizer, scaler, weights, runtime, config, step, 12 * 864
        )
        sync(runtime)
        durations.append(time.monotonic() - started)
    del iterator, data, optimizer, scaler
    _, probe_seconds = timed(
        lambda: score_patches(model, val, panel, runtime, config, budget), runtime
    )
    scene = "4_12"
    val.profile_times = {}
    for batch in (4, 2, 1):
        try:
            cached, scene_seconds = timed(
                lambda batch=batch: score_scenes(
                    model,
                    val,
                    runtime,
                    config,
                    budget,
                    scratch,
                    scene_ids=[scene],
                    profile=True,
                    batch_size=batch,
                ),
                runtime,
            )
            break
        except torch.cuda.OutOfMemoryError:
            if batch == 1:
                raise
            torch.cuda.empty_cache()
    cached_io = val.profile_times
    val.profile_times = None
    tta, tta_seconds = timed(
        lambda: score_scenes(
            model,
            val,
            runtime,
            config,
            budget,
            scratch,
            scene_ids=[scene],
            profile=True,
            policy="hflip",
            batch_size=batch,
        ),
        runtime,
    )
    result = {
        "update_seconds": float(np.median(durations[8:])),
        "update_samples": durations,
        "probe_seconds": probe_seconds,
        "scene_seconds": scene_seconds,
        "tta_scene_seconds": tta_seconds,
        "setup_seconds": setup_seconds,
        "batch_size": batch,
        "cached_scene": cached,
        "cache_io_preprocessing_seconds": cached_io,
        "tta_scene": tta,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(runtime.device),
    }
    if calibration:
        with RasterioReferenceAccess(train.bundle, calibration, auxiliary=True) as raw:
            for window in (
                WindowSpec(0, 0, 512, 512),
                WindowSpec(5632, 5632, 512, 512, "constant"),
            ):
                a, b = raw.read_sample(scene, window), val.read_sample(scene, window)
                np.testing.assert_array_equal(a.labels, b.labels)
                np.testing.assert_array_equal(a.valid_mask, b.valid_mask)
                np.testing.assert_allclose(a.image, b.image, atol=1e-7, rtol=0)
            raw.profile_times = {}
            raster, raster_seconds = timed(
                lambda: score_scenes(
                    model,
                    raw,
                    runtime,
                    config,
                    budget,
                    scratch,
                    scene_ids=[scene],
                    profile=True,
                    batch_size=batch,
                ),
                runtime,
            )
            raster_io = raw.profile_times
        if raster["confusion_matrix"] != cached["confusion_matrix"]:
            raise ValueError(
                "Cache and Rasterio changed the full-scene prediction matrix"
            )
        nonoverlap, nonoverlap_seconds = timed(
            lambda: score_scenes(
                model,
                val,
                runtime,
                config,
                budget,
                scratch,
                scene_ids=[scene],
                profile=True,
                no_overlap=True,
                batch_size=batch,
            ),
            runtime,
        )
        result.update(
            {
                "rasterio_scene": raster,
                "rasterio_seconds": raster_seconds,
                "rasterio_io_preprocessing_seconds": raster_io,
                "cache_speedup": raster_seconds / scene_seconds,
                "nonoverlap_scene": nonoverlap,
                "nonoverlap_seconds": nonoverlap_seconds,
                "overlap_latency_ratio": scene_seconds / nonoverlap_seconds,
            }
        )
    return result


def preflight(args):
    config = load_config(args.config)
    if (args.root / "protocol.json").exists():
        raise FileExistsError("Preflight protocol is already frozen")
    runtime = configure_torch_runtime(
        args.device, num_workers=config["execution"]["num_workers"]
    )
    device = device_binding(runtime)
    out = args.root / "preflight"
    with Budget(
        out, config["gpu_hours"]["preflight"], device["allocated_devices"]
    ) as budget:
        started = time.monotonic()
        bundle = CacheBundle.load(args.cache)
        if (
            bundle.manifest["included_splits"] != ["train", "validation"]
            or bundle.split.content_hash != config["selection_split_hash"]
        ):
            raise ValueError(
                "Preflight requires only the locked train/validation cache"
            )
        cache_setup = time.monotonic() - started
        with (
            bundle.access("train", auxiliary=True) as train,
            bundle.access("validation", auxiliary=True) as val,
        ):
            index = training_index(train)
            if len(index["base"]) != 2592:
                raise ValueError("Expected 2,592 original training windows")
            if (
                sum(
                    bundle.manifest["scenes"][s]["valid_pixels"]
                    for s in bundle.split.validation_scene_ids
                )
                != config["validation_pixels"]
            ):
                raise ValueError("Validation support differs from the locked reference")
            freeze(args.root / "training-index.json", index)
            train_panel, probe = fixed_panel(train, 2, 151), fixed_panel(val, 12, 271)
            weights = torch.as_tensor(
                inverse_sqrt_class_weights(index["class_support"]), dtype=torch.float32
            )
            report = {
                "runtime": device,
                "models": {},
                "cache_verify_seconds": cache_setup,
            }
            timings, model_bindings = {}, {}
            for size, pretrained in (
                ("B", args.pretrained),
                ("L", args.pretrained_large),
            ):
                if pretrained is None:
                    report["models"][size] = {
                        "status": "skipped",
                        "reason": "Pinned optional snapshot is unavailable",
                    }
                    continue
                if size == "L" and budget.remaining() < 1.25 * (
                    3 * (512 + 48) * timings["B"]["update_seconds"]
                    + 9 * timings["B"]["tta_scene_seconds"]
                ):
                    report["models"][size] = {
                        "status": "skipped",
                        "reason": "Preflight budget cannot cover the optional L pilot",
                    }
                    continue
                model = None
                try:
                    budget.check()
                    seed_everything(42)
                    torch.cuda.reset_peak_memory_stats(runtime.device)
                    tick = time.monotonic()
                    model, binding = load_model(
                        pretrained, size, True, runtime.device, checkpointing=True
                    )
                    setup = time.monotonic() - tick + cache_setup
                    sample = train.read_sample(
                        bundle.split.train_scene_ids[0],
                        WindowSpec(1024, 1024, 224, 224),
                    )
                    smoke = real_smoke(model, sample)
                    initial = distribution(model, train, train_panel, runtime)
                    small = fit_small(
                        model, train, index, runtime, config, budget, weights
                    )
                    measured = {
                        "status": "passed" if small["passed"] else "failed",
                        "binding": binding,
                        "smoke": smoke,
                        "small_fit": small,
                        "initial_inputs": initial,
                        "fit_inputs": distribution(model, train, train_panel, runtime),
                    }
                    report["models"][size] = measured
                    write_json(out / "report.json", report)
                    if not small["passed"]:
                        if size == "B":
                            raise RuntimeError(
                                "DINOv3-B failed the fixed eight-patch fit gate"
                            )
                        continue
                    timings[size] = _profile(
                        model,
                        train,
                        val,
                        index,
                        probe,
                        runtime,
                        config,
                        budget,
                        out / ".scratch",
                        args.calibration if size == "B" else None,
                        setup,
                    )
                    model_bindings[size] = binding
                    measured["timing"] = timings[size]
                    LOGGER.info(
                        "Pilot %s: update=%.3fs scene=%.2fs",
                        size,
                        timings[size]["update_seconds"],
                        timings[size]["scene_seconds"],
                    )
                except (
                    FileNotFoundError,
                    ValueError,
                    torch.cuda.OutOfMemoryError,
                ) as error:
                    if size == "B":
                        raise
                    report["models"][size] = {
                        "status": "skipped",
                        "reason": f"{type(error).__name__}: {error}",
                    }
                finally:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    write_json(out / "report.json", report)
            resolved = choose_schedule(
                timings, device["allocated_devices"], include_large="L" in timings
            )
            payload = {
                **config,
                **resolved,
                "source_hash": source_hash(),
                "runtime": device,
                "selection_cache_hash": bundle.manifest["cache_hash"],
                "source_bundle_sha256": bundle.manifest["source_bundle_sha256"],
                "rgbir_snapshot_hash": bundle.manifest["rgbir_snapshot_hash"],
                "dsm_manifest_sha256": bundle.manifest["dsm_manifest_sha256"],
                "relief_scale": bundle.manifest["relief_scale"],
                "feature_recipe": bundle.manifest["feature_recipe"],
                "training_index_hash": canonical_hash(index),
                "train_class_support": index["class_support"],
                "input_panel": describe(train_panel),
                "probe_panel": describe(probe),
                "full_validation_passes": [resolved["passes"] // 2, resolved["passes"]],
                "pretrained_bindings": model_bindings,
                "timings": timings,
                "selection_assets": bundle.manifest["scenes"],
                "selection_file_hashes": {
                    f["path"]: f["sha256"] for f in bundle.manifest["files"]
                },
            }
            budget.check()
            freeze_protocol(args.root, payload)
    artifact_index(out)
    return report
