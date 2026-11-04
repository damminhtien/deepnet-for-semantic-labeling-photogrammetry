"""Execution helpers for the Potsdam selection, refit and held-out phases."""

from __future__ import annotations

import logging
import time
from dataclasses import replace

import numpy as np
import torch

from deepnet.data.dataset import collate_torch_samples
from deepnet.data.evaluation import (
    confusion_matrix,
    evaluate_scenes,
    summarize_confusion,
)
from deepnet.models.segmenter import MultimodalSegmenter
from deepnet.models.specs import load_model_spec
from deepnet.potsdam_data import STRIDE, WINDOW, augment_patch, dsm_model_view
from deepnet.training import _autocast, _optimizer_step
from deepnet.training_utils import (
    learning_rate_factor,
    make_optimizer,
    verify_pretrained,
)

LOGGER = logging.getLogger(__name__)
MODEL_ID = "dinov2-vitb14-multimodal-v3"
CONFIG = {
    "execution": {"mixed_precision": "float16", "gradient_clip_norm": 1.0},
    "loss": {"ignore_index": 6, "cross_entropy_weight": 0.7, "soft_dice_weight": 0.3},
}
DINO3_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
DINO3_FILES = {
    "config.json": "3c9cc418f4622fd6d5587fd142b6f3cba0ba6a69f67ced907d8b7f26118451ec",
    "model.safetensors": (
        "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b"
    ),
}


def model_spec(recipe, model_id=MODEL_ID):
    spec = load_model_spec(model_id)
    return replace(
        spec,
        input_adapter=replace(spec.input_adapter, spectral_index=recipe["spectral"]),
    )


def load_model(bundle, pretrained, runtime, recipe, model_id=MODEL_ID, spec=None):
    spec = spec or model_spec(recipe, model_id)
    domain, model_input = dsm_model_view(bundle.model_input, recipe["dsm"])
    model = MultimodalSegmenter.from_pretrained(
        domain, model_input, spec, pretrained_path=pretrained
    ).to(runtime.device)
    if runtime.device_type == "cuda":
        model.backbone.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def verified_weights(pretrained, spec):
    manifest = verify_pretrained(
        pretrained, spec.backbone.model_id, spec.backbone.revision
    )
    if not {"config.json", "model.safetensors"} <= {
        file["path"] for file in manifest["files"]
    }:
        raise ValueError(
            "Pretrained manifest must cover the actual model and configuration"
        )
    return manifest


def optimizer_and_scaler(model, runtime, lr_multiplier=1):
    optimizer = make_optimizer(
        model,
        {
            "learning_rates": {
                "lora": 1e-4 * lr_multiplier,
                "input_adapter_and_decoder": 3e-4 * lr_multiplier,
            },
            "weight_decay": 0.01,
        },
    )
    scaler = torch.amp.GradScaler(
        "cuda", init_scale=1024, enabled=runtime.device_type == "cuda"
    )
    return optimizer, scaler


def update(model, samples, optimizer, scaler, weights, runtime, step, total):
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * learning_rate_factor(step, total, 0.05)
    return _optimizer_step(
        model,
        [(0, collate_torch_samples([sample])) for sample in samples],
        optimizer,
        scaler,
        [weights],
        runtime,
        CONFIG,
    )


def sync(runtime):
    if runtime.device_type == "cuda":
        torch.cuda.synchronize(runtime.device)
    elif runtime.device_type == "mps":
        torch.mps.synchronize()


def describe_panel(panel):
    return [
        [scene, window.row, window.col, window.height, window.width]
        for scene, window in panel
    ]


@torch.no_grad()
def input_distribution(model, access, panel, runtime):
    """Measure only real, non-ignore pixels on an unchanged panel."""
    model.eval()
    adapter = model.input_adapter
    samples = (access.read_sample(scene, window) for scene, window in panel)
    views = [[], []]
    for sample in samples:
        valid = sample.valid_mask & (sample.labels != 6)
        x = torch.from_numpy(sample.image[:3][None]).to(runtime.device)
        for storage, value in zip(views, adapter.encoder_views(x), strict=True):
            storage.append(value[0].float().cpu().numpy()[:, valid])
    mean = adapter.backbone_mean.flatten().cpu().numpy()
    std = adapter.backbone_std.flatten().cpu().numpy()
    output = {}
    for name, chunks, low, high in zip(
        ("projected", "normalized"),
        views,
        (np.zeros(3), -mean / std),
        (np.ones(3), (1 - mean) / std),
        strict=True,
    ):
        output[name] = []
        for channel, lo, hi in zip(
            np.concatenate(chunks, axis=1), low, high, strict=True
        ):
            x = channel[np.isfinite(channel)]
            if not x.size:
                raise ValueError("Backbone input has no finite scored pixels")
            output[name].append(
                {
                    "samples": int(channel.size),
                    "nonfinite": int(channel.size - x.size),
                    "mean": float(x.mean(dtype=np.float64)),
                    "std": float(x.std(dtype=np.float64)),
                    "min": float(x.min()),
                    "max": float(x.max()),
                    "percentiles": dict(
                        zip(
                            ("p01", "p05", "p50", "p95", "p99"),
                            np.percentile(x, [1, 5, 50, 95, 99]).tolist(),
                            strict=True,
                        )
                    ),
                    "fraction_outside_identity": float(
                        ((x < lo - 1e-6) | (x > hi + 1e-6)).mean()
                    ),
                }
            )
    return output


def require_finite_distribution(distribution):
    if any(
        channel["nonfinite"]
        for channels in distribution.values()
        for channel in channels
    ):
        raise FloatingPointError("Non-finite backbone input on the fixed train panel")


@torch.no_grad()
def score_patches(model, access, panel, runtime, budget=None):
    matrix = np.zeros((6, 6), dtype=np.int64)
    model.eval()
    for scene, window in panel:
        if budget is not None:
            budget.check()
        sample = access.read_sample(scene, window)
        image = torch.from_numpy(sample.image[None]).to(runtime.device)
        with _autocast(runtime.device, CONFIG):
            logits = model(image)
        if not logits.isfinite().all():
            raise FloatingPointError("Non-finite probe logits")
        matrix += confusion_matrix(
            sample.labels,
            logits[0].argmax(0).cpu().numpy(),
            num_classes=6,
            ignore_label=6,
            valid_mask=sample.valid_mask,
        )
    result = summarize_confusion(matrix)
    result["class_support"] = matrix.sum(1).tolist()
    return result


def score_scenes(
    model,
    access,
    runtime,
    budget,
    scratch,
    *,
    scene_ids=None,
    profile=False,
    no_overlap=False,
):
    with _autocast(runtime.device, CONFIG):
        return evaluate_scenes(
            model,
            access,
            scene_ids or access.split.scenes_for(access.split_name),
            window_size=WINDOW,
            stride=None if no_overlap else STRIDE,
            batch_size=4 if runtime.device_type == "cuda" else 1,
            device=runtime.device,
            scratch_dir=scratch,
            profile=profile,
            diagnostic_no_overlap=no_overlap,
            check_budget=budget.check,
        )


def sampled_batch(access, schedule, step, rng, *, strong):
    samples = []
    for scene, window, rare in schedule[4 * step : 4 * step + 4]:
        sample = access.read_sample(scene, window)
        samples.append(augment_patch(sample, rng.getrandbits(32), strong, rare))
    return samples


def budget_estimate(
    fit_update_seconds,
    scene_seconds,
    probe_seconds,
    arm_hours,
    refit_hours,
    heldout_hours,
):
    """Use measured timings plus 25% headroom; resolve length before selection."""
    per_pass = 864 * fit_update_seconds + probe_seconds
    for passes in (6, 4):
        if (
            1.25 * (passes * per_pass + passes // 2 * 6 * scene_seconds)
            <= arm_hours * 3600
            and 1.25 * 1152 * fit_update_seconds <= refit_hours * 3600
            and 1.25 * 14 * scene_seconds <= heldout_hours * 3600
        ):
            return passes
    raise TimeoutError("Pilot does not fit the reserved budget, even with four passes")


def timed(call, runtime):
    sync(runtime)
    start = time.monotonic()
    result = call()
    sync(runtime)
    return result, time.monotonic() - start
