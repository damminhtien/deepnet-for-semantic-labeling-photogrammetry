"""Shared optimization, diagnostics, and inference for the RGBIR study."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from deepnet.data.contracts import WindowSpec
from deepnet.data.dataset import collate_torch_samples
from deepnet.data.evaluation import (
    confusion_matrix,
    evaluate_scenes,
    summarize_confusion,
)
from deepnet.potsdam_run import input_distribution, require_finite_distribution
from deepnet.training import _autocast, _optimizer_step
from deepnet.training_utils import learning_rate_factor, make_optimizer

from .sampling import Draw, fit_panel
from .types import PanelRecord, TrainingIndex

LOGGER = logging.getLogger(__name__)


def pairs(panel: Sequence[Draw]) -> list[tuple[str, WindowSpec]]:
    return [(d.scene, d.window) for d in panel]


def _pair(values: Sequence[int]) -> tuple[int, int]:
    if len(values) != 2:
        raise ValueError("Spatial settings must contain two dimensions")
    return values[0], values[1]


def describe(panel: Sequence[Draw]) -> list[PanelRecord]:
    return [
        {
            "scene": draw.scene,
            "window": {
                "row": draw.window.row,
                "col": draw.window.col,
                "height": draw.window.height,
                "width": draw.window.width,
                "padding": draw.window.padding,
            },
        }
        for draw in panel
    ]


def restore_panel(records: Sequence[PanelRecord]) -> list[Draw]:
    from deepnet.data.contracts import WindowSpec

    return [Draw(d["scene"], WindowSpec(**d["window"])) for d in records]


def distribution(model: Any, access: Any, panel: Sequence[Draw], runtime: Any) -> dict:
    values = input_distribution(model, access, pairs(panel), runtime)
    require_finite_distribution(values)
    return values


def optimizer_and_scaler(
    model: Any, runtime: Any, config: Mapping[str, Any], multiplier: float = 1
):
    options = {
        **config["optimizer"],
        "learning_rates": {
            k: v * multiplier for k, v in config["optimizer"]["learning_rates"].items()
        },
    }
    return make_optimizer(model, options), torch.amp.GradScaler(
        "cuda", init_scale=1024, enabled=runtime.device_type == "cuda"
    )


def train_step(
    model: Any,
    batches: list[dict[str, Any]],
    optimizer: Any,
    scaler: Any,
    weights: torch.Tensor,
    runtime: Any,
    config: Mapping[str, Any],
    step: int,
    total: int,
) -> tuple[float, np.ndarray]:
    """Use shared AMP retry logic and retain metrics from the successful attempt."""
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * learning_rate_factor(step, total, 0.05)
    matrices = deque(maxlen=len(batches))
    ordinal = 0

    def collect(_model, _inputs, logits):
        nonlocal ordinal
        batch = batches[ordinal % len(batches)]
        ordinal += 1
        matrices.append(
            confusion_matrix(
                batch["labels"].numpy(),
                logits.detach().argmax(1).cpu().numpy(),
                num_classes=6,
                ignore_label=6,
                valid_mask=batch["valid_mask"].numpy(),
            )
        )

    hook = model.register_forward_hook(collect)
    try:
        loss = _optimizer_step(
            model,
            [(0, batch) for batch in batches],
            optimizer,
            scaler,
            [weights],
            runtime,
            config,
        )
    finally:
        hook.remove()
    if len(matrices) != len(batches):
        raise RuntimeError("Training metrics did not cover the successful update")
    return loss, sum(matrices, np.zeros((6, 6), dtype=np.int64))


@torch.no_grad()
def score_patches(
    model: Any,
    access: Any,
    panel: Sequence[Draw],
    runtime: Any,
    config: Mapping[str, Any],
    budget: Any | None = None,
) -> dict[str, Any]:
    model.eval()
    matrix = np.zeros((6, 6), dtype=np.int64)
    for draw in panel:
        if budget:
            budget.check()
        sample = access.read_sample(draw.scene, draw.window)
        image = torch.from_numpy(sample.image[None]).to(runtime.device)
        with _autocast(runtime.device, config):
            logits = model(image)
        if not logits.isfinite().all():
            raise FloatingPointError("Nonfinite probe logits")
        matrix += confusion_matrix(
            sample.labels,
            logits[0].argmax(0).cpu().numpy(),
            num_classes=6,
            ignore_label=6,
            valid_mask=sample.valid_mask,
        )
    return summarize_confusion(matrix)


class InferenceModel(nn.Module):
    def __init__(self, model, policy):
        super().__init__()
        if policy not in {"identity", "hflip"}:
            raise ValueError("Unknown inference policy")
        self.model, self.policy = model, policy

    def forward(self, images):
        logits = self.model(images)
        if self.policy == "hflip":
            mirrored = self.model(images.flip(-1)).flip(-1)
            return (logits.float() + mirrored.float()) / 2
        return logits


def score_scenes(
    model: Any,
    access: Any,
    runtime: Any,
    config: Mapping[str, Any],
    budget: Any,
    scratch: Any,
    *,
    scene_ids=None,
    policy="identity",
    profile=False,
    no_overlap=False,
    batch_size: int = 4,
) -> dict[str, Any]:
    with _autocast(runtime.device, config):
        return evaluate_scenes(
            InferenceModel(model, policy),
            access,
            scene_ids or access.split.scenes_for(access.split_name),
            window_size=_pair(config["window_size"]),
            stride=None if no_overlap else _pair(config["stride"]),
            batch_size=batch_size,
            device=runtime.device,
            scratch_dir=scratch,
            profile=profile,
            diagnostic_no_overlap=no_overlap,
            check_budget=budget.check,
        )


def fit_small(
    model: Any,
    access: Any,
    index: TrainingIndex,
    runtime: Any,
    config: Mapping[str, Any],
    budget: Any,
    weights: torch.Tensor,
) -> dict[str, Any]:
    panel = fit_panel(index)
    samples = [access.read_sample(d.scene, d.window) for d in panel]
    classes = set(
        np.concatenate(
            [s.labels[s.valid_mask & (s.labels < 6)] for s in samples]
        ).tolist()
    )
    if classes != set(range(6)):
        raise ValueError("The fixed eight-patch fit panel must cover all six classes")
    batches = [collate_torch_samples([s]) for s in samples]
    optimizer, scaler = optimizer_and_scaler(model, runtime, config)
    history = []
    maximum = config["small_fit"]["max_updates"]
    for step in range(maximum):
        budget.check()
        model.train()
        start = (4 * step) % 8
        loss, _ = train_step(
            model,
            batches[start : start + 4],
            optimizer,
            scaler,
            weights,
            runtime,
            config,
            step,
            maximum,
        )
        if (step + 1) % 32 == 0 or step + 1 == maximum:
            measured = score_patches(model, access, panel, runtime, config, budget)
            history.append(
                {"updates": step + 1, "loss": loss, "train_metrics": measured}
            )
            LOGGER.info(
                "Small fit update=%d train_mIoU=%.6f loss=%.5f",
                step + 1,
                measured["mean_iou"],
                loss,
            )
            if measured["mean_iou"] >= config["small_fit"]["minimum_miou"]:
                break
    return {
        "passed": history[-1]["train_metrics"]["mean_iou"]
        >= config["small_fit"]["minimum_miou"],
        "panel": describe(panel),
        "history": history,
    }


def sync(runtime: Any) -> None:
    if runtime.device_type == "cuda":
        torch.cuda.synchronize(runtime.device)


def timed(call: Any, runtime: Any) -> tuple[Any, float]:
    sync(runtime)
    begin = time.monotonic()
    result = call()
    sync(runtime)
    return result, time.monotonic() - begin
