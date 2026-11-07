"""Immutable selection, inference, continuation, and allocated-GPU budgets."""

from __future__ import annotations

import importlib.metadata
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from typing_extensions import Self

from deepnet.data.contracts import canonical_hash
from deepnet.data.raster import sha256_file
from deepnet.potsdam_protocol import freeze, matrix_metrics, read_json
from deepnet.training_utils import write_json

from .types import (
    Candidate,
    DeviceBinding,
    EvaluationMetrics,
    InferencePolicy,
    ModelTiming,
    PhaseManifest,
    Protocol,
    Recipe,
    Schedule,
    SelectionLock,
    TTAValidation,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/training/potsdam-dinov3-rgbir-v2.json"
ARM_ORDER = "ABCLD"


def source_hash() -> str:
    files = [
        *sorted((ROOT / "deepnet").rglob("*.py")),
        ROOT / "semantic_labeling/__init__.py",
        ROOT / "semantic_labeling/constants.py",
        ROOT / "semantic_labeling/losses.py",
        *sorted((ROOT / "configs/models").glob("dinov3-*-rgbir*-v2.json")),
    ]
    return canonical_hash({str(p.relative_to(ROOT)): sha256_file(p) for p in files})


def device_binding(runtime: Any) -> DeviceBinding:
    if runtime.device_type != "cuda":
        raise ValueError("Campaign phases require CUDA; use the separate offline smoke")
    return {
        "accelerator": torch.cuda.get_device_name(runtime.device),
        "allocated_devices": torch.cuda.device_count(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "peft",
                "numpy",
                "rasterio",
            )
        },
    }


def scores(metrics: EvaluationMetrics) -> tuple[float, float, float, float]:
    car_iou, clutter_iou = metrics["per_class_iou"][4:6]
    if car_iou is None or clutter_iou is None:
        raise ValueError("Car and clutter IoU must be defined for ranking")
    return (
        metrics["overall_accuracy"],
        metrics["mean_iou"],
        car_iou,
        clutter_iou,
    )


def candidate(
    metrics: EvaluationMetrics, protocol: Protocol, arm: str, step: int
) -> Candidate:
    measured = cast(
        EvaluationMetrics,
        matrix_metrics(metrics["confusion_matrix"], protocol["validation_pixels"]),
    )
    reference = cast(
        EvaluationMetrics,
        matrix_metrics(
            protocol["reference_validation_confusion"], protocol["validation_pixels"]
        ),
    )
    current, floor = scores(measured), scores(reference)
    return {
        "eligible": current[0] > floor[0]
        and all(a >= b for a, b in zip(current[1:], floor[1:], strict=True)),
        "arm": arm,
        "step": step,
        "overall_accuracy": current[0],
        "mean_iou": current[1],
        "car_iou": current[2],
        "clutter_iou": current[3],
    }


def ranking(record: Candidate) -> tuple[bool, float, float, int, int]:
    primary = (
        (record["overall_accuracy"], record["mean_iou"])
        if record["eligible"]
        else (record["mean_iou"], record["overall_accuracy"])
    )
    return (
        record["eligible"],
        *primary,
        -record["step"],
        -ARM_ORDER.index(record["arm"]),
    )


def complete_arm(root: Path, protocol: Protocol, arm: str) -> PhaseManifest:
    manifest = cast(PhaseManifest, read_json(root / arm / "manifest.json"))
    best = manifest["best"]
    validation = manifest["best_validation"]
    if (
        manifest["status"] != "complete"
        or manifest["protocol_hash"] != protocol["protocol_hash"]
        or manifest["arm"] != arm
        or manifest["updates"] != protocol["passes"] * 864
        or manifest["completed_passes"] != protocol["passes"]
        or best is None
        or validation is None
        or sha256_file(root / arm / "best.pt") != best["checkpoint_sha256"]
    ):
        raise ValueError(f"Selection arm is not complete and verified: {arm}")
    expected = candidate(validation, protocol, arm, best["step"])
    if any(best[key] != value for key, value in expected.items()):
        raise ValueError("Checkpoint eligibility differs from its confusion matrix")
    return manifest


def winner(
    root: Path, protocol: Protocol, arms: Sequence[str] | None = None
) -> PhaseManifest:
    arms = arms or protocol["arms"]
    manifests = [complete_arm(root, protocol, arm) for arm in arms]

    def rank_manifest(manifest: PhaseManifest) -> tuple[bool, float, float, int, int]:
        best = manifest["best"]
        if best is None:
            raise ValueError("Complete selection manifest has no best checkpoint")
        return ranking(best)

    return max(manifests, key=rank_manifest)


def recipe_for(root: Path, protocol: Protocol, arm: str) -> Recipe:
    if arm == "A":
        return {"backbone": "B", "auxiliary": False, "enhanced": False}
    if arm == "B":
        return {"backbone": "B", "auxiliary": True, "enhanced": False}
    if arm == "C":
        return {**winner(root, protocol, "AB")["recipe"], "enhanced": True}
    if arm == "L" and "L" in protocol["arms"]:
        return {**winner(root, protocol, "ABC")["recipe"], "backbone": "L"}
    if arm == "D":
        return winner(root, protocol)["recipe"]
    raise ValueError(f"Unexpected study arm: {arm}")


def choose_tta(
    base42: EvaluationMetrics,
    flip42: EvaluationMetrics,
    base43: EvaluationMetrics,
    flip43: EvaluationMetrics,
    protocol: Protocol,
) -> InferencePolicy:
    for base, flip in ((base42, flip42), (base43, flip43)):
        b = scores(
            cast(
                EvaluationMetrics,
                matrix_metrics(base["confusion_matrix"], protocol["validation_pixels"]),
            )
        )
        f = scores(
            cast(
                EvaluationMetrics,
                matrix_metrics(flip["confusion_matrix"], protocol["validation_pixels"]),
            )
        )
        if f[0] - b[0] < protocol["inference"]["minimum_tta_oa_gain"] or any(
            x < y for x, y in zip(f[1:], b[1:], strict=True)
        ):
            return "identity"
    return "hflip"


def lock_selection(
    root: Path,
    protocol: Protocol,
    inference: InferencePolicy,
    tta: TTAValidation | None = None,
) -> SelectionLock:
    selected, confirmation = winner(root, protocol), complete_arm(root, protocol, "D")
    selected_best = selected["best"]
    confirmation_best = confirmation["best"]
    selected_validation = selected["best_validation"]
    confirmation_validation = confirmation["best_validation"]
    if (
        selected_best is None
        or confirmation_best is None
        or selected_validation is None
        or confirmation_validation is None
        or not selected_best["eligible"]
        or not confirmation_best["eligible"]
        or selected["recipe"] != confirmation["recipe"]
        or selected["seed"] != 42
        or confirmation["seed"] != 43
    ):
        raise ValueError(
            "Refit requires an eligible seed-42 winner and seed-43 confirmation"
        )
    if inference not in {"identity", "hflip"}:
        raise ValueError("Unknown inference policy")
    if inference == "hflip" and (
        not tta
        or choose_tta(
            selected_validation,
            tta["seed42"],
            confirmation_validation,
            tta["seed43"],
            protocol,
        )
        != "hflip"
    ):
        raise ValueError("TTA did not pass both seed validation gates")
    lock: SelectionLock = {
        "protocol_hash": protocol["protocol_hash"],
        "arm": selected["arm"],
        "recipe": selected["recipe"],
        "selection_binding": selected["binding"],
        "selection_checkpoint_sha256": selected_best["checkpoint_sha256"],
        "confirmation_checkpoint_sha256": confirmation_best["checkpoint_sha256"],
        "validation": selected_validation,
        "inference": inference,
        "tta_validation": tta,
        "refit_updates": 1152,
        "lr_multiplier": 0.5,
        "selection_cache_hash": protocol["selection_cache_hash"],
    }
    freeze(root / "selection-lock.json", dict(lock))
    return lock


def load_protocol(root: Path) -> Protocol:
    p = cast(Protocol, read_json(root / "protocol.json"))
    payload = {k: v for k, v in p.items() if k != "protocol_hash"}
    if canonical_hash(payload) != p["protocol_hash"] or p["passes"] not in (6, 8, 12):
        raise ValueError("Frozen protocol hash/schedule mismatch")
    if p["source_hash"] != source_hash():
        raise ValueError("Study source differs from the frozen preflight")
    return p


def freeze_protocol(root: Path, payload: Mapping[str, Any]) -> Protocol:
    value = {**payload, "protocol_hash": canonical_hash(payload)}
    freeze(root / "protocol.json", value)
    return cast(Protocol, value)


class Budget:
    """Count all allocated GPUs, including this job's bootstrap and past attempts."""

    def __init__(
        self, root: Path, hours: float, devices: int, *, started: float | None = None
    ) -> None:
        self.path = root / "budget.json"
        self.limit = hours * 3600
        self.devices = devices
        self.started = (
            float(os.environ.get("POTSDAM_GPU_STARTED", time.monotonic()))
            if started is None
            else started
        )
        self.previous = 0.0

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            data = read_json(self.path)
            self.previous = data["gpu_seconds"]
            if data["in_progress"]:
                raise ValueError(
                    "Interrupted budget needs verified remote duration reconciliation"
                )
        self.check()
        write_json(self.path, {"gpu_seconds": self.limit, "in_progress": True})
        return self

    @property
    def elapsed(self) -> float:
        return self.devices * (time.monotonic() - self.started)

    def remaining(self) -> float:
        return (self.limit - self.previous - self.elapsed) / self.devices

    def check(self) -> None:
        if self.remaining() <= 0:
            raise TimeoutError("Allocated GPU budget reached")

    def __exit__(self, *_: object) -> None:
        write_json(
            self.path,
            {"gpu_seconds": self.previous + self.elapsed, "in_progress": False},
        )


def choose_schedule(
    timings: Mapping[str, ModelTiming], devices: int, *, include_large: bool
) -> Schedule:
    """Resolve the whole common schedule before any selectable training run."""
    arms = "ABCL" if include_large else "ABC"
    hours = 16 / (len(arms) + 1)
    candidates = [timings["B"], *([timings["L"]] if include_large else [])]
    for passes in (12, 8, 6):
        if all(
            1.25
            * devices
            * (
                passes * (864 * t["update_seconds"] + t["probe_seconds"])
                + 2 * 6 * t["scene_seconds"]
                + 6 * t["tta_scene_seconds"]
                + t["setup_seconds"]
            )
            <= hours * 3600
            and 1.25 * devices * 1152 * t["update_seconds"] <= 2 * 3600
            and 1.25 * devices * 14 * t["tta_scene_seconds"] <= 2 * 3600
            for t in candidates
        ):
            return {"passes": passes, "arms": list(arms), "per_arm_gpu_hours": hours}
    if include_large:
        return choose_schedule(timings, devices, include_large=False)
    raise TimeoutError("Even six ViT-B passes do not fit the reserved GPU budget")
