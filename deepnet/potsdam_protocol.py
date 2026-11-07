"""Selection rules, immutable phase bindings, and budget checks."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from deepnet.data.contracts import canonical_hash
from deepnet.data.evaluation import summarize_confusion
from deepnet.data.raster import sha256_file
from deepnet.training_utils import write_json

CONFIG = "potsdam-dinov2-coverage-refit-v1.json"
KEYS = ("overall_accuracy", "mean_iou", "car_iou", "clutter_iou")


def read_json(path: Path):
    return json.loads(path.read_text())


def freeze(path: Path, value: dict):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen artifact differs: {path}")
    else:
        write_json(path, value)


def matrix_metrics(matrix, expected_pixels: int):
    values = np.asarray(matrix)
    if values.shape != (6, 6) or values.dtype.kind not in "iu" or np.any(values < 0):
        raise ValueError("Expected nonnegative integer six-class confusion matrix")
    result = summarize_confusion(values)
    if result["pixels_scored"] != expected_pixels:
        raise ValueError("Scored pixels differ from the locked evaluation support")
    if any(value is None for value in result["per_class_iou"]):
        raise ValueError("Full-scene scoring must represent all six classes")
    return result


def _scores(metrics):
    return (
        metrics["overall_accuracy"],
        metrics["mean_iou"],
        metrics["per_class_iou"][4],
        metrics["per_class_iou"][5],
    )


def candidate(metrics: dict, protocol: dict, arm: str, step: int):
    measured = matrix_metrics(
        metrics["confusion_matrix"], protocol["validation_pixels"]
    )
    sam = matrix_metrics(
        protocol["sam_validation_confusion"], protocol["validation_pixels"]
    )
    eligible = all(
        current >= target
        for current, target in zip(_scores(measured), _scores(sam), strict=True)
    )
    return {
        "eligible": eligible,
        "arm": arm,
        "step": step,
        **dict(zip(KEYS, _scores(measured), strict=True)),
    }


def ranking(record: dict):
    oa, miou = record["overall_accuracy"], record["mean_iou"]
    primary = (oa, miou) if record["eligible"] else (miou, oa)
    return record["eligible"], *primary, -record["step"], -"ABCD".index(record["arm"])


def load_protocol(root: Path):
    path = root / "protocol.json"
    protocol = read_json(path)
    payload = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    if canonical_hash(payload) != protocol["protocol_hash"]:
        raise ValueError("Campaign protocol hash does not match")
    if protocol["passes"] not in (4, 6):
        raise ValueError("Invalid campaign length")
    return protocol


def complete_arm(root: Path, protocol: dict, arm: str):
    manifest = read_json(root / arm / "manifest.json")
    if (
        manifest["status"] != "complete"
        or manifest["arm"] != arm
        or manifest["protocol_hash"] != protocol["protocol_hash"]
        or manifest["updates"] != protocol["passes"] * 864
        or sha256_file(root / arm / "best.pt") != manifest["best"]["checkpoint_sha256"]
    ):
        raise ValueError(f"Expected a completed, verified selection run: {arm}")
    expected = candidate(
        manifest["best_validation"], protocol, arm, manifest["best"]["step"]
    )
    if any(manifest["best"][key] != value for key, value in expected.items()):
        raise ValueError(f"Selection results differ from the confusion matrix: {arm}")
    return manifest


def winner(root, protocol, arms="ABC"):
    return max(
        (complete_arm(root, protocol, arm) for arm in arms),
        key=lambda x: ranking(x["best"]),
    )


def lock_winner(root: Path, protocol: dict):
    chosen = winner(root, protocol)
    confirm = complete_arm(root, protocol, "D")
    if (
        not chosen["best"]["eligible"]
        or not confirm["best"]["eligible"]
        or chosen["recipe"] != confirm["recipe"]
        or confirm["seed"] != 43
        or chosen["seed"] != 42
    ):
        raise ValueError("Refit requires a validation winner confirmed with seed 43")
    result = {
        "protocol_hash": protocol["protocol_hash"],
        "arm": chosen["arm"],
        "recipe": chosen["recipe"],
        "selection_checkpoint_sha256": chosen["best"]["checkpoint_sha256"],
        "selection_binding": chosen["binding"],
        "validation": chosen["best_validation"],
        "confirmation_checkpoint_sha256": confirm["best"]["checkpoint_sha256"],
        "refit_updates": 1152,
        "refit_seed": 42,
        "lr_multiplier": 0.5,
    }
    freeze(root / "selection-lock.json", result)
    return result


class Budget:
    """Charge each invocation to a phase, including failed attempts."""

    def __init__(self, directory: Path, hours: float, device_count: int = 1):
        self.directory = directory
        self.limit = hours * 3600
        self.devices = device_count
        self.started = time.monotonic()

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "budget.json"
        # Reserve the remaining budget so a killed attempt stays charged.
        self.previous = (
            read_json(self.path)["gpu_seconds"] if self.path.exists() else 0.0
        )
        self.check()
        write_json(self.path, {"gpu_seconds": self.limit, "in_progress": True})
        return self

    def remaining(self):
        return (self.limit - self.previous) / self.devices - (
            time.monotonic() - self.started
        )

    def check(self):
        if self.remaining() <= 0:
            raise TimeoutError(f"GPU budget reached: {self.directory.name}")

    def __exit__(self, *_):
        elapsed = self.devices * (time.monotonic() - self.started)
        write_json(
            self.path, {"gpu_seconds": self.previous + elapsed, "in_progress": False}
        )
