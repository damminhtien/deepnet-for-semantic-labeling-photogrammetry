"""Small persistence and optimization helpers for joint segmentation training."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deepnet.data.raster import sha256_file


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["mps"] is not None:
        torch.mps.set_rng_state(state["mps"])


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    progress: dict[str, Any],
    binding: str,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any = None,
) -> str:
    payload = {
        "binding": binding,
        "model": model.state_dict(),
        "progress": progress,
        "rng": capture_rng(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return sha256_file(path)


def load_checkpoint(path: Path, binding: str) -> dict[str, Any]:
    # Only load checkpoints created in this run's trusted output directory.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["binding"] != binding:
        raise ValueError("Checkpoint data/model/configuration binding differs")
    return payload


def learning_rate_factor(step: int, total: int, warmup_fraction: float) -> float:
    warmup = max(1, round(total * warmup_fraction))
    if step < warmup:
        return (step + 1) / warmup
    fraction = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1 + math.cos(math.pi * fraction))


def make_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    groups: dict[str, list] = {"lora": [], "input_adapter_and_decoder": []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = "lora" if "lora_" in name else "input_adapter_and_decoder"
            groups[group].append(parameter)
    rates = config["learning_rates"]
    return torch.optim.AdamW(
        [
            {"params": parameters, "lr": rates[name], "initial_lr": rates[name]}
            for name, parameters in groups.items()
            if parameters
        ],
        weight_decay=config["weight_decay"],
    )


def verify_pretrained(path: Path, model_id: str, revision: str | None) -> dict:
    manifest = json.loads((path / "pretrained_manifest.json").read_text())
    if manifest["model_id"] != model_id or manifest["revision"] != revision:
        raise ValueError("Pretrained snapshot differs from the locked model spec")
    for record in manifest["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError("Unsafe pretrained file path")
        source = path / relative
        if (
            source.stat().st_size != record["bytes"]
            or sha256_file(source) != record["sha256"]
        ):
            raise ValueError(f"Pretrained checksum mismatch: {relative}")
    return manifest


def artifact_index(root: Path) -> None:
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != "artifact_sha256_manifest.json"
        and not path.name.endswith(".tmp")
        and ".scratch" not in path.parts
    ]
    write_json(root / "artifact_sha256_manifest.json", {"files": records})
