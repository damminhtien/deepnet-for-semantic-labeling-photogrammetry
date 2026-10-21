"""Runtime selection helpers for local machines and hosted GPU notebooks."""

from __future__ import annotations

import os
import platform
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def _torch() -> Any:
    """Import Torch only when execution needs it; data inspection stays lightweight."""
    import torch

    return torch


@dataclass(frozen=True, slots=True)
class TorchRuntime:
    """Resolved device and input-pipeline options for one process."""

    device: Any
    num_workers: int
    pin_memory: bool
    non_blocking: bool

    @property
    def device_type(self) -> str:
        """Return a stable device name without serializing a Torch object."""
        return str(self.device.type)

    def dataloader_options(self) -> dict[str, Any]:
        """Return options accepted by ``make_torch_dataloader``."""
        return {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible runtime settings for run provenance."""
        values = asdict(self)
        values["device"] = str(self.device)
        return values


def resolve_device(requested: str | None = None) -> Any:
    """Choose an available accelerator, preferring CUDA, then Apple MPS, then CPU.

    An explicitly requested but unavailable accelerator raises an error instead
    of silently moving a run to a different device.
    """
    torch = _torch()
    if requested is not None:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {device} is unavailable")
        if device.type == "mps" and not _mps_available(torch):
            raise RuntimeError("Requested Apple MPS device is unavailable")
        if device.type not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported execution device: {device.type}")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available(torch):
        return torch.device("mps")
    return torch.device("cpu")


def _mps_available(torch: Any) -> bool:
    """Check MPS defensively for Torch builds without an MPS backend."""
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    return backend is not None and backend.is_available()


def configure_torch_runtime(
    requested_device: str | None = None,
    *,
    num_workers: int | None = None,
) -> TorchRuntime:
    """Resolve device and safe default DataLoader settings for this host.

    CUDA hosts default to a small, configurable worker pool. CPU and MPS hosts
    default to synchronous loading, which avoids process overhead for local
    development and remains compatible with macOS multiprocessing semantics.
    """
    device = resolve_device(requested_device)
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = min(4, max(1, cpu_count // 2)) if device.type == "cuda" else 0
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    uses_cuda = device.type == "cuda"
    return TorchRuntime(
        device=device,
        num_workers=num_workers,
        pin_memory=uses_cuda,
        non_blocking=uses_cuda,
    )


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch RNGs before model and loader construction."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch = _torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def runtime_metadata(runtime: TorchRuntime) -> dict[str, Any]:
    """Describe the selected runtime for a reproducible run manifest."""
    torch = _torch()
    return {
        **runtime.to_dict(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "mps_available": _mps_available(torch),
    }
