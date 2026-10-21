"""Thin model adapters that consume shared data contracts."""

from .adapter import TorchSegmentationAdapter
from .specs import (
    BackboneSpec,
    InputAdapterSpec,
    LoRASpec,
    SegmentationModelSpec,
    load_model_spec,
)

__all__ = [
    "BackboneSpec",
    "InputAdapterSpec",
    "LoRASpec",
    "SegmentationModelSpec",
    "TorchSegmentationAdapter",
    "load_model_spec",
]
