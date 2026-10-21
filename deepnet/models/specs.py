"""Versioned, dependency-light specifications for configurable segmentation models."""

from __future__ import annotations

import json
import math
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepnet.data.contracts import canonical_hash


def _model_config_dir() -> Path:
    repository_path = Path(__file__).resolve().parents[2] / "configs" / "models"
    if repository_path.is_dir():
        return repository_path
    return (
        Path(sysconfig.get_path("data"))
        / "share"
        / "deepnet-satellite-labeling"
        / "configs"
        / "models"
    )


DEFAULT_MODEL_CONFIG_DIR = _model_config_dir()


@dataclass(frozen=True, slots=True)
class LoRASpec:
    """LoRA settings whose target module names remain model-configuration data."""

    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.rank <= 0 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be within [0, 1)")
        if not self.target_modules or any(
            not name.strip() for name in self.target_modules
        ):
            raise ValueError("LoRA target modules must be explicitly configured")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LoRASpec:
        return cls(
            rank=int(value["rank"]),
            alpha=int(value["alpha"]),
            dropout=float(value["dropout"]),
            target_modules=tuple(value["target_modules"]),
        )


@dataclass(frozen=True, slots=True)
class BackboneSpec:
    """A replaceable feature encoder and the feature levels it exposes."""

    backend: str
    model_id: str
    patch_size: int
    feature_blocks: tuple[int, ...]
    feature_channels: int
    lora: LoRASpec | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"huggingface", "injected"}:
            raise ValueError(f"Unsupported backbone backend: {self.backend}")
        if not self.model_id.strip():
            raise ValueError("Backbone model_id must not be empty")
        if self.patch_size <= 0 or self.feature_channels <= 0:
            raise ValueError("Backbone patch size and feature width must be positive")
        if len(self.feature_blocks) != 4 or any(
            block <= 0 for block in self.feature_blocks
        ):
            raise ValueError("Exactly four positive transformer block IDs are required")
        if tuple(sorted(set(self.feature_blocks))) != self.feature_blocks:
            raise ValueError("Feature blocks must be unique and increasing")

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "patch_size": self.patch_size,
            "feature_blocks": list(self.feature_blocks),
            "feature_channels": self.feature_channels,
            "lora": self.lora.to_dict() if self.lora is not None else None,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BackboneSpec:
        lora = value.get("lora")
        return cls(
            backend=value["backend"],
            model_id=value["model_id"],
            patch_size=int(value["patch_size"]),
            feature_blocks=tuple(int(item) for item in value["feature_blocks"]),
            feature_channels=int(value["feature_channels"]),
            lora=LoRASpec.from_dict(lora) if lora is not None else None,
        )


@dataclass(frozen=True, slots=True)
class InputAdapterSpec:
    """Spectral identity mapping and normalization for the pretrained encoder."""

    identity_channel_order: tuple[str, ...]
    projection_init: str
    backbone_mean: tuple[float, ...]
    backbone_std: tuple[float, ...]
    spatial_channels: int = 64

    def __post_init__(self) -> None:
        if self.projection_init not in {"identity", "kaiming_uniform"}:
            raise ValueError(
                f"Unsupported projection initializer: {self.projection_init}"
            )
        if not self.backbone_mean or len(self.backbone_mean) != len(self.backbone_std):
            raise ValueError("Backbone normalization needs matching channel values")
        if self.identity_channel_order and len(self.identity_channel_order) != len(
            self.backbone_mean
        ):
            raise ValueError("Identity channel order must match backbone input width")
        if any(not math.isfinite(value) for value in self.backbone_mean):
            raise ValueError("Backbone means must be finite")
        if any(not math.isfinite(value) or value <= 0 for value in self.backbone_std):
            raise ValueError("Backbone standard deviations must be finite and positive")
        if self.spatial_channels < 4:
            raise ValueError("Spatial branch width must be at least four channels")

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_channel_order": list(self.identity_channel_order),
            "projection_init": self.projection_init,
            "backbone_mean": list(self.backbone_mean),
            "backbone_std": list(self.backbone_std),
            "spatial_channels": self.spatial_channels,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> InputAdapterSpec:
        return cls(
            identity_channel_order=tuple(value["identity_channel_order"]),
            projection_init=value["projection_init"],
            backbone_mean=tuple(float(item) for item in value["backbone_mean"]),
            backbone_std=tuple(float(item) for item in value["backbone_std"]),
            spatial_channels=int(value.get("spatial_channels", 64)),
        )


@dataclass(frozen=True, slots=True)
class SegmentationModelSpec:
    """Complete model architecture and task-head contract."""

    model_id: str
    version: int
    backbone: BackboneSpec
    input_adapter: InputAdapterSpec
    decoder_width: int
    num_classes: int
    ignore_index: int
    label_ontology: str
    license_id: str
    license_url: str

    def __post_init__(self) -> None:
        if not self.model_id.strip() or self.version <= 0:
            raise ValueError("Model ID and positive version are required")
        if self.decoder_width <= 0 or self.num_classes <= 0:
            raise ValueError("Decoder width and class count must be positive")
        if self.ignore_index < self.num_classes:
            raise ValueError("ignore_index must not overlap model output classes")
        if not self.label_ontology.strip() or not self.license_id.strip():
            raise ValueError("Label ontology and license ID are required")
        if not self.license_url.strip():
            raise ValueError("A license reference URL is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "version": self.version,
            "backbone": self.backbone.to_dict(),
            "input_adapter": self.input_adapter.to_dict(),
            "decoder_width": self.decoder_width,
            "num_classes": self.num_classes,
            "ignore_index": self.ignore_index,
            "label_ontology": self.label_ontology,
            "license_id": self.license_id,
            "license_url": self.license_url,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SegmentationModelSpec:
        if int(value.get("schema_version", 1)) != 1:
            raise ValueError("Unsupported model specification schema version")
        return cls(
            model_id=value["model_id"],
            version=int(value["version"]),
            backbone=BackboneSpec.from_dict(value["backbone"]),
            input_adapter=InputAdapterSpec.from_dict(value["input_adapter"]),
            decoder_width=int(value["decoder_width"]),
            num_classes=int(value["num_classes"]),
            ignore_index=int(value["ignore_index"]),
            label_ontology=value["label_ontology"],
            license_id=value["license_id"],
            license_url=value["license_url"],
        )

    @property
    def content_hash(self) -> str:
        """Return a stable identifier for the complete architecture config."""
        return canonical_hash(self.to_dict())


def load_model_spec(
    model_id: str,
    config_dir: Path = DEFAULT_MODEL_CONFIG_DIR,
) -> SegmentationModelSpec:
    """Load and validate a model specification by stable ID from JSON."""
    path = Path(config_dir) / f"{model_id}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    return SegmentationModelSpec.from_dict(value)
