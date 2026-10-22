"""Replaceable dense-feature backbones with optional Hugging Face loading."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn

from .specs import BackboneSpec, LoRASpec


@runtime_checkable
class FeatureBackbone(Protocol):
    """Minimal interface required by the shared segmentation decoder."""

    patch_size: int
    feature_channels: int

    def forward_features(self, images: torch.Tensor) -> Sequence[torch.Tensor]:
        """Return four NCHW feature maps in shallow-to-deep order."""


def _find_lora_target_modules(
    model: nn.Module,
    target_modules: Sequence[str],
) -> tuple[str, ...]:
    """Resolve configured PEFT suffixes against a model's actual module names."""
    module_names = tuple(name for name, _ in model.named_modules() if name)
    unmatched_targets = tuple(
        target
        for target in target_modules
        if not any(
            name == target or name.endswith(f".{target}") for name in module_names
        )
    )
    if unmatched_targets:
        available_linear_names = sorted(
            {
                name.rsplit(".", maxsplit=1)[-1]
                for name, module in model.named_modules()
                if isinstance(module, nn.Linear)
            }
        )
        raise ValueError(
            f"LoRA target modules {list(unmatched_targets)!r} do not match modules "
            f"in {type(model).__name__}. Available linear module names: "
            f"{available_linear_names!r}"
        )

    return tuple(
        name
        for name in module_names
        if any(
            name == target or name.endswith(f".{target}") for target in target_modules
        )
    )


def apply_lora(model: nn.Module, spec: LoRASpec) -> nn.Module:
    """Wrap a caller-supplied encoder using LoRA names from its model config."""
    matched_module_names = _find_lora_target_modules(model, spec.target_modules)
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as error:
        raise ImportError(
            "Install the 'models' extra to configure LoRA adapters"
        ) from error

    lora_config = LoraConfig(
        r=spec.rank,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        target_modules=list(spec.target_modules),
        bias="none",
    )
    adapted_model = get_peft_model(model, lora_config)
    adapted_layer_count = sum(
        hasattr(module, "lora_A") and hasattr(module, "lora_B")
        for module in adapted_model.modules()
    )
    if adapted_layer_count != len(matched_module_names):
        raise RuntimeError(
            "PEFT adapted an unexpected number of modules: expected "
            f"{len(matched_module_names)}, got {adapted_layer_count}"
        )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in adapted_model.parameters()
        if parameter.requires_grad
    )
    if trainable_parameter_count == 0:
        raise RuntimeError("PEFT added no trainable LoRA parameters")
    return adapted_model


class TransformerFeatureBackbone(nn.Module):
    """Adapt a hidden-state transformer to the ``FeatureBackbone`` contract."""

    def __init__(self, model: nn.Module, spec: BackboneSpec) -> None:
        super().__init__()
        if spec.backend != "huggingface":
            raise ValueError("TransformerFeatureBackbone requires a Hugging Face spec")
        self.model = model
        self.patch_size = spec.patch_size
        self.feature_blocks = spec.feature_blocks
        self.feature_channels = spec.feature_channels

    @classmethod
    def from_pretrained(cls, spec: BackboneSpec) -> TransformerFeatureBackbone:
        """Load configured weights only when explicitly requested by a caller."""
        if spec.backend != "huggingface":
            raise ValueError("Only Hugging Face specs can load pretrained weights")
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise ImportError(
                "Install the 'models' extra to load Hugging Face backbones"
            ) from error

        model = AutoModel.from_pretrained(spec.model_id)
        if spec.lora is not None:
            model = apply_lora(model, spec.lora)
        return cls(model, spec)

    @staticmethod
    def _hidden_states(output: Any) -> Sequence[torch.Tensor]:
        """Read hidden states from standard ModelOutput or mapping results."""
        if isinstance(output, Mapping):
            hidden_states = output.get("hidden_states")
        else:
            hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None:
            raise ValueError(
                "Backbone output has no hidden_states; enable transformer outputs"
            )
        return hidden_states

    def forward_features(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Select configured transformer blocks and remove prefix tokens by grid."""
        if images.ndim != 4:
            raise ValueError(f"Expected NCHW input, got shape {tuple(images.shape)}")
        height, width = images.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                "Backbone inputs must be padded to the configured patch size"
            )
        output = self.model(
            pixel_values=images,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = self._hidden_states(output)
        if len(hidden_states) <= max(self.feature_blocks):
            raise ValueError("Backbone returned fewer hidden states than configured")

        grid_height = height // self.patch_size
        grid_width = width // self.patch_size
        patch_count = grid_height * grid_width
        features = []
        for block in self.feature_blocks:
            tokens = hidden_states[block]
            prefix_count = tokens.shape[1] - patch_count
            if prefix_count < 0:
                raise ValueError(
                    f"Block {block} has fewer tokens than the input patch grid"
                )
            patch_tokens = tokens[:, prefix_count:, :]
            if patch_tokens.shape[-1] != self.feature_channels:
                raise ValueError(
                    f"Block {block} has width {patch_tokens.shape[-1]}; expected "
                    f"{self.feature_channels}"
                )
            feature_map = patch_tokens.reshape(
                tokens.shape[0],
                grid_height,
                grid_width,
                self.feature_channels,
            )
            features.append(feature_map.permute(0, 3, 1, 2).contiguous())
        return tuple(features)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run the encoder and return its configured dense feature maps."""
        return self.forward_features(images)
