"""Composable multimodal semantic-segmentation model."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional

from deepnet.data.contracts import DomainSpec, ModelInputSpec

from .adapters import MultimodalInputAdapter
from .backbones import FeatureBackbone, TransformerFeatureBackbone
from .decoders import DPTFusionDecoder
from .specs import SegmentationModelSpec


class MultimodalSegmenter(nn.Module):
    """Join domain input adaptation, a replaceable backbone, and dense decoding."""

    def __init__(
        self,
        domain: DomainSpec,
        model_input: ModelInputSpec,
        spec: SegmentationModelSpec,
        backbone: nn.Module,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("Backbone must be a torch.nn.Module")
        if not isinstance(backbone, FeatureBackbone):
            raise TypeError(
                "Backbone must implement forward_features and feature specs"
            )
        if backbone.patch_size != spec.backbone.patch_size:
            raise ValueError("Backbone patch size differs from the model specification")
        if backbone.feature_channels != spec.backbone.feature_channels:
            raise ValueError(
                "Backbone feature width differs from the model specification"
            )

        self.input_adapter = MultimodalInputAdapter(
            domain,
            model_input,
            spec.input_adapter,
        )
        self.backbone = backbone
        self.decoder = DPTFusionDecoder(
            feature_channels=spec.backbone.feature_channels,
            spatial_channels=spec.input_adapter.spatial_channels,
            decoder_width=spec.decoder_width,
            num_classes=spec.num_classes,
        )
        self.patch_size = backbone.patch_size
        self.input_channels = model_input.channels
        self.num_classes = spec.num_classes
        self.ignore_index = spec.ignore_index
        self.model_spec_hash = spec.content_hash

    @classmethod
    def from_pretrained(
        cls,
        domain: DomainSpec,
        model_input: ModelInputSpec,
        spec: SegmentationModelSpec,
    ) -> MultimodalSegmenter:
        """Explicitly load the configured backbone and compose its segmentation head."""
        backbone = TransformerFeatureBackbone.from_pretrained(spec.backbone)
        return cls(domain, model_input, spec, backbone)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits with the same height and width as the input batch."""
        if images.ndim != 4:
            raise ValueError(f"Expected NCHW input, got shape {tuple(images.shape)}")
        if images.shape[1] != len(self.input_channels):
            raise ValueError(
                f"Input has {images.shape[1]} channels; model expects "
                f"{len(self.input_channels)}"
            )
        original_height, original_width = images.shape[-2:]
        padded_height = math.ceil(original_height / self.patch_size) * self.patch_size
        padded_width = math.ceil(original_width / self.patch_size) * self.patch_size
        pad_height = padded_height - original_height
        pad_width = padded_width - original_width
        if pad_height or pad_width:
            images = functional.pad(
                images, (0, pad_width, 0, pad_height), mode="replicate"
            )

        backbone_inputs, spatial_features = self.input_adapter(images)
        feature_maps = self.backbone.forward_features(backbone_inputs)
        logits = self.decoder(
            feature_maps,
            spatial_features,
            (padded_height, padded_width),
        )
        return logits[..., :original_height, :original_width]
