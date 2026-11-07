"""Multi-scale transformer feature fusion and high-resolution segmentation head."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as functional


def _conv_refinement(channels: int, groups: int) -> nn.Sequential:
    """Build a convolutional refinement block with optional GroupNorm."""
    layers = []
    for _ in range(2):
        layers.append(nn.Conv2d(channels, channels, 3, padding=1))
        if groups:
            layers.append(nn.GroupNorm(groups, channels))
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


class DPTFusionDecoder(nn.Module):
    """Fuse four reassembled transformer stages with a stride-four skip branch."""

    _feature_strides = (4, 8, 16, 32)

    def __init__(
        self,
        *,
        feature_channels: int,
        spatial_channels: int,
        decoder_width: int,
        num_classes: int,
        norm_groups: int = 0,
        normalize_fusion: bool = False,
    ) -> None:
        super().__init__()
        if min(feature_channels, spatial_channels, decoder_width, num_classes) <= 0:
            raise ValueError("Decoder channel counts and class count must be positive")
        self.lateral = nn.ModuleList(
            nn.Conv2d(feature_channels, decoder_width, 1) for _ in self._feature_strides
        )
        self.stage_refinement = nn.ModuleList(
            _conv_refinement(decoder_width, norm_groups) for _ in self._feature_strides
        )
        self.top_down_fusion = nn.ModuleList(
            _conv_refinement(decoder_width, norm_groups)
            for _ in self._feature_strides[:-1]
        )
        self.spatial_projection = nn.Conv2d(spatial_channels, decoder_width, 1)
        fusion = []
        for source in (decoder_width * 2, decoder_width):
            fusion.append(nn.Conv2d(source, decoder_width, 3, padding=1))
            if normalize_fusion and norm_groups:
                fusion.append(nn.GroupNorm(norm_groups, decoder_width))
            fusion.append(nn.GELU())
        self.output_fusion = nn.Sequential(*fusion)
        head_width = max(decoder_width // 2, num_classes)
        classifier = [nn.Conv2d(decoder_width, head_width, 3, padding=1)]
        if normalize_fusion and norm_groups:
            classifier.append(nn.GroupNorm(norm_groups, head_width))
        classifier.extend((nn.GELU(), nn.Conv2d(head_width, num_classes, 1)))
        self.classifier = nn.Sequential(*classifier)

    def _stage_size(self, height: int, width: int, stride: int) -> tuple[int, int]:
        return math.ceil(height / stride), math.ceil(width / stride)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        spatial_features: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Return dense logits at the requested input resolution."""
        if len(features) != len(self._feature_strides):
            raise ValueError(
                f"Expected four backbone feature levels, received {len(features)}"
            )
        height, width = output_size
        if height <= 0 or width <= 0:
            raise ValueError("Decoder output dimensions must be positive")
        batch_size = spatial_features.shape[0]
        stages = []
        for index, (feature, stride) in enumerate(
            zip(features, self._feature_strides, strict=True)
        ):
            if feature.ndim != 4 or feature.shape[:2] != (
                batch_size,
                self.lateral[index].in_channels,
            ):
                raise ValueError(f"Backbone feature level {index} has invalid shape")
            size = self._stage_size(height, width, stride)
            stage = self.lateral[index](feature)
            stage = functional.interpolate(
                stage,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
            stages.append(self.stage_refinement[index](stage))

        decoded = stages[-1]
        for index in range(len(stages) - 2, -1, -1):
            decoded = functional.interpolate(
                decoded,
                size=stages[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            decoded = self.top_down_fusion[index](decoded + stages[index])

        spatial = functional.interpolate(
            spatial_features,
            size=stages[0].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.output_fusion(
            torch.cat((decoded, self.spatial_projection(spatial)), dim=1)
        )
        logits = self.classifier(decoded)
        return functional.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
