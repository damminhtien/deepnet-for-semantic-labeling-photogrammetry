"""Configurable spectral projections and spatial feature branches."""

from __future__ import annotations

import torch
from torch import nn

from deepnet.data.contracts import DomainSpec, ModelInputSpec

from .specs import InputAdapterSpec


def _spatial_branch(input_channels: int, output_channels: int) -> nn.Sequential:
    """Create a stride-four convolutional branch without batch-size assumptions."""
    hidden_channels = max(output_channels // 2, 2)
    return nn.Sequential(
        nn.Conv2d(input_channels, hidden_channels, 3, stride=2, padding=1),
        nn.GELU(),
        nn.Conv2d(hidden_channels, output_channels, 3, stride=2, padding=1),
        nn.GELU(),
    )


class MultimodalInputAdapter(nn.Module):
    """Map named sensor bands to backbone inputs and high-resolution features."""

    def __init__(
        self,
        domain: DomainSpec,
        model_input: ModelInputSpec,
        spec: InputAdapterSpec,
    ) -> None:
        super().__init__()
        if model_input.layout != "CHW":
            raise ValueError("The Torch model adapter requires CHW model inputs")
        domain.validate_model_input(model_input)
        self.channel_names = model_input.channels
        band_roles = {band.name: band.role for band in domain.bands}
        self.optical_indices = tuple(
            index
            for index, name in enumerate(model_input.channels)
            if band_roles[name] == "optical"
        )
        self.auxiliary_indices = tuple(
            index
            for index, name in enumerate(model_input.channels)
            if band_roles[name] == "auxiliary"
        )
        if not self.optical_indices:
            raise ValueError("A segmentation model requires optical input channels")

        backbone_channels = len(spec.backbone_mean)
        selected_optical_order = tuple(
            model_input.channels[index] for index in self.optical_indices
        )
        if (
            spec.identity_channel_order
            and selected_optical_order == spec.identity_channel_order
        ):
            self.spectral_projection: nn.Module = nn.Identity()
        else:
            projection = nn.Conv2d(
                len(self.optical_indices),
                backbone_channels,
                kernel_size=1,
            )
            self._initialize_projection(projection, spec.projection_init)
            self.spectral_projection = projection

        self.register_buffer(
            "backbone_mean",
            torch.tensor(spec.backbone_mean, dtype=torch.float32).view(1, -1, 1, 1),
        )
        self.register_buffer(
            "backbone_std",
            torch.tensor(spec.backbone_std, dtype=torch.float32).view(1, -1, 1, 1),
        )

        spatial_channels = spec.spatial_channels
        self.optical_detail = _spatial_branch(backbone_channels, spatial_channels)
        self.auxiliary_spatial = (
            _spatial_branch(len(self.auxiliary_indices), spatial_channels)
            if self.auxiliary_indices
            else None
        )
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(spatial_channels, spatial_channels, 3, padding=1),
            nn.GELU(),
        )

    @staticmethod
    def _initialize_projection(projection: nn.Conv2d, initializer: str) -> None:
        """Initialize a configurable channel projection without naming sensors."""
        if initializer == "identity":
            with torch.no_grad():
                projection.weight.zero_()
                projection.bias.zero_()
                diagonal = min(projection.in_channels, projection.out_channels)
                indices = torch.arange(diagonal)
                projection.weight[indices, indices, 0, 0] = 1.0
            return
        if initializer == "kaiming_uniform":
            nn.init.kaiming_uniform_(projection.weight, a=5**0.5)
            nn.init.zeros_(projection.bias)
            return
        raise ValueError(f"Unsupported projection initializer: {initializer}")

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized encoder inputs and stride-four spatial features."""
        if images.ndim != 4:
            raise ValueError(f"Expected NCHW input, got shape {tuple(images.shape)}")
        if images.shape[1] != len(self.channel_names):
            raise ValueError(
                f"Input has {images.shape[1]} channels; domain contract declares "
                f"{len(self.channel_names)}"
            )
        optical = images[:, self.optical_indices]
        projected = self.spectral_projection(optical)
        encoder_inputs = (projected - self.backbone_mean) / self.backbone_std

        spatial = self.optical_detail(encoder_inputs)
        if self.auxiliary_spatial is not None:
            auxiliary = images[:, self.auxiliary_indices]
            spatial = spatial + self.auxiliary_spatial(auxiliary)
        return encoder_inputs, self.spatial_fusion(spatial)
