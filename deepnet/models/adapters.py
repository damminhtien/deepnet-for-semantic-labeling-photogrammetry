"""Configurable spectral projections and spatial feature branches."""

from __future__ import annotations

import torch
from torch import nn

from deepnet.data.contracts import DomainSpec, ModelInputSpec

from .specs import InputAdapterSpec


def _spatial_branch(
    input_channels: int, output_channels: int, groups: int = 0
) -> nn.Sequential:
    """Create a stride-four convolutional branch without batch-size assumptions."""
    hidden_channels = max(output_channels // 2, 2)
    layers = []
    for source, target in (
        (input_channels, hidden_channels),
        (hidden_channels, output_channels),
    ):
        layers.append(nn.Conv2d(source, target, 3, stride=2, padding=1))
        if groups:
            layers.append(nn.GroupNorm(groups, target))
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


class ResidualIdentityProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.delta = nn.Conv2d(3, 3, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, optical: torch.Tensor) -> torch.Tensor:
        return optical + 0.1 * self.delta(optical)


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
        if spec.encoder_bands:
            if any(
                name not in model_input.channels or band_roles[name] != "optical"
                for name in spec.encoder_bands
            ):
                raise ValueError("Encoder bands must be available optical channels")
            self.optical_indices = tuple(
                model_input.channels.index(name) for name in spec.encoder_bands
            )
        else:
            self.optical_indices = tuple(
                index
                for index, name in enumerate(model_input.channels)
                if band_roles[name] == "optical"
            )
        self.auxiliary_indices = tuple(
            index
            for index, name in enumerate(model_input.channels)
            if index not in self.optical_indices
        )
        if not self.optical_indices:
            raise ValueError("A segmentation model requires optical input channels")

        backbone_channels = len(spec.backbone_mean)
        selected_optical_order = tuple(
            model_input.channels[index] for index in self.optical_indices
        )
        self.spectral_index = spec.spectral_index
        self.index_indices = ()
        if self.spectral_index:
            pair = spec.spectral_index_bands or ("ir", "red")
            if not spec.spectral_index_bands and selected_optical_order != (
                "ir",
                "red",
                "green",
            ):
                raise ValueError("IR/red index requires IRRG channel order")
            if any(name not in model_input.channels for name in pair):
                raise ValueError("Spectral index bands are missing from model input")
            self.index_indices = tuple(
                model_input.channels.index(name) for name in pair
            )
        if spec.projection_init == "residual_identity":
            if len(self.optical_indices) != 3 or backbone_channels != 3:
                raise ValueError("Residual identity requires three optical channels")
            self.spectral_projection = ResidualIdentityProjection()
        elif (
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
        groups = spec.spatial_norm_groups
        self.optical_detail = _spatial_branch(
            backbone_channels, spatial_channels, groups
        )
        auxiliary_channels = len(self.auxiliary_indices) + int(self.spectral_index)
        self.auxiliary_spatial = None
        if self.spectral_index:
            # Keep shared A/B weights paired even when B adds a spatial branch.
            with torch.random.fork_rng(devices=[]):
                self.auxiliary_spatial = _spatial_branch(
                    auxiliary_channels, spatial_channels, groups
                )
        elif auxiliary_channels:
            self.auxiliary_spatial = _spatial_branch(
                auxiliary_channels, spatial_channels, groups
            )
        fusion = [nn.Conv2d(spatial_channels, spatial_channels, 3, padding=1)]
        if groups:
            fusion.append(nn.GroupNorm(groups, spatial_channels))
        fusion.append(nn.GELU())
        self.spatial_fusion = nn.Sequential(*fusion)
        self.auxiliary_gate = (
            nn.Parameter(torch.zeros(()))
            if spec.auxiliary_gate and self.auxiliary_spatial is not None
            else None
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

    def encoder_views(self, optical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.spectral_projection(optical)
        return projected, (projected - self.backbone_mean) / self.backbone_std

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
        _, encoder_inputs = self.encoder_views(optical)

        spatial = self.optical_detail(encoder_inputs)
        if self.auxiliary_spatial is not None:
            auxiliary = images[:, self.auxiliary_indices]
            if self.spectral_index:
                first, second = self.index_indices
                ir, red = images[:, first : first + 1], images[:, second : second + 1]
                index = (ir - red) / (ir + red + 1e-6)
                auxiliary = torch.cat((index, auxiliary), dim=1)
            extra = self.auxiliary_spatial(auxiliary)
            if self.auxiliary_gate is not None:
                extra = self.auxiliary_gate.tanh() * extra
            spatial = spatial + extra
        return encoder_inputs, self.spatial_fusion(spatial)
