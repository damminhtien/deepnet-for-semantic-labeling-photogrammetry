"""SAM ViT-B with attention LoRA and a six-class semantic decoder."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .constants import N_CLASSES


class LoRALinear(nn.Module):
    """Frozen linear projection with a trainable low-rank update."""

    def __init__(self, base_layer, rank=4, alpha=8):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRA can only wrap nn.Linear modules")

        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(
            base_layer.weight.new_empty((rank, base_layer.in_features))
        )
        self.lora_b = nn.Parameter(
            base_layer.weight.new_zeros((base_layer.out_features, rank))
        )
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs):
        low_rank = F.linear(F.linear(inputs, self.lora_a), self.lora_b)
        return self.base_layer(inputs) + low_rank * self.scaling


class ConvNormAct(nn.Sequential):
    """Small convolution block for the semantic mask decoder."""

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )


class SAMViTSemanticSegmenter(nn.Module):
    """Use SAM ViT-B image features for Potsdam six-class segmentation.

    Inputs are three-channel uint8-range IRRG values represented as float32.
    The pretrained SAM image encoder is frozen except for rank-r LoRA updates
    on each attention qkv and output projection. A lightweight four-stage
    decoder predicts the six Potsdam classes.
    """

    def __init__(
        self,
        image_encoder,
        pixel_mean,
        pixel_std,
        num_classes=N_CLASSES,
        rank=4,
        alpha=8,
        gradient_checkpointing=True,
    ):
        super().__init__()
        if num_classes != N_CLASSES:
            raise ValueError(f"Expected {N_CLASSES} Potsdam classes, got {num_classes}")
        self.image_encoder = image_encoder
        self.gradient_checkpointing = gradient_checkpointing
        self.register_buffer(
            "pixel_mean", pixel_mean.detach().clone().view(1, 3, 1, 1)
        )
        self.register_buffer(
            "pixel_std", pixel_std.detach().clone().view(1, 3, 1, 1)
        )

        for parameter in self.image_encoder.parameters():
            parameter.requires_grad_(False)
        for block in self.image_encoder.blocks:
            block.attn.qkv = LoRALinear(block.attn.qkv, rank=rank, alpha=alpha)
            block.attn.proj = LoRALinear(block.attn.proj, rank=rank, alpha=alpha)

        embedding_channels = self.image_encoder.neck[0].out_channels
        self.decoder = nn.ModuleList(
            [
                ConvNormAct(embedding_channels, 128),
                ConvNormAct(128, 64),
                ConvNormAct(64, 32),
                ConvNormAct(32, 32),
            ]
        )
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    @classmethod
    def from_pretrained(cls, checkpoint_path, **kwargs):
        """Build ViT-B from the official Segment Anything checkpoint."""
        from segment_anything import sam_model_registry

        sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))
        return cls(
            image_encoder=sam.image_encoder,
            pixel_mean=sam.pixel_mean,
            pixel_std=sam.pixel_std,
            **kwargs,
        )

    def _encode(self, normalized_inputs):
        encoder = self.image_encoder
        features = encoder.patch_embed(normalized_inputs)
        if encoder.pos_embed is not None:
            position = encoder.pos_embed
            if position.shape[1:3] != features.shape[1:3]:
                position = F.interpolate(
                    position.permute(0, 3, 1, 2),
                    size=features.shape[1:3],
                    mode="bicubic",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
            features = features + position.to(dtype=features.dtype)

        for block in encoder.blocks:
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                features = checkpoint(block, features, use_reentrant=False)
            else:
                features = block(features)

        return encoder.neck(features.permute(0, 3, 1, 2))

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"Expected NCHW three-band IRRG inputs, got {tuple(images.shape)}"
            )
        if images.dtype != torch.float32:
            images = images.float()

        output_size = images.shape[-2:]
        normalized = (images - self.pixel_mean) / self.pixel_std
        features = self._encode(normalized)

        for decoder_stage in self.decoder:
            features = decoder_stage(features)
            features = F.interpolate(
                features, scale_factor=2, mode="bilinear", align_corners=False
            )
        logits = self.classifier(features)
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(
                logits, size=output_size, mode="bilinear", align_corners=False
            )
        return logits

    def trainable_parameters(self):
        """Return only LoRA adapter and semantic decoder parameters."""
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def trainable_parameter_count(self):
        """Count parameters updated by the Potsdam fine-tuning run."""
        return sum(parameter.numel() for parameter in self.trainable_parameters())
