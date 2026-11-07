from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional

from deepnet.data.contracts import (
    BandSpec,
    DomainSpec,
    ModelInputSpec,
    NormalizationRule,
)
from deepnet.models.adapters import MultimodalInputAdapter
from deepnet.models.backbones import TransformerFeatureBackbone, apply_lora
from deepnet.models.segmenter import MultimodalSegmenter
from deepnet.models.specs import (
    BackboneSpec,
    InputAdapterSpec,
    LoRASpec,
    SegmentationModelSpec,
    load_model_spec,
)


def _domain() -> DomainSpec:
    return DomainSpec(
        "synthetic-multimodal",
        1,
        "synthetic sensor",
        (
            BandSpec("ir", "optical", value_range=(0.0, 1.0)),
            BandSpec("red", "optical", value_range=(0.0, 1.0)),
            BandSpec("green", "optical", value_range=(0.0, 1.0)),
            BandSpec("dsm", "auxiliary", unit="metres"),
            BandSpec("ndsm", "auxiliary", unit="metres"),
        ),
    )


def _model_input(channels: tuple[str, ...]) -> ModelInputSpec:
    return ModelInputSpec(
        "synthetic-input",
        1,
        channels,
        tuple((name, "synthetic", name) for name in channels),
        tuple(NormalizationRule(name, "identity") for name in channels),
        pixel_semantics="Synthetic model channels.",
    )


def _adapter_spec() -> InputAdapterSpec:
    return InputAdapterSpec(
        identity_channel_order=("red", "green", "blue"),
        projection_init="identity",
        backbone_mean=(0.0, 0.0, 0.0),
        backbone_std=(1.0, 1.0, 1.0),
        spatial_channels=8,
    )


def _model_spec(patch_size: int = 16) -> SegmentationModelSpec:
    return SegmentationModelSpec(
        model_id="synthetic-segmenter",
        version=1,
        backbone=BackboneSpec(
            "injected",
            "synthetic-backbone",
            patch_size,
            (3, 6, 9, 12),
            8,
        ),
        input_adapter=_adapter_spec(),
        decoder_width=16,
        num_classes=6,
        ignore_index=6,
        label_ontology="isprs-six-class-v1",
        license_id="synthetic",
        license_url="https://example.com/license",
    )


class TinyFeatureBackbone(nn.Module):
    feature_channels = 8

    def __init__(self, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(3, self.feature_channels, 1)

    def forward_features(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        pooled = functional.avg_pool2d(
            self.projection(images),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        return tuple(pooled + index * 0.01 for index in range(4))


class FakeTransformer(nn.Module):
    def forward(self, pixel_values, output_hidden_states, return_dict):
        assert output_hidden_states and return_dict
        batch = pixel_values.shape[0]
        patch_count = (pixel_values.shape[-2] // 16) * (pixel_values.shape[-1] // 16)
        prefix = torch.full((batch, 2, 8), -999.0)
        states = [
            torch.cat((prefix, torch.full((batch, patch_count, 8), float(i))), dim=1)
            for i in range(13)
        ]
        return SimpleNamespace(hidden_states=tuple(states))


def test_huggingface_backbone_removes_prefix_tokens_using_patch_grid():
    backbone = TransformerFeatureBackbone(
        FakeTransformer(),
        BackboneSpec("huggingface", "fake", 16, (3, 6, 9, 12), 8),
    )

    features = backbone.forward_features(torch.zeros(2, 3, 32, 48))

    assert len(features) == 4
    assert all(feature.shape == (2, 8, 2, 3) for feature in features)
    for feature, block in zip(features, (3, 6, 9, 12), strict=True):
        assert torch.all(feature == block)


def test_apply_lora_rejects_unmatched_configured_target_modules():
    model = nn.Sequential(nn.Linear(3, 4))
    spec = LoRASpec(rank=2, alpha=4, dropout=0.0, target_modules=("qkv",))

    with pytest.raises(ValueError, match="qkv"):
        apply_lora(model, spec)


@pytest.mark.parametrize(
    "filename",
    (
        "dinov2-vitb14-multimodal-v1.json",
        "dinov3-vitb16-multimodal-v1.json",
    ),
)
def test_dino_specs_target_huggingface_attention_projections(filename: str):
    model_spec = load_model_spec(filename.removesuffix(".json"))

    assert model_spec.backbone.lora is not None
    assert model_spec.backbone.lora.target_modules == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )


def test_model_spec_rejects_incompatible_normalized_decoder_head_width():
    with pytest.raises(ValueError, match="Decoder head width"):
        replace(
            _model_spec(),
            decoder_width=24,
            decoder_norm_groups=8,
            normalize_fusion=True,
        )


def test_rgb_uses_identity_while_irrg_uses_learnable_spectral_projection():
    rgb_domain = DomainSpec(
        "rgb",
        1,
        "visible camera",
        (
            BandSpec("red", "optical", value_range=(0.0, 1.0)),
            BandSpec("green", "optical", value_range=(0.0, 1.0)),
            BandSpec("blue", "optical", value_range=(0.0, 1.0)),
        ),
    )
    rgb_adapter = MultimodalInputAdapter(
        rgb_domain,
        _model_input(("red", "green", "blue")),
        _adapter_spec(),
    )
    irrg_adapter = MultimodalInputAdapter(
        _domain(),
        _model_input(("ir", "red", "green", "dsm", "ndsm")),
        _adapter_spec(),
    )

    assert isinstance(rgb_adapter.spectral_projection, nn.Identity)
    assert isinstance(irrg_adapter.spectral_projection, nn.Conv2d)
    assert irrg_adapter.auxiliary_spatial is not None
    assert irrg_adapter.spectral_projection.weight.requires_grad


@pytest.mark.parametrize(
    ("channels", "patch_size"), [(3, 16), (5, 16), (3, 14), (5, 14)]
)
def test_multimodal_segmenter_forward_backward_and_ignore_index(
    channels: int,
    patch_size: int,
):
    names = (
        ("ir", "red", "green")
        if channels == 3
        else ("ir", "red", "green", "dsm", "ndsm")
    )
    model = MultimodalSegmenter(
        _domain(),
        _model_input(names),
        _model_spec(patch_size),
        TinyFeatureBackbone(patch_size),
    )
    images = torch.rand(2, channels, 65, 67, requires_grad=True)
    labels = torch.randint(0, 6, (2, 65, 67), dtype=torch.long)
    labels[:, :4, :5] = model.ignore_index

    logits = model(images)
    loss = functional.cross_entropy(logits, labels, ignore_index=model.ignore_index)
    loss.backward()

    assert logits.shape == (2, 6, 65, 67)
    assert torch.isfinite(loss)
    assert images.grad is not None and torch.isfinite(images.grad).all()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize(
    "filename",
    (
        "dinov3-vitb16-multimodal-v1.json",
        "dinov2-vitb14-multimodal-v1.json",
        "sam-vitb-injected-baseline-v1.json",
    ),
)
def test_model_configs_load_without_loading_pretrained_weights(filename: str):
    model_spec = load_model_spec(filename.removesuffix(".json"))

    assert len(model_spec.backbone.feature_blocks) == 4
    assert model_spec.num_classes == 6
    assert model_spec.content_hash


def test_named_rgb_routing_preserves_rgb_and_gates_multimodal_features():
    names = ("ir", "relief", "blue", "red", "dsm_available", "green")
    domain = DomainSpec(
        "rgbir",
        1,
        "ISPRS",
        tuple(
            BandSpec(
                name, "auxiliary" if name in {"relief", "dsm_available"} else "optical"
            )
            for name in names
        ),
    )
    spec = load_model_spec("dinov3-vitb16-rgbir-v2").input_adapter
    torch.manual_seed(42)
    rgb = MultimodalInputAdapter(
        domain,
        _model_input(("red", "green", "blue")),
        replace(spec, spectral_index=False, spectral_index_bands=()),
    )
    torch.manual_seed(42)
    enriched = MultimodalInputAdapter(domain, _model_input(names), spec)
    image = torch.rand(1, len(names), 32, 48)
    visible = image[:, [3, 5, 2]]
    projected, _ = enriched.encoder_views(visible)
    assert torch.equal(projected, visible)
    captured = []
    handle = enriched.auxiliary_spatial.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach())
    )
    base_encoder, base_spatial = rgb(visible)
    encoder, spatial = enriched(image)
    handle.remove()
    assert torch.equal(encoder, base_encoder)
    assert torch.equal(spatial, base_spatial)
    expected = (image[:, :1] - image[:, 3:4]) / (image[:, :1] + image[:, 3:4] + 1e-6)
    torch.testing.assert_close(captured[0][:, :1], expected)
    spatial.square().mean().backward()
    assert enriched.auxiliary_gate.grad is not None
    assert torch.isfinite(enriched.auxiliary_gate.grad)


def test_dinov3_uses_pretrained_norm_and_rejects_unexpected_prefix_count():
    class V3(FakeTransformer):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(8)
            self.config = SimpleNamespace(num_register_tokens=1)

    model = V3()
    spec = BackboneSpec(
        "huggingface",
        "fake-v3",
        16,
        (3, 6, 9, 12),
        8,
        normalize_features=True,
    )
    backbone = TransformerFeatureBackbone(model, spec)
    features = backbone(torch.zeros(1, 3, 32, 48))
    assert all(torch.equal(feature, torch.zeros_like(feature)) for feature in features)
    model.config.num_register_tokens = 4
    with pytest.raises(ValueError, match="CLS/register"):
        backbone(torch.zeros(1, 3, 32, 48))
