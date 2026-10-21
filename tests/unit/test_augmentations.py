from __future__ import annotations

import numpy as np
import pytest

from deepnet.data.augmentations import AlbumentationsAugmenter
from deepnet.data.catalog import load_augmentation_spec, load_domain_spec
from deepnet.data.contracts import (
    AugmentationSpec,
    BandSpec,
    DomainSpec,
    ModelInputSpec,
    NormalizationRule,
    TransformSpec,
)


def _view(channels: tuple[str, ...]) -> ModelInputSpec:
    return ModelInputSpec(
        model_id="synthetic-view",
        version=1,
        channels=channels,
        channel_sources=tuple((name, "asset", name) for name in channels),
        normalization=tuple(NormalizationRule(name, "identity") for name in channels),
        pixel_semantics="Synthetic aligned bands for augmentation tests.",
    )


def test_geometry_synchronizes_optical_auxiliary_labels_and_validity():
    pytest.importorskip("albumentations")
    bands = (
        BandSpec("ir", "optical", value_range=(0.0, 255.0)),
        BandSpec("red", "optical", value_range=(0.0, 255.0)),
        BandSpec("green", "optical", value_range=(0.0, 255.0)),
        BandSpec("dsm", "auxiliary", unit="metres"),
    )
    domain = DomainSpec("synthetic-aerial", 1, "synthetic", bands)
    policy = AugmentationSpec(
        "synthetic-geometry",
        1,
        domain.domain_id,
        geometry=(TransformSpec("square_symmetry"),),
    )
    augmenter = AlbumentationsAugmenter(
        domain,
        _view(("ir", "red", "green", "dsm")),
        policy,
        ignore_label=255,
    )
    grid = np.arange(49, dtype=np.float32).reshape(7, 7)
    channels = {
        "ir": grid,
        "red": grid + 1,
        "green": grid + 2,
        "dsm": grid + 10,
    }
    labels = grid.astype(np.uint8)
    valid = np.ones(grid.shape, dtype=bool)
    valid[1, 3] = False

    transformed, transformed_labels, transformed_valid = augmenter(
        channels,
        labels,
        valid,
        seed=72,
    )

    np.testing.assert_array_equal(
        transformed["ir"][transformed_valid],
        transformed_labels[transformed_valid],
    )
    np.testing.assert_array_equal(
        transformed["red"][transformed_valid],
        transformed_labels[transformed_valid] + 1,
    )
    np.testing.assert_array_equal(
        transformed["green"][transformed_valid],
        transformed_labels[transformed_valid] + 2,
    )
    np.testing.assert_array_equal(
        transformed["dsm"][transformed_valid],
        transformed_labels[transformed_valid] + 10,
    )
    assert np.count_nonzero(~transformed_valid) == 1
    assert np.all(transformed_labels[~transformed_valid] == 255)


def test_radiometry_changes_only_optical_bands_and_is_seeded():
    pytest.importorskip("albumentations")
    domain = DomainSpec(
        "synthetic-radiometry",
        1,
        "synthetic",
        (
            BandSpec("nir", "optical", value_range=(0.0, 1.0)),
            BandSpec("red", "optical", value_range=(0.0, 1.0)),
            BandSpec("green", "optical", value_range=(0.0, 1.0)),
            BandSpec("dsm", "auxiliary", unit="metres"),
        ),
    )
    policy = AugmentationSpec(
        "synthetic-radiometry",
        1,
        domain.domain_id,
        optical=(
            TransformSpec(
                "random_brightness_contrast",
                probability=1.0,
                parameters=(
                    ("brightness_limit", (0.2, 0.2)),
                    ("contrast_limit", (0.0, 0.0)),
                ),
            ),
        ),
    )
    augmenter = AlbumentationsAugmenter(
        domain,
        _view(("nir", "red", "green", "dsm")),
        policy,
        ignore_label=6,
    )
    channels = {
        name: np.full((8, 8), 0.3, dtype=np.float32) for name in ("nir", "red", "green")
    }
    channels["dsm"] = np.full((8, 8), 12.0, dtype=np.float32)
    labels = np.arange(64, dtype=np.uint8).reshape(8, 8) % 6
    valid = np.ones(labels.shape, dtype=bool)

    first = augmenter(channels, labels, valid, seed=991)
    repeated = augmenter(channels, labels, valid, seed=991)

    for name in ("nir", "red", "green"):
        assert not np.allclose(first[0][name], channels[name])
        np.testing.assert_array_equal(first[0][name], repeated[0][name])
    np.testing.assert_array_equal(first[0]["dsm"], channels["dsm"])
    np.testing.assert_array_equal(first[1], labels)
    np.testing.assert_array_equal(first[2], valid)


def test_potsdam_policy_json_builds_with_optional_augmentation_dependency():
    pytest.importorskip("albumentations")
    domain = load_domain_spec("isprs-potsdam-multimodal-v1")
    model_input = _view(("ir", "red", "green", "dsm", "ndsm"))
    spec = load_augmentation_spec("isprs-potsdam-v1")
    augmenter = AlbumentationsAugmenter(
        domain,
        model_input,
        spec,
        ignore_label=6,
    )
    image = np.ones((32, 32), dtype=np.float32)
    channels = {name: image.copy() for name in model_input.channels}
    labels = np.zeros(image.shape, dtype=np.uint8)
    valid = np.ones(image.shape, dtype=bool)

    transformed = augmenter(channels, labels, valid, seed=18)

    assert transformed[0]["ir"].shape == image.shape
    assert transformed[0]["dsm"].shape == image.shape
    assert transformed[1].shape == image.shape
