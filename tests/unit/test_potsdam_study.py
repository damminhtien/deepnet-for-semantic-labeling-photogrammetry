from dataclasses import replace

import numpy as np
import pytest
import torch

from deepnet.data.contracts import Sample
from deepnet.data.sampling import WindowSpec
from deepnet.models.adapters import ResidualIdentityProjection
from deepnet.potsdam_data import augment_patch, patch_schedule, refit_split
from deepnet.potsdam_protocol import candidate, ranking
from deepnet.potsdam_study import _split


def test_spectral_residual_is_exact_identity_and_learns():
    model = ResidualIdentityProjection()
    optical = torch.rand(1, 3, 8, 8)
    projected = model(optical)
    assert torch.equal(projected, optical)
    projected.sum().backward()
    assert model.delta.weight.grad is not None
    assert torch.count_nonzero(model.delta.weight.grad) == 9


def test_refit_has_exact_parent_union_and_no_evaluation_role():
    parent = _split()
    refit = refit_split(parent)
    assert refit.refit_parent_hash == parent.content_hash
    assert refit.train_scene_ids == parent.train_scene_ids + parent.validation_scene_ids
    assert set(refit.train_scene_ids).isdisjoint(parent.test_scene_ids)
    assert dict(refit.label_assets) == {"train": "label_participant"}
    with pytest.raises(ValueError, match="train-only"):
        replace(refit, test_scene_ids=parent.test_scene_ids)


def test_schedule_keeps_all_base_windows_and_adds_rare_quota():
    indexed = [
        ("scene", WindowSpec(0, col, 1, 1), [0, 0, 0, 0, col == 4, col == 5])
        for col in range(6)
    ]
    schedule = patch_schedule(indexed, 42, 0)
    assert schedule == patch_schedule(indexed, 42, 0)
    assert sorted(window.col for _, window, rare in schedule if not rare) == list(
        range(6)
    )
    assert sorted(window.col for _, window, rare in schedule if rare) == [4, 5]


def test_strong_augmentation_does_not_interpolate_missing_dsm_or_create_padding():
    image = np.ones((5, 64, 64), dtype=np.float32)
    image[4, :, :32] = 0
    image[3] = image[4] * -0.5
    labels = np.full((64, 64), 4)
    valid = np.ones((64, 64), dtype=bool)
    valid[:12] = False
    sample = Sample(image, labels, valid, "scene", WindowSpec(0, 0, 64, 64), "CHW")
    affine_seen = False
    for seed in range(20):
        out = augment_patch(sample, seed, strong=True, rare=False)
        rare = augment_patch(sample, seed, strong=True, rare=True)
        affine_seen |= int(out.valid_mask.sum()) != int(valid.sum())
        assert int(rare.valid_mask.sum()) == int(valid.sum())
        assert set(np.unique(out.labels)) <= {4, 6}
        assert (out.image[:, ~out.valid_mask] == 0).all()
        assert set(np.unique(out.image[4])) <= {0, 1}
        np.testing.assert_allclose(out.image[3], out.image[4] * -0.5, atol=1e-6)
    assert affine_seen


def test_checkpoint_floors_reject_class_collapse_and_prefer_earlier_ties():
    sam = np.eye(6, dtype=np.int64) * 5
    sam[0, 1] += 5
    proto = {"sam_validation_confusion": sam.tolist(), "validation_pixels": 35}
    worse_clutter = sam.copy()
    worse_clutter[0, 0], worse_clutter[0, 1] = 10, 0
    worse_clutter[5, 0], worse_clutter[5, 5] = 1, 4
    bad = candidate({"confusion_matrix": worse_clutter}, proto, "B", 1)
    early = candidate({"confusion_matrix": sam}, proto, "A", 2)
    late = candidate({"confusion_matrix": sam}, proto, "A", 4)
    assert bad["overall_accuracy"] > early["overall_accuracy"]
    assert not bad["eligible"]
    assert max([bad, late, early], key=ranking) == early
