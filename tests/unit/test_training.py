from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin
from torch import nn

from deepnet import training
from deepnet.data.bundles import export_model_bundle
from deepnet.data.catalog import load_model_input_spec, load_preprocessing_spec
from deepnet.data.contracts import SceneRecord, SplitManifest
from deepnet.data.quality import validate_data_quality
from deepnet.data.raster import inspect_raster, sha256_file
from deepnet.training_utils import artifact_index, load_checkpoint, write_json


@pytest.fixture
def training_run(tmp_path, monkeypatch):
    repository = Path(__file__).resolve().parents[2]
    config = json.loads(
        (repository / "configs/training/isprs-dinov2-joint-v1.json").read_text()
    )
    protocol = json.loads(
        (repository / "configs/evaluations/isprs-dinov2-v1.json").read_text()
    )
    config["execution"].update(
        max_epochs=2,
        optimizer_steps_per_epoch=2,
        gradient_accumulation_steps=2,
        checkpoint_interval_optimizer_steps=1,
        num_workers=0,
        gradient_checkpointing=False,
        mixed_precision="float32",
        evaluation_batch_size=2,
    )
    preprocessing = load_preprocessing_spec(config["preprocessing_id"])
    model_input = load_model_input_spec(config["model_input_id"])
    palette = np.asarray(
        [color[:3] for color in preprocessing.label_palette], dtype=np.uint8
    )
    labels = np.indices((12, 12)).sum(axis=0) % 6
    labels[0, 0] = 6
    source = tmp_path / "source"
    source.mkdir()
    root = tmp_path / "bundles"
    for item in config["datasets"]:
        item.update(train_window_size=[8, 8], validation_window_size=[8, 8])
        name = item["dataset_id"].removeprefix("isprs-")
        records = []
        scene_ids = tuple(name + "-" + role for role in ("train", "validation", "test"))
        for scene_id in scene_ids:
            assets = {}
            for asset, pixels, bands in (
                ("irrg", np.full((3, 12, 12), 120, np.uint8), ("ir", "red", "green")),
                (
                    "labels",
                    palette[labels].transpose(2, 0, 1),
                    ("red", "green", "blue"),
                ),
            ):
                path = source / f"{scene_id}-{asset}.tif"
                with rasterio.open(
                    path,
                    "w",
                    driver="GTiff",
                    width=12,
                    height=12,
                    count=3,
                    dtype="uint8",
                    crs="EPSG:32632",
                    transform=from_origin(100, 200, 1, 1),
                ) as target:
                    target.write(pixels)
                assets[asset] = inspect_raster(
                    path,
                    asset_id=asset,
                    uri=path.name,
                    bands=bands,
                    grid_id=scene_id,
                    hash_file=True,
                )
            records.append(
                SceneRecord(
                    item["dataset_id"],
                    scene_id,
                    1,
                    assets["irrg"].grid,
                    preprocessing.label_ontology,
                    tuple(assets.items()),
                )
            )
        split = SplitManifest(
            item["dataset_id"],
            item["split_id"],
            item["split_version"],
            (scene_ids[0],),
            (scene_ids[1],),
            (scene_ids[2],),
            spatial_groups=tuple((scene_id, scene_id) for scene_id in scene_ids),
            spatial_scope="Synthetic scene-held-out fixture",
            label_assets=tuple(
                (role, "labels") for role in ("train", "validation", "test")
            ),
        )
        protocol["split_manifests"][item["dataset_id"]]["content_hash"] = (
            split.content_hash
        )
        for roles, destination in (
            (("train", "validation"), item["bundle_path"]),
            (("test",), name + "/post-selection-test-bundle"),
        ):
            export_model_bundle(
                records,
                source,
                split,
                preprocessing,
                model_input,
                root / destination,
                include_splits=roles,
            )
        quality = validate_data_quality(
            tuple(records),
            source,
            split,
            preprocessing,
            {"irrg-view": ("irrg",)},
            tile_size=8,
            include_splits=("train",),
        )
        write_json(root / item["quality_path"], quality)
    config_path, protocol_path = tmp_path / "config.json", tmp_path / "protocol.json"
    write_json(config_path, config)
    protocol["training_config"]["sha256"] = sha256_file(config_path)
    write_json(protocol_path, protocol)
    pretrained = tmp_path / "pretrained"
    pretrained.mkdir()
    (pretrained / "weights.bin").write_bytes(b"synthetic test weights")
    write_json(
        pretrained / "pretrained_manifest.json",
        {
            "model_id": config["hub_preflight"]["model_id"],
            "revision": config["hub_preflight"]["revision"],
            "files": [
                {
                    "path": "weights.bin",
                    "bytes": 22,
                    "sha256": sha256_file(pretrained / "weights.bin"),
                }
            ],
        },
    )

    def tiny_model(*_args, **_kwargs):
        return nn.Sequential(
            nn.Conv2d(3, 8, 1), nn.GELU(), nn.Dropout(0.2), nn.Conv2d(8, 6, 1)
        )

    monkeypatch.setattr(training.MultimodalSegmenter, "from_pretrained", tiny_model)

    def run(output, **kwargs):
        return training.run_training(
            config_path, protocol_path, root, output, pretrained, device="cpu", **kwargs
        )

    return run


def test_joint_training_freezes_selection_before_loading_each_test_once(
    training_run, tmp_path, monkeypatch
):
    output = tmp_path / "run"
    original = training.load_bundle
    test_reads = []

    def checked_load(path):
        if path.name == "post-selection-test-bundle":
            assert (output / "selection.json").is_file()
            test_reads.append(path.parent.name)
        return original(path)

    monkeypatch.setattr(training, "load_bundle", checked_load)
    manifest = training_run(output)
    assert manifest["status"] == "complete"
    assert test_reads == ["potsdam", "vaihingen"]
    metrics = json.loads((output / "metrics.json").read_text())
    for result in metrics["test"].values():
        assert result["pixels_scored"] == 143
        np.testing.assert_array_equal(
            result["scenes"][0]["confusion_matrix"], result["confusion_matrix"]
        )
    for epoch in manifest["progress"]["epoch_histograms"]:
        assert all(value["patch_count"] == 2 for value in epoch["datasets"].values())
    training_run(output, resume=True)
    assert test_reads == ["potsdam", "vaihingen"]
    with pytest.raises(FileExistsError):
        training_run(output)
    artifact_index(output)
    audit = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts/watch_dinov2_kaggle.py")
    )["audit"]
    assert audit(output)["status"] == "verified"


def test_interrupted_training_resume_matches_uninterrupted_weights(
    training_run, tmp_path, monkeypatch
):
    baseline = tmp_path / "baseline"
    baseline_manifest = training_run(baseline)
    interrupted = tmp_path / "interrupted"
    original = training.save_checkpoint

    def fail_after_first_update(path, model, progress, binding, **kwargs):
        checksum = original(path, model, progress, binding, **kwargs)
        if path.name == "last.pt" and progress["global_step"] == 1:
            raise RuntimeError("simulated interruption")
        return checksum

    monkeypatch.setattr(training, "save_checkpoint", fail_after_first_update)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        training_run(interrupted)
    monkeypatch.setattr(training, "save_checkpoint", original)
    resumed = training_run(interrupted, resume=True)
    before = load_checkpoint(baseline / "last.pt", baseline_manifest["binding"])
    after = load_checkpoint(interrupted / "last.pt", resumed["binding"])
    assert before["progress"] == after["progress"]
    for name in before["model"]:
        torch.testing.assert_close(
            before["model"][name], after["model"][name], rtol=0, atol=0
        )


def test_interrupted_test_is_not_silently_repeated(training_run, tmp_path, monkeypatch):
    output = tmp_path / "run"
    original = training._evaluate

    def fail_test(model, bundle, role, *args):
        if role == "test":
            raise RuntimeError("test interrupted")
        return original(model, bundle, role, *args)

    monkeypatch.setattr(training, "_evaluate", fail_test)
    with pytest.raises(RuntimeError, match="test interrupted"):
        training_run(output)
    monkeypatch.setattr(training, "_evaluate", original)
    with pytest.raises(RuntimeError, match="requires explicit review"):
        training_run(output, resume=True)
