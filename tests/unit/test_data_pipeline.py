from __future__ import annotations

import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from deepnet import runtime
from deepnet.data.bundles import export_model_bundle, load_bundle, verify_bundle
from deepnet.data.catalog import DatasetCatalog, load_split_manifest
from deepnet.data.contracts import (
    AlignmentError,
    GridSpec,
    ModelInputSpec,
    NormalizationRule,
    PreprocessingSpec,
    RasterAsset,
    SceneRecord,
    SplitManifest,
    WindowSpec,
    canonical_hash,
)
from deepnet.data.dataset import (
    SceneAccess,
    ScenePatchDataset,
    build_class_pixel_index,
    collate_samples,
)
from deepnet.data.evaluation import confusion_matrix, evaluate_scenes
from deepnet.data.lineage import build_run_manifest
from deepnet.data.preprocessing import (
    decode_rgb_labels,
    fit_model_statistics,
    fit_train_statistics,
)
from deepnet.data.raster import RasterioSceneReader, inspect_raster
from deepnet.data.sampling import (
    ClassAwareSampler,
    UniformSampler,
    augment_aligned,
    coverage_counts,
    grid_windows,
    sample_rng,
)
from deepnet.models.adapter import TorchSegmentationAdapter


@pytest.fixture
def pipeline_fixture(tmp_path: Path):
    data_root = tmp_path / "source"
    data_root.mkdir()
    shape = (5, 7)
    transform = from_origin(100, 200, 1, 1)
    palette = {
        0: (255, 0, 0),
        1: (0, 255, 0),
        6: (0, 0, 0),
    }
    labels = np.indices(shape).sum(axis=0).astype(np.uint8) % 2
    label_rgb = np.empty((*shape, 3), dtype=np.uint8)
    for class_id, color in palette.items():
        label_rgb[labels == class_id] = color

    records = []
    scene_ids = ("train-a", "validation-a", "test-a")
    for scene_id in scene_ids:
        image_path = data_root / f"{scene_id}-image.tif"
        label_path = data_root / f"{scene_id}-label.tif"
        image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
        image[0, 0] = 255
        with rasterio.open(
            image_path,
            "w",
            driver="GTiff",
            width=shape[1],
            height=shape[0],
            count=1,
            dtype="uint8",
            crs="EPSG:32632",
            transform=transform,
            nodata=255,
        ) as target:
            target.write(image, 1)
        encoded_labels = label_rgb.copy()
        with rasterio.open(
            label_path,
            "w",
            driver="GTiff",
            width=shape[1],
            height=shape[0],
            count=3,
            dtype="uint8",
            crs="EPSG:32632",
            transform=transform,
        ) as target:
            target.write(encoded_labels.transpose(2, 0, 1))
            label_mask = np.full(shape, 255, dtype=np.uint8)
            label_mask[0, 0] = 0
            target.write_mask(label_mask)

        grid_id = f"synthetic:{scene_id}"
        image_asset = inspect_raster(
            image_path,
            asset_id="irrg",
            uri=image_path.name,
            bands=("red",),
            grid_id=grid_id,
        )
        label_asset = inspect_raster(
            label_path,
            asset_id="labels",
            uri=label_path.name,
            bands=("red", "green", "blue"),
            grid_id=grid_id,
        )
        records.append(
            SceneRecord(
                dataset_id="synthetic",
                scene_id=scene_id,
                schema_version=1,
                reference_grid=image_asset.grid,
                label_ontology="synthetic-two-class-v1",
                assets=(("irrg", image_asset), ("labels", label_asset)),
                provenance=(("provider", "test fixture"),),
            )
        )

    split = SplitManifest(
        dataset_id="synthetic",
        split_id="synthetic-v1",
        version=1,
        train_scene_ids=("train-a",),
        validation_scene_ids=("validation-a",),
        test_scene_ids=("test-a",),
        spatial_groups=(
            ("train-a", "scene:train-a"),
            ("validation-a", "scene:validation-a"),
            ("test-a", "scene:test-a"),
        ),
        label_assets=(
            ("train", "labels"),
            ("validation", "labels"),
            ("test", "labels"),
        ),
        spatial_scope="scene-level synthetic fixture",
    )
    preprocessing = PreprocessingSpec(
        preprocessing_id="synthetic-canonical",
        version=1,
        label_ontology="synthetic-two-class-v1",
        alignment_policy="require-matching-grid",
        continuous_resampling="bilinear",
        categorical_resampling="nearest",
        nodata_policy="separate-mask",
        label_palette=(
            (255, 0, 0, 0),
            (0, 255, 0, 1),
            (0, 0, 0, 6),
        ),
        ignore_label=6,
    )
    model_input = ModelInputSpec(
        model_id="synthetic-model",
        version=1,
        channels=("red",),
        channel_sources=(("red", "irrg", "red"),),
        normalization=(NormalizationRule("red", "scale", scale=255.0),),
        pixel_semantics="Synthetic uint8 red band scaled to [0, 1].",
    )
    return data_root, tuple(records), split, preprocessing, model_input


def test_potsdam_legacy_split_is_frozen_scene_disjoint():
    manifest_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "splits"
        / "potsdam-legacy-v1.json"
    )
    split = load_split_manifest(manifest_path)
    assert (
        len(split.train_scene_ids),
        len(split.validation_scene_ids),
        len(split.test_scene_ids),
    ) == (
        18,
        6,
        14,
    )
    assert len(split.scene_ids) == len(set(split.scene_ids)) == 38
    assert "not a region-held-out split" in split.spatial_scope


def test_catalog_resolves_initial_isprs_datasets():
    catalog = DatasetCatalog.load()

    assert catalog.get("isprs-potsdam").provider == "ISPRS"
    assert catalog.get("isprs-vaihingen").provider == "ISPRS"


def test_confusion_matrix_accepts_only_declared_classes_and_ignore():
    matrix = confusion_matrix(
        np.array([[0, 1, 6]]),
        np.array([[1, 1, 0]]),
        num_classes=2,
        ignore_label=6,
    )
    np.testing.assert_array_equal(matrix, [[0, 1], [0, 1]])
    with pytest.raises(ValueError, match="undeclared class IDs"):
        confusion_matrix(
            np.array([[7]]),
            np.array([[0]]),
            num_classes=2,
            ignore_label=6,
        )


def test_split_rejects_scene_and_spatial_group_leakage():
    with pytest.raises(ValueError, match="occurs in both"):
        SplitManifest(
            dataset_id="x",
            split_id="bad",
            version=1,
            train_scene_ids=("a",),
            validation_scene_ids=("a",),
            test_scene_ids=("c",),
            spatial_groups=(("a", "g1"), ("c", "g2")),
            label_assets=(("train", "l"), ("validation", "l"), ("test", "l")),
            spatial_scope="scene-level",
        )

    with pytest.raises(ValueError, match="leaks"):
        SplitManifest(
            dataset_id="x",
            split_id="bad",
            version=1,
            train_scene_ids=("a",),
            validation_scene_ids=("b",),
            test_scene_ids=("c",),
            spatial_groups=(("a", "shared"), ("b", "shared"), ("c", "test")),
            label_assets=(("train", "l"), ("validation", "l"), ("test", "l")),
            spatial_scope="group-level",
        )


def test_alignment_never_uses_matching_dimensions_alone():
    reference = GridSpec("reference", 7, 5, "EPSG:32632", (1, 0, 100, 0, -1, 200))
    shifted = GridSpec("shifted", 7, 5, "EPSG:32632", (1, 0, 101, 0, -1, 200))
    asset = RasterAsset("labels", "labels.tif", ("r", "g", "b"), "uint8", shifted)
    scene = SceneRecord(
        "x",
        "scene-a",
        1,
        reference,
        "ontology-v1",
        (("labels", asset),),
    )
    with pytest.raises(AlignmentError, match="matching dimensions alone"):
        scene.require_aligned(("labels",))

    pixel_reference = GridSpec("reference-pixels", 7, 5)
    reference_asset = RasterAsset(
        "irrg", "irrg.tif", ("red",), "uint8", pixel_reference
    )
    unreferenced_scene = SceneRecord(
        "x",
        "scene-c",
        1,
        pixel_reference,
        "ontology-v1",
        (("irrg", reference_asset),),
    )
    unreferenced_scene.require_aligned(("irrg",))

    same_shape_unverified = RasterAsset(
        "same-shape-labels",
        "same-shape-labels.tif",
        ("red",),
        "uint8",
        GridSpec("separate-pixel-grid", 7, 5),
    )
    unreferenced_scene = SceneRecord(
        "x",
        "scene-d",
        1,
        pixel_reference,
        "ontology-v1",
        (("irrg", reference_asset), ("same-shape-labels", same_shape_unverified)),
    )
    with pytest.raises(AlignmentError, match="matching dimensions alone"):
        unreferenced_scene.require_aligned(("same-shape-labels",))

    pixel_local = RasterAsset(
        "pixel-local-labels",
        "pixel-local-labels.tif",
        ("red",),
        "uint8",
        GridSpec("pixel-local", 6, 5),
        alignment_evidence="Source pair is known but dimensions differ.",
    )
    pixel_local_scene = SceneRecord(
        "x",
        "scene-b",
        1,
        reference,
        "ontology-v1",
        (("pixel-local-labels", pixel_local),),
    )
    with pytest.raises(AlignmentError):
        pixel_local_scene.require_aligned(("pixel-local-labels",))


def test_label_decoder_rejects_unknown_valid_colors_and_preserves_ignore(
    pipeline_fixture,
):
    _, _, _, preprocessing, _ = pipeline_fixture
    image = np.array(
        [[[255, 0, 0], [1, 2, 3]], [[0, 255, 0], [0, 0, 0]]], dtype=np.uint8
    )
    valid = np.array([[True, False], [True, True]])
    decoded = decode_rgb_labels(image, preprocessing, valid_mask=valid)
    np.testing.assert_array_equal(decoded, [[0, 6], [1, 6]])
    with pytest.raises(ValueError, match="undeclared RGB colors"):
        decode_rgb_labels(image, preprocessing)


def test_fitted_statistics_use_train_scenes_only(pipeline_fixture):
    _, _, split, preprocessing, _ = pipeline_fixture
    samples = [
        ("train-a", {"elevation": np.array([[1.0, 3.0]])}, np.array([[True, True]]))
    ]
    fitted = fit_train_statistics(
        preprocessing,
        split,
        samples,
        channels=("elevation",),
    )
    assert fitted.fitted_statistics[0][0] == "elevation"
    assert fitted.fitted_statistics[0][1] == pytest.approx(2.0)
    assert fitted.fit_scene_ids == ("train-a",)
    with pytest.raises(ValueError, match="non-training scene"):
        fit_train_statistics(
            preprocessing,
            split,
            [("test-a", {"elevation": np.array([[1.0, 3.0]])}, np.ones((1, 2), bool))],
            channels=("elevation",),
        )


def test_model_statistics_stream_train_windows_and_version_the_spec(
    pipeline_fixture,
):
    data_root, scenes, split, preprocessing, _ = pipeline_fixture
    model_input = ModelInputSpec(
        model_id="synthetic-elevation-model",
        version=1,
        channels=("elevation",),
        channel_sources=(("elevation", "irrg", "red"),),
        normalization=(NormalizationRule("elevation", "zscore"),),
        pixel_semantics="Synthetic test values standardized from training pixels.",
    )
    fitted = fit_model_statistics(
        scenes,
        data_root,
        split,
        preprocessing,
        model_input,
        tile_size=2,
    )
    assert fitted.version == preprocessing.version + 1
    assert fitted.fit_scene_ids == split.train_scene_ids
    assert fitted.fit_split_hash == split.content_hash
    assert fitted.fitted_statistics[0][0] == "elevation"
    assert fitted.fitted_statistics[0][1] == pytest.approx(17.5)


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected_device"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_runtime_prefers_available_device_and_sets_transfer_options(
    monkeypatch,
    cuda_available,
    mps_available,
    expected_device,
):
    class FakeDevice:
        def __init__(self, value):
            self.type = value.split(":", maxsplit=1)[0]

        def __str__(self):
            return self.type

    fake_torch = SimpleNamespace(
        device=FakeDevice,
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available)
        ),
    )
    monkeypatch.setattr(runtime, "_torch", lambda: fake_torch)

    selected = runtime.configure_torch_runtime(num_workers=2)

    assert selected.device_type == expected_device
    assert selected.num_workers == 2
    assert selected.pin_memory is (expected_device == "cuda")
    assert selected.non_blocking is (expected_device == "cuda")


def test_torch_adapter_checks_model_input_contract(monkeypatch, pipeline_fixture):
    _, _, _, _, model_input = pipeline_fixture

    class FakeTensor:
        def __init__(self, shape):
            self.shape = tuple(shape)
            self.ndim = len(self.shape)

        def to(self, **_kwargs):
            return self

    fake_torch = SimpleNamespace(
        Tensor=FakeTensor,
        float32="float32",
        as_tensor=lambda value: FakeTensor(value.shape),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    adapter = TorchSegmentationAdapter(
        model=lambda _images: FakeTensor((2, 2, 4, 4)),
        input_spec=model_input,
    )

    logits = adapter.predict_logits(np.zeros((2, 1, 4, 4)), device="cpu")

    assert logits.shape == (2, 2, 4, 4)
    with pytest.raises(ValueError, match="model contract declares"):
        adapter.prepare_images(np.zeros((2, 3, 4, 4)), device="cpu")


def test_grid_coverage_and_sampling_are_deterministic():
    windows = grid_windows(13, 17, (6, 5), (4, 3))
    coverage = coverage_counts(13, 17, windows)
    assert np.all(coverage > 0)
    assert windows == grid_windows(13, 17, (6, 5), (4, 3))

    uniform = UniformSampler({"train-a": (32, 32)}, ("train-a",), (8, 8))
    first = uniform.sample(sample_rng(42, 3, 11))
    second = uniform.sample(sample_rng(42, 3, 11))
    assert first == second


def test_class_aware_sampler_rejects_validation_candidates(pipeline_fixture):
    _, _, split, _, _ = pipeline_fixture
    with pytest.raises(ValueError, match="non-training scenes"):
        ClassAwareSampler(
            {"train-a": (10, 10), "validation-a": (10, 10)},
            {1: {"validation-a": [4]}},
            split,
            (4, 4),
        )


def test_aligned_augmentation_keeps_label_pixels_in_register():
    image = np.arange(25).reshape(5, 5, 1)
    labels = image[..., 0] % 2
    valid = np.ones((5, 5), dtype=bool)
    transformed_image, transformed_labels, transformed_valid = augment_aligned(
        image,
        labels,
        valid,
        random.Random(5),
    )
    np.testing.assert_array_equal(transformed_image[..., 0] % 2, transformed_labels)
    assert transformed_valid.all()


def test_window_reader_preserves_nodata_mask_and_boundless_padding(pipeline_fixture):
    data_root, scenes, _, _, _ = pipeline_fixture
    scene = scenes[0]
    reader = RasterioSceneReader(scene, data_root)
    array, valid = reader.read_window_with_mask(
        "labels",
        WindowSpec(0, 0, 4, 5),
    )
    assert array.shape == (4, 5, 3)
    assert not valid[0, 0]
    edge, edge_valid = reader.read_window_with_mask(
        "irrg",
        WindowSpec(3, 5, 4, 4, padding="constant"),
    )
    assert edge.shape == (4, 4, 1)
    assert edge_valid[:2, :2].all()
    assert not edge_valid[2:, :].any()
    assert not edge_valid[:, 2:].any()
    reader.close()


def test_shared_dataset_and_bundle_round_trip_are_reproducible(
    pipeline_fixture, tmp_path
):
    data_root, scenes, split, preprocessing, model_input = pipeline_fixture
    source_bytes = {
        path.name: path.read_bytes() for path in data_root.iterdir() if path.is_file()
    }
    access = SceneAccess(scenes, data_root, split, "train", preprocessing, model_input)
    dataset = ScenePatchDataset(
        access,
        (4, 4),
        3,
        seed=9,
        augmentation=False,
    )
    sample = dataset[0]
    assert sample.image.shape == (1, 4, 4)
    batch = collate_samples([sample, dataset[0]])
    assert batch["image"].shape == (2, 1, 4, 4)
    repeated = dataset[0]
    np.testing.assert_array_equal(sample.image, repeated.image)
    np.testing.assert_array_equal(sample.labels, repeated.labels)
    with pytest.raises(ValueError, match="outside the selected train split"):
        access.read_labels("test-a", WindowSpec(0, 0, 4, 4))
    access.close()

    index = build_class_pixel_index(
        SceneAccess(scenes, data_root, split, "train", preprocessing, model_input),
        (0, 1),
        tile_size=3,
        max_pixels_per_scene=2,
        seed=1,
    )
    assert all(set(by_scene) == {"train-a"} for by_scene in index.values())
    assert all(
        len(points) <= 2 for by_scene in index.values() for points in by_scene.values()
    )

    output_dir = tmp_path / "bundle"
    manifest = export_model_bundle(
        scenes,
        data_root,
        split,
        preprocessing,
        model_input,
        output_dir,
        source_revision="test-revision",
    )
    assert verify_bundle(output_dir)["bundle_sha256"] == manifest["bundle_sha256"]
    bundle = load_bundle(output_dir)
    run_manifest = build_run_manifest(
        bundle,
        run_id="synthetic-run",
        seed=42,
        sampler={"strategy": "uniform"},
        configuration={"epochs": 1},
        runtime={"device": "cpu"},
    )
    assert run_manifest["bundle_sha256"] == manifest["bundle_sha256"]
    assert run_manifest["split_hash"] == split.content_hash
    assert run_manifest["runtime"]["device"] == "cpu"
    bundled_access = bundle.scene_access("validation")
    bundled_sample = bundled_access.read_sample("validation-a", WindowSpec(0, 0, 4, 4))
    assert bundled_sample.image.shape == (1, 4, 4)
    bundled_access.close()
    assert source_bytes == {
        path.name: path.read_bytes() for path in data_root.iterdir() if path.is_file()
    }

    changed_file = next(
        output_dir / record["path"]
        for record in manifest["files"]
        if record["path"].endswith("-image.tif")
    )
    changed_file.write_bytes(changed_file.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="size mismatch|checksum mismatch"):
        verify_bundle(output_dir)


def test_scene_evaluation_uses_full_grid_and_ignores_nodata(
    pipeline_fixture, monkeypatch, tmp_path
):
    data_root, scenes, split, preprocessing, model_input = pipeline_fixture

    class FakeTensor:
        def __init__(self, array):
            self.array = np.asarray(array)

        @property
        def shape(self):
            return self.array.shape

        def to(self, **_kwargs):
            return self

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    fake_torch = SimpleNamespace(
        device=lambda name: name,
        float32="float32",
        from_numpy=FakeTensor,
        no_grad=nullcontext,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class PatternModel:
        def eval(self):
            return self

        def parameters(self):
            return iter(())

        def __call__(self, batch):
            values = np.rint(batch.array[:, 0] * 255).astype(np.int64)
            classes = (values // 7 + values % 7) % 2
            logits = np.zeros((len(values), 2, *values.shape[1:]), dtype=np.float32)
            np.put_along_axis(logits, classes[:, None, ...], 10.0, axis=1)
            return FakeTensor(logits)

    access = SceneAccess(
        scenes,
        data_root,
        split,
        "validation",
        preprocessing,
        model_input,
    )
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()
    result = evaluate_scenes(
        PatternModel(),
        access,
        ("validation-a",),
        window_size=(4, 4),
        stride=(3, 3),
        batch_size=2,
        scratch_dir=scratch_dir,
    )

    assert result["mean_iou"] == pytest.approx(1.0), result
    assert result["pixels_scored"] == 34
    assert result["scenes"][0]["window_count"] == 6
    assert not list(scratch_dir.iterdir())
    access.close()


def test_bundle_rejects_path_traversal_in_manifest(pipeline_fixture, tmp_path):
    data_root, scenes, split, preprocessing, model_input = pipeline_fixture
    output_dir = tmp_path / "bundle"
    export_model_bundle(
        scenes, data_root, split, preprocessing, model_input, output_dir
    )
    path = output_dir / "bundle_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][0]["path"] = "../outside"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe bundle path"):
        verify_bundle(output_dir)


def test_canonical_hash_is_independent_of_mapping_order():
    assert canonical_hash({"a": 1, "b": [2, 3]}) == canonical_hash(
        {"b": [2, 3], "a": 1}
    )
