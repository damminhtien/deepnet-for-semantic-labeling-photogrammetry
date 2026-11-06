"""Regression witnesses for RGBIR masks, reproducibility, and experiment gates."""

import io
import runpy
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deepnet.data.contracts import GridSpec, SceneRecord, WindowSpec, canonical_hash
from deepnet.data.raster import sha256_file
from deepnet.models.adapters import MultimodalInputAdapter
from deepnet.models.specs import load_model_spec
from deepnet.potsdam_data import refit_split
from deepnet.potsdam_rgbir.augmentation import augment
from deepnet.potsdam_rgbir.data import CacheBundle, model_view
from deepnet.potsdam_rgbir.engine import optimizer_and_scaler, train_step
from deepnet.potsdam_rgbir.phases import validate_final
from deepnet.potsdam_rgbir.protocol import (
    CONFIG,
    candidate,
    choose_schedule,
    choose_tta,
    ranking,
    read_json,
)
from deepnet.potsdam_rgbir.sampling import Draw, loader, schedule, training_index
from deepnet.potsdam_study import _split
from deepnet.training_utils import write_json


def test_kaggle_source_archive_imports_without_worktree_dependencies(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/run_potsdam_rgbir_kaggle.py"
    source = runpy.run_path(str(script))["_source"]()
    with zipfile.ZipFile(io.BytesIO(source)) as archive:
        archive.extractall(tmp_path)
    code = """
import pathlib, sys
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root))
import deepnet.potsdam_rgbir.study as study
import semantic_labeling.losses as losses
import semantic_labeling.constants as constants
for module in (study, losses, constants):
    assert pathlib.Path(module.__file__).is_relative_to(root), module.__file__
sys.argv = ['study', '--help']
study.main()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_gpu_accounting_includes_provider_wrapper_after_worker_exit(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/run_potsdam_rgbir_kaggle.py"
    account = runpy.run_path(str(script))["execution_accounting"]
    write_json(tmp_path / "kernel.log", [{"time": 5}, {"time": 27}])
    result = account(tmp_path, {"allocated_devices": 2, "execution_gpu_seconds": 44})
    assert result["execution_gpu_hours"] * 3600 == 54
    assert result["worker_gpu_hours"] * 3600 == 44
    assert result["execution_log_sha256"] == sha256_file(tmp_path / "kernel.log")


def test_private_inputs_wait_for_ready_without_reupload(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/run_potsdam_rgbir_kaggle.py"
    wait = runpy.run_path(str(script))["wait_inputs"]
    write_json(
        tmp_path / "kaggle/input-refit-receipt.json",
        {"status": "accepted", "cache_hash": "locked"},
    )
    results = iter(
        [
            subprocess.TimeoutExpired("dataset status", 45),
            SimpleNamespace(returncode=0, stdout="processing\n", stderr=""),
            SimpleNamespace(returncode=1, stdout="", stderr="403 processing"),
            SimpleNamespace(returncode=0, stdout="CLI warning\nready\n", stderr=""),
        ]
    )
    calls, sleeps = [], []

    def remote(*args, **kwargs):
        calls.append(args)
        assert kwargs == {"timeout": 45, "check": False}
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setitem(wait.__globals__, "kaggle", remote)
    monkeypatch.setattr(wait.__globals__["time"], "sleep", sleeps.append)
    wait(tmp_path, "refit", "locked")
    assert (
        calls
        == [("datasets", "status", "damminhtien/potsdam-dinov3-rgbir-v2-refit")] * 4
    )
    assert sleeps == [30, 30, 30]


@pytest.mark.parametrize("status", ["error", "processing"])
def test_private_inputs_stop_before_submission_when_not_ready(
    tmp_path, monkeypatch, status
):
    script = Path(__file__).resolve().parents[2] / "scripts/run_potsdam_rgbir_kaggle.py"
    wait = runpy.run_path(str(script))["wait_inputs"]
    write_json(
        tmp_path / "kaggle/input-heldout-receipt.json",
        {"status": "accepted", "cache_hash": "locked"},
    )
    monkeypatch.setitem(
        wait.__globals__,
        "kaggle",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=status, stderr=""),
    )
    error = RuntimeError if status == "error" else TimeoutError
    with pytest.raises(error, match="Private heldout dataset"):
        wait(tmp_path, "heldout", "locked", timeout=0)


@pytest.mark.parametrize(
    "receipt",
    [
        {"status": "submitting", "cache_hash": "locked"},
        {"status": "accepted", "cache_hash": "different"},
    ],
)
def test_private_inputs_require_confirmed_matching_receipt(
    tmp_path, monkeypatch, receipt
):
    script = Path(__file__).resolve().parents[2] / "scripts/run_potsdam_rgbir_kaggle.py"
    wait = runpy.run_path(str(script))["wait_inputs"]
    write_json(tmp_path / "kaggle/input-refit-receipt.json", receipt)
    monkeypatch.setitem(
        wait.__globals__,
        "kaggle",
        lambda *args, **kwargs: pytest.fail("Unconfirmed upload"),
    )
    with pytest.raises(ValueError, match="needs reconciliation"):
        wait(tmp_path, "refit", "locked")


@pytest.mark.parametrize("phase", ["refit", "heldout"])
def test_phase_package_separates_notebooks_from_existing_input_datasets(
    tmp_path, monkeypatch, phase
):
    script = Path(__file__).resolve().parents[2] / "scripts/run_potsdam_rgbir_kaggle.py"
    package = runpy.run_path(str(script))["package"]
    monkeypatch.setitem(
        package.__globals__, "load_protocol", lambda _: {"arms": list("ABC")}
    )
    monkeypatch.setitem(
        package.__globals__,
        "CacheBundle",
        SimpleNamespace(
            load=lambda _: SimpleNamespace(manifest={"cache_hash": "locked"})
        ),
    )
    monkeypatch.setitem(package.__globals__, "validate_final", lambda *args: None)
    write_json(tmp_path / "study/selection-lock.json", {})
    write_json(tmp_path / "study/refit/manifest.json", {})
    (tmp_path / "study/refit/final.pt").write_bytes(b"final-gate-tested-separately")
    folder = package(tmp_path, phase, tmp_path / "cache")
    metadata = read_json(folder / "kernel-metadata.json")
    prefix = "damminhtien/potsdam-dinov3-rgbir-v2-"
    assert metadata["dataset_sources"] == [prefix + phase]
    assert metadata["id"] == prefix + phase + "-run"
    assert metadata["id"] not in metadata["dataset_sources"]
    assert metadata["title"].endswith(phase + " run") and metadata["is_private"]
    if phase == "heldout":
        assert metadata["kernel_sources"][-1] == prefix + "refit-run"


def test_phase_watch_recovers_status_timeout_without_resubmitting(
    tmp_path, monkeypatch
):
    script = Path(__file__).resolve().parents[2] / "scripts/run_potsdam_rgbir_kaggle.py"
    watch = runpy.run_path(str(script))["await_phase"]
    receipt = {"slug": "owner/existing-run", "version": 1, "status": "accepted"}
    write_json(tmp_path / "kaggle/refit/submission.json", receipt)
    calls, sleeps = [], []

    def remote(*args, **kwargs):
        calls.append(args)
        assert args[0] == "kernels" and args[1] in {"status", "logs", "output"}
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(args, 45)
        text = "KernelWorkerStatus.COMPLETE" if args[1] == "status" else ""
        return subprocess.CompletedProcess(args, 0, text, "")

    monkeypatch.setitem(watch.__globals__, "kaggle", remote)
    monkeypatch.setitem(watch.__globals__, "_verify_index", lambda *args: None)
    monkeypatch.setattr(watch.__globals__["time"], "sleep", sleeps.append)
    assert watch(tmp_path, "refit") == tmp_path / "kaggle/refit/outputs"
    assert sleeps == [25]
    assert read_json(tmp_path / "kaggle/refit/submission.json") == receipt
    terminal = read_json(tmp_path / "kaggle/refit/terminal.json")
    assert terminal == {"state": "COMPLETE", "version": "owner/existing-run/1"}
    assert [c[1] for c in calls] == ["status", "status", "logs", "output"]


def _cache(root, split=None):
    parent = _split()
    split = split or parent
    roles = ["train"] if split.refit_parent_hash else ["train", "validation"]
    ids = [scene for role in roles for scene in split.scenes_for(role)]
    root.mkdir()
    write_json(root / "split.json", split.to_dict())
    prep = read_json(
        Path(__file__).resolve().parents[2]
        / "configs/preprocessing/isprs-canonical-v1.json"
    )
    write_json(root / "preprocessing.json", prep)
    scenes, records = [], {}
    for scene in ids:
        role = "train" if scene in split.train_scene_ids else "validation"
        folder = root / scene
        folder.mkdir()
        optical = np.broadcast_to(
            np.array([50, 100, 150, 200], dtype=np.uint8)[:, None, None], (4, 48, 48)
        ).copy()
        labels = np.broadcast_to(
            np.arange(48, dtype=np.uint8)[None] % 6, (48, 48)
        ).copy()
        valid = np.ones((48, 48), dtype=np.uint8)
        valid[:3] = 0
        labels[:3] = 6
        mask = np.ones((48, 48), dtype=np.uint8)
        mask[:, 24:] = 0
        arrays = {
            "optical": optical,
            "relief": mask.astype(np.float32) * -0.4,
            "dsm_available": mask,
            "labels": labels,
            "valid": valid,
        }
        for name, values in arrays.items():
            np.save(folder / f"{name}.npy", values, allow_pickle=False)
        grid = GridSpec(scene, 48, 48)
        scenes.append(
            SceneRecord(
                "isprs-potsdam", scene, 1, grid, "isprs-six-class-v1", ()
            ).to_dict()
        )
        records[scene] = {
            "role": role,
            "shape": [48, 48],
            "label_asset": split.label_asset_for(role),
            "valid_pixels": int(valid.sum()),
            "class_support": np.bincount(
                labels[valid.astype(bool)], minlength=6
            ).tolist(),
        }
    write_json(root / "catalog.json", scenes)
    files = [
        {
            "path": p.relative_to(root).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]
    manifest = {
        "kind": "potsdam-rgbir-memmap-v2",
        "schema_version": 1,
        "selection_split_hash": parent.content_hash,
        "split_hash": split.content_hash,
        "included_splits": roles,
        "included_scene_ids": ids,
        "scenes": records,
        "files": files,
    }
    manifest["cache_hash"] = canonical_hash(manifest)
    write_json(root / "cache_manifest.json", manifest)
    return CacheBundle.load(root)


def test_cache_keeps_missing_dsm_pixels_and_rejects_changed_bytes(tmp_path):
    bundle = _cache(tmp_path / "cache")
    with (
        bundle.access("train", auxiliary=True) as full,
        bundle.access("train", auxiliary=False) as rgb,
    ):
        scene = bundle.split.train_scene_ids[0]
        window = WindowSpec(32, 32, 32, 32, "constant")
        enriched, visible = (
            full.read_sample(scene, window),
            rgb.read_sample(scene, window),
        )
        assert enriched.valid_mask.sum() == 16 * 16
        np.testing.assert_array_equal(enriched.labels, visible.labels)
        np.testing.assert_array_equal(enriched.valid_mask, visible.valid_mask)
        assert not enriched.image[4:].any()
        assert (enriched.labels[~enriched.valid_mask] == 6).all()
        with pytest.raises(ValueError, match="outside selected"):
            full.read_sample(bundle.split.test_scene_ids[0], window)
    path = bundle.root / scene / "labels.npy"
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Cache file mismatch"):
        CacheBundle.load(bundle.root)


def test_accumulated_update_counts_only_successful_valid_presentations():
    config = read_json(CONFIG)
    config["execution"]["mixed_precision"] = "float32"
    runtime = SimpleNamespace(
        device=torch.device("cpu"), device_type="cpu", non_blocking=False
    )
    model = torch.nn.Conv2d(3, 6, 1)
    initial = model.weight.detach().clone()
    optimizer, scaler = optimizer_and_scaler(model, runtime, config)
    labels = (torch.arange(64).reshape(1, 8, 8) % 6).long()
    valid = torch.ones_like(labels, dtype=torch.bool)
    valid[0, 0, 0] = False
    batch = {"image": torch.rand(1, 3, 8, 8), "labels": labels, "valid_mask": valid}
    loss, matrix = train_step(
        model, [batch] * 4, optimizer, scaler, torch.ones(6), runtime, config, 0, 16
    )
    assert np.isfinite(loss) and matrix.sum() == 4 * 63
    assert not torch.equal(initial, model.weight)
    assert not model._forward_hooks


def test_mask_aware_augmentation_and_post_optical_spectral_index(tmp_path):
    bundle = _cache(tmp_path / "cache")
    with bundle.access("train", auxiliary=True) as access:
        scene = bundle.split.train_scene_ids[0]
        draw = Draw(scene, WindowSpec(8, 8, 32, 32))
        recipe = read_json(CONFIG)["augmentation"]["enhanced"]
        for seed in range(24):
            out = augment(access, draw, seed=seed, recipe=recipe)
            assert set(np.unique(out.image[5])) <= {0, 1}
            assert (out.image[:, ~out.valid_mask] == 0).all()
            assert (out.labels[~out.valid_mask] == 6).all()
            np.testing.assert_allclose(out.image[4], out.image[5] * -0.4, atol=1e-6)
        recipe = {
            **recipe,
            "affine_probability": 0,
            "gain_probability": 1,
            "gain_range": [2, 2],
            "gamma_probability": 0,
            "noise_probability": 0,
            "dsm_dropout": 0,
        }
        out = augment(access, replace(draw, rare_class=4), seed=42, recipe=recipe)
        domain, view = model_view(True)
        adapter = MultimodalInputAdapter(
            domain, view, load_model_spec("dinov3-vitb16-rgbir-v2").input_adapter
        )
        seen = []
        handle = adapter.auxiliary_spatial.register_forward_pre_hook(
            lambda _m, inputs: seen.append(inputs[0].detach())
        )
        adapter(torch.from_numpy(out.image[None]))
        handle.remove()
        expected = (out.image[3] - out.image[0]) / (out.image[3] + out.image[0] + 1e-6)
        np.testing.assert_allclose(seen[0][0, 0].numpy(), expected)
        assert abs(float(expected[out.valid_mask].mean()) - 0.6) > 0.1


def test_sampling_covers_base_grid_and_worker_resume_is_deterministic(tmp_path):
    bundle = _cache(tmp_path / "cache")
    with bundle.access("train", auxiliary=True) as access:
        index = training_index(access)
        draws = schedule(index, access, 42, 0)
        assert len(draws) == 24
        assert len([d for d in draws if d.rare_class is None]) == 18
        for draw in draws:
            if draw.rare_class is not None:
                labels, valid = access.read_labels(draw.scene, draw.window)
                assert np.any(valid & (labels == draw.rare_class))
        scene = bundle.split.train_scene_ids[0]
        draws = [Draw(scene, WindowSpec(8, 8, 32, 32), 4) for _ in range(8)]
        recipe = read_json(CONFIG)["augmentation"]["enhanced"]
        rng = torch.get_rng_state().clone()
        reference = list(loader(access, draws, 42, 0, recipe, workers=0))
        assert torch.equal(torch.get_rng_state(), rng)
        resumed = list(loader(access, draws, 42, 0, recipe, start=4, workers=2))
        for a, b in zip(reference[4:], resumed, strict=True):
            for key in ("image", "labels", "valid_mask"):
                assert torch.equal(a[key], b[key])


def test_new_selection_requires_strict_oa_and_preserves_class_floors():
    matrix = np.eye(6, dtype=np.int64) * 5
    matrix[0, 1] += 5
    proto = {
        "reference_validation_confusion": matrix.tolist(),
        "validation_pixels": 35,
        "inference": {"minimum_tta_oa_gain": 0.001},
    }
    base = {"confusion_matrix": matrix.tolist()}
    assert not candidate(base, proto, "A", 1)["eligible"]
    good = matrix.copy()
    good[0, 0] += 2
    good[0, 1] -= 2
    scored = {"confusion_matrix": good.tolist()}
    early, late = candidate(scored, proto, "B", 2), candidate(scored, proto, "A", 4)
    assert early["eligible"] and max((early, late), key=ranking) == early
    bad = good.copy()
    bad[5, 5] -= 1
    bad[5, 0] += 1
    assert not candidate({"confusion_matrix": bad.tolist()}, proto, "C", 2)["eligible"]
    assert choose_tta(base, scored, base, scored, proto) == "hflip"
    assert choose_tta(base, scored, base, base, proto) == "identity"


def test_refit_cache_rejects_a_heldout_scene_even_with_consistent_inventory(tmp_path):
    parent = _split()
    refit = refit_split(parent)
    good = _cache(tmp_path / "refit", refit)
    assert set(good.manifest["included_scene_ids"]) == set(
        parent.train_scene_ids + parent.validation_scene_ids
    )
    ids = (parent.test_scene_ids[0], *refit.train_scene_ids[1:])
    bad = replace(
        refit,
        train_scene_ids=ids,
        spatial_groups=tuple((s, dict(parent.spatial_groups)[s]) for s in sorted(ids)),
    )
    with pytest.raises(ValueError, match="exactly 24"):
        _cache(tmp_path / "bad-refit", bad)


def test_budget_pilot_excludes_large_and_rejects_an_unaffordable_base():
    base = {
        "update_seconds": 0.1,
        "probe_seconds": 1,
        "scene_seconds": 1,
        "tta_scene_seconds": 2,
        "setup_seconds": 1,
    }
    resolved = choose_schedule(
        {"B": base, "L": {**base, "update_seconds": 10}}, 2, include_large=True
    )
    assert resolved["arms"] == list("ABC") and resolved["passes"] == 12
    with pytest.raises(TimeoutError, match="six ViT-B"):
        choose_schedule({"B": {**base, "update_seconds": 10}}, 2, include_large=False)


def test_benchmark_requires_exact_final_refit_and_protocol():
    lock = {
        "protocol_hash": "protocol",
        "recipe": {"backbone": "B"},
        "selection_checkpoint_sha256": "selected",
    }
    manifest = {
        "status": "complete",
        "protocol_hash": "protocol",
        "updates": 1152,
        "completed_passes": 1,
        "arm": "refit",
        "recipe": lock["recipe"],
        "selection_checkpoint_sha256": "selected",
        "final_checkpoint_sha256": "final",
    }
    validate_final(manifest, lock, "final")
    for changes in (
        {"updates": 1151},
        {"protocol_hash": "other"},
        {"final_checkpoint_sha256": "wrong"},
    ):
        with pytest.raises(ValueError, match="update-1,152"):
            validate_final({**manifest, **changes}, lock, "final")
