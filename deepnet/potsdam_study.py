"""Command-line Potsdam experiment with separated selection, refit and test jobs."""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch

from deepnet.data.bundles import load_bundle
from deepnet.data.catalog import load_scene_catalog
from deepnet.data.contracts import SplitManifest, canonical_hash
from deepnet.data.raster import sha256_file
from deepnet.data.sampling import WindowSpec
from deepnet.potsdam_data import (
    STRIDE,
    WINDOW,
    DsmAccess,
    augment_patch,
    export_refit,
    fit_panel,
    fixed_panel,
    index_training,
    patch_schedule,
    prepare_dsm,
    refit_split,
    verify_dsm,
    verify_scope,
)
from deepnet.potsdam_protocol import (
    CONFIG,
    Budget,
    candidate,
    freeze,
    load_protocol,
    lock_winner,
    matrix_metrics,
    ranking,
    read_json,
    winner,
)
from deepnet.potsdam_run import (
    DINO3_FILES,
    DINO3_REVISION,
    budget_estimate,
    describe_panel,
    input_distribution,
    load_model,
    model_spec,
    optimizer_and_scaler,
    require_finite_distribution,
    sampled_batch,
    score_patches,
    score_scenes,
    timed,
    update,
    verified_weights,
)
from deepnet.runtime import configure_torch_runtime, seed_everything
from deepnet.training_utils import (
    artifact_index,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
    write_json,
)
from semantic_labeling.losses import inverse_sqrt_class_weights

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)
BASE = {"spectral": False, "dsm": False, "strong": False}


def _source_hash():
    files = (
        *sorted((ROOT / "deepnet").rglob("*.py")),
        ROOT / "semantic_labeling/losses.py",
        ROOT / "configs/models/dinov2-vitb14-multimodal-v3.json",
        ROOT / "configs/training" / CONFIG,
    )
    return canonical_hash(
        {str(path.relative_to(ROOT)): sha256_file(path) for path in files}
    )


def _split():
    return SplitManifest.from_dict(
        read_json(ROOT / "configs/splits/potsdam-dinov3-v2.json")
    )


def _runtime(args):
    runtime = configure_torch_runtime(args.device, num_workers=0)
    if runtime.device_type != "cuda":
        raise ValueError(
            "The study uses a measured CUDA budget; local smoke is a separate command"
        )
    return runtime


def _device_binding(runtime):
    return {
        "accelerator": torch.cuda.get_device_name(runtime.device),
        "allocated_devices": torch.cuda.device_count(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "rasterio", "numpy")
        },
    }


def _selection_assets(bundle):
    return {
        scene.scene_id: {
            name: asset.sha256
            for name, asset in scene.assets
            if name in {"irrg", "label_participant"}
        }
        for scene in bundle.scenes
    }


def _check_refit_assets(bundle, protocol):
    expected = protocol["selection_asset_hashes"]
    actual = _selection_assets(bundle)
    for scene, assets in expected.items():
        if any(actual[scene][name] != digest for name, digest in assets.items()):
            raise ValueError(f"Refit changed a previously locked source: {scene}")


def _attach_dsm(access, root, manifest):
    return DsmAccess(access, root, manifest) if manifest else access


def _fit_small(model, train, indexed, runtime, budget, weights):
    panel = fit_panel(indexed)
    samples = [train.read_sample(scene, window) for scene, window in panel]
    support = np.bincount(
        np.concatenate(
            [
                sample.labels[sample.valid_mask & (sample.labels < 6)]
                for sample in samples
            ]
        ),
        minlength=6,
    )
    if min(support) <= 0:
        raise ValueError("The eight-patch fit panel does not cover all classes")
    optimizer, scaler = optimizer_and_scaler(model, runtime)
    history = []
    for step in range(512):
        budget.check()
        model.train()
        start = (step % 2) * 4
        loss = update(
            model,
            samples[start : start + 4],
            optimizer,
            scaler,
            weights,
            runtime,
            step,
            512,
        )
        if (step + 1) % 32 == 0:
            metrics = score_patches(model, train, panel, runtime, budget)
            history.append({"update": step + 1, "loss": loss, **metrics})
            LOGGER.info("small fit update=%d mIoU=%.6f", step + 1, metrics["mean_iou"])
            if metrics["mean_iou"] >= 0.90:
                break
    return {
        "panel": describe_panel(panel),
        "history": history,
        "passed": history[-1]["mean_iou"] >= 0.90,
    }


def _legacy_stats(args, bundle, train, panel, runtime, config):
    if sha256_file(args.old_checkpoint) != config["old_checkpoint_sha256"]:
        raise ValueError("Historical DINOv2 checkpoint differs from audited SHA-256")
    old = load_model(
        bundle, args.pretrained, runtime, BASE, model_id="dinov2-vitb14-multimodal-v2"
    )
    checkpoint = torch.load(args.old_checkpoint, map_location="cpu", weights_only=False)
    old.load_state_dict(checkpoint["model"])
    return input_distribution(old, train, panel, runtime)


def _audit_augmentation(access, panel, budget):
    counts = {
        "patches": 0,
        "car_before": 0,
        "car_after": 0,
        "clutter_before": 0,
        "clutter_after": 0,
    }
    for idx, (scene, window) in enumerate(panel):
        budget.check()
        sample = access.read_sample(scene, window)
        for rare in (False, True):
            transformed = augment_patch(sample, idx, strong=True, rare=rare)
            if (transformed.image[:, ~transformed.valid_mask] != 0).any() or not set(
                np.unique(transformed.labels)
            ) <= set(range(7)):
                raise ValueError("Augmentation changed padding or the label ontology")
            counts["patches"] += 1
            for name, class_id in (("car", 4), ("clutter", 5)):
                before = int((sample.labels == class_id).sum())
                after = int((transformed.labels == class_id).sum())
                counts[name + "_before"] += before
                counts[name + "_after"] += after
                if rare and before != after:
                    raise ValueError("Rare sampling augmentation changed label support")
    return counts


def _pilot_runtime(
    args, budget, runtime, bundle, train, validation, indexed, weights, dsm
):
    recipe = {"spectral": True, "dsm": bool(dsm), "strong": True}
    model = load_model(bundle, args.pretrained, runtime, recipe)
    optimizer, scaler = optimizer_and_scaler(model, runtime)
    schedule = patch_schedule(indexed, 42, 0)
    rng = random.Random(42)
    elapsed = []
    for step in range(9):
        budget.check()

        def work(step=step):
            model.train()
            samples = sampled_batch(train, schedule, step, rng, strong=True)
            return update(
                model, samples, optimizer, scaler, weights, runtime, step, 864
            )

        loss, seconds = timed(work, runtime)
        if step == 0 and dsm:
            side = model.input_adapter.auxiliary_spatial
            if not any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in side.parameters()
            ):
                raise ValueError("DSM spatial branch did not receive a gradient")
        LOGGER.info("pilot update=%d loss=%.5f seconds=%.2f", step + 1, loss, seconds)
        elapsed.append(seconds)

    panel = fixed_panel(validation, 12, 252)
    _, warmup = timed(
        lambda: score_patches(model, validation, panel[:4], runtime, budget), runtime
    )
    _, probe = timed(
        lambda: score_patches(model, validation, panel, runtime, budget), runtime
    )
    # Profile the middle scene without using its score for selection.
    validation_ids = validation.split.scenes_for("validation")
    scene_id = validation_ids[len(validation_ids) // 2]
    options = dict(scene_ids=[scene_id], profile=True)
    overlap, overlap_seconds = timed(
        lambda: score_scenes(
            model, validation, runtime, budget, args.root / ".scratch", **options
        ),
        runtime,
    )
    plain, plain_seconds = timed(
        lambda: score_scenes(
            model,
            validation,
            runtime,
            budget,
            args.root / ".scratch",
            no_overlap=True,
            **options,
        ),
        runtime,
    )
    return {
        "warmup_seconds": warmup,
        "probe_seconds": probe,
        "update_seconds": elapsed,
        "overlap_seconds": overlap_seconds,
        "no_overlap_seconds": plain_seconds,
        "overlap_profile": overlap["scenes"][0]["profile_seconds"],
        "no_overlap_profile": plain["scenes"][0]["profile_seconds"],
        "overlap_windows": overlap["scenes"][0]["window_count"],
        "no_overlap_windows": plain["scenes"][0]["window_count"],
        "profile_scene": scene_id,
        "probe_panel": describe_panel(panel),
    }


def preflight(args):
    config = read_json(ROOT / "configs/training" / CONFIG)
    split = _split()
    if split.content_hash != config["selection_split_hash"]:
        raise ValueError("Unexpected selection split")
    runtime = _runtime(args)
    device = _device_binding(runtime)
    pilot = args.root / "preflight"
    if (args.root / "protocol.json").exists():
        raise FileExistsError("Pilot protocol is already frozen")
    with Budget(
        pilot, config["gpu_hours"]["preflight"], device["allocated_devices"]
    ) as budget:
        smoke = read_json(args.dinov3_smoke)
        if (
            smoke.get("revision") != DINO3_REVISION
            or smoke.get("lora_b_gradients") != 48
        ):
            raise ValueError("Preflight requires the verified real DINOv3 smoke result")
        bundle = load_bundle(args.bundle)
        verify_scope(bundle, split, ("train", "validation"))
        spec = model_spec(BASE)
        weights_manifest = verified_weights(args.pretrained, spec)
        seed_everything(42)
        dsm = (
            verify_dsm(args.dsm, split, bundle.manifest["included_scene_ids"])
            if args.dsm
            else None
        )
        with (
            bundle.scene_access("train") as train,
            bundle.scene_access("validation") as val,
        ):
            indexed, counts = index_training(train)
            if len(indexed) != 2592:
                raise ValueError("Expected exactly 2592 non-overlapping train tiles")
            panel = fixed_panel(train, 2, 151)
            weights = inverse_sqrt_class_weights(counts)
            model = load_model(bundle, args.pretrained, runtime, BASE)
            init = input_distribution(model, train, panel, runtime)
            require_finite_distribution(init)
            old = _legacy_stats(args, bundle, train, panel, runtime, config)
            small_fit = _fit_small(model, train, indexed, runtime, budget, weights)
            final = input_distribution(model, train, panel, runtime)
            require_finite_distribution(final)
            report = {
                "backbone_input_panel": describe_panel(panel),
                "input_distributions": {
                    "initial": init,
                    "small_fit": final,
                    "historical": old,
                },
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "small_fit": small_fit,
                "train_class_support": counts.tolist(),
            }
            write_json(pilot / "report.json", report)
            if not small_fit["passed"]:
                raise RuntimeError("Small-fit gate did not reach 90% training mIoU")
            del model
            train_view = _attach_dsm(train, args.dsm, dsm)
            val_view = _attach_dsm(val, args.dsm, dsm)
            if dsm:
                scene, window = panel[0]
                base, enriched = (
                    train.read_sample(scene, window),
                    train_view.read_sample(scene, window),
                )
                if not np.array_equal(base.valid_mask, enriched.valid_mask):
                    raise ValueError("DSM changed the optical and label scoring mask")
            report["augmentation_audit"] = _audit_augmentation(
                train_view, panel + fit_panel(indexed), budget
            )
            report["timing"] = _pilot_runtime(
                args,
                budget,
                runtime,
                bundle,
                train_view,
                val_view,
                indexed,
                weights,
                dsm,
            )
            write_json(pilot / "report.json", report)
        timing = report["timing"]
        hours = config["gpu_hours"]
        count = device["allocated_devices"]
        passes = budget_estimate(
            float(np.median(timing["update_seconds"][1:])),
            timing["overlap_seconds"],
            timing["probe_seconds"],
            hours["per_selection_run"] / count,
            hours["refit"] / count,
            hours["heldout"] / count,
        )
        protocol = {
            **config,
            "source_hash": _source_hash(),
            "passes": passes,
            "full_validation_passes": list(range(2, passes + 1, 2)),
            "window_size": list(WINDOW),
            "stride": list(STRIDE),
            "runtime": device,
            "preflight_report_sha256": sha256_file(pilot / "report.json"),
            "pretrained_hash": canonical_hash(weights_manifest),
            "selection_bundle_sha256": bundle.manifest["bundle_sha256"],
            "selection_asset_hashes": _selection_assets(bundle),
            "model_input_hash": canonical_hash(bundle.model_input.to_dict()),
            "preprocessing_hash": bundle.preprocessing.content_hash,
            "dsm_manifest_sha256": sha256_file(args.dsm / "manifest.json")
            if dsm
            else None,
            "dinov3_smoke_sha256": sha256_file(args.dinov3_smoke),
            "train_class_support": report["train_class_support"],
            "input_panel": report["backbone_input_panel"],
            "probe_panel": timing["probe_panel"],
            "probe_panel_sha256": canonical_hash(timing["probe_panel"]),
        }
        protocol["protocol_hash"] = canonical_hash(protocol)
        budget.check()
        freeze(args.root / "protocol.json", protocol)
    artifact_index(pilot)


def _bound_phase(args, runtime, bundle, protocol, recipe):
    if protocol["source_hash"] != _source_hash():
        raise ValueError("Study source changed since the pilot")
    if protocol["runtime"] != _device_binding(runtime):
        raise ValueError("GPU or package versions differ from the budgeted pilot")
    if (
        bundle.preprocessing.content_hash != protocol["preprocessing_hash"]
        or canonical_hash(bundle.model_input.to_dict()) != protocol["model_input_hash"]
    ):
        raise ValueError("Bundle model input or preprocessing differs")
    weights = verified_weights(args.pretrained, model_spec(recipe))
    if canonical_hash(weights) != protocol["pretrained_hash"]:
        raise ValueError("Pretrained snapshot differs from the pilot")
    if not recipe["dsm"]:
        return None
    if (
        args.dsm is None
        or sha256_file(args.dsm / "manifest.json") != protocol["dsm_manifest_sha256"]
    ):
        raise ValueError("DSM feature store differs from the pilot")
    return verify_dsm(args.dsm, _split(), bundle.manifest["included_scene_ids"])


def _recipe(root, protocol, arm):
    if arm == "A":
        return BASE.copy()
    if arm == "B":
        return {
            **BASE,
            "spectral": True,
            "dsm": protocol["dsm_manifest_sha256"] is not None,
        }
    if arm == "C":
        return {**winner(root, protocol, "AB")["recipe"], "strong": True}
    if arm == "D":
        chosen = winner(root, protocol)
        if not chosen["best"]["eligible"]:
            raise ValueError("No selection run meets the SAM validation floors")
        return chosen["recipe"]
    raise ValueError(f"Unknown selection arm: {arm}")


def _panel(records):
    return [
        (scene, WindowSpec(row, col, h, w, padding="constant"))
        for scene, row, col, h, w in records
    ]


def _sample_counts(samples):
    return sum(int((s.valid_mask & (s.labels != 6)).sum()) for s in samples)


def _save_last(out, model, state, binding, optimizer, scaler):
    return save_checkpoint(
        out / "last.pt", model, state, binding, optimizer=optimizer, scaler=scaler
    )


def _train(
    args, protocol, runtime, bundle, recipe, budget, dsm, *, selection_lock=None
):
    refitting = selection_lock is not None
    arm = "refit" if refitting else args.arm
    out = args.root / arm
    steps = 1152 if refitting else 864
    passes = 1 if refitting else protocol["passes"]
    total = steps * passes
    seed = 43 if arm == "D" else 42
    seed_everything(seed)
    binding = canonical_hash(
        {
            "protocol_hash": protocol["protocol_hash"],
            "bundle": bundle.manifest["bundle_sha256"],
            "arm": arm,
            "recipe": recipe,
            "seed": seed,
            "selection": selection_lock,
        }
    )
    model = load_model(bundle, args.pretrained, runtime, recipe)
    optimizer, scaler = optimizer_and_scaler(model, runtime, 0.5 if refitting else 1)
    state = {
        "updates": 0,
        "completed_passes": 0,
        "history": [],
        "best": None,
        "best_validation": None,
        "valid_pixel_presentations": 0,
        "validation_seconds": 0.0,
        "training_seconds": 0.0,
    }
    last = out / "last.pt"
    if last.exists():
        saved = load_checkpoint(last, binding)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        state = saved["progress"]
    elif refitting:
        checkpoint = args.root / selection_lock["arm"] / "best.pt"
        if sha256_file(checkpoint) != selection_lock["selection_checkpoint_sha256"]:
            raise ValueError("Refit initial checkpoint differs from frozen selection")
        model.load_state_dict(
            load_checkpoint(checkpoint, selection_lock["selection_binding"])["model"]
        )

    with bundle.scene_access("train") as train:
        indexed, counts = index_training(train)
        if len(indexed) != (3456 if refitting else 2592):
            raise ValueError("Unexpected count of base training tiles")
        if state["updates"] == 0 and not refitting:
            initial = input_distribution(
                model, train, _panel(protocol["input_panel"]), runtime
            )
            require_finite_distribution(initial)
            write_json(out / "initial-input-distribution.json", initial)
        if not refitting and counts.tolist() != protocol["train_class_support"]:
            raise ValueError("Selection training support differs from the pilot")
        weights = inverse_sqrt_class_weights(protocol["train_class_support"])
        train_view = _attach_dsm(train, args.dsm, dsm)
        for pass_idx in range(state["completed_passes"], passes):
            schedule = patch_schedule(indexed, seed, pass_idx)
            rng = random.Random(f"augmentation/{seed}/{pass_idx}")
            offset = state["updates"] - pass_idx * steps
            for _ in range(offset * 4):
                rng.getrandbits(32)
            model.train()
            for pos in range(offset, steps):
                budget.check()
                started = time.monotonic()
                samples = sampled_batch(
                    train_view, schedule, pos, rng, strong=recipe["strong"]
                )
                loss = update(
                    model,
                    samples,
                    optimizer,
                    scaler,
                    weights,
                    runtime,
                    state["updates"],
                    total,
                )
                state["updates"] += 1
                state["valid_pixel_presentations"] += _sample_counts(samples)
                state["training_seconds"] += time.monotonic() - started
                if state["updates"] % 64 == 0:
                    LOGGER.info(
                        "%s update=%d/%d loss=%.5f", arm, state["updates"], total, loss
                    )
                    _save_last(out, model, state, binding, optimizer, scaler)
            # Persist completed training before potentially expensive validation.
            _save_last(out, model, state, binding, optimizer, scaler)
            record = {"pass": pass_idx + 1, "updates": state["updates"]}
            if not refitting:
                started = time.monotonic()
                with bundle.scene_access("validation") as validation:
                    access = _attach_dsm(validation, args.dsm, dsm)
                    record["probe"] = score_patches(
                        model, access, _panel(protocol["probe_panel"]), runtime, budget
                    )
                    if pass_idx + 1 in protocol["full_validation_passes"]:
                        metrics = score_scenes(
                            model, access, runtime, budget, out / ".scratch"
                        )
                        record["full_validation"] = metrics
                        measured = candidate(metrics, protocol, arm, state["updates"])
                        if state["best"] is None or ranking(measured) > ranking(
                            state["best"]
                        ):
                            digest = save_checkpoint(
                                out / "best.pt", model, state, binding
                            )
                            state["best"] = {**measured, "checkpoint_sha256": digest}
                            state["best_validation"] = metrics
                        distribution = input_distribution(
                            model, train, _panel(protocol["input_panel"]), runtime
                        )
                        require_finite_distribution(distribution)
                        record["backbone_inputs"] = distribution
                state["validation_seconds"] += time.monotonic() - started
            state["history"].append(record)
            state["completed_passes"] += 1
            _save_last(out, model, state, binding, optimizer, scaler)

        manifest = {
            **state,
            "arm": arm,
            "recipe": recipe,
            "seed": seed,
            "status": "complete",
            "binding": binding,
            "protocol_hash": protocol["protocol_hash"],
            "bundle_sha256": bundle.manifest["bundle_sha256"],
            "train_scene_ids": list(bundle.split.train_scene_ids),
            "train_class_support": counts.tolist(),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }
        if refitting:
            distribution = input_distribution(
                model, train, _panel(protocol["input_panel"]), runtime
            )
            require_finite_distribution(distribution)
            manifest["backbone_inputs"] = distribution
            manifest["selection_checkpoint_sha256"] = selection_lock[
                "selection_checkpoint_sha256"
            ]
            budget.check()
            manifest["final_checkpoint_sha256"] = save_checkpoint(
                out / "final.pt", model, state, binding
            )
        elapsed = state["training_seconds"] + state["validation_seconds"]
        manifest["validation_fraction"] = state["validation_seconds"] / max(
            elapsed, 1e-9
        )
        budget.check()
        write_json(out / "manifest.json", manifest)
    return manifest


def run_arm(args):
    protocol = load_protocol(args.root)
    runtime = _runtime(args)
    if (args.root / args.arm / "manifest.json").exists():
        raise FileExistsError(f"Selection arm already completed: {args.arm}")
    hours = protocol["gpu_hours"]["per_selection_run"]
    with Budget(args.root / args.arm, hours, torch.cuda.device_count()) as budget:
        bundle = load_bundle(args.bundle)
        verify_scope(bundle, _split(), ("train", "validation"))
        if bundle.manifest["bundle_sha256"] != protocol["selection_bundle_sha256"]:
            raise ValueError("Selection bundle differs from the pilot")
        recipe = _recipe(args.root, protocol, args.arm)
        dsm = _bound_phase(args, runtime, bundle, protocol, recipe)
        manifest = _train(args, protocol, runtime, bundle, recipe, budget, dsm)
    artifact_index(args.root / args.arm)
    return manifest


def run_refit(args):
    protocol = load_protocol(args.root)
    lock = lock_winner(args.root, protocol)
    runtime = _runtime(args)
    if (args.root / "refit" / "manifest.json").exists():
        raise FileExistsError("Refit already completed")
    with Budget(
        args.root / "refit", protocol["gpu_hours"]["refit"], torch.cuda.device_count()
    ) as budget:
        bundle = load_bundle(args.bundle)
        verify_scope(bundle, refit_split(_split()), ("train",))
        _check_refit_assets(bundle, protocol)
        dsm = _bound_phase(args, runtime, bundle, protocol, lock["recipe"])
        manifest = _train(
            args,
            protocol,
            runtime,
            bundle,
            lock["recipe"],
            budget,
            dsm,
            selection_lock=lock,
        )
    artifact_index(args.root / "refit")
    return manifest


def run_heldout(args):
    protocol = load_protocol(args.root)
    lock = lock_winner(args.root, protocol)
    out = args.root / "heldout"
    marker = out / "attempt.json"
    runtime = _runtime(args)
    refit = read_json(args.root / "refit" / "manifest.json")
    checkpoint = args.root / "refit" / "final.pt"
    if (
        refit["status"] != "complete"
        or refit["updates"] != 1152
        or refit["protocol_hash"] != protocol["protocol_hash"]
        or refit["selection_checkpoint_sha256"] != lock["selection_checkpoint_sha256"]
        or refit["recipe"] != lock["recipe"]
        or sha256_file(checkpoint) != refit["final_checkpoint_sha256"]
    ):
        raise ValueError(
            "Held-out evaluation requires the locked final refit checkpoint"
        )
    out.mkdir(parents=True, exist_ok=True)
    # Atomically consume the single attempt before opening the held-out bundle.
    with marker.open("x") as stream:
        stream.write(canonical_hash(refit) + "\n")
    with Budget(
        out, protocol["gpu_hours"]["heldout"], torch.cuda.device_count()
    ) as budget:
        bundle = load_bundle(args.bundle)
        verify_scope(bundle, _split(), ("test",))
        dsm = _bound_phase(args, runtime, bundle, protocol, lock["recipe"])
        model = load_model(bundle, args.pretrained, runtime, lock["recipe"])
        state = load_checkpoint(checkpoint, refit["binding"])
        if (
            state["progress"]["updates"] != 1152
            or state["progress"]["completed_passes"] != 1
        ):
            raise ValueError("Final refit checkpoint is incomplete")
        model.load_state_dict(state["model"])
        with bundle.scene_access("test") as access:
            metrics = score_scenes(
                model,
                _attach_dsm(access, args.dsm, dsm),
                runtime,
                budget,
                out / ".scratch",
            )
        aggregate = np.sum(
            [np.asarray(scene["confusion_matrix"]) for scene in metrics["scenes"]],
            axis=0,
        )
        measured = matrix_metrics(aggregate, protocol["heldout_pixels"])
        if measured["confusion_matrix"] != metrics["confusion_matrix"]:
            raise ValueError("Held-out confusion matrix differs from sum of scenes")
        report = {
            **metrics,
            **measured,
            "checkpoint_sha256": refit["final_checkpoint_sha256"],
            "protocol_hash": protocol["protocol_hash"],
            "heldout_evaluation_count": 1,
        }
        write_json(out / "metrics.json", report)
    artifact_index(out)
    return report


def smoke_dinov3(args):
    for name, digest in DINO3_FILES.items():
        if sha256_file(args.pretrained / name) != digest:
            raise ValueError(f"DINOv3 snapshot differs from locked revision: {name}")
    bundle = load_bundle(args.bundle)
    verify_scope(bundle, _split(), ("train", "validation"))
    runtime = configure_torch_runtime(args.device, num_workers=0)
    seed_everything(42)
    spec = model_spec(BASE, "dinov3-vitb16-multimodal-v1")
    from dataclasses import replace

    spec = replace(spec, backbone=replace(spec.backbone, revision=DINO3_REVISION))
    model = load_model(bundle, args.pretrained, runtime, BASE, spec=spec)
    with bundle.scene_access("train") as access:
        sample = access.read_sample("2_10", WindowSpec(1024, 1024, 224, 224))
    model.train()
    image = torch.from_numpy(sample.image[None]).to(runtime.device)
    labels = torch.from_numpy(sample.labels[None]).to(runtime.device)
    logits = model(image)
    if logits.shape != (1, 6, 224, 224) or not logits.isfinite().all():
        raise ValueError("Unexpected logits from real DINOv3 snapshot")
    loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=6)
    loss.backward()
    parameters = list(model.backbone.named_parameters())
    lora = [p for name, p in parameters if ".lora_B." in name]
    targets = [
        name.rsplit(".", 1)[-1]
        for name, module in model.backbone.named_modules()
        if hasattr(module, "lora_A") and hasattr(module, "lora_B")
    ]
    if (
        sorted(targets) != sorted(("q_proj", "k_proj", "v_proj", "o_proj") * 12)
        or len(lora) != 48
        or any(
            p.grad is None or not p.grad.isfinite().all() or p.grad.abs().sum() == 0
            for p in lora
        )
        or any(
            p.requires_grad or p.grad is not None
            for n, p in parameters
            if ".lora_" not in n
        )
    ):
        raise ValueError("Real DINOv3 LoRA does not match the locked backend contract")
    report = {
        "revision": DINO3_REVISION,
        "sha256": DINO3_FILES,
        "lora_b_gradients": len(lora),
        "logit_shape": list(logits.shape),
        "loss": float(loss.detach()),
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
    }
    freeze(args.root / "dinov3-smoke.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, required=True)
    common.add_argument("--bundle", type=Path, required=True)
    common.add_argument("--pretrained", type=Path, required=True)
    common.add_argument("--device", default="cuda")
    common.add_argument("--dsm", type=Path)
    pilot = commands.add_parser("preflight", parents=[common])
    pilot.add_argument("--old-checkpoint", type=Path, required=True)
    pilot.add_argument("--dinov3-smoke", type=Path, required=True)
    arm = commands.add_parser("select", parents=[common])
    arm.add_argument("--arm", choices="ABCD", required=True)
    commands.add_parser("refit", parents=[common])
    commands.add_parser("heldout", parents=[common])
    commands.add_parser("smoke-dinov3", parents=[common])
    pack = commands.add_parser("pack-refit")
    pack.add_argument("--root", type=Path, required=True)
    pack.add_argument("--bundle", type=Path, required=True)
    pack.add_argument("--raw", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    dsm = commands.add_parser("prepare-dsm")
    dsm.add_argument("--catalog", type=Path, required=True)
    dsm.add_argument("--raw", type=Path, required=True)
    dsm.add_argument("--inventory", type=Path, required=True)
    dsm.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.command == "pack-refit":
        protocol = load_protocol(args.root)
        lock_winner(args.root, protocol)
        bundle = load_bundle(args.bundle)
        verify_scope(bundle, _split(), ("train", "validation"))
        if bundle.manifest["bundle_sha256"] != protocol["selection_bundle_sha256"]:
            raise ValueError("Cannot derive refit from a different selection bundle")
        export_refit(bundle, args.raw, args.output)
    elif args.command == "prepare-dsm":
        prepare_dsm(
            _split(),
            load_scene_catalog(args.catalog),
            args.raw,
            args.inventory,
            args.output,
        )
    else:
        {
            "preflight": preflight,
            "select": run_arm,
            "refit": run_refit,
            "heldout": run_heldout,
            "smoke-dinov3": smoke_dinov3,
        }[args.command](args)


if __name__ == "__main__":
    main()
