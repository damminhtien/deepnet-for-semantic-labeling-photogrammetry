"""Joint training and post-selection evaluation over immutable data bundles."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deepnet.data.augmentations import AlbumentationsAugmenter
from deepnet.data.bundles import BundleData, load_bundle
from deepnet.data.catalog import (
    load_augmentation_spec,
    load_domain_spec,
    load_model_input_spec,
    load_preprocessing_spec,
)
from deepnet.data.contracts import canonical_hash
from deepnet.data.dataset import (
    ScenePatchDataset,
    build_class_pixel_index,
    make_torch_dataloader,
)
from deepnet.data.evaluation import evaluate_scenes
from deepnet.data.lineage import build_run_manifest
from deepnet.data.raster import sha256_file
from deepnet.data.sampling import ClassAwareSampler, MixedSampler, UniformSampler
from deepnet.models.segmenter import MultimodalSegmenter
from deepnet.models.specs import load_model_spec
from deepnet.runtime import configure_torch_runtime, runtime_metadata, seed_everything
from deepnet.training_utils import (
    artifact_index,
    capture_rng,
    learning_rate_factor,
    load_checkpoint,
    make_optimizer,
    restore_rng,
    save_checkpoint,
    verify_pretrained,
    write_json,
)
from semantic_labeling.losses import (
    inverse_sqrt_class_weights,
    weighted_ce_macro_dice_loss,
)

LOGGER = logging.getLogger(__name__)


def _load_training_bundle(root: Path, dataset: dict, protocol: dict) -> BundleData:
    bundle = load_bundle(root / dataset["bundle_path"])
    split = bundle.split
    expected = protocol["split_manifests"][dataset["dataset_id"]]
    if (
        split.dataset_id != dataset["dataset_id"]
        or split.split_id != dataset["split_id"]
        or split.version != dataset["split_version"]
        or split.content_hash != expected["content_hash"]
    ):
        raise ValueError("Training bundle differs from the locked split")
    included = set(bundle.manifest["included_scene_ids"])
    if set(bundle.manifest["included_splits"]) != {
        "train",
        "validation",
    } or included != set((*split.train_scene_ids, *split.validation_scene_ids)):
        raise ValueError("Training bundle must contain exactly train and validation")
    return bundle


def _training_statistics(root: Path, dataset: dict, bundle: BundleData) -> dict:
    path = root / dataset["quality_path"]
    report = json.loads(path.read_text())
    training = report["splits"]["train"]
    if (
        report["split_hash"] != bundle.split.content_hash
        or report["preprocessing_hash"] != bundle.preprocessing.content_hash
        or training["scene_ids"] != list(bundle.split.train_scene_ids)
    ):
        raise ValueError("Training quality statistics differ from the bundle")
    counts = [training["class_counts"].get(str(index), 0) for index in range(6)]
    if min(counts) <= 0 or sum(counts) != training["scored_pixels"]:
        raise ValueError("Training counts must cover all six semantic classes")
    return {
        "class_counts": counts,
        "quality_sha256": sha256_file(path),
        "bundle_sha256": bundle.manifest["bundle_sha256"],
        "scene_ids": list(bundle.split.train_scene_ids),
    }


def _training_dataset(
    bundle: BundleData, dataset: dict, config: dict, weights: torch.Tensor
) -> ScenePatchDataset:
    access = bundle.scene_access("train")
    window = tuple(dataset["train_window_size"])
    alpha = config["sampling"]["class_aware_fraction"]
    aware = None
    if alpha:
        LOGGER.info("Building train-only class index: %s", dataset["dataset_id"])
        candidates = build_class_pixel_index(access, range(6), seed=config["seed"])
        aware = ClassAwareSampler(
            access.scene_shapes,
            candidates,
            bundle.split,
            window,
            class_weights=dict(enumerate(weights.tolist())),
        )
    access.close()
    uniform = UniformSampler(access.scene_shapes, bundle.split.train_scene_ids, window)
    augmenter = AlbumentationsAugmenter(
        load_domain_spec(config["domain_id"]),
        bundle.model_input,
        load_augmentation_spec(config["augmentation_id"]),
        ignore_label=bundle.preprocessing.ignore_label,
    )
    execution = config["execution"]
    samples = (
        execution["optimizer_steps_per_epoch"]
        * execution["gradient_accumulation_steps"]
        * execution["micro_batch_size"]
        // len(config["datasets"])
    )
    return ScenePatchDataset(
        access,
        window,
        samples,
        seed=config["seed"],
        sampler=MixedSampler(uniform, alpha, aware),
        augmenter=augmenter,
    )


def _autocast(device: torch.device, config: dict) -> Any:
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda"
        and config["execution"]["mixed_precision"] == "float16",
    )


def _optimizer_step(
    model, batches, optimizer, scaler, weights, runtime, config
) -> float:
    for attempt in range(5):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for domain, batch in batches:
            image = batch["image"].to(
                runtime.device, dtype=torch.float32, non_blocking=runtime.non_blocking
            )
            labels = batch["labels"].to(
                runtime.device, dtype=torch.long, non_blocking=runtime.non_blocking
            )
            valid = batch["valid_mask"].to(runtime.device)
            labels = labels.masked_fill(~valid, config["loss"]["ignore_index"])
            with _autocast(runtime.device, config):
                logits = model(image)
                loss = weighted_ce_macro_dice_loss(
                    logits,
                    labels,
                    weights[domain].to(runtime.device),
                    ignore_index=config["loss"]["ignore_index"],
                    ce_weight=config["loss"]["cross_entropy_weight"],
                    dice_weight=config["loss"]["soft_dice_weight"],
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss / len(batches)).backward()
            losses.append(float(loss.detach()))
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            config["execution"]["gradient_clip_norm"],
        )
        if torch.isfinite(norm):
            scaler.step(optimizer)
            scaler.update()
            return float(np.mean(losses))
        if not scaler.is_enabled():
            raise FloatingPointError("Non-finite training gradients")
        LOGGER.warning(
            "AMP overflow; retrying update %d with a lower scale", attempt + 1
        )
        scaler.update(new_scale=scaler.get_scale() / 2)
    raise FloatingPointError("Training gradients overflowed after five AMP retries")


def _evaluate(model, bundle, role, dataset, config, runtime, output) -> dict:
    LOGGER.info("Full-scene %s evaluation: %s", role, dataset["dataset_id"])
    access = bundle.scene_access(role)
    try:
        with _autocast(runtime.device, config):
            result = evaluate_scenes(
                model,
                access,
                bundle.split.scenes_for(role),
                window_size=tuple(dataset["validation_window_size"]),
                batch_size=config["execution"]["evaluation_batch_size"],
                device=runtime.device,
                scratch_dir=output / ".scratch",
            )
    finally:
        access.close()
    LOGGER.info(
        "%s %s mIoU=%.6f OA=%.6f pixels=%d",
        dataset["dataset_id"],
        role,
        result["mean_iou"],
        result["overall_accuracy"],
        result["pixels_scored"],
    )
    return result


def _post_selection(
    model, bundles, config, protocol, root, output, runtime, binding, manifest
):
    best = output / "best.pt"
    if sha256_file(best) != manifest["best_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint checksum differs from the run manifest")
    selected = load_checkpoint(best, binding)
    selection = {
        "checkpoint_sha256": sha256_file(best),
        "binding": binding,
        "epoch": selected["progress"]["best_epoch"],
        "validation_mean_dataset_miou": selected["progress"]["best_score"],
    }
    selection_path = output / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Frozen checkpoint selection differs from the current best")
    write_json(selection_path, selection)
    model.load_state_dict(selected["model"])
    manifest.update(status="evaluating", selection=selection)
    manifest.setdefault("test_evaluations", {})
    write_json(output / "run_manifest.json", manifest)
    tests = {}
    for dataset, training in zip(config["datasets"], bundles, strict=True):
        dataset_id = dataset["dataset_id"]
        result_path = output / f"test-{dataset_id}.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result["selection"] != selection:
                raise ValueError("Saved test result belongs to another checkpoint")
            tests[dataset_id] = result
            continue
        if dataset_id in manifest["test_evaluations"]:
            raise RuntimeError(
                "An interrupted test evaluation requires explicit review"
            )
        manifest["test_evaluations"][dataset_id] = {"status": "started"}
        write_json(output / "run_manifest.json", manifest)
        evaluation = next(
            item
            for item in protocol["post_selection_evaluations"]
            if item["dataset_id"] == dataset_id
        )
        # Test files are first opened only after selection.json is durable.
        bundle = load_bundle(root / evaluation["bundle_path"])
        if (
            bundle.manifest["included_splits"] != ["test"]
            or set(bundle.manifest["included_scene_ids"])
            != set(training.split.test_scene_ids)
            or bundle.split.content_hash != training.split.content_hash
            or bundle.preprocessing.content_hash != training.preprocessing.content_hash
            or bundle.model_input.to_dict() != training.model_input.to_dict()
        ):
            raise ValueError("Test bundle differs from the frozen data contracts")
        result = _evaluate(model, bundle, "test", dataset, config, runtime, output)
        result.update(
            selection=selection, bundle_sha256=bundle.manifest["bundle_sha256"]
        )
        if "benchmark_table_metrics" in evaluation:
            ids = evaluation["benchmark_table_metrics"]["per_class_f1_ids"]
            result["benchmark_table"] = {
                "class_ids": ids,
                "per_class_f1": [result["per_class_f1"][index] for index in ids],
                "overall_accuracy": result["overall_accuracy"],
                "reference": "local eroded/no-boundary",
            }
        write_json(result_path, result)
        tests[dataset_id] = result
        manifest["test_evaluations"][dataset_id] = {"status": "complete", "runs": 1}
        write_json(output / "run_manifest.json", manifest)
    metrics = {
        "selection": selection,
        "validation": selected["progress"]["best_validation"],
        "test": tests,
    }
    write_json(output / "metrics.json", metrics)
    manifest.update(
        status="complete", metrics_sha256=sha256_file(output / "metrics.json")
    )
    write_json(output / "run_manifest.json", manifest)


def run_training(
    config_path: Path,
    protocol_path: Path,
    bundle_root: Path,
    output: Path,
    pretrained_path: Path,
    *,
    device: str | None = None,
    resume: bool = False,
    preflight: bool = False,
    source_manifest: Path | None = None,
) -> dict:
    config = json.loads(config_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if protocol["training_config"]["sha256"] != sha256_file(config_path):
        raise ValueError("Training configuration differs from the locked protocol")
    execution = config["execution"]
    domains = config["datasets"]
    if len(domains) != 2 or any(item["sampling_weight"] != 0.5 for item in domains):
        raise ValueError("This joint runner requires two equally weighted datasets")
    if execution["gradient_accumulation_steps"] % len(domains):
        raise ValueError("Accumulation must contain complete balanced domain cycles")
    if (
        output.exists()
        and any(path.name != "training.log" for path in output.iterdir())
        and not resume
    ):
        raise FileExistsError(
            "Use a new output directory or explicitly resume this run"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / ".scratch").mkdir(exist_ok=True)
    seed_everything(config["seed"])
    runtime = configure_torch_runtime(device, num_workers=execution["num_workers"])
    model_spec = load_model_spec(config["model_id"])
    pretrained = verify_pretrained(
        pretrained_path, model_spec.backbone.model_id, model_spec.backbone.revision
    )
    bundles = [_load_training_bundle(bundle_root, item, protocol) for item in domains]
    expected_input = load_model_input_spec(config["model_input_id"])
    expected_preprocessing = load_preprocessing_spec(config["preprocessing_id"])
    if any(
        bundle.preprocessing.content_hash != expected_preprocessing.content_hash
        or bundle.model_input.to_dict() != expected_input.to_dict()
        or bundle.preprocessing.ignore_label != model_spec.ignore_index
        or bundle.model_input.to_dict() != bundles[0].model_input.to_dict()
        for bundle in bundles
    ):
        raise ValueError("Training bundles differ from the requested model contracts")
    statistics = [
        _training_statistics(bundle_root, item, bundle)
        for item, bundle in zip(domains, bundles, strict=True)
    ]
    augmentation = load_augmentation_spec(config["augmentation_id"])
    provenance = {
        "configuration_sha256": sha256_file(config_path),
        "protocol_sha256": sha256_file(protocol_path),
        "model_spec_hash": model_spec.content_hash,
        "augmentation_hash": augmentation.content_hash,
        "domain_hash": canonical_hash(load_domain_spec(config["domain_id"]).to_dict()),
        "training_statistics": statistics,
        "pretrained": pretrained,
    }
    if source_manifest is not None:
        provenance["source_package"] = json.loads(source_manifest.read_text())
    binding = canonical_hash(provenance)
    manifest_path = output / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if resume and manifest_path.exists()
        else {"run_id": config["run_id"], "binding": binding}
    )
    if manifest["binding"] != binding:
        raise ValueError("Run data/model/configuration binding changed")
    if manifest.get("status") == "complete":
        return manifest
    model = MultimodalSegmenter.from_pretrained(
        load_domain_spec(config["domain_id"]),
        bundles[0].model_input,
        model_spec,
        pretrained_path=pretrained_path,
    ).to(runtime.device)
    if execution["gradient_checkpointing"]:
        model.backbone.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    weights = [
        inverse_sqrt_class_weights(
            item["class_counts"],
            min_weight=config["sampling"]["class_weight_min"],
            max_weight=config["sampling"]["class_weight_max"],
        )
        for item in statistics
    ]
    optimizer = make_optimizer(model, config["optimizer"])
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=1024,
        enabled=runtime.device_type == "cuda"
        and execution["mixed_precision"] == "float16",
    )
    manifest.update(
        provenance=provenance,
        runtime={
            **runtime_metadata(runtime),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("transformers", "peft", "albumentations", "rasterio")
            },
            "accelerator": torch.cuda.get_device_name()
            if runtime.device_type == "cuda"
            else runtime.device_type,
        },
        datasets=[
            build_run_manifest(
                bundle,
                run_id=config["run_id"],
                seed=config["seed"],
                sampler=config["sampling"],
                configuration=config,
            )
            for bundle in bundles
        ],
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    )
    write_json(output / "training-config.json", config)
    write_json(output / "evaluation-protocol.json", protocol)
    write_json(output / "model-spec.json", model_spec.to_dict())
    write_json(output / "training-statistics.json", statistics)
    progress = {
        "epoch": 0,
        "update": 0,
        "global_step": 0,
        "best_score": -1.0,
        "best_epoch": None,
        "bad_epochs": 0,
        "history": [],
        "best_validation": {},
        "epoch_histograms": [],
        "histogram_state": [],
    }
    if resume and (output / "last.pt").exists():
        checkpoint = load_checkpoint(output / "last.pt", binding)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        progress = checkpoint["progress"]
        restore_rng(checkpoint["rng"])
    manifest.update(status="preflight" if preflight else "training")
    write_json(manifest_path, manifest)
    datasets = [
        _training_dataset(bundle, item, config, weight)
        for bundle, item, weight in zip(bundles, domains, weights, strict=True)
    ]
    with ExitStack() as stack:
        for dataset in datasets:
            stack.callback(dataset.access.close)
        if preflight:
            from deepnet.data.dataset import collate_torch_samples

            model.train()
            batches = [
                (index, collate_torch_samples([dataset[0]]))
                for index, dataset in enumerate(datasets)
            ]
            loss = _optimizer_step(
                model, batches, optimizer, scaler, weights, runtime, config
            )
            manifest.update(status="preflight_passed", preflight_loss=loss)
            write_json(manifest_path, manifest)
            return manifest
        start = time.monotonic()
        deadline = start + execution["maximum_run_hours"] * 3600
        total_steps = execution["max_epochs"] * execution["optimizer_steps_per_epoch"]
        for epoch in range(progress["epoch"], execution["max_epochs"]):
            if progress["bad_epochs"] >= execution["early_stopping_patience"]:
                break
            snapshots = capture_rng()
            iterators = []
            histograms = []
            for dataset in datasets:
                dataset.set_epoch(epoch)
                loader = make_torch_dataloader(
                    dataset,
                    batch_size=execution["micro_batch_size"],
                    **runtime.dataloader_options(),
                )
                iterator = iter(loader)
                for _ in range(
                    progress["update"]
                    * execution["gradient_accumulation_steps"]
                    // len(datasets)
                ):
                    next(iterator)
                iterators.append(iterator)
                histogram = dataset.new_label_histogram()
                if progress["update"] and progress["histogram_state"]:
                    histogram.__dict__.update(
                        progress["histogram_state"][len(histograms)]
                    )
                histograms.append(histogram)
            restore_rng(snapshots)
            model.train()
            for update in range(
                progress["update"], execution["optimizer_steps_per_epoch"]
            ):
                if time.monotonic() >= deadline:
                    progress["histogram_state"] = [
                        histogram.__dict__ for histogram in histograms
                    ]
                    manifest.update(status="paused_time_budget", progress=progress)
                    save_checkpoint(
                        output / "last.pt",
                        model,
                        progress,
                        binding,
                        optimizer=optimizer,
                        scaler=scaler,
                    )
                    write_json(manifest_path, manifest)
                    return manifest
                factor = learning_rate_factor(
                    progress["global_step"],
                    total_steps,
                    config["scheduler"]["warmup_fraction"],
                )
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * factor
                batches = []
                for micro in range(execution["gradient_accumulation_steps"]):
                    domain = micro % len(datasets)
                    batch = next(iterators[domain])
                    histograms[domain].update(
                        batch["labels"].numpy(), batch["valid_mask"].numpy()
                    )
                    batches.append((domain, batch))
                loss = _optimizer_step(
                    model, batches, optimizer, scaler, weights, runtime, config
                )
                progress.update(
                    update=update + 1, global_step=progress["global_step"] + 1
                )
                if (update + 1) % 8 == 0 or update == 0:
                    LOGGER.info(
                        "epoch=%d update=%d/%d loss=%.6f",
                        epoch + 1,
                        update + 1,
                        execution["optimizer_steps_per_epoch"],
                        loss,
                    )
                if (
                    progress["global_step"]
                    % execution["checkpoint_interval_optimizer_steps"]
                    == 0
                ):
                    progress["histogram_state"] = [
                        histogram.__dict__ for histogram in histograms
                    ]
                    manifest["last_checkpoint_sha256"] = save_checkpoint(
                        output / "last.pt",
                        model,
                        progress,
                        binding,
                        optimizer=optimizer,
                        scaler=scaler,
                    )
                    manifest["progress"] = progress
                    write_json(manifest_path, manifest)
            del iterators
            validation = {
                item["dataset_id"]: _evaluate(
                    model, bundle, "validation", item, config, runtime, output
                )
                for item, bundle in zip(domains, bundles, strict=True)
            }
            score = float(np.mean([value["mean_iou"] for value in validation.values()]))
            progress["history"].append(
                {"epoch": epoch + 1, "mean_dataset_miou": score, "datasets": validation}
            )
            progress["epoch_histograms"].append(
                {
                    "epoch": epoch + 1,
                    "datasets": {
                        item["dataset_id"]: histogram.summary(epoch=epoch + 1)
                        for item, histogram in zip(domains, histograms, strict=True)
                    },
                }
            )
            progress.update(epoch=epoch + 1, update=0, histogram_state=[])
            if score > progress["best_score"]:
                progress.update(
                    best_score=score,
                    best_epoch=epoch + 1,
                    bad_epochs=0,
                    best_validation=validation,
                )
                manifest["best_checkpoint_sha256"] = save_checkpoint(
                    output / "best.pt", model, progress, binding
                )
            else:
                progress["bad_epochs"] += 1
            manifest["last_checkpoint_sha256"] = save_checkpoint(
                output / "last.pt",
                model,
                progress,
                binding,
                optimizer=optimizer,
                scaler=scaler,
            )
            manifest["progress"] = progress
            write_json(manifest_path, manifest)
            LOGGER.info(
                "epoch=%d validation_mean_miou=%.6f best_epoch=%s",
                epoch + 1,
                score,
                progress["best_epoch"],
            )
        _post_selection(
            model,
            bundles,
            config,
            protocol,
            bundle_root,
            output,
            runtime,
            binding,
            manifest,
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error("Output directory already has files; choose a new run or --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(args.output / "training.log"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )
    try:
        manifest = run_training(
            args.config,
            args.protocol,
            args.bundle_root,
            args.output,
            args.pretrained,
            device=args.device,
            resume=args.resume,
            preflight=args.preflight,
            source_manifest=args.source_manifest,
        )
        LOGGER.info("Run status: %s", manifest["status"])
    except Exception:
        LOGGER.exception("Training flow failed")
        path = args.output / "run_manifest.json"
        manifest = json.loads(path.read_text()) if path.exists() else {}
        manifest["status"] = "failed"
        write_json(path, manifest)
        raise
    finally:
        for handler in handlers:
            handler.flush()
        artifact_index(args.output)


if __name__ == "__main__":
    main()
