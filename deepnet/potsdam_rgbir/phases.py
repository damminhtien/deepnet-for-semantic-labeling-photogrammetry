"""Resumable selection, two-seed locking, fixed continuation, and final scoring."""

from __future__ import annotations

import argparse
import gc
import logging
import time
from typing import cast

import numpy as np
import torch
from torch import nn

from deepnet.data.contracts import canonical_hash
from deepnet.data.evaluation import summarize_confusion
from deepnet.data.raster import sha256_file
from deepnet.runtime import TorchRuntime, configure_torch_runtime, seed_everything
from deepnet.training_utils import (
    artifact_index,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
    write_json,
)
from semantic_labeling.losses import inverse_sqrt_class_weights

from .data import CacheBundle
from .engine import (
    distribution,
    optimizer_and_scaler,
    restore_panel,
    score_patches,
    score_scenes,
    train_step,
)
from .model import load_model
from .protocol import (
    Budget,
    candidate,
    choose_tta,
    complete_arm,
    device_binding,
    freeze,
    load_protocol,
    lock_selection,
    matrix_metrics,
    ranking,
    read_json,
    recipe_for,
    winner,
)
from .sampling import loader, schedule, training_index
from .types import (
    BestCandidate,
    CheckpointBinding,
    EvaluationMetrics,
    InferencePolicy,
    ModelBinding,
    PassRecord,
    PhaseManifest,
    PhaseState,
    Protocol,
    Recipe,
    SelectionLock,
    TrainingIndex,
    TTAValidation,
)

LOGGER = logging.getLogger(__name__)


def _bound_inputs(
    args: argparse.Namespace, protocol: Protocol, phase: str
) -> tuple[TorchRuntime, CacheBundle]:
    runtime = configure_torch_runtime(
        args.device, num_workers=protocol["execution"]["num_workers"]
    )
    if device_binding(runtime) != protocol["runtime"]:
        raise ValueError("Runtime differs from the budgeted preflight")
    cache = CacheBundle.load(args.cache)
    expected_roles = (
        ["train"]
        if phase == "refit"
        else ["test"]
        if phase == "heldout"
        else ["train", "validation"]
    )
    if cache.manifest["included_splits"] != expected_roles:
        raise ValueError("Cache roles differ from this study phase")
    if (
        phase not in {"refit", "heldout"}
        and cache.manifest["cache_hash"] != protocol["selection_cache_hash"]
    ):
        raise ValueError("Selection cache differs from preflight")
    for name in (
        "selection_split_hash",
        "rgbir_snapshot_hash",
        "dsm_manifest_sha256",
        "relief_scale",
        "feature_recipe",
    ):
        if cache.manifest[name] != protocol[name]:
            raise ValueError(f"Cache feature/provenance differs: {name}")
    return runtime, cache


def _model(
    args: argparse.Namespace,
    recipe: Recipe,
    protocol: Protocol,
    runtime: TorchRuntime,
) -> tuple[nn.Module, ModelBinding]:
    size = recipe["backbone"]
    pretrained = args.pretrained_large if size == "L" else args.pretrained
    if pretrained is None:
        raise FileNotFoundError("The selected pinned pretrained snapshot is required")
    model, binding = load_model(
        pretrained, size, recipe["auxiliary"], runtime.device, checkpointing=True
    )
    if (
        binding["pretrained_hash"]
        != protocol["pretrained_bindings"][size]["pretrained_hash"]
    ):
        raise ValueError("Pretrained manifest differs from preflight")
    return model, binding


def check_refit_cache(cache: CacheBundle, protocol: Protocol) -> None:
    if cache.split.refit_parent_hash != protocol["selection_split_hash"]:
        raise ValueError("Refit cache does not derive from the selected split")
    actual = {f["path"]: f["sha256"] for f in cache.manifest["files"]}
    for scene, original in protocol["selection_assets"].items():
        current = cache.manifest["scenes"][scene]
        for field in ("source_rgbir_sha256", "source_dsm_sha256", "shape"):
            if current[field] != original[field]:
                raise ValueError(f"Refit changed a locked source: {scene}/{field}")
        names = ("optical", "relief", "dsm_available")
        if original["role"] == "train":
            names += ("labels", "valid")
        for name in names:
            key = f"{scene}/{name}.npy"
            if actual[key] != protocol["selection_file_hashes"][key]:
                raise ValueError(f"Refit changed a locked cache array: {key}")


def _train(
    args: argparse.Namespace,
    protocol: Protocol,
    runtime: TorchRuntime,
    cache: CacheBundle,
    recipe: Recipe,
    budget: Budget,
    *,
    selection_lock: SelectionLock | None = None,
) -> PhaseManifest:
    refit = selection_lock is not None
    arm = "refit" if refit else args.arm
    out = args.root / arm
    out.mkdir(parents=True, exist_ok=True)
    seed = protocol["confirmation_seed"] if arm == "D" else protocol["seed"]
    seed_everything(seed)
    model, model_binding = _model(args, recipe, protocol, runtime)
    binding_fields: CheckpointBinding = {
        "protocol_hash": protocol["protocol_hash"],
        "cache_hash": cache.manifest["cache_hash"],
        "arm": arm,
        "recipe": recipe,
        "seed": seed,
        "model": model_binding,
        "selection_lock": selection_lock,
    }
    binding = canonical_hash(binding_fields)
    optimizer, scaler = optimizer_and_scaler(
        model, runtime, protocol, 0.5 if refit else 1
    )
    state: PhaseState = {
        "updates": 0,
        "completed_passes": 0,
        "history": [],
        "best": None,
        "best_validation": None,
        "valid_pixel_presentations": 0,
        "training_seconds": 0.0,
        "validation_seconds": 0.0,
        "pass_matrix": np.zeros((6, 6), dtype=np.int64).tolist(),
        "pass_loss": 0.0,
    }
    if (out / "last.pt").exists():
        saved = load_checkpoint(out / "last.pt", binding)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        state = cast(PhaseState, saved["progress"])
    elif selection_lock is not None:
        path = args.root / selection_lock["arm"] / "best.pt"
        if sha256_file(path) != selection_lock["selection_checkpoint_sha256"]:
            raise ValueError("Refit initial checkpoint differs from the selection lock")
        model.load_state_dict(
            load_checkpoint(path, selection_lock["selection_binding"])["model"]
        )

    def save_last():
        return save_checkpoint(
            out / "last.pt", model, state, binding, optimizer=optimizer, scaler=scaler
        )

    passes, steps = (1, 1152) if refit else (protocol["passes"], 864)
    total = passes * steps
    with cache.access("train", auxiliary=recipe["auxiliary"]) as train:
        if selection_lock is not None:
            index = training_index(train)
        else:
            index = cast(TrainingIndex, read_json(args.root / "training-index.json"))
            if canonical_hash(index) != protocol["training_index_hash"]:
                raise ValueError("Train-only index differs from the preflight")
        if len(index["base"]) != (3456 if refit else 2592):
            raise ValueError("Unexpected source coverage window count")
        weights = torch.as_tensor(
            inverse_sqrt_class_weights(protocol["train_class_support"]),
            dtype=torch.float32,
        )
        panel = restore_panel(protocol["input_panel"])
        if state["updates"] == 0:
            write_json(
                out / "initial-input-distribution.json",
                distribution(model, train, panel, runtime),
            )
        augmentation = protocol["augmentation"][
            "enhanced" if recipe["enhanced"] else "baseline"
        ]
        for pass_number in range(state["completed_passes"], passes):
            draws = schedule(
                index,
                train,
                seed,
                pass_number,
                jitter=protocol["sampling"]["center_jitter"],
            )
            if len(draws) != 4 * steps:
                raise ValueError(
                    "Training schedule does not match the locked optimizer budget"
                )
            offset = state["updates"] - pass_number * steps
            data = loader(
                train,
                draws,
                seed,
                pass_number,
                augmentation,
                start=offset * 4,
                workers=runtime.num_workers,
                cuda=True,
            )
            iterator = iter(data)
            model.train()
            for _ in range(offset, steps):
                budget.check()
                tick = time.monotonic()
                batches = [next(iterator) for _ in range(4)]
                loss, matrix = train_step(
                    model,
                    batches,
                    optimizer,
                    scaler,
                    weights,
                    runtime,
                    protocol,
                    state["updates"],
                    total,
                )
                state["updates"] += 1
                state["pass_loss"] += loss
                state["pass_matrix"] = (
                    np.asarray(state["pass_matrix"]) + matrix
                ).tolist()
                state["valid_pixel_presentations"] += int(matrix.sum())
                state["training_seconds"] += time.monotonic() - tick
                if state["updates"] % 64 == 0:
                    LOGGER.info(
                        "%s update=%d/%d loss=%.5f", arm, state["updates"], total, loss
                    )
                    save_last()
            del iterator, data
            save_last()
            record: PassRecord = {
                "pass": pass_number + 1,
                "updates": state["updates"],
                "train_loss": state["pass_loss"] / steps,
                "train_metrics": cast(
                    EvaluationMetrics,
                    summarize_confusion(
                        np.asarray(state["pass_matrix"], dtype=np.int64)
                    ),
                ),
            }
            if selection_lock is None:
                tick = time.monotonic()
                with cache.access("validation", auxiliary=recipe["auxiliary"]) as val:
                    record["probe"] = score_patches(
                        model,
                        val,
                        restore_panel(protocol["probe_panel"]),
                        runtime,
                        protocol,
                        budget,
                    )
                    if pass_number + 1 in protocol["full_validation_passes"]:
                        metrics = cast(
                            EvaluationMetrics,
                            score_scenes(
                                model,
                                val,
                                runtime,
                                protocol,
                                budget,
                                out / ".scratch",
                                batch_size=protocol["timings"][recipe["backbone"]][
                                    "batch_size"
                                ],
                            ),
                        )
                        measured = candidate(metrics, protocol, arm, state["updates"])
                        record["full_validation"] = metrics
                        if state["best"] is None or ranking(measured) > ranking(
                            state["best"]
                        ):
                            digest = save_checkpoint(
                                out / "best.pt", model, state, binding
                            )
                            state["best"] = cast(
                                BestCandidate,
                                {**measured, "checkpoint_sha256": digest},
                            )
                            state["best_validation"] = metrics
                        record["backbone_inputs"] = distribution(
                            model, train, panel, runtime
                        )
                state["validation_seconds"] += time.monotonic() - tick
            state["history"].append(record)
            state["completed_passes"] += 1
            state["pass_matrix"] = np.zeros((6, 6), dtype=np.int64).tolist()
            state["pass_loss"] = 0.0
            save_last()
        elapsed = state["training_seconds"] + state["validation_seconds"]
        manifest: PhaseManifest = {
            **state,
            "status": "complete",
            "arm": arm,
            "seed": seed,
            "recipe": recipe,
            "binding": binding,
            "model_binding": model_binding,
            "protocol_hash": protocol["protocol_hash"],
            "cache_hash": cache.manifest["cache_hash"],
            "train_scene_ids": list(cache.split.train_scene_ids),
            "train_class_support": index["class_support"],
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "validation_fraction": state["validation_seconds"] / max(elapsed, 1e-9),
        }
        if selection_lock is not None:
            manifest["backbone_inputs"] = distribution(model, train, panel, runtime)
            manifest["selection_checkpoint_sha256"] = selection_lock[
                "selection_checkpoint_sha256"
            ]
            if state["updates"] != 1152 or state["completed_passes"] != 1:
                raise ValueError("Refit has not reached the declared final checkpoint")
            manifest["final_checkpoint_sha256"] = save_checkpoint(
                out / "final.pt", model, state, binding
            )
        budget.check()
        write_json(out / "manifest.json", manifest)
    return manifest


def _tta(
    args: argparse.Namespace,
    protocol: Protocol,
    runtime: TorchRuntime,
    cache: CacheBundle,
    budget: Budget,
) -> None:
    selected, confirmation = (
        winner(args.root, protocol),
        complete_arm(args.root, protocol, "D"),
    )
    selected_best = selected["best"]
    confirmation_best = confirmation["best"]
    if (
        selected_best is None
        or confirmation_best is None
        or not selected_best["eligible"]
        or not confirmation_best["eligible"]
    ):
        write_json(
            args.root / "D/gate.json",
            {"status": "validation-ineligible", "heldout_opened": False},
        )
        return
    size = selected["recipe"]["backbone"]
    predicted = 1.25 * 12 * protocol["timings"][size]["tta_scene_seconds"]
    final_fits = (
        1.25
        * protocol["runtime"]["allocated_devices"]
        * 14
        * protocol["timings"][size]["tta_scene_seconds"]
        <= 2 * 3600
    )
    tta: TTAValidation | None = None
    policy: InferencePolicy = "identity"
    if budget.remaining() >= predicted and final_fits:
        model, _ = _model(args, selected["recipe"], protocol, runtime)
        tta_results: dict[str, EvaluationMetrics] = {}
        with cache.access(
            "validation", auxiliary=selected["recipe"]["auxiliary"]
        ) as val:
            for key, manifest in (("seed42", selected), ("seed43", confirmation)):
                path = args.root / manifest["arm"] / "best.pt"
                model.load_state_dict(
                    load_checkpoint(path, manifest["binding"])["model"]
                )
                tta_results[key] = cast(
                    EvaluationMetrics,
                    score_scenes(
                        model,
                        val,
                        runtime,
                        protocol,
                        budget,
                        args.root / "D/.scratch",
                        policy="hflip",
                        batch_size=protocol["timings"][size]["batch_size"],
                    ),
                )
        tta = {
            "seed42": tta_results["seed42"],
            "seed43": tta_results["seed43"],
        }
        selected_metrics = selected["best_validation"]
        confirmation_metrics = confirmation["best_validation"]
        if selected_metrics is None or confirmation_metrics is None:
            raise ValueError("Validation metrics are required for TTA selection")
        policy = choose_tta(
            selected_metrics,
            tta["seed42"],
            confirmation_metrics,
            tta["seed43"],
            protocol,
        )
    lock_selection(args.root, protocol, policy, tta)


def select(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.root)
    runtime, cache = _bound_inputs(args, protocol, args.arm)
    recipe = recipe_for(args.root, protocol, args.arm)
    if args.arm == "D":
        selected_best = winner(args.root, protocol)["best"]
        if selected_best is None or not selected_best["eligible"]:
            out = args.root / "D"
            out.mkdir(exist_ok=True)
            write_json(
                out / "gate.json",
                {"status": "seed42-ineligible", "heldout_opened": False},
            )
            artifact_index(out)
            return
    with Budget(
        args.root / args.arm,
        protocol["per_arm_gpu_hours"],
        protocol["runtime"]["allocated_devices"],
    ) as budget:
        if (args.root / args.arm / "manifest.json").exists():
            raise FileExistsError("A completed selection phase must not be rerun")
        _train(args, protocol, runtime, cache, recipe, budget)
    gc.collect()
    torch.cuda.empty_cache()
    if args.arm == "D":
        spent = sum(
            read_json(args.root / arm / "budget.json")["gpu_seconds"]
            for arm in (*protocol["arms"], "D")
        )
        remaining = protocol["gpu_hours"]["selection"] - spent / 3600
        if remaining > 0:
            with Budget(
                args.root / "tta",
                remaining,
                protocol["runtime"]["allocated_devices"],
                started=time.monotonic(),
            ) as tta_budget:
                _tta(args, protocol, runtime, cache, tta_budget)
            artifact_index(args.root / "tta")
        else:
            lock_selection(args.root, protocol, "identity")
    artifact_index(args.root / args.arm)


def refit(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.root)
    lock = cast(SelectionLock, read_json(args.root / "selection-lock.json"))
    # Recompute the lock, including both seed gates, before touching refit data.
    lock_selection(args.root, protocol, lock["inference"], lock["tta_validation"])
    runtime, cache = _bound_inputs(args, protocol, "refit")
    check_refit_cache(cache, protocol)
    freeze(
        args.root / "refit-binding.json",
        {
            "protocol_hash": protocol["protocol_hash"],
            "selection_lock_hash": canonical_hash(lock),
            "refit_cache_hash": cache.manifest["cache_hash"],
        },
    )
    if (args.root / "refit/manifest.json").exists():
        raise FileExistsError("Completed refit must not be repeated")
    with Budget(
        args.root / "refit",
        protocol["gpu_hours"]["refit"],
        protocol["runtime"]["allocated_devices"],
    ) as budget:
        _train(
            args, protocol, runtime, cache, lock["recipe"], budget, selection_lock=lock
        )
    artifact_index(args.root / "refit")


def validate_final(manifest: PhaseManifest, lock: SelectionLock, digest: str) -> None:
    if (
        manifest["status"] != "complete"
        or manifest["protocol_hash"] != lock["protocol_hash"]
        or manifest["updates"] != 1152
        or manifest["completed_passes"] != 1
        or manifest["arm"] != "refit"
        or manifest["recipe"] != lock["recipe"]
        or manifest.get("selection_checkpoint_sha256")
        != lock["selection_checkpoint_sha256"]
        or manifest.get("final_checkpoint_sha256") != digest
    ):
        raise ValueError("Benchmark requires the complete, locked update-1,152 refit")


def heldout(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.root)
    lock = cast(SelectionLock, read_json(args.root / "selection-lock.json"))
    lock_selection(args.root, protocol, lock["inference"], lock["tta_validation"])
    manifest = cast(PhaseManifest, read_json(args.root / "refit/manifest.json"))
    path = args.root / "refit/final.pt"
    digest = sha256_file(path)
    validate_final(manifest, lock, digest)
    runtime, cache = _bound_inputs(args, protocol, "heldout")
    if (
        sum(s["valid_pixels"] for s in cache.manifest["scenes"].values())
        != protocol["heldout_pixels"]
    ):
        raise ValueError("Benchmark pixel support differs from the frozen protocol")
    out = args.root / "heldout"
    out.mkdir(exist_ok=True)
    marker = out / "attempt.json"
    if marker.exists():
        raise FileExistsError("The final benchmark checkpoint has already been opened")
    freeze(
        marker,
        {
            "protocol_hash": protocol["protocol_hash"],
            "checkpoint_sha256": digest,
            "cache_hash": cache.manifest["cache_hash"],
            "inference": lock["inference"],
            "evaluation_count": 1,
        },
    )
    with Budget(
        out, protocol["gpu_hours"]["heldout"], protocol["runtime"]["allocated_devices"]
    ) as budget:
        model, _ = _model(args, lock["recipe"], protocol, runtime)
        saved = load_checkpoint(path, manifest["binding"])
        if (
            saved["progress"]["updates"] != 1152
            or saved["progress"]["completed_passes"] != 1
        ):
            raise ValueError(
                "Checkpoint progress differs from the final refit manifest"
            )
        model.load_state_dict(saved["model"])
        with cache.access("test", auxiliary=lock["recipe"]["auxiliary"]) as access:
            started = time.monotonic()
            metrics = score_scenes(
                model,
                access,
                runtime,
                protocol,
                budget,
                out / ".scratch",
                policy=lock["inference"],
                profile=True,
                batch_size=protocol["timings"][lock["recipe"]["backbone"]][
                    "batch_size"
                ],
            )
            inference_seconds = time.monotonic() - started
        checked = matrix_metrics(
            metrics["confusion_matrix"], protocol["heldout_pixels"]
        )
        report = {
            **metrics,
            "protocol_hash": protocol["protocol_hash"],
            "checkpoint_sha256": digest,
            "inference": lock["inference"],
            "heldout_evaluation_count": 1,
            "inference_seconds": inference_seconds,
            "remaining_errors": checked["pixels_scored"]
            - sum(checked["confusion_matrix"][i][i] for i in range(6)),
            "oa_goal": protocol["oa_goal"],
            "oa_shortfall_percentage_points": max(
                0, 100 * (protocol["oa_goal"] - checked["overall_accuracy"])
            ),
            "benchmark_scope": (
                "Previously observed historical 14-scene benchmark; not a fresh holdout"
            ),
        }
        write_json(out / "metrics.json", report)
    artifact_index(out)
