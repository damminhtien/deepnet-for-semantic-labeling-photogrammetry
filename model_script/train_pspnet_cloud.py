"""Resume the Potsdam PSPNet experiment on a CUDA runtime."""

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semantic_labeling.constants import IGNORE_LABEL, INPUT_CHANNELS, N_CLASSES
from semantic_labeling.dataset import ISPRS_dataset
from semantic_labeling.inference import run_inference
from semantic_labeling.losses import CrossEntropy2d
from semantic_labeling.pspnet import PSPNet

VALID_IDS = ["3_12", "4_10", "4_11", "4_12", "5_11", "6_12"]
WINDOW_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCH_SIZE = 10_000
NUM_EPOCHS = 16
BASE_LR = 0.001
WEIGHT_DECAY = 0.0001
AUX_LOSS_WEIGHT = 0.4


def atomic_json_write(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_bundle(data_root, bundle_manifest):
    file_records = bundle_manifest.get("files")
    if not file_records:
        raise ValueError("Bundle manifest has no file inventory")
    fingerprint_payload = json.dumps(
        file_records, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    expected_bundle_sha256 = hashlib.sha256(fingerprint_payload).hexdigest()
    if expected_bundle_sha256 != bundle_manifest.get("bundle_sha256"):
        raise RuntimeError("Bundle manifest fingerprint is invalid")

    for record in file_records:
        relative_path = Path(record["path"])
        path = (data_root / relative_path).resolve()
        if not path.is_relative_to(data_root):
            raise ValueError(f"Bundle path escapes its root: {relative_path}")
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise RuntimeError(f"Bundle file is missing or has the wrong size: {relative_path}")
        with path.open("rb") as file_handle:
            actual_sha256 = hashlib.file_digest(file_handle, "sha256").hexdigest()
        if actual_sha256 != record["sha256"]:
            raise RuntimeError(f"Bundle checksum mismatch: {relative_path}")


def to_cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu_state(item) for item in value)
    return value


def dsm_scene_id(scene_id):
    first, second = scene_id.split("_")
    return f"{first}_{int(second):02d}"


def scene_ids(directory, suffix):
    prefix = "top_potsdam_"
    result = set()
    for path in directory.glob(f"*{suffix}"):
        if path.name.startswith(prefix):
            result.add(path.name[len(prefix) : -len(suffix)])
        else:
            result.add(path.name[: -len(suffix)])
    return result


def checkpoint_rng_state(device):
    state = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": None,
        "mps_rng_state": None,
    }
    if device.type == "cuda":
        state["cuda_rng_states"] = [
            rng.cpu() for rng in torch.cuda.get_rng_state_all()
        ]
    elif device.type == "mps":
        state["mps_rng_state"] = torch.mps.get_rng_state().cpu()
    return state


def restore_rng_state(checkpoint, device, seed):
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if device.type == "cuda":
        saved_states = checkpoint.get("cuda_rng_states")
        if saved_states and len(saved_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(saved_states)
        else:
            torch.cuda.manual_seed_all(seed)
    elif device.type == "mps" and checkpoint.get("mps_rng_state") is not None:
        torch.mps.set_rng_state(checkpoint["mps_rng_state"])


def smoke_test(net, dataset, device, seed):
    rng_state = checkpoint_rng_state(device)
    weights = {
        name: value.detach().cpu().clone()
        for name, value in net.state_dict().items()
    }
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
        )
    )
    inputs, targets = (tensor.to(device, non_blocking=True) for tensor in batch)
    net.train()
    main_logits, auxiliary_logits = net(inputs)
    loss = CrossEntropy2d(main_logits, targets) + AUX_LOSS_WEIGHT * CrossEntropy2d(
        auxiliary_logits, targets
    )
    if not torch.isfinite(loss):
        raise RuntimeError("CUDA forward/backward smoke produced a non-finite loss")
    loss.backward()
    gradients = [
        parameter.grad for parameter in net.parameters() if parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("CUDA backward smoke produced no finite gradients")
    result = {
        "input_shape": list(inputs.shape),
        "main_output_shape": list(main_logits.shape),
        "auxiliary_output_shape": list(auxiliary_logits.shape),
        "loss": float(loss.detach().cpu()),
    }
    net.zero_grad(set_to_none=True)
    net.load_state_dict(weights)
    net.eval()
    restore_rng_state(rng_state, device, seed)
    del batch, inputs, targets, main_logits, auxiliary_logits, loss
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("colab", "kaggle"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    resume_from = args.resume_from.expanduser().resolve()
    bundle_manifest_path = args.bundle_manifest.expanduser().resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the cloud training job")
    for path in (data_root, resume_from, bundle_manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)
    bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    checkpoint_record = bundle_manifest.get("checkpoint", {})
    if not bundle_manifest.get("bundle_sha256") or not checkpoint_record:
        raise ValueError("Bundle manifest is missing its fingerprint or checkpoint record")
    verify_bundle(data_root, bundle_manifest)

    potsdam_root = data_root / "potsdamvaihingen"
    label_root = data_root / "prepared_labels"
    participant_label_files = label_root / "participant" / "{}.npz"
    no_boundary_label_files = label_root / "no_boundary" / "{}.npz"
    data_files = potsdam_root / "3_ortho_irrg/3_Ortho_IRRG/top_potsdam_{}_IRRG.tif"
    dsm_files = potsdam_root / "1_dsm/1_DSM/dsm_potsdam_0{}.tif"
    ndsm_files = (
        potsdam_root
        / "1_dsm_normalisation/1_DSM_normalisation/"
        "dsm_potsdam_0{}_normalized_lastools.jpg"
    )

    participant_ids = scene_ids(
        participant_label_files.parent,
        ".npz",
    )
    no_boundary_ids = scene_ids(no_boundary_label_files.parent, ".npz")
    train_ids = sorted(participant_ids - set(VALID_IDS))
    test_ids = sorted(no_boundary_ids - set(VALID_IDS))
    if (len(participant_ids), len(no_boundary_ids), len(train_ids), len(test_ids)) != (
        18,
        20,
        18,
        14,
    ):
        raise ValueError(
            "Unexpected prepared Potsdam split: "
            f"participant={len(participant_ids)}, no_boundary={len(no_boundary_ids)}, "
            f"train={len(train_ids)}, test={len(test_ids)}"
        )
    if not set(VALID_IDS) <= no_boundary_ids:
        raise ValueError("Prepared no-boundary validation labels are incomplete")
    expected_split = bundle_manifest.get("split", {})
    actual_split = {
        "train_ids": train_ids,
        "validation_ids": VALID_IDS,
        "held_out_test_ids": test_ids,
    }
    if any(
        set(actual_split[key]) != set(expected_split.get(key, ()))
        for key in actual_split
    ):
        raise RuntimeError("Prepared label files do not match the bundle split manifest")

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    checkpoint_path = run_dir / "last.pt"
    if any(run_dir.iterdir()) and not manifest_path.is_file():
        raise FileExistsError(f"Refusing to reuse run directory without a manifest: {run_dir}")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_id") != args.run_id:
            raise RuntimeError(f"Run directory belongs to {manifest.get('run_id')}")
    else:
        manifest = {}

    device = torch.device("cuda:0")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    net = PSPNet(num_classes=N_CLASSES, use_aux=True).to(device)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY
    )
    class_weights = torch.ones(N_CLASSES, dtype=torch.float32, device=device)
    completed_epoch = 0
    checkpoint_source = checkpoint_path if checkpoint_path.is_file() else resume_from
    with checkpoint_source.open("rb") as checkpoint_file:
        checkpoint_sha256 = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    checkpoint = torch.load(checkpoint_source, map_location="cpu", weights_only=False)
    if checkpoint_path.is_file():
        if checkpoint.get("run_id") != args.run_id:
            raise RuntimeError("Saved cloud checkpoint belongs to a different run")
    else:
        if checkpoint_sha256 != checkpoint_record.get("sha256"):
            raise RuntimeError(
                "Resume checkpoint checksum does not match the bundle manifest"
            )
        if checkpoint.get("run_id") != checkpoint_record.get("run_id"):
            raise RuntimeError(
                "Resume checkpoint run ID does not match the bundle manifest"
            )
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    completed_epoch = int(checkpoint["epoch"])
    if not checkpoint_path.is_file() and completed_epoch != checkpoint_record.get("epoch"):
        raise RuntimeError("Resume checkpoint epoch does not match the bundle manifest")
    if not 0 <= completed_epoch <= NUM_EPOCHS:
        raise RuntimeError(f"Checkpoint epoch is outside the configured run: {completed_epoch}")
    if manifest and checkpoint_path.is_file():
        manifest_epoch = int(manifest.get("last_completed_epoch", 0))
        if manifest_epoch > completed_epoch:
            raise RuntimeError("Run manifest is ahead of the available cloud checkpoint")
        if manifest_epoch < completed_epoch:
            manifest["last_completed_epoch"] = completed_epoch
            manifest["last_epoch_loss"] = checkpoint.get("epoch_loss")
    elif manifest and manifest.get("last_completed_epoch", 0) > completed_epoch:
        raise RuntimeError("Run manifest is ahead of the available cloud checkpoint")
    restore_rng_state(checkpoint, device, args.seed)

    train_dataset = ISPRS_dataset(
        train_ids,
        [dsm_scene_id(scene_id) for scene_id in train_ids],
        data_files=str(data_files),
        label_files=str(participant_label_files),
        dsm_files=str(dsm_files),
        ndsm_files=str(ndsm_files),
        cache=True,
        augmentation=True,
        window_size=WINDOW_SIZE,
        epoch_size=EPOCH_SIZE,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    smoke = smoke_test(net, train_dataset, device, args.seed)
    source_run_id = checkpoint.get("run_id")
    if not manifest:
        manifest = {
            "run_id": args.run_id,
            "status": "prepared",
            "created_utc": datetime.now(UTC).isoformat(),
            "backend": args.backend,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_count": torch.cuda.device_count(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "source_run_id": source_run_id,
            "source_checkpoint_epoch": completed_epoch,
            "source_checkpoint_sha256": checkpoint_sha256,
            "bundle_sha256": bundle_manifest.get("bundle_sha256"),
            "source_git_revision": bundle_manifest.get("source_git_revision"),
            "seed": args.seed,
            "initialization": "random at epoch 0; resumed from the saved MPS epoch-2 model and Adam state",
            "train_ids": train_ids,
            "validation_ids": VALID_IDS,
            "test_ids": test_ids,
            "input_channels": INPUT_CHANNELS,
            "classes": N_CLASSES,
            "ignore_label": IGNORE_LABEL,
            "window_size": list(WINDOW_SIZE),
            "batch_size": BATCH_SIZE,
            "epoch_size": EPOCH_SIZE,
            "epochs": NUM_EPOCHS,
            "optimizer": "Adam",
            "learning_rate": BASE_LR,
            "weight_decay": WEIGHT_DECAY,
            "auxiliary_loss_weight": AUX_LOSS_WEIGHT,
            "input_scale": 255.0,
            "augmentation": "independent random vertical flip and horizontal mirror, each p=0.5",
            "cuda_smoke": smoke,
            "dataset_root": str(data_root),
            "run_directory": str(run_dir),
        }

    manifest["status"] = "running"
    manifest["started_utc"] = manifest.get(
        "started_utc", datetime.now(UTC).isoformat()
    )
    manifest["batches_per_epoch"] = len(train_loader)
    manifest["optimizer_steps_planned"] = NUM_EPOCHS * len(train_loader)
    manifest.setdefault(
        "optimizer_steps_completed_before_cloud",
        checkpoint_record["epoch"] * len(train_loader),
    )
    manifest["optimizer_steps_completed_at_resume"] = (
        completed_epoch * len(train_loader)
    )
    manifest["optimizer_steps_remaining"] = (
        NUM_EPOCHS - completed_epoch
    ) * len(train_loader)
    atomic_json_write(manifest_path, manifest)
    print(
        f"backend={args.backend} device={device} gpu={manifest['gpu_name']} "
        f"resume_epoch={completed_epoch} target_epochs={NUM_EPOCHS}",
        flush=True,
    )

    for epoch in range(completed_epoch + 1, NUM_EPOCHS + 1):
        epoch_started = time.perf_counter()
        net.train()
        rolling_loss = torch.zeros((), dtype=torch.float32, device=device)
        rolling_batches = 0
        epoch_loss_sum = 0.0

        for batch_index, (batch_inputs, batch_targets) in enumerate(
            train_loader, start=1
        ):
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            main_logits, auxiliary_logits = net(batch_inputs)
            loss = CrossEntropy2d(
                main_logits, batch_targets, weight=class_weights
            ) + AUX_LOSS_WEIGHT * CrossEntropy2d(
                auxiliary_logits, batch_targets, weight=class_weights
            )
            loss.backward()
            optimizer.step()

            rolling_loss.add_(loss.detach())
            rolling_batches += 1
            if batch_index % 100 == 0 or batch_index == len(train_loader):
                average_loss = rolling_loss.item() / rolling_batches
                epoch_loss_sum += average_loss * rolling_batches
                print(
                    f"epoch {epoch:02d}/{NUM_EPOCHS} "
                    f"batch {batch_index:04d}/{len(train_loader)} "
                    f"window_loss={average_loss:.5f} "
                    f"elapsed={time.perf_counter() - epoch_started:.0f}s",
                    flush=True,
                )
                rolling_loss.zero_()
                rolling_batches = 0

        torch.cuda.synchronize(device)
        epoch_loss = epoch_loss_sum / len(train_loader)
        epoch_seconds = time.perf_counter() - epoch_started
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": {
                name: value.detach().cpu() for name, value in net.state_dict().items()
            },
            "optimizer_state_dict": to_cpu_state(optimizer.state_dict()),
            **checkpoint_rng_state(device),
            "epoch_loss": epoch_loss,
            "run_id": args.run_id,
        }
        temporary_checkpoint = run_dir / "last.pt.tmp"
        torch.save(checkpoint, temporary_checkpoint)
        os.replace(temporary_checkpoint, checkpoint_path)
        if epoch % 4 == 0 or epoch == NUM_EPOCHS:
            torch.save(
                checkpoint["model_state_dict"],
                run_dir / f"model_epoch_{epoch:02d}.pt",
            )

        with (run_dir / "training_log.jsonl").open("a", encoding="utf-8") as log:
            log.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "mean_loss": epoch_loss,
                        "seconds": epoch_seconds,
                        "completed_utc": datetime.now(UTC).isoformat(),
                    }
                )
                + "\n"
            )
        manifest["last_completed_epoch"] = epoch
        manifest["last_epoch_loss"] = epoch_loss
        atomic_json_write(manifest_path, manifest)
        print(
            f"completed epoch {epoch:02d}/{NUM_EPOCHS}: "
            f"mean_loss={epoch_loss:.5f}, elapsed={epoch_seconds / 60:.1f} min",
            flush=True,
        )

    final_model_path = run_dir / "final_model_state.pt"
    torch.save(
        {name: value.detach().cpu() for name, value in net.state_dict().items()},
        final_model_path,
    )
    net.eval()
    validation_accuracy = run_inference(
        net,
        VALID_IDS,
        data_files=str(data_files),
        dsm_files=str(dsm_files),
        ndsm_files=str(ndsm_files),
        eroded_label_files=str(no_boundary_label_files),
        stride=256,
        batch_size=6,
        window_size=WINDOW_SIZE,
    )
    test_accuracy = run_inference(
        net,
        test_ids,
        data_files=str(data_files),
        dsm_files=str(dsm_files),
        ndsm_files=str(ndsm_files),
        eroded_label_files=str(no_boundary_label_files),
        stride=256,
        batch_size=6,
        window_size=WINDOW_SIZE,
    )
    metrics = {
        "validation_accuracy_percent": validation_accuracy,
        "held_out_test_accuracy_percent": test_accuracy,
        "validation_ids": VALID_IDS,
        "test_ids": test_ids,
        "metric": "overall pixel accuracy, ignoring label 6",
        "evaluated_utc": datetime.now(UTC).isoformat(),
    }
    atomic_json_write(run_dir / "metrics.json", metrics)
    manifest.update(
        {
            "status": "complete",
            "final_model": str(final_model_path),
            "metrics_file": str(run_dir / "metrics.json"),
            "metrics": metrics,
            "completed_utc": datetime.now(UTC).isoformat(),
        }
    )
    atomic_json_write(manifest_path, manifest)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
