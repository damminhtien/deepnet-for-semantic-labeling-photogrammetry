"""Verify private inputs, preserve Kaggle Torch, then execute the full DINOv2 run."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

EXPECTED_INPUTS = {}
EXPANDED_FILES = {}
RUN_ID = "isprs-dinov2-joint-v1-20261031"
ATTEMPT_ID = RUN_ID
LOGGER = logging.getLogger(__name__)


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract(path: Path, root: Path) -> None:
    with tarfile.open(path) as source:
        for member in source.getmembers():
            relative = Path(member.name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"Unsafe archive member: {member.name}")
        source.extractall(root, filter="data")


def verify_file(path: Path, record: dict) -> None:
    if path.stat().st_size != record["bytes"] or checksum(path) != record["sha256"]:
        raise ValueError(f"Input checksum mismatch: {path}")


def verify_expanded(root: Path, records: list[dict]) -> None:
    expected = {record["path"] for record in records}
    observed = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed != expected:
        raise ValueError(f"Expanded archive inventory differs: {root}")
    for record in records:
        verify_file(root / record["path"], record)


def prepare_lora_runtime() -> str | None:
    # PEFT probes torchao even for floating-point LoRA.
    try:
        version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return None
    LOGGER.info("Removing unused torchao %s from the ephemeral runtime", version)
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "--yes", "torchao"], check=True
    )
    return version


def main() -> None:
    working = Path("/kaggle/working")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(working / "bootstrap.log"),
        ],
    )
    inputs = {}
    for path in Path("/kaggle/input").rglob("deepnet-input-manifest.json"):
        manifest = json.loads(path.read_text())
        role = manifest["role"]
        if manifest["run_id"] != RUN_ID or role not in EXPECTED_INPUTS:
            continue
        if role in inputs or checksum(path) != EXPECTED_INPUTS[role]:
            raise ValueError(f"Input manifest differs: {role}")
        for record in manifest["files"]:
            relative = Path(record["path"])
            if relative.is_absolute() or len(relative.parts) != 1:
                raise ValueError("Unsafe input file path")
            payload = path.parent / relative
            if payload.exists():
                verify_file(payload, record)
            elif payload.name.endswith(".tar.gz"):
                verify_expanded(
                    payload.with_name(payload.name.removesuffix(".tar.gz")),
                    EXPANDED_FILES[role][payload.name],
                )
            else:
                raise FileNotFoundError(payload)
        inputs[role] = path.parent
        LOGGER.info("Verified input %s", role)
    if set(inputs) != {"source", "training", "test"}:
        raise ValueError("All three locked private inputs are required")
    root = Path("/tmp/deepnet-inputs")
    root.mkdir()
    for role in ("training", "test", "source"):
        for name in EXPANDED_FILES[role]:
            path = inputs[role] / name
            if path.exists():
                LOGGER.info("Extracting %s/%s", role, path.name)
                extract(path, root)
                continue
            dataset = name.removesuffix(".tar.gz")
            relative = (
                Path("pretrained")
                if role == "source"
                else Path(dataset)
                / (
                    "train-validation-bundle"
                    if role == "training"
                    else "post-selection-test-bundle"
                )
            )
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(
                inputs[role] / dataset / relative, target_is_directory=True
            )
    for dataset in ("potsdam", "vaihingen"):
        shutil.copy2(
            inputs["training"] / f"{dataset}-quality.json",
            root / dataset / "quality.json",
        )
    constraints = Path("/tmp/deepnet-torch-constraints.txt")
    accelerator_versions = {
        name: importlib.metadata.version(name) for name in ("torch", "torchvision")
    }
    constraints.write_text(
        "".join(
            f"{name}=={version}\n" for name, version in accelerator_versions.items()
        )
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--constraint",
            str(constraints),
            "transformers==5.20.2",
            "peft==0.21.2",
            "albumentations==2.0.8",
            "rasterio==1.5.2",
        ],
        check=True,
    )
    (wheel,) = inputs["source"].glob("*.whl")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", str(wheel)], check=True
    )
    removed_torchao = prepare_lora_runtime()
    if any(
        importlib.metadata.version(name) != version
        for name, version in accelerator_versions.items()
    ):
        raise RuntimeError("Kaggle accelerator packages changed")
    (working / "bootstrap-runtime.json").write_text(
        json.dumps(
            {
                "experiment_id": RUN_ID,
                "attempt_id": ATTEMPT_ID,
                "removed_unused_torchao": removed_torchao,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "torchvision",
                        "transformers",
                        "peft",
                        "albumentations",
                        "rasterio",
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; from peft import LoraConfig, get_peft_model; "
            "assert torch.cuda.is_available(); "
            "model = get_peft_model(torch.nn.Sequential(torch.nn.Linear(4, 4)), "
            "LoraConfig(r=2, target_modules=['0'])); "
            "model(torch.ones(1, 4)).sum().backward(); "
            "assert any(p.grad is not None for n,p in model.named_parameters() "
            "if 'lora_B' in n); "
            "print('LoRA CPU forward/backward passed; GPU:', "
            "torch.cuda.get_device_name())",
        ],
        check=True,
    )
    source_manifest = working / "source-package.json"
    shutil.copy2(inputs["source"] / "deepnet-input-manifest.json", source_manifest)
    LOGGER.info("Starting full DINOv2 run")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "deepnet.training",
            "--config",
            str(inputs["source"] / "isprs-dinov2-joint-v1.json"),
            "--protocol",
            str(inputs["source"] / "isprs-dinov2-v1.json"),
            "--bundle-root",
            str(root),
            "--pretrained",
            str(root / "pretrained"),
            "--source-manifest",
            str(source_manifest),
            "--output",
            str(working / ATTEMPT_ID),
            "--device",
            "cuda",
        ],
        check=True,
    )
    LOGGER.info(
        "DINOv2 flow exited successfully; inspect run_manifest.json for final status"
    )


if __name__ == "__main__":
    main()
