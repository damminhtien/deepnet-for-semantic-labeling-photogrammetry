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
RUN_ID = "isprs-dinov2-joint-v1-20261031"
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
            if (
                payload.stat().st_size != record["bytes"]
                or checksum(payload) != record["sha256"]
            ):
                raise ValueError(f"Input checksum mismatch: {payload}")
        inputs[role] = path.parent
        LOGGER.info("Verified input %s", role)
    if set(inputs) != {"source", "training", "test"}:
        raise ValueError("All three locked private inputs are required")
    root = Path("/tmp/deepnet-inputs")
    root.mkdir()
    for role in ("training", "test", "source"):
        for path in inputs[role].glob("*.tar.gz"):
            LOGGER.info("Extracting %s/%s", role, path.name)
            extract(path, root)
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
    if any(
        importlib.metadata.version(name) != version
        for name, version in accelerator_versions.items()
    ):
        raise RuntimeError("Kaggle accelerator packages changed")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; assert torch.cuda.is_available(); "
            "print(torch.cuda.get_device_name())",
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
            str(working / RUN_ID),
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
