"""Bootstrap one private, role-scoped RGBIR campaign phase on Kaggle."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

STARTED = time.monotonic()
SETTINGS = {}
PAYLOAD = ""
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def _one(paths, description):
    matches = list(paths)
    if len(matches) != 1:
        raise ValueError(f"Expected one {description}, found {len(matches)}")
    return matches[0]


def _setup():
    source = base64.b85decode(PAYLOAD)
    if hashlib.sha256(source).hexdigest() != SETTINGS["source_sha256"]:
        raise ValueError("Packaged RGBIR source differs")
    root = Path("/tmp/potsdam-rgbir-source")
    with zipfile.ZipFile(io.BytesIO(source)) as archive:
        if any(
            Path(name).is_absolute() or ".." in Path(name).parts
            for name in archive.namelist()
        ):
            raise ValueError("Unsafe source archive")
        archive.extractall(root)
    installed = {
        name: importlib.metadata.version(name) for name in ("torch", "torchvision")
    }
    constraints = Path("/tmp/potsdam-rgbir-constraints.txt")
    constraints.write_text(
        "".join(f"{name}=={value}\n" for name, value in installed.items())
    )
    pip = [sys.executable, "-m", "pip"]
    subprocess.run(
        pip
        + [
            "install",
            "--constraint",
            str(constraints),
            "transformers==5.20.2",
            "peft==0.21.2",
            "albumentations==2.0.8",
            "rasterio==1.5.2",
            "numpy==2.2.6",
            "opencv-python-headless==4.11.0.86",
        ],
        check=True,
    )
    subprocess.run(pip + ["uninstall", "--yes", "torchao"], check=True)
    if any(importlib.metadata.version(n) != v for n, v in installed.items()):
        raise ValueError("Kaggle accelerator packages changed")
    return root


def _inputs(phase):
    roles = (
        ["train"]
        if phase == "refit"
        else ["test"]
        if phase == "heldout"
        else ["train", "validation"]
    )
    caches = list(INPUT.rglob("cache_manifest.json"))
    if len(caches) != 1:
        raise ValueError("Exactly one role-scoped cache must be attached")
    manifest = json.loads(caches[0].read_text())
    if (
        manifest["included_splits"] != roles
        or manifest["cache_hash"] != SETTINGS["cache_hash"]
    ):
        raise ValueError("Attached cache has different roles or provenance")
    models = {}
    for path in INPUT.rglob("pretrained_manifest.json"):
        model_id = json.loads(path.read_text())["model_id"]
        if model_id in models:
            raise ValueError("Duplicate pretrained snapshot")
        models[model_id] = path.parent
    return caches[0].parent, models


def _previous(study):
    records = {}
    for marker in INPUT.rglob("rgbir-phase.json"):
        data = json.loads(marker.read_text())
        phase = data["phase"]
        if phase not in SETTINGS["expected_phases"]:
            raise ValueError(f"Unexpected attached campaign output: {phase}")
        if phase in records or data["status"] != "complete":
            raise ValueError(f"Duplicate or unsuccessful prerequisite: {phase}")
        records[phase] = marker.parent / "study"
    if set(records) != set(SETTINGS["expected_phases"]):
        raise ValueError("Missing prerequisite campaign outputs")
    for phase, root in records.items():
        for name in ("protocol.json", "training-index.json", "selection-lock.json"):
            source, target = root / name, study / name
            if source.exists():
                if target.exists() and target.read_bytes() != source.read_bytes():
                    raise ValueError(f"Prerequisite metadata differs: {name}")
                shutil.copy2(source, target)
        target = study / phase
        target.mkdir()
        if phase == "preflight":
            names = ("report.json", "budget.json")
        elif phase == "refit":
            names = ("manifest.json", "final.pt", "budget.json")
        else:
            names = ("manifest.json", "best.pt", "budget.json")
        for name in names:
            (target / name).symlink_to(root / phase / name)


def _execute(phase):
    cache, models = _inputs(phase)
    source = _setup()
    import torch

    actual = torch.cuda.device_count()
    if actual != SETTINGS["allocation"]:
        raise ValueError(f"Expected {SETTINGS['allocation']} GPUs, found {actual}")
    print(f"Allocated GPUs: {actual}; all devices count toward the budget", flush=True)
    study = WORK / "study"
    study.mkdir(exist_ok=False)
    _previous(study)
    b = models["facebook/dinov3-vitb16-pretrain-lvd1689m"]
    command = ["select", "--arm", phase] if phase in "ABCLD" else [phase]
    options = ["--root", str(study), "--cache", str(cache), "--pretrained", str(b)]
    large = models.get("facebook/dinov3-vitl16-pretrain-sat493m")
    if large is not None:
        options += ["--pretrained-large", str(large)]
    if phase == "preflight":
        marker = _one(INPUT.rglob("calibration.json"), "Rasterio calibration panel")
        options += ["--calibration", str(marker.parent)]
    with (WORK / "runner.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "deepnet.potsdam_rgbir.study",
                *command,
                *options,
            ],
            cwd=source,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise RuntimeError(
                f"RGBIR phase failed with exit code {process.returncode}"
            )


def main():
    phase, status = SETTINGS["phase"], "failed"
    os.environ["POTSDAM_GPU_STARTED"] = str(STARTED)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        _execute(phase)
        status = "complete"
    finally:
        for previous in SETTINGS["expected_phases"]:
            path = WORK / "study" / previous
            if path.exists():
                shutil.rmtree(path)
        files = []
        for path in sorted((WORK / "study").rglob("*")):
            if path.is_file() and ".scratch" not in path.parts:
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                files.append(
                    {
                        "path": path.relative_to(WORK).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": digest,
                    }
                )
        (WORK / "output-index.json").write_text(json.dumps({"files": files}) + "\n")
        torch = sys.modules.get("torch")
        devices = (
            torch.cuda.device_count() if torch is not None else SETTINGS["allocation"]
        )
        marker = {
            "study_id": "potsdam-dinov3-rgbir-v2",
            "phase": phase,
            "status": status,
            "source_sha256": SETTINGS["source_sha256"],
            "allocated_devices": devices,
            "allocation_verified": torch is not None,
            "execution_gpu_seconds": devices * (time.monotonic() - STARTED),
        }
        (WORK / "rgbir-phase.json").write_text(json.dumps(marker, indent=2) + "\n")


if __name__ == "__main__":
    main()
