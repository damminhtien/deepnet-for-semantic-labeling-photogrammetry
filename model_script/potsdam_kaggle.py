"""Bootstrap exactly one private Potsdam study phase on Kaggle."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

SETTINGS = {}
PAYLOAD = ""
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")


def _one(paths, description):
    matches = list(paths)
    if len(matches) != 1:
        raise ValueError(f"Expected one {description}, found {len(matches)}")
    return matches[0]


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _bundle(phase):
    role = (
        ["test"]
        if phase == "heldout"
        else ["train"]
        if phase == "refit"
        else ["train", "validation"]
    )
    candidates = []
    for path in INPUT.rglob("bundle_manifest.json"):
        data = json.loads(path.read_text())
        if data["dataset_id"] != "isprs-potsdam":
            continue
        if data["included_splits"] == ["test"] and phase != "heldout":
            raise ValueError("An evaluation bundle was attached to a training job")
        if set(data["included_splits"]) == set(role):
            candidates.append(path.parent)
    return _one(candidates, "role-scoped Potsdam bundle")


def _setup():
    source = base64.b85decode(PAYLOAD)
    if hashlib.sha256(source).hexdigest() != SETTINGS["source_sha256"]:
        raise ValueError("Packaged study source differs")
    root = Path("/tmp/potsdam-study-source")
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
    constraints = Path("/tmp/potsdam-torch-constraints.txt")
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
        ],
        check=True,
    )
    subprocess.run(pip + ["uninstall", "--yes", "torchao"], check=True)
    if any(
        importlib.metadata.version(name) != value for name, value in installed.items()
    ):
        raise ValueError("Kaggle accelerator packages changed")
    return root


def _previous(study):
    records = {}
    for marker in INPUT.rglob("potsdam-phase.json"):
        phase = json.loads(marker.read_text())["phase"]
        if phase in SETTINGS["expected_phases"]:
            if phase in records:
                raise ValueError(f"Duplicate previous phase: {phase}")
            records[phase] = marker.parent / "study"
    if set(records) != set(SETTINGS["expected_phases"]):
        raise ValueError("Missing prerequisite campaign outputs")
    for phase, root in records.items():
        if phase == "preflight":
            shutil.copy2(root / "protocol.json", study / "protocol.json")
            continue
        target = study / phase
        target.mkdir()
        for name in ("manifest.json", "final.pt" if phase == "refit" else "best.pt"):
            (target / name).symlink_to(root / phase / name)


def main():
    phase = SETTINGS["phase"]
    bundle = _bundle(phase)
    root = _setup()
    import torch

    if torch.cuda.device_count() != SETTINGS["allocation"]:
        raise ValueError("Kaggle GPU allocation differs from the package budget")
    study = WORK / "study"
    study.mkdir(exist_ok=False)
    _previous(study)
    pretrained = _one(
        (
            path.parent
            for path in INPUT.rglob("pretrained_manifest.json")
            if json.loads(path.read_text())["model_id"] == "facebook/dinov2-base"
        ),
        "pretrained DINOv2",
    )
    command = ["select", "--arm", phase] if phase in "ABCD" else [phase]
    options = [
        "--root",
        str(study),
        "--bundle",
        str(bundle),
        "--pretrained",
        str(pretrained),
    ]
    if phase == "preflight":
        marker = _one(INPUT.rglob("potsdam-calibration.json"), "calibration input")
        files = json.loads(marker.read_text())
        for name, digest in files.items():
            if _sha(marker.parent / name) != digest:
                raise ValueError(f"Calibration file differs: {name}")
        options += [
            "--old-checkpoint",
            str(marker.parent / "old-checkpoint.pt"),
            "--dinov3-smoke",
            str(marker.parent / "dinov3-smoke.json"),
        ]
    if SETTINGS["include_dsm"]:
        marker = _one(INPUT.rglob("potsdam-dsm-role.json"), "role-scoped DSM input")
        data = json.loads(marker.read_text())
        role = "heldout" if phase == "heldout" else "selection"
        if (
            data["role"] != role
            or _sha(marker.parent / "manifest.json") != data["manifest_sha256"]
        ):
            raise ValueError("DSM input is not the declared role snapshot")
        options += ["--dsm", str(marker.parent)]
    try:
        subprocess.run(
            [sys.executable, "-m", "deepnet.potsdam_study", *command, *options],
            cwd=root,
            check=True,
        )
    finally:
        # Keep only this phase in the output; previous phases stay in their own jobs.
        for previous in SETTINGS["expected_phases"]:
            if previous != "preflight":
                shutil.rmtree(study / previous)
    (WORK / "potsdam-phase.json").write_text(json.dumps({"phase": phase}) + "\n")


if __name__ == "__main__":
    main()
