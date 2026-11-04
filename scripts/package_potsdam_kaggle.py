"""Stage private Kaggle inputs and a single Potsdam study phase."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import shutil
import zipfile
from pathlib import Path

from deepnet.data.bundles import load_bundle
from deepnet.data.raster import sha256_file
from deepnet.potsdam_data import refit_split, verify_scope
from deepnet.potsdam_study import ROOT, _split
from deepnet.training_utils import write_json

BOOTSTRAP = ROOT / "model_script/potsdam_kaggle.py"
PHASES = ("preflight", "A", "B", "C", "D", "refit", "heldout")
PREVIOUS = {
    "preflight": (),
    "A": ("preflight",),
    "B": ("preflight",),
    "C": ("preflight", "A", "B"),
    "D": ("preflight", "A", "B", "C"),
    "refit": ("preflight", "A", "B", "C", "D"),
    "heldout": ("preflight", "A", "B", "C", "D", "refit"),
}


def _dataset(directory: Path, owner: str, slug: str):
    directory.mkdir(parents=True, exist_ok=False)
    write_json(
        directory / "dataset-metadata.json",
        {
            "id": f"{owner}/{slug}",
            "title": slug.replace("-", " "),
            "licenses": [{"name": "other"}],
        },
    )


def _link_or_copy(source: Path, destination: Path):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _source_bytes():
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for pattern in (
            "deepnet/**/*.py",
            "semantic_labeling/**/*.py",
            "configs/**/*.json",
        ):
            for path in sorted(ROOT.glob(pattern)):
                zipped.write(path, path.relative_to(ROOT))
    return archive.getvalue()


def _inputs(args):
    slug = f"{args.prefix}-calibration"
    calibration = args.output / "calibration"
    _dataset(calibration, args.owner, slug)
    old, smoke = args.old_checkpoint, args.dinov3_smoke
    _link_or_copy(old, calibration / "old-checkpoint.pt")
    shutil.copy2(smoke, calibration / "dinov3-smoke.json")
    write_json(
        calibration / "potsdam-calibration.json",
        {
            "old-checkpoint.pt": sha256_file(old),
            "dinov3-smoke.json": sha256_file(smoke),
        },
    )
    if args.dsm:
        manifest = args.dsm / "manifest.json"
        data = json.loads(manifest.read_text())
        split = _split()
        roles = {
            "selection": split.train_scene_ids + split.validation_scene_ids,
            "heldout": split.test_scene_ids,
        }
        for role, scenes in roles.items():
            directory = args.output / f"dsm-{role}"
            _dataset(directory, args.owner, f"{args.prefix}-dsm-{role}")
            shutil.copy2(manifest, directory / "manifest.json")
            write_json(
                directory / "potsdam-dsm-role.json",
                {
                    "role": role,
                    "manifest_sha256": sha256_file(manifest),
                },
            )
            for scene in scenes:
                source = args.dsm / f"{scene}.tif"
                if sha256_file(source) != data["scenes"][scene]["sha256"]:
                    raise ValueError(f"Cannot package unverified DSM: {scene}")
                _link_or_copy(source, directory / source.name)


def _refit_input(args):
    bundle = load_bundle(args.bundle)
    verify_scope(bundle, refit_split(_split()), ("train",))
    _dataset(args.output, args.owner, f"{args.prefix}-refit-bundle")
    shutil.copytree(
        args.bundle, args.output / "refit-bundle", copy_function=_link_or_copy
    )


def _kernel(args):
    phase = args.phase
    owner = args.owner
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    data = [f"{owner}/dinov2-joint-source-20261031"]
    if phase == "heldout":
        data += [f"{owner}/isprs-irrg-heldout-20261031"]
    elif phase == "refit":
        data += [f"{owner}/{args.prefix}-refit-bundle"]
    else:
        data += [f"{owner}/isprs-irrg-train-validation-20261031"]
    if phase == "preflight":
        data += [f"{owner}/{args.prefix}-calibration"]
    if args.with_dsm and (phase != "heldout" or args.heldout_dsm):
        role = "heldout" if phase == "heldout" else "selection"
        data += [f"{owner}/{args.prefix}-dsm-{role}"]
    payload = _source_bytes()
    config = {
        "phase": phase,
        "expected_phases": PREVIOUS[phase],
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "include_dsm": args.with_dsm and (phase != "heldout" or args.heldout_dsm),
        "allocation": args.allocated_devices,
    }
    code = BOOTSTRAP.read_text().replace("SETTINGS = {}", f"SETTINGS = {config!r}")
    code = code.replace(
        'PAYLOAD = ""', f"PAYLOAD = {base64.b85encode(payload).decode()!r}"
    )
    (args.output / "potsdam_kaggle.py").write_text(code)
    write_json(
        args.output / "kernel-metadata.json",
        {
            "id": f"{owner}/{args.prefix}-{phase.lower()}",
            "title": f"{args.prefix} {phase}",
            "code_file": "potsdam_kaggle.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": data,
            "competition_sources": [],
            "model_sources": [],
            "kernel_sources": [
                f"{owner}/{args.prefix}-{p.lower()}" for p in PREVIOUS[phase]
            ],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="damminhtien")
    parser.add_argument("--prefix", default="potsdam-dinov2-coverage-refit-v1")
    commands = parser.add_subparsers(dest="command", required=True)
    inputs = commands.add_parser("inputs")
    inputs.add_argument("--old-checkpoint", required=True, type=Path)
    inputs.add_argument("--dinov3-smoke", required=True, type=Path)
    inputs.add_argument("--dsm", type=Path)
    inputs.add_argument("--output", required=True, type=Path)
    refit = commands.add_parser("refit-input")
    refit.add_argument("--bundle", required=True, type=Path)
    refit.add_argument("--output", required=True, type=Path)
    kernel = commands.add_parser("kernel")
    kernel.add_argument("phase", choices=PHASES)
    kernel.add_argument("--with-dsm", action="store_true")
    kernel.add_argument("--heldout-dsm", action="store_true")
    kernel.add_argument(
        "--allocated-devices",
        type=int,
        choices=(1, 2),
        default=2,
        help="Physical GPUs allocated by Kaggle (default: two Tesla T4s)",
    )
    kernel.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    {"inputs": _inputs, "kernel": _kernel, "refit-input": _refit_input}[args.command](
        args
    )


if __name__ == "__main__":
    main()
