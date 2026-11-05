"""Package, submit once, retrieve, and audit private RGBIR Kaggle phases.

Run from the repository with ``python -m scripts.run_potsdam_rgbir_kaggle``.
All receipts, inputs, source payloads, and GPU artifacts live outside Git.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from deepnet.data.contracts import canonical_hash
from deepnet.data.raster import sha256_file
from deepnet.potsdam_rgbir.data import CacheBundle
from deepnet.potsdam_rgbir.phases import check_refit_cache, validate_final
from deepnet.potsdam_rgbir.prepare import build_cache
from deepnet.potsdam_rgbir.protocol import (
    ROOT,
    complete_arm,
    freeze,
    load_protocol,
    matrix_metrics,
    read_json,
    source_hash,
    winner,
)
from deepnet.training_utils import load_checkpoint, write_json

OWNER = "damminhtien"
STUDY = "potsdam-dinov3-rgbir-v2"


def log(message):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), message, flush=True)


def kaggle(*arguments, timeout=120, check=True):
    result = subprocess.run(
        ["kaggle", *map(str, arguments)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result


def _slug(phase):
    return f"{OWNER}/{STUDY}-{phase.lower()}"


def _source():
    stream = io.BytesIO()
    paths = [
        *sorted((ROOT / "deepnet").rglob("*.py")),
        ROOT / "semantic_labeling/__init__.py",
        ROOT / "semantic_labeling/losses.py",
        *sorted((ROOT / "configs").rglob("*.json")),
    ]
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
    return stream.getvalue()


def package(run, phase, cache_path, allocation=2):
    import hashlib

    cache = CacheBundle.load(cache_path)
    study = run / "study"
    protocol = None if phase == "preflight" else load_protocol(study)
    previous = [] if protocol is None else ["preflight"]
    if protocol is not None:
        if phase in "ABCL":
            previous += protocol["arms"][: protocol["arms"].index(phase)]
        elif phase == "D":
            previous += protocol["arms"]
        else:
            previous += [*protocol["arms"], "D"]
            if not (study / "selection-lock.json").exists():
                raise ValueError("Refit/benchmark requires the frozen two-seed lock")
            if phase == "heldout":
                previous += ["refit"]
                validate_final(
                    read_json(study / "refit/manifest.json"),
                    read_json(study / "selection-lock.json"),
                    sha256_file(study / "refit/final.pt"),
                )
    source = _source()
    settings = {
        "phase": phase,
        "cache_hash": cache.manifest["cache_hash"],
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "expected_phases": previous,
        "allocation": allocation,
    }
    code = (ROOT / "model_script/potsdam_rgbir_kaggle.py").read_text()
    code = code.replace("SETTINGS = {}", f"SETTINGS = {settings!r}", 1)
    code = code.replace(
        'PAYLOAD = ""', f"PAYLOAD = {base64.b85encode(source).decode()!r}", 1
    )
    compile(code, "potsdam_rgbir_kaggle.py", "exec")
    role = phase if phase in ("refit", "heldout") else "selection"
    metadata = {
        "id": _slug(phase),
        "title": f"{STUDY} {phase}",
        "code_file": "potsdam_rgbir_kaggle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{OWNER}/{STUDY}-{role}"],
        "competition_sources": [],
        "model_sources": [],
        "kernel_sources": [_slug(p) for p in previous],
    }
    out = run / "kaggle" / phase
    out.mkdir(parents=True, exist_ok=True)
    if (out / "submission.json").exists():
        raise FileExistsError("A submission receipt exists; recover that job")
    (out / "potsdam_rgbir_kaggle.py").write_text(code)
    write_json(out / "kernel-metadata.json", metadata)
    write_json(out / "package.json", {**settings, "source_hash": source_hash()})
    return out


def submit(run, phase, cache_path, allocation=2):
    out = package(run, phase, cache_path, allocation)
    protocol = None if phase == "preflight" else load_protocol(run / "study")
    hours = (
        2
        if protocol is None
        else (protocol["per_arm_gpu_hours"] if phase in "ABCLD" else 2)
    )
    freeze(out / "submission.json", {"slug": _slug(phase), "status": "submitting"})
    # The backend limit also counts bootstrap; the in-run budget is the primary guard.
    result = kaggle(
        "kernels",
        "push",
        "-p",
        out,
        "--accelerator",
        "NvidiaTeslaT4",
        "-t",
        round(hours * 3600 / allocation + 120),
        check=False,
    )
    (out / "submission.log").write_text(result.stdout + result.stderr)
    match = re.search(r"Kernel version (\d+) successfully pushed", result.stdout)
    if result.returncode or not match:
        raise RuntimeError(
            "Submission is unconfirmed; inspect its receipt before retrying"
        )
    write_json(
        out / "submission.json",
        {
            "slug": _slug(phase),
            "version": int(match[1]),
            "status": "accepted",
            "package_sha256": sha256_file(out / "package.json"),
        },
    )
    log(f"Submitted {_slug(phase)}/{match[1]}")


def await_phase(run, phase):
    out = run / "kaggle" / phase
    receipt = read_json(out / "submission.json")
    if receipt["status"] != "accepted":
        raise ValueError("Unconfirmed submission needs remote reconciliation")
    previous = None
    while True:
        result = kaggle("kernels", "status", receipt["slug"], timeout=45, check=False)
        match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", result.stdout)
        state = match[1] if match else "UNAVAILABLE"
        if state != previous:
            log(f"{phase}: {state}")
            previous = state
        if state in {"COMPLETE", "ERROR", "CANCELLED", "CANCELED"}:
            version = f"{receipt['slug']}/{receipt['version']}"
            remote = kaggle("kernels", "logs", version, check=False)
            (out / "remote.log").write_text(remote.stdout + remote.stderr)
            target = out / "outputs"
            target.mkdir(exist_ok=True)
            kaggle(
                "kernels",
                "output",
                version,
                "-p",
                target,
                "--page-size",
                200,
                "--file-pattern",
                r"^(study/.*|rgbir-phase\.json|output-index\.json|runner\.log)$",
                timeout=1800,
            )
            write_json(out / "terminal.json", {"state": state, "version": version})
            _verify_index(target, "output-index.json")
            if state != "COMPLETE":
                marker = read_json(target / "rgbir-phase.json")
                write_json(
                    out / "audit.json",
                    {
                        "status": "failed",
                        "phase": phase,
                        "terminal_state": state,
                        "execution_gpu_hours": marker["execution_gpu_seconds"] / 3600,
                    },
                )
                raise RuntimeError(
                    f"{phase} ended with {state}; later phases remain closed"
                )
            return target
        time.sleep(25)


def _verify_index(root, name="artifact_sha256_manifest.json"):
    records = read_json(root / name)["files"]
    for entry in records:
        relative = Path(entry["path"])
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.stat().st_size != entry["bytes"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError(f"Downloaded artifact differs: {relative}")
    return len(records)


def _verify_metrics(metrics, pixels=None):
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    checked = matrix_metrics(matrix.tolist(), pixels or int(matrix.sum()))
    for key in ("overall_accuracy", "mean_iou", "per_class_iou", "per_class_f1"):
        if key in metrics and not np.allclose(
            metrics[key], checked[key], rtol=0, atol=1e-12
        ):
            raise ValueError(
                f"Persisted metric differs from its confusion matrix: {key}"
            )
    if "scenes" in metrics:
        total = np.zeros((6, 6), dtype=np.int64)
        for scene in metrics["scenes"]:
            _verify_metrics(scene)
            total += np.asarray(scene["confusion_matrix"])
        np.testing.assert_array_equal(matrix, total)
    return checked


def audit(run, phase, output):
    marker = read_json(output / "rgbir-phase.json")
    packaged = read_json(run / "kaggle" / phase / "package.json")
    if (
        marker["phase"] != phase
        or marker["status"] != "complete"
        or marker["source_sha256"] != packaged["source_sha256"]
        or packaged["source_hash"] != source_hash()
    ):
        raise ValueError("Remote phase/source identity differs")
    root = output / "study"
    protocol = load_protocol(root)
    count = _verify_index(root / phase)
    budget = read_json(root / phase / "budget.json")
    if budget["in_progress"] or budget["gpu_seconds"] <= 0:
        raise ValueError("Incomplete GPU budget ledger")
    if phase == "preflight":
        report = read_json(root / "preflight/report.json")
        if (
            not report["models"]["B"]["small_fit"]["passed"]
            or protocol["source_hash"] != packaged["source_hash"]
            or canonical_hash(read_json(root / "training-index.json"))
            != protocol["training_index_hash"]
        ):
            raise ValueError("Preflight quality/source gate failed")
    elif phase in "ABCLD":
        manifest = complete_arm(root, protocol, phase)
        for item in manifest["history"]:
            if "full_validation" in item:
                _verify_metrics(item["full_validation"], protocol["validation_pixels"])
        payload = load_checkpoint(root / phase / "best.pt", manifest["binding"])
        if payload["progress"]["updates"] != manifest["best"]["step"]:
            raise ValueError("Selected checkpoint step differs")
    elif phase == "refit":
        manifest = read_json(root / "refit/manifest.json")
        lock = read_json(root / "selection-lock.json")
        validate_final(manifest, lock, sha256_file(root / "refit/final.pt"))
        payload = load_checkpoint(root / "refit/final.pt", manifest["binding"])
        if payload["progress"]["updates"] != 1152:
            raise ValueError("Final checkpoint progress differs")
    else:
        _verify_metrics(
            read_json(root / "heldout/metrics.json"), protocol["heldout_pixels"]
        )
    if (phase in "ABCLD" or phase == "refit") and any(
        not torch.isfinite(v).all() for v in payload["model"].values()
    ):
        raise ValueError("Checkpoint contains nonfinite model tensors")
    target = run / "study"
    target.mkdir(exist_ok=True)
    for path in root.iterdir():
        destination = target / path.name
        if path.is_dir():
            if destination.exists():
                raise FileExistsError(f"Audited phase already exists: {path.name}")
            shutil.copytree(path, destination, copy_function=os.link)
        else:
            freeze(destination, read_json(path))
    evidence = {
        "status": "verified",
        "phase": phase,
        "indexed_files": count,
        "protocol_hash": protocol["protocol_hash"],
        "execution_gpu_hours": marker["execution_gpu_seconds"] / 3600,
        "runner_gpu_hours": budget["gpu_seconds"] / 3600,
    }
    write_json(run / "kaggle" / phase / "audit.json", evidence)
    log(
        f"Audited {phase}: {count} files; "
        f"{evidence['execution_gpu_hours']:.4f} GPU-hours"
    )
    return evidence


def execution_hours(run):
    return sum(
        read_json(path)["execution_gpu_hours"]
        for path in (run / "kaggle").glob("*/audit.json")
    )


def _phase(run, phase, cache, allowance):
    if (run / "kaggle" / phase / "audit.json").exists():
        status = read_json(run / "kaggle" / phase / "audit.json")["status"]
        if status != "verified":
            raise RuntimeError(f"{phase} failed; reconcile its existing attempt")
        return
    if execution_hours(run) + allowance > 24:
        raise TimeoutError("The next phase cannot fit the remaining 24 GPU-hours")
    if not (run / "kaggle" / phase / "submission.json").exists():
        submit(run, phase, cache)
    audit(run, phase, await_phase(run, phase))


def phase_inputs(data, run, phase):
    """Create private refit/benchmark inputs only after the relevant model gate."""
    protocol = load_protocol(run / "study")
    lock = read_json(run / "study/selection-lock.json")
    if phase == "heldout":
        validate_final(
            read_json(run / "study/refit/manifest.json"),
            lock,
            sha256_file(run / "study/refit/final.pt"),
        )
    old = data / "runs/potsdam-dinov2-coverage-refit-v1"
    source = (
        old / "locked-refit-bundle"
        if phase == "refit"
        else data / "prepared/potsdam/post-selection-test-bundle"
    )
    cache_path = run / f"{phase}-cache"
    if not (cache_path / "cache_manifest.json").exists():
        build_cache(
            source, data / "inputs/potsdam-rgbir-v2", old / "dsm-features", cache_path
        )
    cache = CacheBundle.load(cache_path)
    if phase == "refit":
        check_refit_cache(cache, protocol)
    package_root = run / "kaggle" / f"input-{phase}"
    receipt = run / "kaggle" / f"input-{phase}-receipt.json"
    if not receipt.exists():
        package_root.mkdir(exist_ok=False)
        shutil.copytree(cache_path, package_root / "cache", copy_function=os.link)
        for size, folder in (
            ("B", "dinov3-vitb16-pretrain-lvd1689m"),
            ("L", "dinov3-vitl16-pretrain-sat493m"),
        ):
            weights = data / "models" / folder
            manifest = read_json(weights / "pretrained_manifest.json")
            target = package_root / f"pretrained-{size}"
            target.mkdir()
            for name in (
                "pretrained_manifest.json",
                *(f["path"] for f in manifest["files"]),
            ):
                os.link(weights / name, target / name)
        metadata = {
            "id": _slug(phase),
            "title": f"{STUDY} private {phase} inputs",
            "licenses": [{"name": "other"}],
        }
        write_json(package_root / "dataset-metadata.json", metadata)
        freeze(
            receipt,
            {"status": "submitting", "cache_hash": cache.manifest["cache_hash"]},
        )
        result = kaggle(
            "datasets",
            "create",
            "-p",
            package_root,
            "-r",
            "zip",
            "-q",
            timeout=7200,
            check=False,
        )
        (run / "kaggle" / f"input-{phase}-upload.log").write_text(
            result.stdout + result.stderr
        )
        if result.returncode or "created" not in result.stdout.lower():
            raise RuntimeError(
                "Dataset upload unconfirmed; reconcile its existing receipt"
            )
        write_json(
            receipt, {"status": "accepted", "cache_hash": cache.manifest["cache_hash"]}
        )
    if read_json(receipt)["status"] != "accepted":
        raise ValueError("An unconfirmed private upload needs reconciliation")
    return cache_path


def campaign(data, run):
    _phase(run, "preflight", run / "selection-cache", 2)
    protocol = load_protocol(run / "study")
    for phase in protocol["arms"]:
        _phase(run, phase, run / "selection-cache", protocol["per_arm_gpu_hours"])
    selected = winner(run / "study", protocol)
    if not selected["best"]["eligible"]:
        write_json(
            run / "campaign-status.json",
            {
                "status": "stopped-at-validation",
                "winner": selected["best"],
                "execution_gpu_hours": execution_hours(run),
                "benchmark_opened": False,
            },
        )
        log("No seed-42 checkpoint passed all validation floors; stopping")
        return
    _phase(run, "D", run / "selection-cache", protocol["per_arm_gpu_hours"])
    if not (run / "study/selection-lock.json").exists():
        write_json(
            run / "campaign-status.json",
            {
                "status": "stopped-at-confirmation",
                "winner": selected["best"],
                "execution_gpu_hours": execution_hours(run),
                "benchmark_opened": False,
            },
        )
        log("Seed 43 did not confirm the validation gate; stopping")
        return
    refit_cache = phase_inputs(data, run, "refit")
    _phase(run, "refit", refit_cache, 2)
    benchmark_cache = phase_inputs(data, run, "heldout")
    _phase(run, "heldout", benchmark_cache, 2)
    write_json(
        run / "campaign-status.json",
        {
            "status": "complete",
            "execution_gpu_hours": execution_hours(run),
            "benchmark_opened": True,
            "final_metrics": "study/heldout/metrics.json",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("package", "submit", "watch", "audit", "campaign")
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", *"ABCLD", "refit", "heldout"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    if args.action == "campaign":
        if args.data is None:
            parser.error("--data is required to run the full campaign")
        campaign(args.data, args.run)
    elif args.phase is None:
        parser.error("--phase is required for this action")
    elif args.action in ("package", "submit"):
        if args.cache is None:
            parser.error("--cache is required for packaging/submission")
        (package if args.action == "package" else submit)(
            args.run, args.phase, args.cache
        )
    elif args.action == "watch":
        output = await_phase(args.run, args.phase)
        audit(args.run, args.phase, output)
    else:
        audit(args.run, args.phase, args.run / "kaggle" / args.phase / "outputs")


if __name__ == "__main__":
    main()
