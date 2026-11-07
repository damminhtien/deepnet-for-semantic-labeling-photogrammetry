"""Submit, audit and compare the private Potsdam PSPNet Kaggle phases."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import re
import time
import zipfile
from pathlib import Path

from deepnet.data.contracts import canonical_hash
from deepnet.data.raster import sha256_file
from deepnet.potsdam_rgbir.protocol import ROOT, freeze, load_protocol, read_json
from deepnet.training_utils import write_json
from model_script.pspnet_rgb import HOURS, PREVIOUS, STUDY, verified_previous
from scripts.run_potsdam_rgbir_kaggle import (
    OWNER,
    _read_status,
    _source,
    _verify_index,
    _verify_metrics,
    execution_accounting,
    kaggle,
    log,
)

DATASETS = {
    "selection": "potsdam-dinov3-rgbir-v2-selection",
    "refit": "potsdam-dinov3-rgbir-v2-refit",
    "benchmark": "potsdam-dinov3-rgbir-v2-heldout",
}
ALLOCATION = 2


def slug(phase):
    return f"{OWNER}/{STUDY}-{phase}-run"


def inputs(reference):
    study = reference / "study"
    protocol = load_protocol(study)
    hashes = {
        phase: read_json(reference / f"{role}-cache/cache_manifest.json")["cache_hash"]
        for phase, role in (
            ("selection", "selection"),
            ("refit", "refit"),
            ("benchmark", "heldout"),
        )
    }
    if hashes["selection"] != protocol["selection_cache_hash"]:
        raise ValueError("Selection cache differs from the frozen DINOv3 preflight")
    if (
        hashes["refit"] != read_json(study / "refit/manifest.json")["cache_hash"]
        or hashes["benchmark"]
        != read_json(study / "heldout/attempt.json")["cache_hash"]
    ):
        raise ValueError("Refit or benchmark cache differs from the audited DINOv3 run")
    return protocol, hashes


def package(run, reference, phase):
    _, hashes = inputs(reference)
    archive = io.BytesIO(_source())
    with zipfile.ZipFile(archive, "a", zipfile.ZIP_DEFLATED) as source:
        source.write(ROOT / "model_script/pspnet_rgb.py", "model_script/pspnet_rgb.py")
        for name in ("protocol.json", "selection-lock.json", "training-index.json"):
            source.write(reference / "study" / name, f"reference/{name}")
    payload = archive.getvalue()
    settings = {
        "phase": phase,
        "allocation": ALLOCATION,
        "cache_hashes": hashes,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "selection_lock_hash": canonical_hash(
            read_json(reference / "study/selection-lock.json")
        ),
    }
    folder = run / phase
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "submission.json").exists():
        raise FileExistsError("Existing submission receipt; recover it instead")
    if phase in PREVIOUS:
        prior = read_json(run / PREVIOUS[phase] / "audit.json")
        if (
            prior["status"] != "verified"
            or prior["source_sha256"] != settings["source_sha256"]
        ):
            raise ValueError("Previous phase is not audited under the same source")
    code = (ROOT / "model_script/pspnet_rgb_bootstrap.py").read_text()
    code = code.replace("SETTINGS = {}", f"SETTINGS = {settings!r}", 1)
    code = code.replace(
        'PAYLOAD = ""', f"PAYLOAD = {base64.b85encode(payload).decode()!r}", 1
    )
    compile(code, "pspnet_rgb_kaggle.py", "exec")
    (folder / "pspnet_rgb_kaggle.py").write_text(code)
    write_json(folder / "package.json", settings)
    write_json(
        folder / "kernel-metadata.json",
        {
            "id": slug(phase),
            "title": f"{STUDY} {phase} run",
            "code_file": "pspnet_rgb_kaggle.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [f"{OWNER}/{DATASETS[phase]}"],
            "competition_sources": [],
            "model_sources": [],
            "kernel_sources": [slug(PREVIOUS[phase])] if phase in PREVIOUS else [],
        },
    )
    return folder


def submit(run, reference, phase):
    folder = package(run, reference, phase)
    status = _read_status("datasets", f"{OWNER}/{DATASETS[phase]}")
    lines = status.stdout.strip().lower().splitlines()
    if status.returncode or lines[-1:] != ["ready"]:
        raise RuntimeError("Required private cache is not ready")
    receipt = folder / "submission.json"
    freeze(receipt, {"slug": slug(phase), "status": "submitting"})
    result = kaggle(
        "kernels",
        "push",
        "-p",
        folder,
        "--accelerator",
        "NvidiaTeslaT4",
        "-t",
        round(HOURS[phase] * 3600 / ALLOCATION + 120),
        check=False,
    )
    (folder / "submission.log").write_text(result.stdout + result.stderr)
    match = re.search(r"Kernel version (\d+) successfully pushed", result.stdout)
    if result.returncode or not match:
        raise RuntimeError(
            "Unconfirmed Kaggle submission; reconcile the receipt before retry"
        )
    write_json(
        receipt,
        {
            "slug": slug(phase),
            "status": "accepted",
            "version": int(match[1]),
            "package_sha256": sha256_file(folder / "package.json"),
        },
    )
    log(f"Submitted {slug(phase)}/{match[1]}")


def await_output(folder):
    receipt = read_json(folder / "submission.json")
    if receipt["status"] != "accepted" or receipt["package_sha256"] != sha256_file(
        folder / "package.json"
    ):
        raise ValueError("Accepted submission/package binding is required")
    last = None
    while True:
        response = _read_status("kernels", receipt["slug"])
        match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", response.stdout)
        status = match[1] if match else "UNAVAILABLE"
        if status != last:
            log(f"{receipt['slug']}: {status}")
            last = status
        if status in {"COMPLETE", "ERROR", "CANCELLED", "CANCELED"}:
            break
        time.sleep(25)
    version = f"{receipt['slug']}/{receipt['version']}"
    write_json(folder / "terminal.json", {"state": status, "version": version})
    output = folder / "outputs"
    kaggle(
        "kernels",
        "output",
        version,
        "-p",
        output,
        "--page-size",
        200,
        "--file-pattern",
        r"^(study/.*|pspnet-phase\.json|output-index\.json|.*\.log)$",
        timeout=1800,
    )
    if status != "COMPLETE":
        raise RuntimeError(
            f"Kaggle {version} ended with {status}; later phases remain closed"
        )
    return output


def audit(run, reference, phase, output):
    folder = run / phase
    settings = read_json(folder / "package.json")
    marker = read_json(output / "pspnet-phase.json")
    if (
        marker["status"] != "complete"
        or marker["phase"] != phase
        or marker["source_sha256"] != settings["source_sha256"]
        or marker["allocated_devices"] != ALLOCATION
    ):
        raise ValueError("Remote phase/package identity differs")
    count = _verify_index(output, "output-index.json")
    protocol, _ = inputs(reference)
    records = output / "study" / phase
    budget = read_json(records / "budget.json")
    if budget["in_progress"] or not 0 < budget["gpu_seconds"] <= HOURS[phase] * 3600:
        raise ValueError("Phase exceeded or did not close its execution budget")
    evidence = {
        "status": "verified",
        "phase": phase,
        "indexed_files": count,
        "source_sha256": settings["source_sha256"],
        **execution_accounting(output, marker),
    }
    if phase != "benchmark":
        info, _ = verified_previous(records, phase, settings, protocol)
        evidence["checkpoint_sha256"] = info["checkpoint_sha256"]
        history = read_json(records / "history.json")
        if phase == "selection":
            if not read_json(records / "pilot-fit.json")["passed"]:
                raise ValueError("Small fit did not pass")
            validations = [h for h in history if "validation" in h]
            if [h["pass"] for h in validations] != protocol["full_validation_passes"]:
                raise ValueError("Full validation schedule differs")
            for record in validations:
                _verify_metrics(record["validation"], protocol["validation_pixels"])
            best = max(
                validations,
                key=lambda r: (
                    r["validation"]["overall_accuracy"],
                    r["validation"]["mean_iou"],
                    -r["updates"],
                ),
            )
            if best["updates"] != info["best_update"]:
                raise ValueError("Selected checkpoint differs from validation ranking")
        elif (
            info["initial_checkpoint_sha256"]
            != read_json(run / "selection/audit.json")["checkpoint_sha256"]
        ):
            raise ValueError("Refit did not start at the selected checkpoint")
    else:
        result = read_json(records / "metrics.json")
        attempt = read_json(records / "attempt.json")
        if (
            any(result[k] != v for k, v in attempt.items())
            or attempt["heldout_evaluation_count"] != 1
            or attempt["checkpoint_sha256"]
            != read_json(run / "refit/audit.json")["checkpoint_sha256"]
            or attempt["cache_hash"] != settings["cache_hashes"]["benchmark"]
            or attempt["inference"] != "identity"
        ):
            raise ValueError("Benchmark attempt binding differs")
        primary = read_json(reference / "study/heldout/metrics.json")
        current = _verify_metrics(result, protocol["heldout_pixels"])
        baseline = _verify_metrics(primary, protocol["heldout_pixels"])
        if (
            result["scene_ids"] != primary["scene_ids"]
            or result["window_size"] != primary["window_size"]
            or result["stride"] != primary["stride"]
            or any(
                sum(row) != sum(other)
                for row, other in zip(
                    current["confusion_matrix"],
                    baseline["confusion_matrix"],
                    strict=True,
                )
            )
        ):
            raise ValueError("Benchmark cohort, tiling or class support differs")
        write_json(
            run / "comparison.json",
            {
                "benchmark": "historical Potsdam 14-scene cohort",
                "pspnet": current,
                "dinov3": baseline,
                "pspnet_minus_dinov3_oa_pp": 100
                * (current["overall_accuracy"] - baseline["overall_accuracy"]),
                "pspnet_minus_dinov3_miou_pp": 100
                * (current["mean_iou"] - baseline["mean_iou"]),
                "pspnet_minus_dinov3_class_iou_pp": [
                    100 * (a - b)
                    for a, b in zip(
                        current["per_class_iou"], baseline["per_class_iou"], strict=True
                    )
                ],
                "pspnet_checkpoint_sha256": attempt["checkpoint_sha256"],
                "dinov3_checkpoint_sha256": primary["checkpoint_sha256"],
                "dinov3_metrics_sha256": sha256_file(
                    reference / "study/heldout/metrics.json"
                ),
                "pspnet_inference_seconds": result["inference_seconds"],
                "dinov3_inference_seconds": primary["inference_seconds"],
                "pspnet_allocated_gpu_hours": evidence["execution_gpu_hours"]
                + sum(
                    read_json(run / phase_name / "audit.json")["execution_gpu_hours"]
                    for phase_name in ("selection", "refit")
                ),
            },
        )
    write_json(folder / "audit.json", evidence)
    log(
        f"Audited {phase}: {count} files, "
        f"{evidence['execution_gpu_hours']:.4f} allocated GPU-hours"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("package", "submit", "campaign"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--phase", choices=HOURS, default="selection")
    args = parser.parse_args()
    if args.command == "package":
        print(package(args.run, args.reference, args.phase))
    elif args.command == "submit":
        submit(args.run, args.reference, args.phase)
    else:
        for phase in HOURS:
            folder = args.run / phase
            if (folder / "audit.json").exists():
                if read_json(folder / "audit.json")["status"] != "verified":
                    raise ValueError(f"Phase {phase} needs reconciliation")
                continue
            if not (folder / "submission.json").exists():
                submit(args.run, args.reference, phase)
            audit(args.run, args.reference, phase, await_output(folder))
        log(f"Completed: {args.run / 'comparison.json'}")


if __name__ == "__main__":
    main()
