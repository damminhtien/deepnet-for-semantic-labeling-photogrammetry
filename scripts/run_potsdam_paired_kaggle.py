"""Package, launch, and audit supplemental paired-prediction inference on Kaggle."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OWNER = "damminhtien"
STUDY = "potsdam-paired-prediction-analysis-v1"
KERNEL = f"{OWNER}/{STUDY}"
DINO_REFIT_KERNEL = "damminhtien/potsdam-dinov3-rgbir-v2-refit-run"
PSPNET_REFIT_KERNEL = "damminhtien/potsdam-pspnet-rgb-v1-refit-run"
HELDOUT_DATASET = "damminhtien/potsdam-dinov3-rgbir-v2-heldout"
SCENE_ORDER = [
    "2_13",
    "2_14",
    "3_13",
    "3_14",
    "4_13",
    "4_14",
    "4_15",
    "5_13",
    "5_14",
    "5_15",
    "6_13",
    "6_14",
    "6_15",
    "7_13",
]
PIXELS = 467_862_855
ALLOCATED_GPUS = 2


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _metrics_summary(path: Path, checkpoint_sha: str) -> dict[str, Any]:
    metrics = read_json(path)
    matrices = [row["confusion_matrix"] for row in metrics["scenes"]]
    if (
        metrics["scene_ids"] != SCENE_ORDER
        or [row["scene_id"] for row in metrics["scenes"]] != SCENE_ORDER
        or metrics["pixels_scored"] != PIXELS
        or sum(sum(sum(row) for row in matrix) for matrix in matrices) != PIXELS
        or metrics["checkpoint_sha256"] != checkpoint_sha
        or metrics["heldout_evaluation_count"] != 1
    ):
        raise ValueError(f"Archived benchmark metric contract differs: {path}")
    return {
        "metrics_sha256": sha256(path),
        "checkpoint_sha256": checkpoint_sha,
        "scene_ids": metrics["scene_ids"],
        "confusion_matrix": metrics["confusion_matrix"],
        "scenes": metrics["scenes"],
    }


def _reference_data(runs_root: Path) -> dict[str, Any]:
    dino_root = runs_root / "potsdam-dinov3-rgbir-v2"
    psp_root = runs_root / "potsdam-pspnet-rgb-v1"
    dino_metrics_path = dino_root / "study/heldout/metrics.json"
    dino_attempt_path = dino_root / "study/heldout/attempt.json"
    psp_metrics_path = psp_root / "benchmark/outputs/study/benchmark/metrics.json"
    dino_metrics = read_json(dino_metrics_path)
    psp_metrics = read_json(psp_metrics_path)
    dino_attempt = read_json(dino_attempt_path)
    dino_protocol = read_json(dino_root / "study/protocol.json")

    dino_audit = read_json(dino_root / "kaggle/refit/audit.json")
    psp_audit = read_json(psp_root / "refit/audit.json")
    if dino_audit.get("status") != "verified" or psp_audit.get("status") != "verified":
        raise ValueError("Both refit phases must have verified checkpoint audits")
    if dino_attempt["evaluation_count"] != 1:
        raise ValueError("DINOv3's archived benchmark record is not a single score")
    if (
        dino_attempt["checkpoint_sha256"] != dino_metrics["checkpoint_sha256"]
        or psp_metrics["checkpoint_sha256"]
        != read_json(psp_root / "refit/outputs/study/refit/manifest.json")[
            "checkpoint_sha256"
        ]
    ):
        raise ValueError("Archived metrics and refit checkpoints are not bound")
    if (
        dino_attempt["cache_hash"] != psp_metrics["cache_hash"]
        or dino_metrics["split_hash"] != psp_metrics["split_hash"]
        or dino_metrics["scene_ids"] != psp_metrics["scene_ids"]
        or dino_metrics["pixels_scored"] != psp_metrics["pixels_scored"]
        or dino_metrics["window_size"] != psp_metrics["window_size"]
        or dino_metrics["stride"] != psp_metrics["stride"]
        or dino_metrics["inference"] != psp_metrics["inference"]
        or dino_metrics["heldout_evaluation_count"] != 1
        or psp_metrics["heldout_evaluation_count"] != 1
    ):
        raise ValueError("DINOv3 and PSPNet original benchmark protocols differ")
    if (
        dino_metrics["pixels_scored"] != PIXELS
        or dino_metrics["scene_ids"] != SCENE_ORDER
        or dino_metrics["window_size"] != [512, 512]
        or dino_metrics["stride"] != [256, 256]
        or dino_metrics["inference"] != "identity"
    ):
        raise ValueError("Archived inference configuration differs from this analysis")

    return {
        "protocol": {
            "cache_hash": dino_attempt["cache_hash"],
            "split_hash": dino_metrics["split_hash"],
            "scene_ids": SCENE_ORDER,
            "pixels_scored": PIXELS,
            "window_size": dino_metrics["window_size"],
            "stride": dino_metrics["stride"],
            "inference": dino_metrics["inference"],
            "batch_size": 4,
            "mixed_precision": dino_protocol["execution"]["mixed_precision"],
        },
        "dinov3": _metrics_summary(
            dino_metrics_path, dino_metrics["checkpoint_sha256"]
        ),
        "pspnet": _metrics_summary(psp_metrics_path, psp_metrics["checkpoint_sha256"]),
    }


def _source_payload() -> bytes:
    stream = io.BytesIO()
    files = [
        *sorted((ROOT / "deepnet").rglob("*.py")),
        *sorted((ROOT / "semantic_labeling").rglob("*.py")),
        *sorted((ROOT / "configs").rglob("*.json")),
        ROOT / "model_script/pspnet_rgb.py",
        ROOT / "scripts/research/paired_kaggle_inference.py",
    ]
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = (
                "paired_kaggle_inference.py"
                if path.name == "paired_kaggle_inference.py"
                else path.relative_to(ROOT).as_posix()
            )
            archive.write(path, name)
    return stream.getvalue()


def attempt_root(run_root: Path, attempt: int) -> Path:
    if attempt < 1:
        raise ValueError("Attempt numbers must be positive")
    return run_root if attempt == 1 else run_root / f"attempt-{attempt}"


def package(run_root: Path, runs_root: Path, attempt: int) -> Path:
    target_root = attempt_root(run_root, attempt)
    if target_root.exists():
        raise FileExistsError(
            f"Supplemental attempt already exists; reconcile it first: {target_root}"
        )
    references = _reference_data(runs_root)
    payload = _source_payload()
    source_hash = hashlib.sha256(payload).hexdigest()
    kernel_dir = target_root / "kaggle"
    kernel_dir.mkdir(parents=True)
    code = (ROOT / "model_script/paired_kaggle_bootstrap.py").read_text()
    settings = {
        "source_sha256": source_hash,
        "allocation": ALLOCATED_GPUS,
        "references": references,
    }
    code = code.replace(
        "SETTINGS = {}",
        f"SETTINGS = {settings!r}",
        1,
    )
    code = code.replace(
        'PAYLOAD = ""', f"PAYLOAD = {base64.b85encode(payload).decode()!r}", 1
    )
    compile(code, "paired_kaggle_bootstrap.py", "exec")
    (kernel_dir / "paired_kaggle_bootstrap.py").write_text(code)
    metadata = {
        "id": KERNEL,
        "title": "Potsdam paired prediction analysis v1",
        "code_file": "paired_kaggle_bootstrap.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [HELDOUT_DATASET],
        "competition_sources": [],
        "kernel_sources": [DINO_REFIT_KERNEL, PSPNET_REFIT_KERNEL],
    }
    write_json(kernel_dir / "kernel-metadata.json", metadata)
    write_json(
        kernel_dir / "package.json",
        {
            "kernel": KERNEL,
            "source_sha256": source_hash,
            "source_bytes": len(payload),
            "references": {
                key: {
                    "checkpoint_sha256": value["checkpoint_sha256"],
                    "metrics_sha256": value["metrics_sha256"],
                }
                for key, value in references.items()
                if key != "protocol"
            },
            "protocol": references["protocol"],
            "training_performed": False,
            "historical_benchmark_preserved": True,
            "allocated_gpu_count": ALLOCATED_GPUS,
        },
    )
    write_json(kernel_dir / "references.json", references)
    return kernel_dir


def _kaggle(*args: str, timeout: int = 120, check: bool = True):
    result = subprocess.run(
        ["kaggle", *args], capture_output=True, text=True, timeout=timeout
    )
    if check and result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result


def submit(kernel_dir: Path) -> None:
    run_root = kernel_dir.parent
    receipt_path = run_root / "submission.json"
    if receipt_path.exists():
        raise FileExistsError("Submission receipt already exists; reconcile it first")
    write_json(receipt_path, {"kernel": KERNEL, "status": "submitting"})
    result = _kaggle(
        "kernels",
        "push",
        "-p",
        str(kernel_dir),
        "--accelerator",
        "NvidiaTeslaT4",
        "-t",
        "7200",
        check=False,
        timeout=1800,
    )
    (run_root / "submission.log").write_text(result.stdout + result.stderr)
    match = re.search(r"Kernel version (\d+) successfully pushed", result.stdout)
    if result.returncode or match is None:
        raise RuntimeError("Submission not confirmed; inspect the existing receipt")
    write_json(
        receipt_path,
        {"kernel": KERNEL, "version": int(match[1]), "status": "accepted"},
    )
    print(f"Submitted {KERNEL}/{match[1]}")


def watch(run_root: Path) -> None:
    receipt = read_json(run_root / "submission.json")
    if receipt.get("status") != "accepted":
        raise ValueError("Kernel submission receipt needs reconciliation")
    previous = None
    while True:
        result = _kaggle("kernels", "status", receipt["kernel"], check=False)
        output = result.stdout + result.stderr
        match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", output)
        state = match[1] if match else "UNAVAILABLE"
        if state != previous:
            print(f"{receipt['kernel']}: {state}", flush=True)
            previous = state
        if state in {"COMPLETE", "ERROR", "CANCELLED", "CANCELED"}:
            run_root.mkdir(parents=True, exist_ok=True)
            (run_root / "remote.log").write_text(
                _kaggle("kernels", "logs", receipt["kernel"], check=False).stdout
            )
            output_dir = run_root / "outputs"
            output_dir.mkdir(exist_ok=True)
            version = f"{receipt['kernel']}/{receipt['version']}"
            _kaggle(
                "kernels",
                "output",
                version,
                "-p",
                str(output_dir),
                "--page-size",
                "200",
                timeout=3600,
            )
            write_json(run_root / "terminal.json", {"state": state, "version": version})
            if state != "COMPLETE":
                raise RuntimeError(f"Supplemental inference ended with {state}")
            audit(run_root)
            return
        time.sleep(25)


def audit(run_root: Path) -> dict[str, Any]:
    output = run_root / "outputs"
    index = read_json(output / "output-index.json")
    verified = []
    for record in index["files"]:
        path = output / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise ValueError(f"Paired output hash mismatch: {record['path']}")
        verified.append(record["path"])
    result = read_json(output / "paired-kernel-run.json")
    if result["status"] != "complete" or result["training_performed"]:
        raise ValueError("Kaggle paired inference marker does not verify completion")
    if (
        result["cache_hash"]
        != read_json(run_root / "kaggle/package.json")["protocol"]["cache_hash"]
    ):
        raise ValueError("Paired predictions used an unexpected cache")
    evidence = {
        "status": "verified",
        "output_file_count": len(verified),
        "output_files": verified,
        "cache_hash": result["cache_hash"],
        "scene_ids": result["scene_ids"],
        "pixels_scored": result["pixels_scored"],
        "wall_seconds": result["kernel_wall_seconds"],
        "execution_gpu_seconds": result["execution_gpu_seconds"],
        "execution_gpu_hours": result["execution_gpu_seconds"] / 3600,
        "models": {
            name: {
                "checkpoint_sha256": model["checkpoint_sha256"],
                "peak_cuda_allocated_bytes": model["peak_cuda_allocated_bytes"],
                "peak_cuda_reserved_bytes": model["peak_cuda_reserved_bytes"],
            }
            for name, model in result["models"].items()
        },
    }
    write_json(run_root / "audit.json", evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("package", "submit", "watch", "audit"))
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs"
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "data/runs/potsdam-paired-error-study-v1",
    )
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.action == "package":
        path = package(args.run_root, args.runs_root, args.attempt)
        print(path)
    elif args.action == "submit":
        submit(attempt_root(args.run_root, args.attempt) / "kaggle")
    elif args.action == "watch":
        watch(attempt_root(args.run_root, args.attempt))
    else:
        print(json.dumps(audit(attempt_root(args.run_root, args.attempt)), indent=2))


if __name__ == "__main__":
    main()
