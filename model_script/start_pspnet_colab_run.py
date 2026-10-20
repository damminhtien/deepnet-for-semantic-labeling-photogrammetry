"""Verify the Drive bundle and start one durable PSPNet Colab run."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

DRIVE_MOUNT = Path("/content/drive")
DRIVE_DATA_DIR = Path(
    "/content/drive/MyDrive/deepnet-for-semantic-labeling-photogrammetry-data"
)
PARTS_DIR = DRIVE_DATA_DIR
ARCHIVE_NAME = "pspnet-colab-shared-bundle-20261019.tar"
RUN_ID = "pspnet-colab-20261020-rerun"
RUN_DIR = Path(f"/content/pspnet-colab-run-{RUN_ID}")
BUNDLE_DIR = Path(f"/content/pspnet-bundle-{RUN_ID}")
RUNNER_PATH = Path("/content/colab_drive_runner.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def verify_and_assemble_archive() -> tuple[Path, dict]:
    manifest_path = PARTS_DIR / "parts_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest.get("parts", [])) != 106:
        raise RuntimeError("Expected the existing 106-part PSPNet Drive bundle")
    archive = Path("/content") / ARCHIVE_NAME
    temporary = archive.with_name(archive.name + ".partial")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    total_bytes = 0
    try:
        with temporary.open("wb") as output:
            for index, item in enumerate(manifest["parts"], start=1):
                part = PARTS_DIR / item["name"]
                if not part.is_file() or part.stat().st_size != item["size"]:
                    raise RuntimeError(f"Missing or wrong-sized Drive chunk: {part.name}")
                part_digest = hashlib.sha256()
                with part.open("rb") as source:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        output.write(chunk)
                        digest.update(chunk)
                        part_digest.update(chunk)
                        total_bytes += len(chunk)
                if part_digest.hexdigest() != item["sha256"]:
                    raise RuntimeError(f"SHA-256 mismatch for Drive chunk: {part.name}")
                print(f"verified Drive bundle part {index}/106", flush=True)
            output.flush()
            os.fsync(output.fileno())
        if total_bytes != manifest["archive_size"]:
            raise RuntimeError("Reconstructed PSPNet archive size does not match manifest")
        if digest.hexdigest() != manifest["archive_sha256"]:
            raise RuntimeError("Reconstructed PSPNet archive SHA-256 does not match")
        os.replace(temporary, archive)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return archive, manifest


def main() -> None:
    if not os.path.ismount(DRIVE_MOUNT):
        raise RuntimeError("Google Drive is not mounted at /content/drive")
    if not (PARTS_DIR / "parts_manifest.json").is_file():
        raise FileNotFoundError(f"Missing PSPNet input parts in {PARTS_DIR}")
    if not RUNNER_PATH.is_file():
        raise FileNotFoundError(f"Missing uploaded Drive runner at {RUNNER_PATH}")
    if RUN_DIR.exists():
        raise FileExistsError("This run ID already has a runtime directory; refusing reuse")

    import torch
    import torchvision

    if not torch.cuda.is_available():
        raise RuntimeError("Colab session has no CUDA GPU")
    gpu_name = torch.cuda.get_device_name(0)
    bundle_manifest_path = BUNDLE_DIR / "bundle_manifest.json"
    if BUNDLE_DIR.exists():
        if not bundle_manifest_path.is_file():
            raise FileExistsError(
                f"Refusing to reuse incomplete extracted bundle: {BUNDLE_DIR}"
            )
        parts_manifest = json.loads(
            (PARTS_DIR / "parts_manifest.json").read_text(encoding="utf-8")
        )
        if len(parts_manifest.get("parts", [])) != 106:
            raise RuntimeError("Expected the existing 106-part PSPNet Drive bundle")
        archive = None
    else:
        archive, parts_manifest = verify_and_assemble_archive()
        BUNDLE_DIR.mkdir(parents=True)
        with tarfile.open(archive, "r:") as bundle:
            bundle.extractall(BUNDLE_DIR, filter="data")

    if not bundle_manifest_path.is_file():
        raise FileNotFoundError("Extracted PSPNet bundle has no bundle_manifest.json")
    bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    checkpoint_record = bundle_manifest["checkpoint"]
    checkpoint_path = BUNDLE_DIR / checkpoint_record["path"]
    if checkpoint_path.stat().st_size != checkpoint_record["bytes"]:
        raise RuntimeError("Source PSPNet checkpoint size does not match its manifest")
    if sha256_file(checkpoint_path) != checkpoint_record["sha256"]:
        raise RuntimeError("Source PSPNet checkpoint SHA-256 does not match its manifest")
    trainer_path = BUNDLE_DIR / "source/model_script/train_pspnet_cloud.py"
    if not trainer_path.is_file():
        raise FileNotFoundError(f"Missing PSPNet trainer: {trainer_path}")

    drive_run_dir = DRIVE_DATA_DIR / "runs" / RUN_ID
    if drive_run_dir.exists():
        existing_entries = {path.name for path in drive_run_dir.iterdir()}
        if existing_entries != {"preflight_manifest.json"}:
            raise FileExistsError(
                f"Refusing to reuse non-empty Drive run directory: {drive_run_dir}"
            )
        preflight_path = drive_run_dir / "preflight_manifest.json"
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        expected_archive_hash = parts_manifest["archive_sha256"]
        if (
            preflight.get("run_id") != RUN_ID
            or preflight.get("status")
            not in {"preflight_verified", "drive_mount_verified"}
            or preflight.get("data_archive", {}).get("sha256")
            != expected_archive_hash
        ):
            raise RuntimeError(
                f"Existing Drive preflight manifest does not match {RUN_ID}"
            )
    else:
        drive_run_dir.mkdir(parents=True, exist_ok=False)
    setup_manifest = {
        "run_id": RUN_ID,
        "status": "bundle_verified",
        "created_utc": datetime.now(UTC).isoformat(),
        "gpu_name": gpu_name,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "drive_folder": str(drive_run_dir),
        "drive_parts": 106,
        "archive_bytes": parts_manifest["archive_size"],
        "archive_sha256": parts_manifest["archive_sha256"],
        "bundle_sha256": bundle_manifest["bundle_sha256"],
        "source_checkpoint": checkpoint_record,
        "trainer_script_sha256": sha256_file(trainer_path),
        "train_ids": bundle_manifest["split"]["train_ids"],
        "validation_ids": bundle_manifest["split"]["validation_ids"],
        "held_out_test_ids": bundle_manifest["split"]["held_out_test_ids"],
    }
    atomic_json_write(drive_run_dir / "setup_manifest.json", setup_manifest)

    (Path("/content") / ARCHIVE_NAME).unlink(missing_ok=True)
    command = [
        sys.executable,
        "-u",
        str(RUNNER_PATH),
        "--train-script",
        str(trainer_path),
        "--data-root",
        str(BUNDLE_DIR),
        "--run-dir",
        str(RUN_DIR),
        "--drive-dir",
        str(drive_run_dir),
        "--resume-from",
        str(checkpoint_path),
        "--bundle-manifest",
        str(bundle_manifest_path),
        "--run-id",
        RUN_ID,
        "--seed",
        "42",
    ]
    runner_log = (drive_run_dir / "drive_runner.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=runner_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd="/content",
    )
    setup_manifest.update(
        {
            "status": "running",
            "runner_pid": process.pid,
            "runner_command": command,
            "started_utc": datetime.now(UTC).isoformat(),
        }
    )
    atomic_json_write(drive_run_dir / "setup_manifest.json", setup_manifest)
    print(
        json.dumps(
            {
                "run_id": RUN_ID,
                "status": "runner_started",
                "runner_pid": process.pid,
                "gpu_name": gpu_name,
                "drive_run_dir": str(drive_run_dir),
                "bundle_sha256": bundle_manifest["bundle_sha256"],
                "source_checkpoint_epoch": checkpoint_record["epoch"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
