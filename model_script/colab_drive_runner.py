"""Run PSPNet in a Colab subprocess and mirror recoverable artifacts to Drive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

SYNC_FILES = (
    "run_manifest.json",
    "training_log.jsonl",
    "last.pt",
    "final_model_state.pt",
    "metrics.json",
    "training.stdout.log",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


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


def atomic_copy(source: Path, destination: Path) -> dict | None:
    if not source.is_file():
        return None
    before = source.stat()
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as src, temporary.open("wb") as dst:
            for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                dst.write(chunk)
                digest.update(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            temporary.unlink(missing_ok=True)
            return None
        expected_hash = digest.hexdigest()
        if sha256_file(temporary) != expected_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Drive copy checksum mismatch for {source.name}")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "bytes": after.st_size,
        "sha256": expected_hash,
        "source_mtime_ns": after.st_mtime_ns,
        "synced_utc": utc_now(),
    }


def sync_artifacts(
    run_dir: Path, stdout_path: Path, drive_dir: Path, state: dict
) -> dict:
    copied = state.setdefault("files", {})
    for filename in SYNC_FILES:
        source = stdout_path if filename == "training.stdout.log" else run_dir / filename
        if not source.is_file():
            continue
        stat = source.stat()
        previous = copied.get(filename, {})
        if (
            previous.get("bytes") == stat.st_size
            and previous.get("source_mtime_ns") == stat.st_mtime_ns
        ):
            continue
        result = atomic_copy(source, drive_dir / filename)
        if result is not None:
            copied[filename] = result
    state["run_id"] = run_dir.name.removeprefix("pspnet-colab-run-")
    state["updated_utc"] = utc_now()
    atomic_json_write(drive_dir / "drive_sync_manifest.json", state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-script", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--drive-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    drive_dir = args.drive_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Refusing to reuse PSPNet run directory: {run_dir}")
    if not drive_dir.is_dir():
        raise FileNotFoundError(f"Drive output folder is not ready: {drive_dir}")
    stdout_path = run_dir.with_name(run_dir.name + ".stdout.log")
    status = {
        "run_id": args.run_id,
        "status": "starting",
        "started_utc": utc_now(),
        "run_directory": str(run_dir),
        "drive_directory": str(drive_dir),
        "trainer_script": str(args.train_script.resolve()),
    }
    atomic_json_write(drive_dir / "colab_runner_manifest.json", status)

    command = [
        sys.executable,
        "-u",
        str(args.train_script.resolve()),
        "--backend",
        "colab",
        "--data-root",
        str(args.data_root.resolve()),
        "--run-dir",
        str(run_dir),
        "--run-id",
        args.run_id,
        "--resume-from",
        str(args.resume_from.resolve()),
        "--bundle-manifest",
        str(args.bundle_manifest.resolve()),
        "--seed",
        str(args.seed),
    ]
    state: dict = {"run_id": args.run_id, "files": {}}
    try:
        with stdout_path.open("w", encoding="utf-8", buffering=1) as stdout:
            process = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            status.update(
                {
                    "status": "running",
                    "trainer_pid": process.pid,
                    "command": command,
                    "started_utc": utc_now(),
                }
            )
            atomic_json_write(drive_dir / "colab_runner_manifest.json", status)
            while process.poll() is None:
                state = sync_artifacts(run_dir, stdout_path, drive_dir, state)
                time.sleep(max(args.poll_seconds, 5))
            return_code = process.wait()

        state = sync_artifacts(run_dir, stdout_path, drive_dir, state)
        training_manifest_path = drive_dir / "run_manifest.json"
        training_manifest = (
            json.loads(training_manifest_path.read_text(encoding="utf-8"))
            if training_manifest_path.is_file()
            else {}
        )
        final_status = "complete" if return_code == 0 else "error"
        if return_code == 0 and training_manifest.get("status") != "complete":
            final_status = "incomplete"
        status.update(
            {
                "status": final_status,
                "trainer_exit_code": return_code,
                "completed_utc": utc_now(),
                "last_completed_epoch": training_manifest.get("last_completed_epoch"),
            }
        )
        atomic_json_write(drive_dir / "colab_runner_manifest.json", status)
        return 0 if final_status == "complete" else (return_code or 1)
    except Exception as error:
        status.update(
            {
                "status": "runner_error",
                "error": f"{type(error).__name__}: {error}",
                "updated_utc": utc_now(),
            }
        )
        atomic_json_write(drive_dir / "colab_runner_manifest.json", status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
