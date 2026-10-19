"""Assemble a checksummed, portable PSPNet cloud-training bundle."""

import argparse
import errno
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def dsm_scene_id(scene_id):
    first, second = scene_id.split("_")
    return f"{first}_{int(second):02d}"


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno not in {
            errno.EACCES,
            errno.EPERM,
            errno.EXDEV,
            errno.ENOTSUP,
        }:
            raise
        shutil.copy2(source, destination)


def sha256_file(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def stage_file(source, output_dir, relative_path):
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = output_dir / relative_path
    link_or_copy(source, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--preprocessing-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-run-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    preprocessing_dir = args.preprocessing_dir.expanduser().resolve()
    checkpoint_path = args.source_checkpoint.expanduser().resolve()
    run_manifest_path = args.source_run_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    staging_dir = output_dir.with_name(output_dir.name + ".tmp")

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle: {output_dir}")
    if staging_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing staging dir: {staging_dir}")
    for path in (data_root, preprocessing_dir, checkpoint_path, run_manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)

    preprocessing_manifest_path = preprocessing_dir / "preprocessing_manifest.json"
    preprocessing_manifest = json.loads(
        preprocessing_manifest_path.read_text(encoding="utf-8")
    )
    split = preprocessing_manifest["split"]
    train_ids = split["train_ids"]
    validation_ids = split["validation_ids"]
    test_ids = split["held_out_test_ids"]
    if (len(train_ids), len(validation_ids), len(test_ids)) != (18, 6, 14):
        raise ValueError("Expected the documented 18/6/14 train/validation/test split")

    source_run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if source_run_manifest.get("status") != "paused_for_cloud":
        raise ValueError("Source run manifest must describe the paused local run")
    if source_run_manifest.get("last_completed_epoch") != 2:
        raise ValueError("Expected the local checkpoint after exactly two epochs")
    if "random" not in source_run_manifest.get("initialization", "").lower():
        raise ValueError("Source run manifest does not confirm random initialization")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_checkpoint = {
        "epoch": source_run_manifest["last_completed_epoch"],
        "run_id": source_run_manifest["run_id"],
    }
    actual_checkpoint = {
        "epoch": checkpoint.get("epoch"),
        "run_id": checkpoint.get("run_id"),
    }
    if actual_checkpoint != expected_checkpoint:
        raise ValueError(
            "Checkpoint identity differs from its run manifest: "
            f"{actual_checkpoint} != {expected_checkpoint}"
        )
    del checkpoint

    staging_dir.mkdir(parents=True)
    try:
        staged_paths = []
        potsdam_root = data_root / "potsdamvaihingen"
        input_files = (
            (
                "3_ortho_irrg/3_Ortho_IRRG",
                "top_potsdam_{}_IRRG.tif",
                None,
            ),
            ("1_dsm/1_DSM", "dsm_potsdam_0{}.tif", dsm_scene_id),
            (
                "1_dsm_normalisation/1_DSM_normalisation",
                "dsm_potsdam_0{}_normalized_lastools.jpg",
                dsm_scene_id,
            ),
        )
        for scene_id in sorted(set(train_ids + validation_ids + test_ids)):
            for folder, filename_template, id_mapper in input_files:
                file_scene_id = id_mapper(scene_id) if id_mapper else scene_id
                source = potsdam_root / folder / filename_template.format(file_scene_id)
                relative_path = Path("potsdamvaihingen") / folder / source.name
                staged_paths.append(stage_file(source, staging_dir, relative_path))

        label_sources = (
            ("participant", train_ids),
            ("no_boundary", validation_ids + test_ids),
        )
        for label_type, scene_ids in label_sources:
            for scene_id in scene_ids:
                source = preprocessing_dir / "labels" / label_type / f"{scene_id}.npz"
                relative_path = Path("prepared_labels") / label_type / source.name
                staged_paths.append(stage_file(source, staging_dir, relative_path))

        staged_paths.append(
            stage_file(
                checkpoint_path,
                staging_dir,
                Path("checkpoint") / "last.pt",
            )
        )
        staged_paths.append(
            stage_file(
                preprocessing_manifest_path,
                staging_dir,
                Path("preprocessing_manifest.json"),
            )
        )
        staged_paths.append(
            stage_file(
                run_manifest_path,
                staging_dir,
                Path("source_run_manifest.json"),
            )
        )

        code_paths = sorted((PROJECT_ROOT / "semantic_labeling").glob("*.py"))
        code_paths.extend(
            PROJECT_ROOT / "model_script" / filename
            for filename in (
                "build_pspnet_cloud_bundle.py",
                "prepare_potsdam_cloud_data.py",
                "train_pspnet_cloud.py",
            )
        )
        for source in code_paths:
            relative_path = Path("source") / source.relative_to(PROJECT_ROOT)
            staged_paths.append(stage_file(source, staging_dir, relative_path))

        try:
            source_git_revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            source_git_revision = None
        try:
            source_worktree_changes = subprocess.run(
                ["git", "status", "--short", "--", "model_script", "semantic_labeling"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        except (OSError, subprocess.CalledProcessError):
            source_worktree_changes = []

        file_records = []
        for path in sorted(staged_paths, key=lambda item: item.relative_to(staging_dir).as_posix()):
            relative_path = path.relative_to(staging_dir).as_posix()
            file_records.append(
                {
                    "path": relative_path,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

        fingerprint_payload = json.dumps(
            file_records, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        checkpoint_file_record = next(
            record
            for record in file_records
            if record["path"] == "checkpoint/last.pt"
        )
        bundle_manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(UTC).isoformat(),
            "bundle_sha256": hashlib.sha256(fingerprint_payload).hexdigest(),
            "source_git_revision": source_git_revision,
            "source_worktree_changes": source_worktree_changes,
            "source_run_id": source_run_manifest["run_id"],
            "source_checkpoint_epoch": source_run_manifest["last_completed_epoch"],
            "checkpoint": {
                "path": "checkpoint/last.pt",
                "bytes": checkpoint_file_record["bytes"],
                "sha256": checkpoint_file_record["sha256"],
                **actual_checkpoint,
            },
            "split": {
                "train_ids": train_ids,
                "validation_ids": validation_ids,
                "held_out_test_ids": test_ids,
            },
            "file_count": len(file_records),
            "total_bytes": sum(record["bytes"] for record in file_records),
            "files": file_records,
        }
        (staging_dir / "bundle_manifest.json").write_text(
            json.dumps(bundle_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging_dir, output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "bundle_dir": str(output_dir),
                "bundle_sha256": bundle_manifest["bundle_sha256"],
                "file_count": bundle_manifest["file_count"],
                "total_bytes": bundle_manifest["total_bytes"],
                "total_gib": round(bundle_manifest["total_bytes"] / 1024**3, 3),
                "checkpoint": bundle_manifest["checkpoint"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
