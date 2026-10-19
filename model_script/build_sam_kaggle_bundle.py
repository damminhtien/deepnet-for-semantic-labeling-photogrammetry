"""Stage a private, checksummed Potsdam IRRG bundle for the Kaggle SAM run."""

import argparse
import errno
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_BUNDLE = Path(
    "/Users/macbook/Data/runs/pspnet-cloud-bundle-20261019"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/macbook/Data/runs/sam-vitb-lora-kaggle-upload-20261019"
)
DATASET_ID = "damminhtien/sam-vitb-lora-potsdam-20261019"


def sha256_file(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


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


def git_revision():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, default=DEFAULT_SOURCE_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source_bundle = args.source_bundle.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    staging_dir = output_dir.with_name(output_dir.name + ".tmp")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_dir}")
    if staging_dir.exists():
        raise FileExistsError(f"Refusing to overwrite staging directory: {staging_dir}")

    source_manifest = json.loads(
        (source_bundle / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    preprocessing_manifest_path = source_bundle / "preprocessing_manifest.json"
    split = source_manifest["split"]
    train_ids = split["train_ids"]
    validation_ids = split["validation_ids"]
    test_ids = split["held_out_test_ids"]
    if (len(train_ids), len(validation_ids), len(test_ids)) != (18, 6, 14):
        raise ValueError("Expected the fixed 18/6/14 Potsdam split")

    staging_dir.mkdir(parents=True)
    bundle_root = staging_dir / "sam_potsdam_bundle"
    records = []
    try:
        split_scenes = {
            "train": (train_ids, "participant"),
            "validation": (validation_ids, "no_boundary"),
            "held_out": (test_ids, "no_boundary"),
        }
        for split_name, (scene_ids, label_source) in split_scenes.items():
            for scene_id in scene_ids:
                image_name = f"top_potsdam_{scene_id}_IRRG.tif"
                image_source = (
                    source_bundle
                    / "potsdamvaihingen"
                    / "3_ortho_irrg"
                    / "3_Ortho_IRRG"
                    / image_name
                )
                label_source_path = (
                    source_bundle
                    / "prepared_labels"
                    / label_source
                    / f"{scene_id}.npz"
                )
                destinations = (
                    (
                        image_source,
                        bundle_root / "images" / split_name / f"{scene_id}.tif",
                    ),
                    (
                        label_source_path,
                        bundle_root / "labels" / split_name / f"{scene_id}.npz",
                    ),
                )
                for source, destination in destinations:
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    link_or_copy(source, destination)
                    records.append(
                        {
                            "path": destination.relative_to(bundle_root).as_posix(),
                            "bytes": destination.stat().st_size,
                            "sha256": sha256_file(destination),
                        }
                    )

        preprocessing_path = bundle_root / "preprocessing_manifest.json"
        link_or_copy(preprocessing_manifest_path, preprocessing_path)
        records.append(
            {
                "path": preprocessing_path.relative_to(bundle_root).as_posix(),
                "bytes": preprocessing_path.stat().st_size,
                "sha256": sha256_file(preprocessing_path),
            }
        )
        records.sort(key=lambda record: record["path"])
        fingerprint_payload = json.dumps(
            records, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        manifest = {
            "schema_version": 1,
            "dataset_id": DATASET_ID,
            "source_pspnet_bundle_sha256": source_manifest["bundle_sha256"],
            "source_git_revision": git_revision(),
            "source_worktree_changes": source_manifest.get(
                "source_worktree_changes", []
            ),
            "bundle_sha256": hashlib.sha256(fingerprint_payload).hexdigest(),
            "split": split,
            "input": "three-channel Potsdam IRRG TIFF only; DSM and nDSM omitted",
            "classes": 6,
            "ignore_label": 6,
            "file_count": len(records),
            "total_bytes": sum(record["bytes"] for record in records),
            "files": records,
        }
        (bundle_root / "bundle_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging_dir / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": "Potsdam IRRG labels for SAM ViT-B LoRA",
                    "id": DATASET_ID,
                    "licenses": [{"name": "other"}],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging_dir, output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "dataset_dir": str(output_dir),
                "dataset_id": DATASET_ID,
                "bundle_sha256": manifest["bundle_sha256"],
                "files": manifest["file_count"],
                "total_bytes": manifest["total_bytes"],
                "total_gib": round(manifest["total_bytes"] / 1024**3, 3),
                "split": {
                    "train_scenes": len(train_ids),
                    "validation_scenes": len(validation_ids),
                    "held_out_scenes": len(test_ids),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
