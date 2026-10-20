"""Stage a private Kaggle dataset containing the SAM model's source modules."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(
    "/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/"
    "runs/sam-vitb-lora-source-20261020"
)
DEFAULT_DATASET_ID = "damminhtien/sam-vitb-lora-source-20261020"
SOURCE_FILES = (
    Path("semantic_labeling/__init__.py"),
    Path("semantic_labeling/constants.py"),
    Path("semantic_labeling/sam_lora.py"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    staging_dir = output_dir.with_name(output_dir.name + ".tmp")
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Kaggle source dataset: {output_dir}")

    staging_dir.mkdir(parents=True)
    try:
        records = []
        for relative_path in SOURCE_FILES:
            source = PROJECT_ROOT / relative_path
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = staging_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            records.append(
                {
                    "path": relative_path.as_posix(),
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )

        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(UTC).isoformat(),
            "dataset_id": args.dataset_id,
            "source_git_revision": source_revision(),
            "files": records,
        }
        manifest["source_sha256"] = hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        (staging_dir / "source_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging_dir / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": "SAM ViT-B LoRA Potsdam source modules",
                    "id": args.dataset_id,
                    "licenses": [{"name": "other"}],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "dataset_dir": str(output_dir),
                "dataset_id": args.dataset_id,
                "source_sha256": manifest["source_sha256"],
                "files": len(records),
                "total_bytes": sum(record["bytes"] for record in records),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
