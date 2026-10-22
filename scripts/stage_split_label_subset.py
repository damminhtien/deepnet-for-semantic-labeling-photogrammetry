"""Copy one manifest-selected label role into an external DVC staging folder."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from deepnet.data.catalog import (
    DatasetCatalog,
    load_split_manifest,
    scene_format_values,
)
from deepnet.data.raster import sha256_file


def _safe_source_file(root: Path, uri: str) -> Path:
    """Resolve a descriptor path without allowing it to escape its data root."""
    relative = PurePosixPath(uri)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe dataset asset path: {uri}")
    path = (root / Path(*relative.parts)).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"Dataset asset path escapes source root: {uri}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def stage_label_subset(
    dataset_id: str,
    asset_name: str,
    split_path: Path,
    split_role: str,
    data_root: Path,
    output_dir: Path,
    *,
    dataset_config_dir: Path | None = None,
) -> dict[str, Any]:
    """Stage only the label files assigned to one split role.

    The source dataset remains read-only. Existing staged files are accepted only
    when their bytes match; unexpected or changed files fail instead of overwriting.
    """
    catalog = DatasetCatalog.load(
        dataset_config_dir
        if dataset_config_dir is not None
        else Path(__file__).resolve().parents[1] / "configs" / "datasets"
    )
    descriptor = catalog.get(dataset_id)
    split = load_split_manifest(split_path)
    if split.dataset_id != dataset_id:
        raise ValueError(
            f"Split dataset {split.dataset_id!r} does not match {dataset_id!r}"
        )
    if split.label_asset_for(split_role) != asset_name:
        raise ValueError(
            f"Split role {split_role!r} uses "
            f"{split.label_asset_for(split_role)!r}, not {asset_name!r}"
        )
    try:
        asset_config = descriptor.asset_configs()[asset_name]
    except KeyError as error:
        raise ValueError(
            f"Dataset {dataset_id!r} has no asset {asset_name!r}"
        ) from error

    source_root = Path(data_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if (
        destination == source_root
        or destination.is_relative_to(source_root)
        or source_root.is_relative_to(destination)
    ):
        raise ValueError("Source data and staging output must not overlap")
    if destination.is_symlink():
        raise ValueError("Staging output must not be a symbolic link")

    destination.mkdir(parents=True, exist_ok=True)
    expected_names: set[str] = set()
    records = []
    for scene_id in split.scenes_for(split_role):
        uri = asset_config["path"].format(**scene_format_values(scene_id))
        source = _safe_source_file(source_root, uri)
        target = destination / source.name
        expected_names.add(target.name)
        source_hash = sha256_file(source)
        if target.exists():
            if not target.is_file() or sha256_file(target) != source_hash:
                raise RuntimeError(f"Staged file differs from source: {target}")
        else:
            shutil.copy2(source, target)
        records.append(
            {
                "scene_id": scene_id,
                "source": str(source),
                "path": target.name,
                "bytes": target.stat().st_size,
                "sha256": source_hash,
            }
        )

    actual_names = {path.name for path in destination.iterdir() if path.is_file()}
    if actual_names != expected_names:
        extra = sorted(actual_names - expected_names)
        missing = sorted(expected_names - actual_names)
        raise RuntimeError(
            f"Staging directory contents do not match the split; "
            f"extra={extra}, missing={missing}"
        )
    return {
        "dataset_id": dataset_id,
        "split_id": split.split_id,
        "split_role": split_role,
        "label_asset": asset_name,
        "scene_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "files": records,
    }


def main() -> int:
    """Parse arguments, stage a subset, and print its checksummed inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--asset-name", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument(
        "--split-role", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-config-dir", type=Path)
    args = parser.parse_args()
    result = stage_label_subset(
        args.dataset_id,
        args.asset_name,
        args.split,
        args.split_role,
        args.data_root,
        args.output_dir,
        dataset_config_dir=args.dataset_config_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
