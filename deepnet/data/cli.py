"""Command-line entry points for catalog creation and data validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bundles import export_model_bundle, verify_bundle
from .catalog import (
    DEFAULT_PREPROCESSING_CONFIG_DIR,
    DatasetCatalog,
    build_scene_catalog,
    load_model_input_spec,
    load_preprocessing_spec,
    load_scene_catalog,
    load_split_manifest,
    validate_scene_catalog,
    write_scene_catalog,
)
from .preprocessing import fit_model_statistics
from .quality import validate_data_quality


def _asset_groups(values: list[str]) -> dict[str, tuple[str, ...]]:
    result = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError("Asset groups must use NAME=ASSET,ASSET")
        name, assets = value.split("=", 1)
        asset_names = tuple(item.strip() for item in assets.split(",") if item.strip())
        if not name.strip() or not asset_names:
            raise argparse.ArgumentTypeError("Asset group name and assets are required")
        result[name.strip()] = asset_names
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build-catalog", help="inspect source rasters and write a scene catalog"
    )
    build.add_argument("--dataset-id", required=True)
    build.add_argument("--data-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--dataset-config-dir", type=Path)
    build.add_argument(
        "--hash-assets",
        action="store_true",
        help="stream SHA-256 for each source raster during ingestion",
    )

    validate = subparsers.add_parser(
        "validate-catalog", help="check catalog/split contracts"
    )
    validate.add_argument("--catalog", type=Path, required=True)
    validate.add_argument("--split", type=Path)
    validate.add_argument("--require-hashes", action="store_true")
    validate.add_argument(
        "--view-assets",
        action="append",
        default=[],
        metavar="NAME=ASSET,ASSET",
    )

    quality = subparsers.add_parser(
        "quality-report", help="scan labels and alignment by windows"
    )
    quality.add_argument("--catalog", type=Path, required=True)
    quality.add_argument("--split", type=Path, required=True)
    quality.add_argument("--preprocessing-id", required=True)
    quality.add_argument("--preprocessing-dir", type=Path)
    quality.add_argument("--data-root", type=Path, required=True)
    quality.add_argument("--output", type=Path, required=True)
    quality.add_argument("--tile-size", type=int, default=1024)
    quality.add_argument(
        "--view-assets",
        action="append",
        required=True,
        metavar="NAME=ASSET,ASSET",
    )

    statistics = subparsers.add_parser(
        "fit-statistics",
        help="fit model-view normalization from train-scene windows only",
    )
    statistics.add_argument("--catalog", type=Path, required=True)
    statistics.add_argument("--split", type=Path, required=True)
    statistics.add_argument("--preprocessing-id", required=True)
    statistics.add_argument("--preprocessing-dir", type=Path)
    statistics.add_argument("--model-input-id", required=True)
    statistics.add_argument("--data-root", type=Path, required=True)
    statistics.add_argument("--output", type=Path, required=True)
    statistics.add_argument("--tile-size", type=int, default=1024)

    export = subparsers.add_parser(
        "export-bundle", help="export one portable model-view bundle"
    )
    export.add_argument("--catalog", type=Path, required=True)
    export.add_argument("--split", type=Path, required=True)
    export.add_argument("--preprocessing-id", required=True)
    export.add_argument("--preprocessing-dir", type=Path)
    export.add_argument("--model-input-id", required=True)
    export.add_argument("--data-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser(
        "verify-bundle", help="verify a portable bundle's hashes"
    )
    verify.add_argument("--bundle", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify-bundle":
        manifest = verify_bundle(args.bundle)
        print(
            json.dumps(
                {
                    "dataset_id": manifest["dataset_id"],
                    "bundle_sha256": manifest["bundle_sha256"],
                    "file_count": manifest["file_count"],
                    "status": "verified",
                },
                indent=2,
            )
        )
        return 0

    if args.command == "build-catalog":
        config = (
            DatasetCatalog.load(args.dataset_config_dir)
            if args.dataset_config_dir
            else DatasetCatalog.load()
        )
        result = build_scene_catalog(
            args.dataset_id,
            args.data_root,
            dataset_catalog=config,
            hash_assets=args.hash_assets,
        )
        write_scene_catalog(result, args.output)
        print(
            json.dumps(
                {
                    "dataset_id": result["dataset_id"],
                    "scene_count": result["scene_count"],
                    "hash_status": result["hash_status"],
                    "catalog_content_hash": result["catalog_content_hash"],
                    "source_version": result["source_version"],
                    "missing_asset_scenes": len(result["missing_assets_by_scene"]),
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "export-bundle":
        scenes = load_scene_catalog(args.catalog)
        split = load_split_manifest(args.split)
        preprocessing = load_preprocessing_spec(
            args.preprocessing_id,
            args.preprocessing_dir or DEFAULT_PREPROCESSING_CONFIG_DIR,
        )
        model_input = load_model_input_spec(args.model_input_id)
        manifest = export_model_bundle(
            scenes,
            args.data_root,
            split,
            preprocessing,
            model_input,
            args.output,
        )
        print(
            json.dumps(
                {
                    "dataset_id": manifest["dataset_id"],
                    "model_input_id": manifest["model_input_id"],
                    "bundle_sha256": manifest["bundle_sha256"],
                    "file_count": manifest["file_count"],
                    "total_bytes": manifest["total_bytes"],
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "fit-statistics":
        if args.output.exists():
            raise FileExistsError(
                f"Refusing to overwrite preprocessing artifact: {args.output}"
            )
        scenes = load_scene_catalog(args.catalog)
        split = load_split_manifest(args.split)
        preprocessing = load_preprocessing_spec(
            args.preprocessing_id,
            args.preprocessing_dir or DEFAULT_PREPROCESSING_CONFIG_DIR,
        )
        model_input = load_model_input_spec(args.model_input_id)
        fitted = fit_model_statistics(
            scenes,
            args.data_root,
            split,
            preprocessing,
            model_input,
            tile_size=args.tile_size,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(fitted.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "preprocessing_id": fitted.preprocessing_id,
                    "preprocessing_version": fitted.version,
                    "preprocessing_hash": fitted.content_hash,
                    "fit_scene_count": len(fitted.fit_scene_ids),
                    "fit_scene_ids": list(fitted.fit_scene_ids),
                    "channels": [channel for channel, _, _ in fitted.fitted_statistics],
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 0

    scenes = load_scene_catalog(args.catalog)
    split = load_split_manifest(args.split) if args.split else None
    if args.command == "validate-catalog":
        asset_groups = _asset_groups(args.view_assets)
        validate_scene_catalog(
            scenes,
            split=split,
            require_hashes=args.require_hashes,
        )
        alignment_results = {}
        for name, asset_names in asset_groups.items():
            for scene in scenes:
                if split is not None and scene.scene_id not in split.scene_ids:
                    continue
                label_asset = (
                    split.label_asset_for(
                        next(
                            split_name
                            for split_name in ("train", "validation", "test")
                            if scene.scene_id in split.scenes_for(split_name)
                        )
                    )
                    if split is not None
                    else None
                )
                required = asset_names + ((label_asset,) if label_asset else ())
                scene.require_aligned(tuple(dict.fromkeys(required)))
            alignment_results[name] = "passed"
        print(
            json.dumps(
                {
                    "dataset_id": scenes[0].dataset_id,
                    "scene_count": len(scenes),
                    "split_id": split.split_id if split else None,
                    "split_hash": split.content_hash if split else None,
                    "alignment_checks": alignment_results,
                    "result": "valid",
                },
                indent=2,
            )
        )
        return 0

    preprocessing = load_preprocessing_spec(
        args.preprocessing_id,
        args.preprocessing_dir or DEFAULT_PREPROCESSING_CONFIG_DIR,
    )
    report = validate_data_quality(
        scenes,
        args.data_root,
        split,
        preprocessing,
        _asset_groups(args.view_assets),
        tile_size=args.tile_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "dataset_id": report["dataset_id"],
                "split_hash": report["split_hash"],
                "output": str(args.output),
                "status": "passed",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
