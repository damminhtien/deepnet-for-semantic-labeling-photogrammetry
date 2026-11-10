"""Export a bounded set of verified paired Potsdam crops for the static Atlas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deepnet.data.preprocessing import decode_rgb_labels  # noqa: E402
from deepnet.potsdam_rgbir.data import CacheBundle  # noqa: E402
from scripts.recompute_public_metrics import sha256  # noqa: E402
from scripts.research.paired_analysis import (  # noqa: E402
    load_prediction,
    verify_inputs,
)
from website.scripts.export_evidence import export as export_evidence  # noqa: E402
from website.scripts.export_visual_assets import export as export_visual_assets  # noqa: E402

RGBIR_ROOT = ROOT / "data/auxiliary/potsdam-rgbir-v2"
LABEL_ROOT = (
    ROOT / "data/raw/inputs/label-all-potsdam-vaihingen/5_labels_all_noboundary"
)
OUTPUT_ROOT = ROOT / "website/public/evidence/scenes"
MANIFEST_ROOT = ROOT / "website/src/generated/public-crops.json"
MODELS = ("dinov3", "pspnet")
CLASS_COUNT = 6
RIGHTS_NOTE = (
    "The project author has approved publication of these selected research excerpts. "
    "The underlying ISPRS Potsdam imagery and labels remain third-party material; this "
    "Atlas does not grant rights to redistribute the original dataset."
)
RIGHTS_URL = (
    "https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/"
    "2d-sem-label-potsdam.aspx"
)
RIGHTS_STATUS = "published_with_project_author_approval; upstream_terms_apply"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_png(path: Path, data: np.ndarray, *, colorinterp: tuple[str, ...]) -> None:
    height, width = data.shape[-2:]
    with rasterio.open(
        path,
        "w",
        driver="PNG",
        width=width,
        height=height,
        count=data.shape[0],
        dtype="uint8",
        interleave="pixel",
        compress="deflate",
    ) as output:
        output.write(data.astype(np.uint8, copy=False))
        output.colorinterp = tuple(
            getattr(rasterio.enums.ColorInterp, name) for name in colorinterp
        )


def raster_digest(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} checksum differs from its audited manifest: {path}")


def export_public_crops(
    *,
    cache_root: Path,
    output_root: Path,
    manifest_path: Path,
    paired_output: Path,
    runs_root: Path,
    rgbir_root: Path,
    labels_root: Path,
) -> dict[str, Any]:
    paired_output = paired_output.resolve()
    export_evidence()
    visual = export_visual_assets()
    evidence = read_json(ROOT / "website/src/generated/evidence.json")
    benchmark = evidence["benchmark"]
    spectral = read_json(ROOT / "artifacts/public/data/potsdam-rgbir-manifest.json")
    cache = CacheBundle.load(cache_root)
    _, paired, _, indexed = verify_inputs(paired_output, cache, runs_root)
    expected_checkpoints = {
        name: benchmark["models"][name]["checkpoint_sha256"] for name in MODELS
    }
    if paired["split_hash"] != benchmark["split_sha256"] or any(
        paired["models"][name]["checkpoint_sha256"] != expected_checkpoints[name]
        or paired["models"][name]["archived_metrics_sha256"]
        != benchmark["models"][name]["source_metrics_sha256"]
        for name in MODELS
    ):
        raise ValueError(
            "Paired inference does not match published benchmark provenance"
        )

    output_root = output_root.resolve()
    if not output_root.is_relative_to(
        (ROOT / "website/public/evidence/scenes").resolve()
    ):
        raise ValueError(
            "Public raster crops must stay under website/public/evidence/scenes"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = {
        name: {
            "rgbir_sha256": record["sha256"],
            "shape": record["shape"],
            "crs": record["crs"],
            "affine": record["affine"],
            "extent": record["extent"],
        }
        for name, record in spectral["scenes"].items()
    }
    paired_records = {
        name: {item["scene_id"]: item for item in paired["models"][name]["scenes"]}
        for name in MODELS
    }
    case_records = visual["case_studies"]
    exported: list[dict[str, Any]] = []
    verified_files: dict[str, str] = {}

    with cache.access("test", auxiliary=False) as access:
        open_scene: str | None = None
        source_rgbir = None
        source_labels = None
        predictions: dict[str, np.ndarray] = {}
        source_label_hash = ""

        for case in case_records:
            scene = case["scene_id"]
            crop = case["pixel_window"]
            row, col = crop["row"], crop["col"]
            height, width = crop["height"], crop["width"]
            if scene != open_scene:
                if source_rgbir is not None:
                    source_rgbir.close()
                    source_labels.close()
                record = scenes.get(scene)
                if record is None:
                    raise ValueError(
                        f"Scene is missing from the audited RGBIR manifest: {scene}"
                    )
                rgbir_path = (
                    rgbir_root / "4_Ortho_RGBIR" / f"top_potsdam_{scene}_RGBIR.tif"
                )
                label_path = labels_root / f"top_potsdam_{scene}_label_noBoundary.tif"
                raster_digest(rgbir_path, record["rgbir_sha256"], "RGBIR scene")
                if not label_path.is_file():
                    raise FileNotFoundError(f"Missing eroded label scene: {label_path}")
                source_label_hash = sha256(label_path)
                source_rgbir = rasterio.open(rgbir_path)
                source_labels = rasterio.open(label_path)
                if (
                    [source_rgbir.height, source_rgbir.width] != record["shape"]
                    or source_rgbir.count != 4
                    or source_rgbir.dtypes != ("uint8",) * 4
                    or source_labels.count != 3
                    or (source_labels.height, source_labels.width)
                    != (source_rgbir.height, source_rgbir.width)
                ):
                    raise ValueError(f"Unexpected source grid or channels for {scene}")
                arrays = access._arrays(scene)
                if tuple(arrays["labels"].shape) != tuple(record["shape"]):
                    raise ValueError(
                        f"Held-out cache grid differs from source for {scene}"
                    )
                predictions = {}
                for model in MODELS:
                    model_record = paired_records[model][scene]
                    relative = "paired/" + model_record["prediction_path"]
                    predictions[model] = load_prediction(
                        paired_output / relative,
                        model_record["prediction_sha256"],
                        indexed[relative],
                    )
                    if predictions[model].shape != tuple(record["shape"]):
                        raise ValueError(
                            f"Prediction grid differs from source for {scene}"
                        )
                open_scene = scene

            window = Window(col, row, width, height)
            band_stack = source_rgbir.read(window=window)
            cached_rgbir = access._arrays(scene)["optical"][
                :,
                row : row + height,
                col : col + width,
            ]
            if band_stack.shape != (4, height, width) or not np.array_equal(
                band_stack, cached_rgbir
            ):
                raise ValueError(
                    f"Source RGBIR values do not match the verified cache: {scene}"
                )

            raw_rgb_labels = np.moveaxis(source_labels.read(window=window), 0, -1)
            truth = decode_rgb_labels(raw_rgb_labels, cache.preprocessing)
            cached_truth = access._arrays(scene)["labels"][
                row : row + height, col : col + width
            ]
            valid = access._arrays(scene)["valid"][
                row : row + height, col : col + width
            ].astype(bool)
            scored = valid & (truth < CLASS_COUNT)
            if not np.array_equal(truth, cached_truth):
                raise ValueError(
                    f"Decoded source labels do not match the verified cache: {scene}"
                )

            model_planes = {
                name: predictions[name][row : row + height, col : col + width]
                for name in MODELS
            }
            if any(
                plane.shape != (height, width) or np.any(plane[scored] >= CLASS_COUNT)
                for plane in model_planes.values()
            ):
                raise ValueError(f"Invalid prediction class IDs in {scene}")
            transition_index = (
                truth[scored].astype(np.int64) * CLASS_COUNT
                + model_planes["dinov3"][scored].astype(np.int64)
            ) * CLASS_COUNT + model_planes["pspnet"][scored].astype(np.int64)
            actual = np.bincount(transition_index, minlength=CLASS_COUNT**3)
            expected = np.zeros(CLASS_COUNT**3, dtype=np.int64)
            for truth_id, dino_id, psp_id, count in case["transitions"]:
                expected[(truth_id * CLASS_COUNT + dino_id) * CLASS_COUNT + psp_id] = (
                    count
                )
            if not np.array_equal(actual, expected):
                raise ValueError(
                    "Exported crop does not reproduce joint class counts: "
                    f"{case['case_id']}"
                )

            asset_stem = case["case_id"]
            rgb_path = output_root / f"{asset_stem}-rgb.png"
            ir_path = output_root / f"{asset_stem}-ir.png"
            ids_path = output_root / f"{asset_stem}-ids.png"
            write_png(rgb_path, band_stack[:3], colorinterp=("red", "green", "blue"))
            write_png(ir_path, band_stack[3:4], colorinterp=("gray",))
            ids = np.stack(
                [truth, model_planes["dinov3"], model_planes["pspnet"]]
            ).astype(np.uint8)
            ids[:3, ~scored] = cache.preprocessing.ignore_label
            write_png(ids_path, ids, colorinterp=("red", "green", "blue"))

            file_records = {}
            for kind, path in (
                ("rgb", rgb_path),
                ("infrared", ir_path),
                ("class_ids", ids_path),
            ):
                digest = sha256(path)
                relative = path.relative_to(ROOT / "website/public").as_posix()
                file_records[kind] = {
                    "url": relative,
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                }
                verified_files[relative] = digest

            transform = record["affine"]
            x0 = transform[0] * col + transform[1] * row + transform[2]
            y0 = transform[3] * col + transform[4] * row + transform[5]
            x1 = (
                transform[0] * (col + width)
                + transform[1] * (row + height)
                + transform[2]
            )
            y1 = (
                transform[3] * (col + width)
                + transform[4] * (row + height)
                + transform[5]
            )
            prediction_sources = {}
            class_palette = cache.preprocessing.label_palette
            for model in MODELS:
                relative = "paired/" + paired_records[model][scene]["prediction_path"]
                prediction_sources[model] = {
                    "path": (paired_output / relative).relative_to(ROOT).as_posix(),
                    "sha256": paired_records[model][scene]["prediction_sha256"],
                }
            exported.append(
                {
                    **case,
                    "asset_type": "pixel_aligned_potsdam_crop",
                    "files": file_records,
                    "ground_sampling_distance_m": 0.05,
                    "channel_mapping": ["red", "green", "blue", "infrared"],
                    "source_grid": {
                        "crs": record["crs"],
                        "affine": record["affine"],
                        "extent": record["extent"],
                        "shape": record["shape"],
                    },
                    "pixel_window_bounds": [
                        min(x0, x1),
                        min(y0, y1),
                        max(x0, x1),
                        max(y0, y1),
                    ],
                    "preprocessing": {
                        "id": cache.preprocessing.preprocessing_id,
                        "version": cache.preprocessing.version,
                        "ignore_label": cache.preprocessing.ignore_label,
                    },
                    "class_colors": [
                        [red, green, blue]
                        for red, green, blue, class_id in class_palette
                        if class_id < CLASS_COUNT
                    ],
                    "source_artifacts": {
                        "rgbir": {
                            "path": rgbir_path.relative_to(ROOT).as_posix(),
                            "sha256": record["rgbir_sha256"],
                        },
                        "eroded_reference": {
                            "path": label_path.relative_to(ROOT).as_posix(),
                            "sha256": source_label_hash,
                            "protocol": benchmark["label_protocol"],
                        },
                        "paired_predictions": prediction_sources,
                        "cache_sha256": cache.manifest["cache_hash"],
                    },
                    "models": {
                        name: {
                            "model_id": benchmark["models"][name]["experiment_id"],
                            "checkpoint_sha256": expected_checkpoints[name],
                            "preprocessing_protocol_sha256": benchmark["models"][name][
                                "protocol_hash"
                            ],
                        }
                        for name in MODELS
                    },
                    "reference_label_protocol": benchmark["label_protocol"],
                    "evaluation_status": {
                        "analysis_kind": paired["analysis_kind"],
                        "training_performed": paired["training_performed"],
                        "independent_benchmark_claim": paired[
                            "independent_benchmark_claim"
                        ],
                        "split_sha256": paired["split_hash"],
                    },
                    "distribution_rights_status": RIGHTS_STATUS,
                    "rights_note": RIGHTS_NOTE,
                    "rights_url": RIGHTS_URL,
                }
            )

        if source_rgbir is not None:
            source_rgbir.close()
            source_labels.close()

    manifest = {
        "schema_version": 1,
        "dataset_id": benchmark["dataset_id"],
        "class_names": benchmark["class_names"],
        "ignore_class_id": cache.preprocessing.ignore_label,
        "cache_sha256": cache.manifest["cache_hash"],
        "split_sha256": paired["split_hash"],
        "models": {
            name: {
                "model_id": benchmark["models"][name]["experiment_id"],
                "checkpoint_sha256": expected_checkpoints[name],
                "preprocessing_protocol_sha256": benchmark["models"][name][
                    "protocol_hash"
                ],
            }
            for name in MODELS
        },
        "assets": exported,
        "published_files": verified_files,
        "distribution_rights_status": RIGHTS_STATUS,
        "rights_note": RIGHTS_NOTE,
        "rights_url": RIGHTS_URL,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    total_bytes = sum(
        path.stat().st_size for path in output_root.iterdir() if path.is_file()
    )
    print(
        f"Exported {len(exported)} count-verified raster crops "
        f"({total_bytes / 1e6:.1f} MB)"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--paired-output", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--rgbir-root", type=Path, default=RGBIR_ROOT)
    parser.add_argument("--labels-root", type=Path, default=LABEL_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_ROOT)
    args = parser.parse_args()
    export_public_crops(
        cache_root=args.cache,
        output_root=args.output,
        manifest_path=args.manifest,
        paired_output=args.paired_output,
        runs_root=args.runs_root,
        rgbir_root=args.rgbir_root,
        labels_root=args.labels_root,
    )


if __name__ == "__main__":
    main()
