"""Validate published paired-figure assets against their source and case metadata."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.recompute_public_metrics import sha256  # noqa: E402
from website.scripts.export_visual_assets import FIGURES, crop_id, read_json  # noqa: E402
from website.scripts.validate_public_crops import validate as validate_public_crops  # noqa: E402


def validate() -> dict:
    manifest_path = ROOT / "website/public/evidence/paired/manifest.json"
    manifest = read_json(manifest_path)
    benchmark = read_json(ROOT / "website/src/generated/evidence.json")["benchmark"]
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported visual evidence manifest schema")
    if manifest.get("dataset_id") != benchmark["dataset_id"]:
        raise ValueError("Visual evidence dataset identity does not match benchmark")
    if len(manifest.get("assets", [])) != len(FIGURES):
        raise ValueError("Visual evidence manifest is missing an allowlisted figure")

    cases = manifest.get("case_studies", [])
    if len(cases) != 20:
        raise ValueError(f"Expected 20 published case records, found {len(cases)}")
    expected_models = {
        name: {
            "model_id": benchmark["models"][name]["experiment_id"],
            "checkpoint_sha256": benchmark["models"][name]["checkpoint_sha256"],
            "preprocessing_protocol_sha256": benchmark["models"][name]["protocol_hash"],
        }
        for name in ("dinov3", "pspnet")
    }
    for asset in manifest["assets"]:
        source = ROOT / asset["source_artifact_path"]
        published = ROOT / "website/public" / asset["published_asset_path"]
        if not source.is_file() or not published.is_file():
            raise FileNotFoundError(
                f"Missing source or published figure: {asset['file']}"
            )
        if sha256(source) != asset["source_artifact_sha256"]:
            raise ValueError(f"Source figure digest mismatch: {asset['file']}")
        if sha256(published) != asset["published_asset_sha256"]:
            raise ValueError(f"Published figure digest mismatch: {asset['file']}")
        if sha256(source) != sha256(published):
            raise ValueError(
                f"Published figure is not byte-identical to its source: {asset['file']}"
            )
        related = [
            case for case in cases if case["figure_asset_id"] == asset["asset_id"]
        ]
        if len(related) != 4 or len(asset["case_ids"]) != 4:
            raise ValueError(f"Wrong number of case rows for {asset['file']}")

    for case in cases:
        if case["case_id"] != crop_id(
            {
                "scene": case["scene_id"],
                "row": case["pixel_window"]["row"],
                "col": case["pixel_window"]["col"],
            }
        ):
            raise ValueError(
                f"Case ID and pixel coordinates disagree: {case['case_id']}"
            )
        if case["models"] != expected_models:
            raise ValueError(
                f"Checkpoint or preprocessing binding differs for {case['case_id']}"
            )
        height, width = case["pixel_window"]["height"], case["pixel_window"]["width"]
        if (
            height <= 0
            or width <= 0
            or not 0 < case["pixel_window"]["valid_pixels"] <= height * width
        ):
            raise ValueError(
                f"Invalid crop geometry or valid support: {case['case_id']}"
            )
        if len(case["panels"]) != 5 or set(case["panels"]) != {
            "rgb",
            "reference",
            "dinov3",
            "pspnet",
            "paired_outcome",
        }:
            raise ValueError(f"Incomplete panel set: {case['case_id']}")
        if (
            sum(row[3] for row in case["transitions"])
            != case["pixel_window"]["valid_pixels"]
        ):
            raise ValueError(
                f"Joint counts do not match valid support: {case['case_id']}"
            )
        for truth, dino, psp, count in case["transitions"]:
            if not all(0 <= value < 6 for value in (truth, dino, psp)) or count < 0:
                raise ValueError(f"Invalid class ID or count: {case['case_id']}")

    manifest["validated_public_crops"] = validate_public_crops()
    return manifest


if __name__ == "__main__":
    result = validate()
    crops = result["validated_public_crops"]
    print(
        f"Validated {len(result['assets'])} figures, {len(result['case_studies'])} "
        f"archived cases, and {crops['assets']} pixel-aligned raster crops"
    )
