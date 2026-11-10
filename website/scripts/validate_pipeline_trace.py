"""Validate the bounded, public training-trace assets without research inputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.recompute_public_metrics import sha256  # noqa: E402
from website.scripts.export_visual_assets import png_size  # noqa: E402

EXPECTED_STAGES = (
    "source",
    "preprocessing",
    "sampling",
    "augmentation",
    "model-input",
    "activations",
    "prediction",
    "evaluation",
)
EXPECTED_BLOCKS = (3, 6, 9, 12)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate() -> dict:
    public_root = ROOT / "website/public/evidence/pipeline-trace"
    manifest = read_json(public_root / "manifest.json")
    generated = read_json(ROOT / "website/src/generated/pipeline-trace.json")
    if generated != manifest:
        raise ValueError("Generated pipeline trace differs from the public manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported pipeline trace manifest schema")
    if manifest.get("dataset_id") != "isprs-potsdam":
        raise ValueError("Pipeline trace dataset identity differs")
    if manifest.get("publication", {}).get("project_owner_authorized") is not True:
        raise ValueError("Pipeline trace publication authorization is missing")

    sample = manifest["sample"]
    window = sample["window"]
    if (
        sample.get("split_role") != "final_refit_training"
        or sample.get("used_for_model_training") is not True
        or sample.get("ignore_id") != 6
        or window.get("height") != 512
        or window.get("width") != 512
        or not sample.get("scene_id")
    ):
        raise ValueError("Pipeline trace sample role or window is invalid")

    stages = manifest.get("stages", [])
    if tuple(stage.get("id") for stage in stages) != EXPECTED_STAGES:
        raise ValueError("Pipeline trace stage order is incomplete")
    assets = manifest.get("assets", {})
    referenced = {key for stage in stages for key in stage.get("assets", [])}
    if referenced != set(assets):
        raise ValueError("Pipeline stages and asset manifest do not match")

    allowed_extensions = {".png"}
    for _key, record in assets.items():
        name = record.get("file", "")
        relative = Path(name)
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.suffix.lower() not in allowed_extensions
            or relative.name != name
        ):
            raise ValueError(f"Unsafe pipeline asset path: {name}")
        path = public_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing pipeline trace asset: {name}")
        if sha256(path) != record.get("sha256") or path.stat().st_size != record.get(
            "bytes"
        ):
            raise ValueError(f"Pipeline trace asset digest/size mismatch: {name}")
        if png_size(path) != (record.get("width_px"), record.get("height_px")):
            raise ValueError(f"Pipeline trace PNG dimensions differ: {name}")
        if record.get("width_px") not in {512, 1024} or record.get(
            "height_px"
        ) != record.get("width_px"):
            raise ValueError(f"Unexpected pipeline trace asset dimensions: {name}")

    if assets["sampling_context"]["width_px"] != 1024:
        raise ValueError("Sampling context must remain a bounded 1024-pixel image")
    if any(
        assets[f"feature_block_{block}"]["width_px"] != 512 for block in EXPECTED_BLOCKS
    ):
        raise ValueError(
            "Activation assets must use the documented 512-pixel display size"
        )

    model = manifest["model"]
    if model.get("feature_blocks") != list(EXPECTED_BLOCKS):
        raise ValueError("Pipeline trace does not show the configured feature blocks")
    activations = model.get("activations", [])
    if tuple(item.get("block") for item in activations) != EXPECTED_BLOCKS:
        raise ValueError("Activation provenance is incomplete")
    for item in activations:
        if (
            item.get("tensor_shape") != [1, 768, 32, 32]
            or item.get("patch_grid") != [32, 32]
            or item.get("prefix_tokens_removed") != 5
            or "PCA" not in item.get("projection", "")
        ):
            raise ValueError(
                f"Invalid activation provenance for block {item.get('block')}"
            )

    matrix = manifest["sample_confusion"][
        "confusion_matrix_reference_rows_prediction_columns"
    ]
    if len(matrix) != 6 or any(len(row) != 6 for row in matrix):
        raise ValueError("Training sample confusion matrix must be six by six")
    scored = sum(sum(row) for row in matrix)
    errors = sum(
        matrix[row][column] for row in range(6) for column in range(6) if row != column
    )
    if (
        scored != manifest["sample_confusion"]["scored_pixels"]
        or errors != manifest["sample_confusion"]["error_pixels"]
        or scored + manifest["sample_confusion"]["ignored_or_invalid_pixels_excluded"]
        != 512 * 512
        or manifest["sample_confusion"]["scope"] != "single_refit_training_draw_only"
    ):
        raise ValueError(
            "Training sample error counts do not reconcile or overstate scope"
        )

    forbidden = (".npy", ".pt", ".safetensors", "/Users/", "/home/")
    if any(token in json.dumps(manifest) for token in forbidden):
        raise ValueError(
            "Pipeline trace contains a raw array, checkpoint, or local path"
        )
    return {
        "assets": len(assets),
        "stages": len(stages),
        "sample_id": sample["sample_id"],
    }


if __name__ == "__main__":
    result = validate()
    print(
        f"Validated {result['assets']} real pipeline assets across "
        f"{result['stages']} stages for {result['sample_id']}"
    )
