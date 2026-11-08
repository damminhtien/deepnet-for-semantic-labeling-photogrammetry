"""Recompute published Potsdam metrics from public confusion matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "artifacts/public/evidence-manifest.json"
RESULTS = ROOT / "docs/results.md"
START = "<!-- BEGIN GENERATED METRICS -->"
END = "<!-- END GENERATED METRICS -->"
EXPECTED_CLASSES = [
    "Impervious surfaces",
    "Building",
    "Low vegetation",
    "Tree",
    "Car",
    "Clutter/background",
]
REQUIRED_CLAIM_FIELDS = {
    "claim_id",
    "statement",
    "experiment_id",
    "dataset_split",
    "model_config",
    "metrics_artifact",
    "protocol_version",
    "source_revision",
    "sha256",
    "limitations",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_file(relative_path: str) -> Path:
    path = (ROOT / relative_path).resolve()
    if ROOT not in path.parents:
        raise ValueError(f"artifact path escapes the repository: {relative_path}")
    return path


def validate_matrix(matrix: Any, label: str) -> list[list[int]]:
    if not isinstance(matrix, list) or len(matrix) != len(EXPECTED_CLASSES):
        raise ValueError(f"{label}: expected a 6x6 confusion matrix")
    normalized: list[list[int]] = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != len(EXPECTED_CLASSES):
            raise ValueError(f"{label}: expected a 6x6 confusion matrix")
        if any(not isinstance(value, int) or value < 0 for value in row):
            raise ValueError(f"{label}: matrix entries must be non-negative integers")
        normalized.append(row)
    return normalized


def add_matrices(matrices: list[list[list[int]]]) -> list[list[int]]:
    total = [[0] * len(EXPECTED_CLASSES) for _ in EXPECTED_CLASSES]
    for matrix in matrices:
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                total[row_index][column_index] += value
    return total


def metrics(matrix: list[list[int]]) -> dict[str, Any]:
    total = sum(sum(row) for row in matrix)
    if total <= 0:
        raise ValueError("confusion matrix has no scored pixels")
    diagonal = [matrix[index][index] for index in range(len(matrix))]
    true_support = [sum(row) for row in matrix]
    predicted_support = [
        sum(matrix[row][column] for row in range(len(matrix)))
        for column in range(len(matrix))
    ]
    ious: list[float] = []
    f1s: list[float] = []
    for index, correct in enumerate(diagonal):
        union = true_support[index] + predicted_support[index] - correct
        denominator = true_support[index] + predicted_support[index]
        ious.append(correct / union if union else 0.0)
        f1s.append(2 * correct / denominator if denominator else 0.0)
    return {
        "pixels": total,
        "correct": sum(diagonal),
        "errors": total - sum(diagonal),
        "oa": sum(diagonal) / total,
        "miou": sum(ious) / len(ious),
        "iou": ious,
        "f1": f1s,
    }


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0, abs_tol=1e-12)


def load_evidence() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = json.loads(REGISTRY.read_text(encoding="utf-8"))
    artifacts = {item["path"]: item for item in manifest.get("artifacts", [])}
    if not artifacts:
        raise ValueError("evidence manifest has no artifact catalog")
    for relative_path, item in artifacts.items():
        path = repo_file(relative_path)
        if not path.is_file():
            raise ValueError(f"missing public artifact: {relative_path}")
        actual = sha256(path)
        if actual != item.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {relative_path}: {actual}")

    claims = manifest.get("claims", [])
    claim_ids: set[str] = set()
    for claim in claims:
        missing = REQUIRED_CLAIM_FIELDS - claim.keys()
        if missing:
            raise ValueError(
                f"{claim.get('claim_id', 'claim')}: missing fields {sorted(missing)}"
            )
        if claim["claim_id"] in claim_ids:
            raise ValueError(f"duplicate claim ID: {claim['claim_id']}")
        claim_ids.add(claim["claim_id"])
        artifact = artifacts.get(claim["metrics_artifact"])
        if artifact is None or claim["sha256"] != artifact["sha256"]:
            raise ValueError(
                f"{claim['claim_id']}: metrics artifact/hash not in registry"
            )
        for evidence_path in claim.get("additional_evidence_artifacts", []):
            if evidence_path not in artifacts:
                raise ValueError(
                    f"{claim['claim_id']}: additional evidence artifact "
                    f"is not registered: {evidence_path}"
                )
        for config_path in claim["model_config"]:
            if not repo_file(config_path).is_file():
                raise ValueError(
                    f"{claim['claim_id']}: missing model/config reference {config_path}"
                )

    metrics_path = manifest.get("benchmark_metrics_artifact")
    if metrics_path not in artifacts:
        raise ValueError(
            "benchmark_metrics_artifact must name a registered matrix artifact"
        )
    bundle = json.loads(repo_file(metrics_path).read_text(encoding="utf-8"))
    if bundle.get("class_names") != EXPECTED_CLASSES:
        raise ValueError(
            "public bundle class names/order differ from the Potsdam six-class contract"
        )

    scene_ids = bundle.get("scene_ids", [])
    if len(scene_ids) != 14 or len(set(scene_ids)) != 14:
        raise ValueError("expected exactly 14 unique benchmark scene IDs")
    model_results: dict[str, dict[str, Any]] = {}
    for model_id, model in bundle.get("models", {}).items():
        if model.get("scene_ids") != scene_ids:
            raise ValueError(
                f"{model_id}: scene order differs from the benchmark contract"
            )
        model_split_hash = model.get("split_sha256")
        if model_split_hash is not None and model_split_hash != bundle.get(
            "split_sha256"
        ):
            raise ValueError(
                f"{model_id}: split hash differs from the shared benchmark"
            )
        if model_split_hash is None and not model.get("split_hash_note"):
            raise ValueError(
                f"{model_id}: missing explanation for the absent split hash"
            )
        scenes = model.get("scenes", [])
        if [scene.get("scene_id") for scene in scenes] != scene_ids:
            raise ValueError(
                f"{model_id}: per-scene matrices are missing or out of order"
            )
        scene_matrices = []
        for scene in scenes:
            matrix = validate_matrix(
                scene.get("confusion_matrix"), f"{model_id}/{scene['scene_id']}"
            )
            if sum(sum(row) for row in matrix) != scene.get("valid_pixels"):
                raise ValueError(
                    f"{model_id}/{scene['scene_id']}: valid pixel count "
                    "differs from matrix"
                )
            scene_matrices.append(matrix)
        aggregate = validate_matrix(
            model.get("aggregate_confusion_matrix"), f"{model_id}/aggregate"
        )
        if add_matrices(scene_matrices) != aggregate:
            raise ValueError(
                f"{model_id}: per-scene matrices do not sum to the aggregate"
            )
        result = metrics(aggregate)
        if result["pixels"] != bundle.get("scored_pixels_expected"):
            raise ValueError(
                f"{model_id}: scored pixel total differs from benchmark contract"
            )
        if result["pixels"] != model.get("scored_pixels"):
            raise ValueError(
                f"{model_id}: declared scored pixel count differs from matrix"
            )
        if not close(result["oa"], model.get("overall_accuracy_source")):
            raise ValueError(f"{model_id}: recomputed OA differs from source summary")
        if not close(result["miou"], model.get("mean_iou_source")):
            raise ValueError(f"{model_id}: recomputed mIoU differs from source summary")
        result["scene_metrics"] = {
            scene["scene_id"]: metrics(
                validate_matrix(scene["confusion_matrix"], model_id)
            )
            for scene in scenes
        }
        result["model"] = model
        model_results[model_id] = result
    if set(model_results) != {"sam", "dinov2", "dinov3", "pspnet"}:
        raise ValueError("expected public matrices for SAM, DINOv2, DINOv3, and PSPNet")
    return manifest, bundle, model_results


def percentage(value: float, places: int = 4) -> str:
    return f"{value * 100:.{places}f}%"


def percentage_points(value: float, places: int = 4) -> str:
    return f"{value * 100:+.{places}f} pp"


def render_metrics(
    manifest: dict[str, Any], bundle: dict[str, Any], results: dict[str, dict[str, Any]]
) -> str:
    claim_by_model: dict[str, str] = {}
    for claim in manifest["claims"]:
        related_models = claim.get("related_models", [])
        if len(related_models) == 1:
            claim_by_model.setdefault(related_models[0], claim["claim_id"])
    order = ["sam", "dinov2", "pspnet", "dinov3"]
    rows = [
        "| Model | OA | Six-class mIoU | Errors / scored pixels | "
        "Recorded full-scene scoring | Evidence |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for model_id in order:
        result = results[model_id]
        model = result["model"]
        runtime = model.get("inference_seconds")
        runtime_text = (
            f"{runtime:.2f} s"
            if runtime is not None
            else "Not reported in public metric artifact"
        )
        rows.append(
            f"| {model['display_name']} | {percentage(result['oa'])} | "
            f"{percentage(result['miou'])} | "
            f"{result['errors']:,} / {result['pixels']:,} | {runtime_text} | "
            f"{claim_by_model.get(model_id, '—')} |"
        )

    class_rows = [
        "| Class | SAM IoU / F1 | DINOv2 IoU / F1 | "
        "PSPNet IoU / F1 | DINOv3 IoU / F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for class_index, class_name in enumerate(bundle["class_names"]):
        values = []
        for model_id in order:
            result = results[model_id]
            values.append(
                f"{percentage(result['iou'][class_index])} / "
                f"{percentage(result['f1'][class_index])}"
            )
        class_rows.append(f"| {class_name} | " + " | ".join(values) + " |")

    psp, d3, d2 = results["pspnet"], results["dinov3"], results["dinov2"]
    scene_rows = [
        "| Scene | DINOv3 OA | PSPNet OA | DINOv3 − PSPNet |",
        "| --- | ---: | ---: | ---: |",
    ]
    for scene_id in bundle["scene_ids"]:
        d3_oa = d3["scene_metrics"][scene_id]["oa"]
        psp_oa = psp["scene_metrics"][scene_id]["oa"]
        scene_rows.append(
            f"| {scene_id} | {percentage(d3_oa)} | {percentage(psp_oa)} | "
            f"{percentage_points(d3_oa - psp_oa)} |"
        )
    per_scene_rows = [
        "| Scene | SAM OA / mIoU | DINOv2 OA / mIoU | "
        "PSPNet OA / mIoU | DINOv3 OA / mIoU |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for scene_id in bundle["scene_ids"]:
        values = []
        for model_id in order:
            scene_result = results[model_id]["scene_metrics"][scene_id]
            values.append(
                f"{percentage(scene_result['oa'])} / {percentage(scene_result['miou'])}"
            )
        per_scene_rows.append(f"| {scene_id} | " + " | ".join(values) + " |")

    return "\n".join(
        [
            "### Aggregate results",
            "",
            "Metrics are recomputed by summing the 14 per-scene confusion matrices "
            "for each model. "
            "Rows are complete recorded systems; only DINOv3 RGB and PSPNet RGB share "
            "the matched input, split, eroded reference, 512×512 windows, stride 256, "
            "batch size, and identity inference recipe. "
            "The benchmark was examined in earlier campaigns.",
            "",
            *rows,
            "",
            "OA is overall accuracy. mIoU is the unweighted mean of the six class "
            "IoUs. "
            "Scoring excludes label ID 6. "
            "Runtime is included only where it was recorded in the source artifact; "
            "it is not imputed.",
            "",
            "### Per-class IoU and F1",
            "",
            "Each cell reports IoU / F1 in percent, recomputed from the same "
            "aggregate matrix.",
            "",
            *class_rows,
            "",
            "### Per-scene OA and mIoU",
            "",
            "Each cell reports OA / mIoU in percent from that scene's confusion "
            "matrix.",
            "",
            *per_scene_rows,
            "",
            "### Matched RGB scene comparison",
            "",
            "DINOv3 minus PSPNet is positive on 12 of 14 scenes. These paired "
            "recipe-level "
            "results do not isolate an architecture-only effect. "
            "Full-scene latency was "
            "405.23 s for DINOv3 and 254.83 s for PSPNet in "
            "the archived benchmark profiles (CLAIM-005).",
            "",
            *scene_rows,
            "",
            "### Recomputed comparisons",
            "",
            f"- DINOv3 vs PSPNet: OA {percentage_points(d3['oa'] - psp['oa'])}, "
            "six-class mIoU "
            f"{percentage_points(d3['miou'] - psp['miou'])} "
            "(matched RGB recipes; CLAIM-005).",
            f"- DINOv3 vs historical DINOv2: OA "
            f"{percentage_points(d3['oa'] - d2['oa'])}, six-class mIoU "
            f"{percentage_points(d3['miou'] - d2['miou'])}; this compares "
            "complete recipes with different inputs and tiling (CLAIM-003).",
            "",
            f"The evidence bundle stores {bundle['scored_pixels_expected']:,} scored "
            "pixels per model. Per-scene and aggregate matrices are in "
            "`artifacts/public/metrics/potsdam-benchmark-confusions.json`.",
        ]
    )


def update_results_document(generated: str, write: bool) -> bool:
    current = RESULTS.read_text(encoding="utf-8")
    if current.count(START) != 1 or current.count(END) != 1:
        raise ValueError(
            f"{RESULTS.relative_to(ROOT)} must contain one generated-metrics "
            "marker pair"
        )
    prefix, rest = current.split(START, 1)
    _, suffix = rest.split(END, 1)
    expected = f"{START}\n\n{generated}\n\n{END}"
    actual = f"{START}{rest.split(END, 1)[0]}{END}"
    if actual != expected:
        if not write:
            print(
                "docs/results.md generated metrics are stale; run with --write",
                file=sys.stderr,
            )
            return False
        RESULTS.write_text(prefix + expected + suffix, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write",
        action="store_true",
        help="write recomputed metrics into docs/results.md",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="verify public hashes and generated metrics",
    )
    args = parser.parse_args()
    try:
        manifest, bundle, results = load_evidence()
        generated = render_metrics(manifest, bundle, results)
        ok = update_results_document(generated, write=args.write)
        if not ok:
            return 1
        print(
            f"Verified {len(results)} models, {len(bundle['scene_ids'])} scenes, "
            f"{bundle['scored_pixels_expected']:,} pixels per model; "
            f"{len(manifest['claims'])} claims and all registered hashes."
        )
        if args.write:
            print(f"Updated {RESULTS.relative_to(ROOT)}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"public metrics verification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
