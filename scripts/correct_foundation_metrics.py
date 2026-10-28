"""Correct the archived foundation report from immutable local metric records.

Run with the original model_results directory as --source-dir. This rewrites only
the public aggregate JSON; source records, predictions, and model weights are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from deepnet.data.evaluation import summarize_confusion


def correct_record(public: dict, source: dict) -> int:
    """Correct published F1 fields and retain their supporting confusion counts."""
    changed = 0
    if "macro_f1" in public:
        matrix = np.asarray(source["confusion_matrix"], dtype=np.int64)
        metrics = summarize_confusion(matrix)
        for key in ("overall_accuracy", "mean_iou"):
            if not np.isclose(public[key], metrics[key], rtol=0, atol=1e-12):
                raise ValueError(f"Published {key} differs from source counts")
        values = [value for value in metrics["per_class_f1"] if value is not None]
        macro_f1 = float(np.mean(values)) if values else None
        changed += not np.isclose(public["macro_f1"], macro_f1, rtol=0, atol=1e-12)
        public["macro_f1"] = macro_f1
        public["per_class_f1"] = dict(
            zip(source["per_class"], metrics["per_class_f1"], strict=True)
        )
        public["confusion_matrix"] = matrix.tolist()
    for key in (
        "per_scene",
        "train_majority_baseline",
        "source_train_majority_baseline",
    ):
        if key not in public:
            continue
        if key == "per_scene":
            for scene_id, record in public[key].items():
                changed += correct_record(record, source[key][scene_id])
        else:
            changed += correct_record(public[key], source[key])
    return changed


def correct_model(public: dict, source: dict) -> int:
    """Map the published summary paths to the original evaluation schema."""
    pairs = []
    for dataset in ("potsdam", "vaihingen"):
        for reference in ("full_reference", "eroded_reference"):
            pairs.append(
                (
                    public[dataset][f"held_out_{reference}"],
                    source[dataset][f"official_held_out_{reference}"],
                )
            )
    oof = source["vaihingen"]["cross_validation"]["eroded_reference"]
    pairs.append((public["vaihingen"]["cross_validation_oof_eroded_reference"], oof))
    for public_fold, source_fold in zip(
        public["vaihingen"]["cross_validation_folds"], oof["folds"], strict=True
    ):
        if public_fold["validation_scene_ids"] != source_fold["validation_scene_ids"]:
            raise ValueError("Published fold membership differs from the source")
        pairs.append(
            (
                public_fold["validation_eroded_reference"],
                source_fold["validation_eroded"],
            )
        )
    for direction, record in public["cross_domain_eroded_reference"].items():
        pairs.append((record, source["cross_domain"][direction]["eroded_reference"]))
    return sum(correct_record(public_record, raw) for public_record, raw in pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "docs/experiments/"
            "potsdam-vaihingen-foundation-probe-2026-10-19-metrics.json"
        ),
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    source_hashes = {}
    corrected = 0
    for name, model in report["models"].items():
        path = args.source_dir / f"{name}_deep_metrics.json"
        data = path.read_bytes()
        source_hashes[path.name] = hashlib.sha256(data).hexdigest()
        corrected += correct_model(model, json.loads(data))
    report["metric_correction"] = {
        "date": "2026-10-28",
        "method": "Per-class F1 = 2*TP/(truth_count+prediction_count); macro averages "
        "all classes with a nonzero denominator, including zero-TP classes.",
        "source_metric_sha256": source_hashes,
        "source_artifacts_modified": False,
        "inference_rerun": False,
        "overall_accuracy_and_mean_iou_unchanged": True,
    }
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Corrected {corrected} published macro-F1 values")


if __name__ == "__main__":
    main()
