"""Recompute Potsdam empirical comparisons from archived metrics and manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from notebooks_src import eda_plots

CLASSES = (
    "Impervious surfaces",
    "Building",
    "Low vegetation",
    "Tree",
    "Car",
    "Clutter/background",
)
SCENES = (
    "2_13",
    "2_14",
    "3_13",
    "3_14",
    "4_13",
    "4_14",
    "4_15",
    "5_13",
    "5_14",
    "5_15",
    "6_13",
    "6_14",
    "6_15",
    "7_13",
)
PIXELS = 467_862_855
VALIDATION_PIXELS = 199_409_714


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_file(files: dict[str, dict[str, Any]], path: Path) -> None:
    files[str(path)] = {"bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_relative_index(directory: Path, index_name: str) -> dict[str, Path]:
    index = read_json(directory / index_name)
    verified = {}
    for record in index["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe indexed path: {relative}")
        path = directory / relative
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise ValueError(f"Artifact index mismatch: {path}")
        verified[relative.as_posix()] = path
    return verified


def verify_absolute_index(index_path: Path, target: Path) -> None:
    index = read_json(index_path)
    record = index.get(str(target))
    if record is None:
        raise ValueError(f"Historical review index does not contain {target}")
    if target.stat().st_size != record["bytes"] or sha256(target) != record["sha256"]:
        raise ValueError(f"Historical input differs from its review index: {target}")


def checked_matrix(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.int64)
    if matrix.shape != (len(CLASSES), len(CLASSES)) or np.any(matrix < 0):
        raise ValueError(f"{label} must be a nonnegative 6x6 matrix")
    return matrix


def metric_values(matrix: np.ndarray) -> dict[str, Any]:
    matrix = checked_matrix(matrix, "confusion matrix")
    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    union = support + predicted - true_positive
    iou = np.divide(
        true_positive,
        union,
        out=np.full(len(CLASSES), np.nan),
        where=union > 0,
    )
    precision = np.divide(
        true_positive,
        predicted,
        out=np.full(len(CLASSES), np.nan),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.full(len(CLASSES), np.nan),
        where=support > 0,
    )
    total = int(matrix.sum())
    return {
        "overall_accuracy": float(true_positive.sum() / total),
        "mean_iou": float(np.nanmean(iou)),
        "per_class_iou": iou,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "support": support.astype(np.int64),
        "predicted": predicted.astype(np.int64),
        "true_positive": true_positive.astype(np.int64),
        "false_negative": (support - true_positive).astype(np.int64),
        "false_positive": (predicted - true_positive).astype(np.int64),
        "pixels_scored": total,
        "errors": total - int(true_positive.sum()),
    }


def metric_from_scene_records(
    metrics: dict[str, Any], model: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if metrics["scene_ids"] != list(SCENES):
        raise ValueError(f"{model} scene order differs from the locked benchmark")
    records = metrics["scenes"]
    if [row["scene_id"] for row in records] != list(SCENES):
        raise ValueError(f"{model} per-scene rows differ from the locked benchmark")
    scene_matrices = np.stack(
        [
            checked_matrix(row["confusion_matrix"], f"{model}/{row['scene_id']}")
            for row in records
        ]
    )
    aggregate = checked_matrix(metrics["confusion_matrix"], f"{model}/aggregate")
    if not np.array_equal(scene_matrices.sum(axis=0), aggregate):
        raise ValueError(f"{model} per-scene matrices do not sum to aggregate")
    if aggregate.sum() != metrics["pixels_scored"]:
        raise ValueError(f"{model} aggregate support differs from its metric record")
    if aggregate.sum() != PIXELS:
        raise ValueError(
            f"{model} scored {aggregate.sum()} rather than {PIXELS} pixels"
        )
    calculated = metric_values(aggregate)
    if not np.isclose(
        calculated["overall_accuracy"], metrics["overall_accuracy"], atol=1e-12
    ):
        raise ValueError(f"{model} OA does not reproduce from its confusion matrix")
    if not np.isclose(calculated["mean_iou"], metrics["mean_iou"], atol=1e-12):
        raise ValueError(f"{model} mIoU does not reproduce from its confusion matrix")
    scene_record_map = {row["scene_id"]: row for row in records}
    return scene_matrices, scene_record_map


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return result


def correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left, right = (
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
    )
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(left, right)[0, 1])
    spearman = float(np.corrcoef(ranks(left), ranks(right))[0, 1])
    return pearson, spearman


def bootstrap_aggregate_deltas(
    d2_scene_matrices: np.ndarray,
    d3_scene_matrices: np.ndarray,
    *,
    replicates: int = 20_000,
    seed: int = 42,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(SCENES), size=(replicates, len(SCENES)))
    d2 = d2_scene_matrices[indices].sum(axis=1)
    d3 = d3_scene_matrices[indices].sum(axis=1)

    def batched(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tp = np.diagonal(matrix, axis1=1, axis2=2)
        rows = matrix.sum(axis=2)
        cols = matrix.sum(axis=1)
        iou = tp / np.maximum(rows + cols - tp, 1)
        return tp.sum(axis=1) / matrix.sum(axis=(1, 2)), iou.mean(axis=1)

    d2_oa, d2_miou = batched(d2)
    d3_oa, d3_miou = batched(d3)
    result = {}
    for name, values in (
        ("global_oa_delta_pp", 100 * (d3_oa - d2_oa)),
        ("global_miou_delta_pp", 100 * (d3_miou - d2_miou)),
    ):
        result[name] = [float(v) for v in np.percentile(values, (2.5, 50, 97.5))]
    return result


def create_error_outputs(
    d2_matrix: np.ndarray,
    d3_matrix: np.ndarray,
    output: Path,
) -> dict[str, Any]:
    data_dir = output / "data"
    figure_dir = output / "figures"
    delta = d2_matrix - d3_matrix
    d2_metrics, d3_metrics = metric_values(d2_matrix), metric_values(d3_matrix)
    total_reduction = d2_metrics["errors"] - d3_metrics["errors"]
    if total_reduction != int(np.diag(d3_matrix - d2_matrix).sum()):
        raise ValueError("OA improvement and error-count reduction disagree")
    if int(delta[~np.eye(6, dtype=bool)].sum()) != total_reduction:
        raise ValueError("Off-diagonal error changes do not sum to error reduction")

    matrix_rows = []
    for row, class_name in enumerate(CLASSES):
        matrix_rows.append(
            {
                "truth": class_name,
                **{CLASSES[col]: int(delta[row, col]) for col in range(6)},
            }
        )
    write_csv(
        data_dir / "potsdam-error-difference-matrix.csv",
        ["truth", *CLASSES],
        matrix_rows,
    )

    transitions = []
    for truth in range(6):
        for prediction in range(6):
            if truth == prediction:
                continue
            fewer = int(delta[truth, prediction])
            transitions.append(
                {
                    "true_class": CLASSES[truth],
                    "predicted_class": CLASSES[prediction],
                    "DINOv2_errors": int(d2_matrix[truth, prediction]),
                    "DINOv3_errors": int(d3_matrix[truth, prediction]),
                    "fewer_errors_in_DINOv3": fewer,
                    "share_of_net_error_reduction_percent": (
                        100 * fewer / total_reduction
                    ),
                }
            )
    transitions.sort(key=lambda row: abs(row["fewer_errors_in_DINOv3"]), reverse=True)
    write_csv(
        data_dir / "potsdam-confusion-transitions.csv",
        list(transitions[0]),
        transitions,
    )

    class_rows = []
    for i, name in enumerate(CLASSES):
        class_rows.append(
            {
                "class": name,
                "ground_truth_support": int(d2_metrics["support"][i]),
                "DINOv2_TP": int(d2_metrics["true_positive"][i]),
                "DINOv3_TP": int(d3_metrics["true_positive"][i]),
                "TP_change_DINOv3_minus_DINOv2": int(
                    d3_metrics["true_positive"][i] - d2_metrics["true_positive"][i]
                ),
                "DINOv2_FN": int(d2_metrics["false_negative"][i]),
                "DINOv3_FN": int(d3_metrics["false_negative"][i]),
                "FN_change_DINOv3_minus_DINOv2": int(
                    d3_metrics["false_negative"][i] - d2_metrics["false_negative"][i]
                ),
                "DINOv2_FP": int(d2_metrics["false_positive"][i]),
                "DINOv3_FP": int(d3_metrics["false_positive"][i]),
                "FP_change_DINOv3_minus_DINOv2": int(
                    d3_metrics["false_positive"][i] - d2_metrics["false_positive"][i]
                ),
                "DINOv2_precision_percent": 100 * d2_metrics["per_class_precision"][i],
                "DINOv3_precision_percent": 100 * d3_metrics["per_class_precision"][i],
                "precision_delta_pp": 100
                * (
                    d3_metrics["per_class_precision"][i]
                    - d2_metrics["per_class_precision"][i]
                ),
                "DINOv2_recall_percent": 100 * d2_metrics["per_class_recall"][i],
                "DINOv3_recall_percent": 100 * d3_metrics["per_class_recall"][i],
                "recall_delta_pp": 100
                * (
                    d3_metrics["per_class_recall"][i]
                    - d2_metrics["per_class_recall"][i]
                ),
                "DINOv2_IoU_percent": 100 * d2_metrics["per_class_iou"][i],
                "DINOv3_IoU_percent": 100 * d3_metrics["per_class_iou"][i],
                "IoU_delta_pp": 100
                * (d3_metrics["per_class_iou"][i] - d2_metrics["per_class_iou"][i]),
            }
        )
    write_csv(
        data_dir / "potsdam-per-class-error-changes.csv",
        list(class_rows[0]),
        class_rows,
    )

    with plt.style.context(eda_plots.STYLE):
        fig, ax = plt.subplots(figsize=(8.4, 6.8), layout="constrained")
        limit = max(1, float(np.max(np.abs(delta))))
        mesh = ax.imshow(
            delta / 1_000_000,
            cmap="RdBu",
            vmin=-limit / 1_000_000,
            vmax=limit / 1_000_000,
        )
        ax.set_xticks(range(6), CLASSES, rotation=28, ha="right")
        ax.set_yticks(range(6), CLASSES)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Ground-truth class")
        ax.set_title("DINOv2 count minus DINOv3 count · millions of pixels", loc="left")
        for row in range(6):
            for col in range(6):
                value = int(delta[row, col])
                ax.text(
                    col,
                    row,
                    f"{value / 1_000_000:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(value) > limit * 0.45 else "black",
                )
        fig.colorbar(mesh, ax=ax, label="DINOv2 − DINOv3 (millions)")
        fig.savefig(
            figure_dir / "potsdam-confusion-difference-scienceplots.png", dpi=220
        )
        plt.close(fig)

    return {
        "pixels_scored": int(d2_matrix.sum()),
        "DINOv2_errors": d2_metrics["errors"],
        "DINOv3_errors": d3_metrics["errors"],
        "net_error_reduction": total_reduction,
        "OA_delta_pp": 100
        * (d3_metrics["overall_accuracy"] - d2_metrics["overall_accuracy"]),
        "mIoU_delta_pp": 100 * (d3_metrics["mean_iou"] - d2_metrics["mean_iou"]),
        "major_pair_reductions": {
            "tree_to_low_vegetation": int(delta[3, 2]),
            "low_vegetation_to_tree": int(delta[2, 3]),
            "combined_tree_low_vegetation": int(delta[3, 2] + delta[2, 3]),
            "clutter_to_impervious": int(delta[5, 0]),
        },
    }


def create_scene_outputs(
    d2_matrices: np.ndarray,
    d3_matrices: np.ndarray,
    output: Path,
) -> dict[str, Any]:
    data_dir, figure_dir = output / "data", output / "figures"
    d2_aggregate = d2_matrices.sum(axis=0)
    d3_aggregate = d3_matrices.sum(axis=0)
    rows, loo_rows = [], []
    for idx, scene in enumerate(SCENES):
        m2, m3 = metric_values(d2_matrices[idx]), metric_values(d3_matrices[idx])
        row = {
            "scene": scene,
            "pixels_scored": m2["pixels_scored"],
            "DINOv2_OA_percent": 100 * m2["overall_accuracy"],
            "DINOv3_OA_percent": 100 * m3["overall_accuracy"],
            "OA_delta_pp": 100 * (m3["overall_accuracy"] - m2["overall_accuracy"]),
            "DINOv2_mIoU_percent": 100 * m2["mean_iou"],
            "DINOv3_mIoU_percent": 100 * m3["mean_iou"],
            "mIoU_delta_pp": 100 * (m3["mean_iou"] - m2["mean_iou"]),
        }
        for class_id, class_name in enumerate(CLASSES):
            row[f"{class_name}_prevalence_percent"] = (
                100 * m2["support"][class_id] / m2["pixels_scored"]
            )
            row[f"{class_name}_IoU_delta_pp"] = 100 * (
                m3["per_class_iou"][class_id] - m2["per_class_iou"][class_id]
            )
        rows.append(row)

        remaining_d2 = d2_aggregate - d2_matrices[idx]
        remaining_d3 = d3_aggregate - d3_matrices[idx]
        loo2, loo3 = metric_values(remaining_d2), metric_values(remaining_d3)
        loo_rows.append(
            {
                "excluded_scene": scene,
                "pixels_remaining": int(remaining_d2.sum()),
                "DINOv2_OA_percent": 100 * loo2["overall_accuracy"],
                "DINOv3_OA_percent": 100 * loo3["overall_accuracy"],
                "OA_delta_pp": 100
                * (loo3["overall_accuracy"] - loo2["overall_accuracy"]),
                "DINOv2_mIoU_percent": 100 * loo2["mean_iou"],
                "DINOv3_mIoU_percent": 100 * loo3["mean_iou"],
                "mIoU_delta_pp": 100 * (loo3["mean_iou"] - loo2["mean_iou"]),
            }
        )
    write_csv(data_dir / "potsdam-scene-robustness.csv", list(rows[0]), rows)
    write_csv(data_dir / "potsdam-leave-one-scene-out.csv", list(loo_rows[0]), loo_rows)

    support_rows = []
    oa_deltas = np.array([row["OA_delta_pp"] for row in rows])
    for class_name in CLASSES:
        prevalence = np.array([row[f"{class_name}_prevalence_percent"] for row in rows])
        iou_delta = np.array([row[f"{class_name}_IoU_delta_pp"] for row in rows])
        pearson_oa, spearman_oa = correlation(prevalence, oa_deltas)
        pearson_iou, spearman_iou = correlation(prevalence, iou_delta)
        support_rows.append(
            {
                "class": class_name,
                "scene_count": len(SCENES),
                "prevalence_vs_scene_OA_delta_pearson_r": pearson_oa,
                "prevalence_vs_scene_OA_delta_spearman_rho": spearman_oa,
                "prevalence_vs_same_class_IoU_delta_pearson_r": pearson_iou,
                "prevalence_vs_same_class_IoU_delta_spearman_rho": spearman_iou,
            }
        )
    write_csv(
        data_dir / "potsdam-scene-prevalence-correlations.csv",
        list(support_rows[0]),
        support_rows,
    )

    macro = {
        "DINOv2_macro_scene_OA_percent": float(
            np.mean([row["DINOv2_OA_percent"] for row in rows])
        ),
        "DINOv3_macro_scene_OA_percent": float(
            np.mean([row["DINOv3_OA_percent"] for row in rows])
        ),
        "macro_scene_OA_delta_pp": float(np.mean(oa_deltas)),
        "DINOv2_macro_scene_mIoU_percent": float(
            np.mean([row["DINOv2_mIoU_percent"] for row in rows])
        ),
        "DINOv3_macro_scene_mIoU_percent": float(
            np.mean([row["DINOv3_mIoU_percent"] for row in rows])
        ),
        "macro_scene_mIoU_delta_pp": float(
            np.mean([row["mIoU_delta_pp"] for row in rows])
        ),
        "DINOv2_global_OA_percent": 100
        * metric_values(d2_aggregate)["overall_accuracy"],
        "DINOv3_global_OA_percent": 100
        * metric_values(d3_aggregate)["overall_accuracy"],
        "global_OA_delta_pp": 100
        * (
            metric_values(d3_aggregate)["overall_accuracy"]
            - metric_values(d2_aggregate)["overall_accuracy"]
        ),
        "DINOv2_global_mIoU_percent": 100 * metric_values(d2_aggregate)["mean_iou"],
        "DINOv3_global_mIoU_percent": 100 * metric_values(d3_aggregate)["mean_iou"],
        "global_mIoU_delta_pp": 100
        * (
            metric_values(d3_aggregate)["mean_iou"]
            - metric_values(d2_aggregate)["mean_iou"]
        ),
    }
    bootstrap = bootstrap_aggregate_deltas(d2_matrices, d3_matrices)
    loo_delta = np.array([row["mIoU_delta_pp"] for row in loo_rows])

    with plt.style.context(eda_plots.STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
        y = np.arange(len(SCENES))
        axes[0].barh(
            y,
            [row["OA_delta_pp"] for row in rows],
            color=["#0072B2" if row["OA_delta_pp"] >= 0 else "#D55E00" for row in rows],
        )
        axes[0].axvline(0, color="0.2", linewidth=0.8)
        axes[0].set_yticks(y, SCENES)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("DINOv3 − DINOv2 overall accuracy (pp)")
        axes[0].set_title("Per-scene OA change", loc="left")

        axes[1].barh(
            y,
            [row["mIoU_delta_pp"] for row in rows],
            color=[
                "#009E73" if row["mIoU_delta_pp"] >= 0 else "#D55E00" for row in rows
            ],
        )
        axes[1].axvline(0, color="0.2", linewidth=0.8)
        axes[1].set_yticks(y, SCENES)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("DINOv3 − DINOv2 six-class mIoU (pp)")
        axes[1].set_title("Per-scene mIoU change", loc="left")
        fig.suptitle("Scene sensitivity · all 14 previously observed benchmark scenes")
        fig.savefig(figure_dir / "potsdam-scene-deltas-scienceplots.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
        ax.axhline(
            100
            * (
                metric_values(d3_aggregate)["mean_iou"]
                - metric_values(d2_aggregate)["mean_iou"]
            ),
            color="0.2",
            linestyle="--",
            label="All 14 scenes",
        )
        ax.scatter(loo_delta, y, color="#0072B2", zorder=3)
        ax.set_yticks(y, [row["excluded_scene"] for row in loo_rows])
        ax.invert_yaxis()
        ax.set_xlabel("DINOv3 − DINOv2 mIoU after omitting one scene (pp)")
        ax.set_title("Leave-one-scene-out sensitivity", loc="left")
        ax.legend()
        fig.savefig(
            figure_dir / "potsdam-leave-one-scene-out-scienceplots.png", dpi=220
        )
        plt.close(fig)

    return {
        "macro_vs_global": macro,
        "positive_OA_scenes": int(np.sum(oa_deltas > 0)),
        "negative_OA_scenes": int(np.sum(oa_deltas < 0)),
        "positive_mIoU_scenes": int(np.sum([row["mIoU_delta_pp"] > 0 for row in rows])),
        "negative_mIoU_scenes": int(np.sum([row["mIoU_delta_pp"] < 0 for row in rows])),
        "scene_OA_delta_range_pp": [float(np.min(oa_deltas)), float(np.max(oa_deltas))],
        "scene_mIoU_delta_range_pp": [
            float(min(row["mIoU_delta_pp"] for row in rows)),
            float(max(row["mIoU_delta_pp"] for row in rows)),
        ],
        "leave_one_out_mIoU_delta_range_pp": [
            float(np.min(loo_delta)),
            float(np.max(loo_delta)),
        ],
        "leave_one_out_sign_changes": bool(np.min(loo_delta) <= 0),
        "scene_cluster_bootstrap_95_percentile_interval": bootstrap,
        "bootstrap_seed": 42,
        "bootstrap_replicates": 20_000,
        "support_correlations": support_rows,
    }


def full_validation(manifest: dict[str, Any], arm: str) -> dict[str, Any]:
    candidates = [
        row
        for row in manifest["history"]
        if row.get("full_validation") and row.get("updates") == 6912
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one pass-8 validation record for {arm}")
    return candidates[0]["full_validation"]


def create_negative_results(
    run_root: Path, output: Path, provenance: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    root = run_root / "potsdam-dinov3-rgbir-v2"
    review = root / "review-selection-abc-20261106"
    audit = read_json(review / "ABC-analysis.json")
    record_file(provenance, review / "ABC-analysis.json")
    if audit["pixels_scored"] != VALIDATION_PIXELS or audit["winner"] != "A":
        raise ValueError("A/B/C audit does not match the recorded selection")
    arms = {}
    for arm in "ABCD":
        manifest_path = root / "study" / arm / "manifest.json"
        manifest = read_json(manifest_path)
        provenance[str(manifest_path)] = {
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256(manifest_path),
        }
        selected = full_validation(manifest, arm)
        matrix = checked_matrix(selected["confusion_matrix"], f"{arm} validation")
        values = metric_values(matrix)
        if values["pixels_scored"] != VALIDATION_PIXELS:
            raise ValueError(
                f"Arm {arm} validation support is not the same 6-scene cohort"
            )
        if arm in audit["arms"]:
            recorded = audit["arms"][arm]["best"]
            if not np.isclose(
                values["overall_accuracy"], recorded["overall_accuracy"], atol=1e-12
            ):
                raise ValueError(f"Arm {arm} OA differs from its review audit")
            if not np.isclose(values["mean_iou"], recorded["mean_iou"], atol=1e-12):
                raise ValueError(f"Arm {arm} mIoU differs from its review audit")
        arms[arm] = {
            "matrix": matrix,
            "values": values,
            "manifest": manifest,
            "validation": selected,
        }

    rows = []
    for arm in "ABCD":
        result = arms[arm]
        recipe = result["manifest"]["recipe"]
        rows.append(
            {
                "arm": arm,
                "seed": result["manifest"].get("seed"),
                "update": 6912,
                "auxiliary_features": recipe.get("auxiliary"),
                "enhanced_augmentation": recipe.get("enhanced"),
                "OA_percent": 100 * result["values"]["overall_accuracy"],
                "mIoU_percent": 100 * result["values"]["mean_iou"],
                "error_pixels": result["values"]["errors"],
                **{
                    f"{CLASSES[i]}_IoU_percent": 100
                    * result["values"]["per_class_iou"][i]
                    for i in range(6)
                },
            }
        )
    write_csv(output / "data/potsdam-negative-results-arms.csv", list(rows[0]), rows)

    contrast_rows = []
    for arm, description in (
        ("B", "Bundled IR/index/relief/mask features"),
        ("C", "Enhanced augmentation"),
        ("D", "A recipe, seed 43"),
    ):
        baseline, current = arms["A"]["values"], arms[arm]["values"]
        contrast_rows.append(
            {
                "contrast": f"{arm} minus A",
                "description": description,
                "OA_delta_pp": 100
                * (current["overall_accuracy"] - baseline["overall_accuracy"]),
                "mIoU_delta_pp": 100 * (current["mean_iou"] - baseline["mean_iou"]),
                "car_IoU_delta_pp": 100
                * (current["per_class_iou"][4] - baseline["per_class_iou"][4]),
                "clutter_IoU_delta_pp": 100
                * (current["per_class_iou"][5] - baseline["per_class_iou"][5]),
                "error_count_delta": current["errors"] - baseline["errors"],
                **{
                    f"{CLASSES[i]}_IoU_delta_pp": 100
                    * (current["per_class_iou"][i] - baseline["per_class_iou"][i])
                    for i in range(6)
                },
            }
        )
    write_csv(
        output / "data/potsdam-negative-results-contrasts.csv",
        list(contrast_rows[0]),
        contrast_rows,
    )

    tta_path = root / "review-confirmation-20261106/TTA-comparison.csv"
    with tta_path.open(encoding="utf-8", newline="") as stream:
        tta_rows = list(csv.DictReader(stream))
    if len(tta_rows) != 2 or any(row["policy"] != "identity" for row in tta_rows):
        raise ValueError("TTA audit rows differ from the frozen inference policy")
    provenance[str(tta_path)] = {
        "bytes": tta_path.stat().st_size,
        "sha256": sha256(tta_path),
    }

    with plt.style.context(eda_plots.STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout="constrained")
        x = np.arange(4)
        labels = ["A\nRGB", "B\n+ auxiliary", "C\nstrong aug.", "D\nseed 43"]
        axes[0].plot(x, [row["OA_percent"] for row in rows], marker="o", label="OA")
        axes[0].plot(x, [row["mIoU_percent"] for row in rows], marker="s", label="mIoU")
        axes[0].set_xticks(x, labels)
        axes[0].set_ylabel("Validation score (%)")
        axes[0].set_title("Aggregate validation", loc="left")
        axes[0].legend()
        for row, color, label in (
            (contrast_rows[0], "#E69F00", "B − A"),
            (contrast_rows[1], "#009E73", "C − A"),
        ):
            deltas = [row[f"{name}_IoU_delta_pp"] for name in CLASSES]
            offset = -0.16 if label.startswith("B") else 0.16
            axes[1].barh(
                np.arange(6) + offset, deltas, height=0.3, color=color, label=label
            )
        axes[1].axvline(0, color="0.2", linewidth=0.8)
        axes[1].set_yticks(np.arange(6), CLASSES)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("IoU difference (percentage points)")
        axes[1].set_title("Per-class differences from A", loc="left")
        axes[1].legend()
        fig.suptitle("Negative results describe tested recipes, not general effects")
        fig.savefig(
            output / "figures/potsdam-negative-results-scienceplots.png", dpi=220
        )
        plt.close(fig)

    return {
        "arms": [
            {key: value for key, value in row.items() if key != "matrix"}
            for row in rows
        ],
        "contrasts": contrast_rows,
        "tta": tta_rows,
        "interpretation": (
            "One seed per A/B/C configuration plus one A-recipe confirmation seed "
            "cannot estimate expected treatment effects or isolate individual "
            "auxiliary modalities."
        ),
    }


def load_training_cost(audit_path: Path) -> dict[str, Any]:
    audit = read_json(audit_path)
    if audit.get("status") != "verified":
        raise ValueError(f"Unverified compute audit: {audit_path}")
    return audit


def create_cost_outputs(
    run_root: Path,
    output: Path,
    provenance: dict[str, dict[str, Any]],
    paired_manifest_path: Path | None,
) -> dict[str, Any]:
    d3_root = run_root / "potsdam-dinov3-rgbir-v2"
    psp_root = run_root / "potsdam-pspnet-rgb-v1"
    d3_path = d3_root / "study/heldout/metrics.json"
    psp_path = psp_root / "benchmark/outputs/study/benchmark/metrics.json"
    d3, psp = read_json(d3_path), read_json(psp_path)
    d3_attempt = read_json(d3_root / "study/heldout/attempt.json")
    if (
        d3_attempt["cache_hash"] != psp["cache_hash"]
        or d3["split_hash"] != psp["split_hash"]
        or d3["scene_ids"] != psp["scene_ids"]
        or d3["pixels_scored"] != psp["pixels_scored"]
        or d3["window_size"] != psp["window_size"]
        or d3["stride"] != psp["stride"]
        or d3["inference"] != psp["inference"]
        or d3["heldout_evaluation_count"] != 1
        or psp["heldout_evaluation_count"] != 1
    ):
        raise ValueError("DINOv3 and PSPNet benchmark protocols are not matched")
    d3_matrices, d3_scene_map = metric_from_scene_records(d3, "DINOv3")
    psp_matrices, psp_scene_map = metric_from_scene_records(psp, "PSPNet")

    d3_preflight_audit = load_training_cost(d3_root / "kaggle/preflight/audit.json")
    d3_selection_audits = tuple(
        load_training_cost(d3_root / f"kaggle/{arm}/audit.json") for arm in "ABCD"
    )
    d3_refit_audit = load_training_cost(d3_root / "kaggle/refit/audit.json")
    d3_benchmark_audit = load_training_cost(d3_root / "kaggle/heldout/audit.json")
    psp_selection_audit = load_training_cost(psp_root / "selection/audit.json")
    psp_refit_audit = load_training_cost(psp_root / "refit/audit.json")
    psp_benchmark_audit = load_training_cost(psp_root / "benchmark/audit.json")

    cost_rows = []
    for (
        model,
        result,
        selection_phases,
        winner_selection,
        refit_phase,
        benchmark_phase,
        preflight_phase,
        checkpoint,
    ) in (
        (
            "DINOv3-B RGB refit",
            d3,
            d3_selection_audits,
            d3_selection_audits[0],
            d3_refit_audit,
            d3_benchmark_audit,
            d3_preflight_audit,
            d3["checkpoint_sha256"],
        ),
        (
            "PSPNet RGB refit",
            psp,
            (psp_selection_audit,),
            psp_selection_audit,
            psp_refit_audit,
            psp_benchmark_audit,
            None,
            psp["checkpoint_sha256"],
        ),
    ):
        selection_hours = sum(
            phase["execution_gpu_hours"] for phase in selection_phases
        )
        winner_selection_hours = winner_selection["execution_gpu_hours"]
        refit_hours = refit_phase["execution_gpu_hours"]
        benchmark_hours = benchmark_phase["execution_gpu_hours"]
        preflight_hours = (
            preflight_phase["execution_gpu_hours"] if preflight_phase else 0.0
        )
        profiles = [row["profile_seconds"] for row in result["scenes"]]
        model_key = "dinov3" if model.startswith("DINOv3") else "pspnet"
        peak_allocated = "not recorded comparably in original benchmark"
        peak_reserved = "not recorded comparably in original benchmark"
        if paired_manifest_path is not None:
            paired_manifest = read_json(paired_manifest_path)
            paired_model = paired_manifest["models"][model_key]
            peak_allocated = paired_model["peak_cuda_allocated_bytes"]
            peak_reserved = paired_model["peak_cuda_reserved_bytes"]
        cost_rows.append(
            {
                "model": model,
                "OA_percent": 100 * result["overall_accuracy"],
                "mIoU_percent": 100 * result["mean_iou"],
                "pixels_scored": result["pixels_scored"],
                "full_scene_scoring_seconds": result["inference_seconds"],
                "model_forward_seconds": sum(x["forward"] for x in profiles),
                "preflight_gpu_hours": preflight_hours,
                "training_selection_gpu_hours": selection_hours,
                "winner_arm_selection_gpu_hours": winner_selection_hours,
                "training_refit_gpu_hours": refit_hours,
                "training_gpu_hours": selection_hours + refit_hours,
                "winner_arm_plus_refit_gpu_hours": winner_selection_hours + refit_hours,
                "benchmark_gpu_hours": benchmark_hours,
                "selection_refit_benchmark_gpu_hours": (
                    selection_hours + refit_hours + benchmark_hours
                ),
                "full_campaign_gpu_hours": (
                    preflight_hours + selection_hours + refit_hours + benchmark_hours
                ),
                "peak_inference_cuda_allocated_bytes": peak_allocated,
                "peak_inference_cuda_reserved_bytes": peak_reserved,
                "checkpoint_sha256": checkpoint,
            }
        )
    write_csv(
        output / "data/potsdam-quality-cost-models.csv", list(cost_rows[0]), cost_rows
    )

    class_rows = []
    for i, name in enumerate(CLASSES):
        class_rows.append(
            {
                "class": name,
                "DINOv3_IoU_percent": 100 * d3["per_class_iou"][i],
                "PSPNet_IoU_percent": 100 * psp["per_class_iou"][i],
                "PSPNet_minus_DINOv3_IoU_pp": 100
                * (psp["per_class_iou"][i] - d3["per_class_iou"][i]),
                "DINOv3_F1_percent": 100 * d3["per_class_f1"][i],
                "PSPNet_F1_percent": 100 * psp["per_class_f1"][i],
                "DINOv3_precision_percent": 100 * d3["per_class_precision"][i],
                "PSPNet_precision_percent": 100 * psp["per_class_precision"][i],
                "DINOv3_recall_percent": 100 * d3["per_class_recall"][i],
                "PSPNet_recall_percent": 100 * psp["per_class_recall"][i],
            }
        )
    write_csv(
        output / "data/potsdam-quality-cost-classes.csv",
        list(class_rows[0]),
        class_rows,
    )

    scene_rows = []
    for scene in SCENES:
        a, b = d3_scene_map[scene], psp_scene_map[scene]
        scene_rows.append(
            {
                "scene": scene,
                "pixels_scored": a["pixels_scored"],
                "DINOv3_OA_percent": 100 * a["overall_accuracy"],
                "PSPNet_OA_percent": 100 * b["overall_accuracy"],
                "PSPNet_minus_DINOv3_OA_pp": 100
                * (b["overall_accuracy"] - a["overall_accuracy"]),
                "DINOv3_mIoU_percent": 100 * a["mean_iou"],
                "PSPNet_mIoU_percent": 100 * b["mean_iou"],
                "PSPNet_minus_DINOv3_mIoU_pp": 100 * (b["mean_iou"] - a["mean_iou"]),
            }
        )
    write_csv(
        output / "data/potsdam-quality-cost-scenes.csv", list(scene_rows[0]), scene_rows
    )

    d2_path = (
        run_root
        / "potsdam-dinov2-coverage-refit-v1"
        / "kaggle/preflight-v2-evidence/outputs/study/heldout/metrics.json"
    )
    d2 = read_json(d2_path)
    quality_rows = []
    historical_review = read_json(
        run_root
        / "potsdam-dinov2-coverage-refit-v1/review-final-20261105/analysis.json"
    )
    sam = historical_review["metrics"]["SAM"]
    for model, values, cost_info in (
        ("Historical SAM ViT-B + LoRA", sam, None),
        ("DINOv2 ViT-B/14 refit", d2, None),
        ("DINOv3-B RGB refit", d3, cost_rows[0]),
        ("PSPNet RGB refit", psp, cost_rows[1]),
    ):
        quality_rows.append(
            {
                "model": model,
                "OA_percent": 100 * values["overall_accuracy"],
                "mIoU_percent": 100 * values["mean_iou"],
                "pixels_scored": values["pixels_scored"],
                "full_scene_scoring_seconds": cost_info["full_scene_scoring_seconds"]
                if cost_info
                else "not comparable or not recorded",
                "training_gpu_hours": cost_info["training_gpu_hours"]
                if cost_info
                else "not harmonized",
                "interpretation_scope": "matched RGB recipe comparison"
                if cost_info
                else "historical complete-recipe reference",
            }
        )
    write_csv(
        output / "data/potsdam-model-quality-overview.csv",
        list(quality_rows[0]),
        quality_rows,
    )

    with plt.style.context(eda_plots.STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), layout="constrained")
        colors = ("#0072B2", "#D55E00")
        for ax, x_name, x_label, title in (
            (
                axes[0],
                "full_scene_scoring_seconds",
                "End-to-end full-scene scoring (s)",
                "Inference cost",
            ),
            (
                axes[1],
                "training_gpu_hours",
                "Selection campaign + refit (allocated GPU-h)",
                "Training cost",
            ),
        ):
            for row, color, marker in zip(cost_rows, colors, ("o", "s"), strict=True):
                ax.scatter(
                    row[x_name],
                    row["mIoU_percent"],
                    color=color,
                    marker=marker,
                    s=70,
                    label=row["model"],
                )
            ax.set_xlabel(x_label)
            ax.set_ylabel("Six-class mIoU (%)")
            ax.set_title(title, loc="left")
            ax.legend(fontsize=8)
        fig.suptitle("Matched RGB systems · neither dominates both quality and cost")
        fig.savefig(
            output / "figures/potsdam-quality-cost-pareto-scienceplots.png", dpi=220
        )
        plt.close(fig)

    provenance.update(
        {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (
                d3_path,
                psp_path,
                d3_root / "study/heldout/attempt.json",
                d3_root / "review-final-20261106/analysis-inventory.json",
                psp_root / "benchmark/outputs/output-index.json",
                psp_root / "benchmark/audit.json",
                run_root
                / "potsdam-dinov2-coverage-refit-v1"
                / "review-final-20261105/analysis.json",
            )
        }
    )
    audit_paths = [
        d3_root / "kaggle/preflight/audit.json",
        *(d3_root / f"kaggle/{arm}/audit.json" for arm in "ABCD"),
        d3_root / "kaggle/refit/audit.json",
        d3_root / "kaggle/heldout/audit.json",
        psp_root / "selection/audit.json",
        psp_root / "refit/audit.json",
        psp_root / "benchmark/audit.json",
    ]
    for path in audit_paths:
        record_file(provenance, path)
    paired_manifest = (
        read_json(paired_manifest_path) if paired_manifest_path is not None else None
    )
    if paired_manifest is not None:
        provenance[str(paired_manifest_path)] = {
            "bytes": paired_manifest_path.stat().st_size,
            "sha256": sha256(paired_manifest_path),
        }
    return {
        "models": cost_rows,
        "full_scene_time_difference_percent_DINOv3_over_PSPNet": 100
        * (
            cost_rows[0]["full_scene_scoring_seconds"]
            / cost_rows[1]["full_scene_scoring_seconds"]
            - 1
        ),
        "PSPNet_scoring_time_reduction_percent_vs_DINOv3": 100
        * (
            1
            - cost_rows[1]["full_scene_scoring_seconds"]
            / cost_rows[0]["full_scene_scoring_seconds"]
        ),
        "DINOv3_mIoU_gain_pp": 100 * (d3["mean_iou"] - psp["mean_iou"]),
        "DINOv3_forward_time_seconds": cost_rows[0]["model_forward_seconds"],
        "PSPNet_forward_time_seconds": cost_rows[1]["model_forward_seconds"],
        "frontier": [row["model"] for row in cost_rows],
        "inference_protocol": {
            "cache_hash": d3_attempt["cache_hash"],
            "pixels_scored": PIXELS,
            "scene_count": len(SCENES),
            "window_size": d3["window_size"],
            "stride": d3["stride"],
            "inference": d3["inference"],
            "batch_size": 4,
        },
        "peak_vram_scope": (
            "Read from the supplemental, exact-matrix-reproduction inference run."
            if paired_manifest is not None
            else "Not recorded comparably in the original benchmark metrics."
        ),
        "scope_limit": (
            "Pareto frontier is only for these two matched RGB recipes; it is not "
            "a frontier over all segmentation systems or deployment hardware."
        ),
    }


def verify_sources(
    run_root: Path, provenance: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    d2_root = run_root / "potsdam-dinov2-coverage-refit-v1"
    d3_root = run_root / "potsdam-dinov3-rgbir-v2"
    psp_root = run_root / "potsdam-pspnet-rgb-v1"
    d2_metrics_path = (
        d2_root / "kaggle/preflight-v2-evidence/outputs/study/heldout/metrics.json"
    )
    d2_sha_index = d2_root / "review-final-20261105/sha256.json"
    verify_absolute_index(d2_sha_index, d2_metrics_path)
    d3_inventory = verify_relative_index(
        d3_root / "review-final-20261106", "analysis-inventory.json"
    )
    required = {"analysis.json", "confusion.csv", "per-class.csv", "per-scene.csv"}
    if not required <= d3_inventory.keys():
        raise ValueError("DINOv3 review inventory lacks required derived tables")
    psp_audit = read_json(psp_root / "benchmark/audit.json")
    if psp_audit.get("status") != "verified":
        raise ValueError("PSPNet benchmark audit is not verified")
    psp_files = verify_relative_index(
        psp_root / "benchmark/outputs", "output-index.json"
    )
    if "study/benchmark/metrics.json" not in psp_files:
        raise ValueError("PSPNet benchmark index does not contain metrics")
    d3_attempt = read_json(d3_root / "study/heldout/attempt.json")
    d3_phase_files = verify_relative_index(
        d3_root / "study/heldout", "artifact_sha256_manifest.json"
    )
    if "metrics.json" not in d3_phase_files:
        raise ValueError("DINOv3 held-out artifact manifest does not contain metrics")
    for path in (
        d2_metrics_path,
        d2_sha_index,
        psp_root / "benchmark/audit.json",
        d3_root / "study/heldout/attempt.json",
    ):
        record_file(provenance, path)
    return {
        "DINOv2_metrics_sha256": sha256(d2_metrics_path),
        "DINOv3_metrics_sha256": sha256(d3_root / "study/heldout/metrics.json"),
        "PSPNet_metrics_sha256": sha256(
            psp_root / "benchmark/outputs/study/benchmark/metrics.json"
        ),
        "DINOv3_attempt": d3_attempt,
        "DINOv3_review_files_verified": sorted(required),
        "PSPNet_review_status": psp_audit["status"],
        "DINOv2_reference_hash_verified": True,
    }


def run(args: argparse.Namespace) -> None:
    run_root = args.runs_root.resolve()
    output = args.output_dir.resolve()
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    provenance: dict[str, dict[str, Any]] = {}
    audit_summary = verify_sources(run_root, provenance)

    d2_path = (
        run_root
        / "potsdam-dinov2-coverage-refit-v1"
        / "kaggle/preflight-v2-evidence/outputs/study/heldout/metrics.json"
    )
    d3_path = run_root / "potsdam-dinov3-rgbir-v2/study/heldout/metrics.json"
    psp_path = (
        run_root
        / "potsdam-pspnet-rgb-v1/benchmark/outputs/study/benchmark/metrics.json"
    )
    d2, d3, psp = read_json(d2_path), read_json(d3_path), read_json(psp_path)
    d2_scenes, _ = metric_from_scene_records(d2, "DINOv2")
    d3_scenes, _ = metric_from_scene_records(d3, "DINOv3")
    psp_scenes, _ = metric_from_scene_records(psp, "PSPNet")
    if not np.array_equal(
        d2_scenes.sum(axis=0),
        checked_matrix(d2["confusion_matrix"], "DINOv2 aggregate"),
    ):
        raise ValueError("DINOv2 scenes fail to reconstruct the aggregate")

    error_summary = create_error_outputs(
        checked_matrix(d2["confusion_matrix"], "DINOv2 aggregate"),
        checked_matrix(d3["confusion_matrix"], "DINOv3 aggregate"),
        output,
    )
    scene_summary = create_scene_outputs(d2_scenes, d3_scenes, output)
    negative_summary = create_negative_results(run_root, output, provenance)
    cost_summary = create_cost_outputs(
        run_root, output, provenance, args.paired_manifest
    )
    paired_summary_path = output / "data/paired-prediction-summary.json"
    paired_pixel_counts = (
        read_json(paired_summary_path)
        if args.paired_manifest is not None and paired_summary_path.is_file()
        else "see paired_analysis.py output"
    )

    for path in (d2_path, d3_path, psp_path):
        record_file(provenance, path)
    run_summary = {
        "status": "verified-from-archived-metrics",
        "generated_at": "runtime-generated; see file modification time",
        "benchmark_scope": (
            "Historical, previously observed Potsdam 14-scene cohort; not a "
            "newly sampled independent test set."
        ),
        "scene_ids": list(SCENES),
        "classes": list(CLASSES),
        "audit_summary": audit_summary,
        "error_decomposition": error_summary,
        "scene_robustness": scene_summary,
        "negative_results": negative_summary,
        "quality_cost": cost_summary,
        "inputs": provenance,
        "paired_predictions": (
            {
                "status": "verified-exact-confusion-reproduction",
                "manifest": str(args.paired_manifest.resolve()),
                "models": {
                    name: {
                        "checkpoint_sha256": model["checkpoint_sha256"],
                        "peak_cuda_allocated_bytes": model["peak_cuda_allocated_bytes"],
                        "peak_cuda_reserved_bytes": model["peak_cuda_reserved_bytes"],
                    }
                    for name, model in read_json(args.paired_manifest)["models"].items()
                },
                "paired_pixel_counts": paired_pixel_counts,
            }
            if args.paired_manifest is not None
            else "not performed by this offline metrics-only step"
        ),
    }
    script_path = Path(__file__).resolve()
    run_summary["analysis_script"] = {
        "path": str(script_path),
        "sha256": sha256(script_path),
    }
    (output / "data/potsdam-empirical-analysis.json").write_text(
        json.dumps(run_summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in run_summary.items()
                if key not in {"inputs", "negative_results"}
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/research"))
    parser.add_argument("--paired-manifest", type=Path)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
