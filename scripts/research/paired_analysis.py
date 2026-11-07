"""Audit paired held-out predictions and produce deterministic case studies."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deepnet.potsdam_rgbir.data import CacheBundle
from notebooks_src import eda_plots
from scripts.research.potsdam_empirical import CLASSES, PIXELS, SCENES, metric_values

BLOCK = 512
CASE_GROUPS = (
    ("dinov3_only_correct", "DINOv3 corrects PSPNet", "dinov3_only_correct"),
    ("pspnet_only_correct", "PSPNet corrects DINOv3", "pspnet_only_correct"),
    ("shared_errors", "Both models are wrong", "both_wrong"),
    (
        "car_clutter_disagreement",
        "Car / clutter disagreement",
        "car_clutter_disagreement",
    ),
    (
        "vegetation_disagreement",
        "Tree / low-vegetation disagreement",
        "vegetation_disagreement",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def verify_output_index(root: Path) -> dict[str, str]:
    index = read_json(root / "output-index.json")
    verified = {}
    for record in index["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe indexed output path: {relative}")
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise ValueError(f"Kaggle output index mismatch: {path}")
        verified[relative.as_posix()] = record["sha256"]
    return verified


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_prediction(path: Path, expected_hash: str, indexed_hash: str) -> np.ndarray:
    if not path.is_file() or indexed_hash != expected_hash:
        raise ValueError(f"Prediction is absent from the verified output index: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["prediction"]:
            raise ValueError(f"Unexpected prediction members in {path}")
        prediction = archive["prediction"]
    if prediction.ndim != 2 or prediction.dtype != np.uint8:
        raise ValueError(f"Invalid prediction map: {path}")
    return prediction


def palette_for(cache: CacheBundle) -> ListedColormap:
    color_by_class = {
        int(class_id): (red / 255, green / 255, blue / 255)
        for red, green, blue, class_id in cache.preprocessing.label_palette
        if int(class_id) < len(CLASSES)
    }
    if set(color_by_class) != set(range(len(CLASSES))):
        raise ValueError("Reference palette does not contain the six scored classes")
    cmap = ListedColormap([color_by_class[i] for i in range(len(CLASSES))])
    cmap.set_bad("#dedede")
    return cmap


def paired_counts(
    truth: np.ndarray,
    dino: np.ndarray,
    psp: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    valid = valid & (truth < len(CLASSES))
    labels = truth[valid].astype(np.int64, copy=False)
    dino_labels = dino[valid].astype(np.int64, copy=False)
    psp_labels = psp[valid].astype(np.int64, copy=False)
    if np.any(dino_labels >= len(CLASSES)) or np.any(psp_labels >= len(CLASSES)):
        raise ValueError("A model emitted an invalid class on a scored pixel")
    dino_ok = dino_labels == labels
    psp_ok = psp_labels == labels
    both_wrong = ~dino_ok & ~psp_ok
    return {
        "valid": int(valid.sum()),
        "both_correct": int(np.count_nonzero(dino_ok & psp_ok)),
        "both_wrong": int(np.count_nonzero(both_wrong)),
        "dinov3_only_correct": int(np.count_nonzero(dino_ok & ~psp_ok)),
        "pspnet_only_correct": int(np.count_nonzero(~dino_ok & psp_ok)),
        "shared_wrong_same_prediction": int(
            np.count_nonzero(both_wrong & (dino_labels == psp_labels))
        ),
        "shared_wrong_different_prediction": int(
            np.count_nonzero(both_wrong & (dino_labels != psp_labels))
        ),
        "prediction_agreement": int(np.count_nonzero(dino_labels == psp_labels)),
        "prediction_disagreement": int(np.count_nonzero(dino_labels != psp_labels)),
        "labels": labels,
        "dinov3_labels": dino_labels,
        "pspnet_labels": psp_labels,
        "dinov3_correct": dino_ok,
        "pspnet_correct": psp_ok,
    }


def verify_inputs(
    output_root: Path, cache: CacheBundle, runs_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    indexed_files = verify_output_index(output_root)
    marker = read_json(output_root / "paired-kernel-run.json")
    paired_root = output_root / "paired"
    paired = read_json(paired_root / "paired-predictions.json")
    if (
        marker.get("status") != "complete"
        or marker.get("training_performed") is not False
        or marker.get("independent_benchmark_claim") is not False
        or paired.get("status") != "verified-exact-confusion-reproduction"
    ):
        raise ValueError("Kaggle paired inference did not pass its completion gate")
    if (
        cache.manifest["included_splits"] != ["test"]
        or cache.manifest["cache_hash"] != paired["cache_hash"]
        or cache.split.content_hash != paired["split_hash"]
        or cache.split.label_asset_for("test") != "label_eroded"
        or paired["scene_ids"] != list(SCENES)
        or paired["pixels_scored"] != PIXELS
        or paired["window_size"] != [512, 512]
        or paired["stride"] != [256, 256]
        or paired["inference"] != "identity"
    ):
        raise ValueError("Paired predictions are not bound to the locked benchmark")

    archived = {
        "dinov3": runs_root / "potsdam-dinov3-rgbir-v2/study/heldout/metrics.json",
        "pspnet": runs_root
        / "potsdam-pspnet-rgb-v1/benchmark/outputs/study/benchmark/metrics.json",
    }
    metrics = {}
    for name, path in archived.items():
        if sha256(path) != paired["models"][name]["archived_metrics_sha256"]:
            raise ValueError(f"Archived metric hash differs for {name}")
        metrics[name] = read_json(path)
    if list(cache.split.test_scene_ids) != list(SCENES):
        raise ValueError("Held-out cache scene order differs from the study")
    return marker, paired, metrics, indexed_files


def paired_attempt_cost(output_root: Path) -> dict[str, Any]:
    run_root = output_root.parent.parent
    attempt_roots = [run_root, *sorted(run_root.glob("attempt-*"))]
    successful_gpu_seconds = 0.0
    failed_gpu_seconds = 0.0
    failed_attempts = []
    for attempt_root in attempt_roots:
        marker_path = attempt_root / "outputs/paired-kernel-run.json"
        if not marker_path.is_file():
            continue
        marker = read_json(marker_path)
        if marker.get("status") == "complete":
            successful_gpu_seconds += float(marker["execution_gpu_seconds"])
        elif marker.get("status") == "failed":
            package_path = attempt_root / "kaggle/package.json"
            allocation = (
                read_json(package_path).get("allocated_gpu_count", 2)
                if package_path.is_file()
                else 2
            )
            failed_gpu_seconds += float(marker["kernel_wall_seconds"]) * allocation
            failed_attempts.append(
                "attempt-1" if attempt_root == run_root else attempt_root.name
            )
    return {
        "successful_gpu_seconds": successful_gpu_seconds,
        "failed_gpu_seconds": failed_gpu_seconds,
        "failed_attempts": failed_attempts,
    }


def analyze(
    output_root: Path,
    cache_root: Path,
    runs_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    cache = CacheBundle.load(cache_root)
    marker, paired, archived, indexed_files = verify_inputs(
        output_root, cache, runs_root
    )
    paired_root = output_root / "paired"
    scene_rows: list[dict[str, Any]] = []
    support = np.zeros(6, dtype=np.int64)
    both_correct = np.zeros(6, dtype=np.int64)
    both_wrong = np.zeros(6, dtype=np.int64)
    dino_only = np.zeros(6, dtype=np.int64)
    psp_only = np.zeros(6, dtype=np.int64)
    dino_correction_transition = np.zeros((6, 6), dtype=np.int64)
    psp_correction_transition = np.zeros((6, 6), dtype=np.int64)
    totals = {
        key: 0
        for key in (
            "valid",
            "both_correct",
            "both_wrong",
            "dinov3_only_correct",
            "pspnet_only_correct",
            "shared_wrong_same_prediction",
            "shared_wrong_different_prediction",
            "prediction_agreement",
            "prediction_disagreement",
        )
    }
    candidates: list[dict[str, Any]] = []
    confusion_by_model = {name: np.zeros((6, 6), dtype=np.int64) for name in archived}

    with cache.access("test", auxiliary=False) as access:
        for scene_id in SCENES:
            scene_meta = paired["models"]["dinov3"]["scenes"]
            dino_record = next(row for row in scene_meta if row["scene_id"] == scene_id)
            psp_record = next(
                row
                for row in paired["models"]["pspnet"]["scenes"]
                if row["scene_id"] == scene_id
            )
            dino = load_prediction(
                paired_root / dino_record["prediction_path"],
                dino_record["prediction_sha256"],
                indexed_files[
                    (Path("paired") / dino_record["prediction_path"]).as_posix()
                ],
            )
            psp = load_prediction(
                paired_root / psp_record["prediction_path"],
                psp_record["prediction_sha256"],
                indexed_files[
                    (Path("paired") / psp_record["prediction_path"]).as_posix()
                ],
            )
            arrays = access._arrays(scene_id)
            labels = arrays["labels"]
            valid_map = arrays["valid"]
            height, width = labels.shape
            if dino.shape != labels.shape or psp.shape != labels.shape:
                raise ValueError(f"Prediction and cache shapes differ for {scene_id}")
            scene_counts = {key: 0 for key in totals}
            matrices = {name: np.zeros((6, 6), dtype=np.int64) for name in archived}
            for row in range(0, height, BLOCK):
                row_end = min(height, row + BLOCK)
                for col in range(0, width, BLOCK):
                    col_end = min(width, col + BLOCK)
                    truth_crop = np.asarray(labels[row:row_end, col:col_end])
                    valid_crop = np.asarray(valid_map[row:row_end, col:col_end]).astype(
                        bool
                    )
                    dino_crop = dino[row:row_end, col:col_end]
                    psp_crop = psp[row:row_end, col:col_end]
                    pair = paired_counts(truth_crop, dino_crop, psp_crop, valid_crop)
                    for key in totals:
                        totals[key] += pair[key]
                        scene_counts[key] += pair[key]
                    truth_v = pair["labels"]
                    dino_v = pair["dinov3_labels"]
                    psp_v = pair["pspnet_labels"]
                    dino_ok = pair["dinov3_correct"]
                    psp_ok = pair["pspnet_correct"]
                    if len(truth_v):
                        matrices["dinov3"] += np.bincount(
                            truth_v * 6 + dino_v, minlength=36
                        ).reshape(6, 6)
                        matrices["pspnet"] += np.bincount(
                            truth_v * 6 + psp_v, minlength=36
                        ).reshape(6, 6)
                        support += np.bincount(truth_v, minlength=6)
                        both_correct += np.bincount(
                            truth_v[dino_ok & psp_ok], minlength=6
                        )
                        both_wrong += np.bincount(
                            truth_v[~dino_ok & ~psp_ok], minlength=6
                        )
                        dino_only += np.bincount(
                            truth_v[dino_ok & ~psp_ok], minlength=6
                        )
                        psp_only += np.bincount(truth_v[~dino_ok & psp_ok], minlength=6)
                        dino_fix = dino_ok & ~psp_ok
                        psp_fix = ~dino_ok & psp_ok
                        dino_correction_transition += np.bincount(
                            truth_v[dino_fix] * 6 + psp_v[dino_fix], minlength=36
                        ).reshape(6, 6)
                        psp_correction_transition += np.bincount(
                            truth_v[psp_fix] * 6 + dino_v[psp_fix], minlength=36
                        ).reshape(6, 6)

                    candidate = {
                        "scene": scene_id,
                        "row": row,
                        "col": col,
                        "height": row_end - row,
                        "width": col_end - col,
                        "valid_pixels": pair["valid"],
                        "dinov3_only_correct": pair["dinov3_only_correct"],
                        "pspnet_only_correct": pair["pspnet_only_correct"],
                        "both_wrong": pair["both_wrong"],
                        "prediction_disagreement": pair["prediction_disagreement"],
                        "car_clutter_disagreement": int(
                            np.count_nonzero(
                                np.isin(pair["labels"], (4, 5))
                                & (pair["dinov3_labels"] != pair["pspnet_labels"])
                            )
                        ),
                        "vegetation_disagreement": int(
                            np.count_nonzero(
                                np.isin(pair["labels"], (2, 3))
                                & (pair["dinov3_labels"] != pair["pspnet_labels"])
                            )
                        ),
                    }
                    candidates.append(candidate)

            for model_name, matrix in matrices.items():
                reference = next(
                    row
                    for row in archived[model_name]["scenes"]
                    if row["scene_id"] == scene_id
                )
                if not np.array_equal(
                    matrix, np.asarray(reference["confusion_matrix"])
                ):
                    raise ValueError(
                        f"Local {model_name} matrix differs for {scene_id}"
                    )
                confusion_by_model[model_name] += matrix
            if scene_counts["valid"] != dino_record["valid_pixels"]:
                raise ValueError(f"Valid-pixel support differs for {scene_id}")
            scene_rows.append(
                {
                    "scene": scene_id,
                    **scene_counts,
                    "DINOv3_OA_percent": 100
                    * metric_values(matrices["dinov3"])["overall_accuracy"],
                    "PSPNet_OA_percent": 100
                    * metric_values(matrices["pspnet"])["overall_accuracy"],
                }
            )
            del dino, psp
            gc.collect()

    if totals["valid"] != PIXELS:
        raise ValueError(
            f"Paired valid support is {totals['valid']}, expected {PIXELS}"
        )
    if (
        totals["both_correct"]
        + totals["both_wrong"]
        + totals["dinov3_only_correct"]
        + totals["pspnet_only_correct"]
        != PIXELS
        or totals["prediction_agreement"] + totals["prediction_disagreement"] != PIXELS
        or totals["shared_wrong_same_prediction"]
        + totals["shared_wrong_different_prediction"]
        != totals["both_wrong"]
    ):
        raise ValueError("Paired correctness states do not partition the scored pixels")
    for model_name in archived:
        if not np.array_equal(
            confusion_by_model[model_name],
            np.asarray(archived[model_name]["confusion_matrix"], dtype=np.int64),
        ):
            raise ValueError(f"Paired {model_name} matrix differs in aggregate")
    if totals["dinov3_only_correct"] - totals["pspnet_only_correct"] != int(
        np.trace(confusion_by_model["dinov3"]) - np.trace(confusion_by_model["pspnet"])
    ):
        raise ValueError("Paired correction counts do not explain the accuracy delta")

    class_rows = []
    for class_id, class_name in enumerate(CLASSES):
        class_rows.append(
            {
                "class": class_name,
                "support": int(support[class_id]),
                "both_correct": int(both_correct[class_id]),
                "both_wrong": int(both_wrong[class_id]),
                "DINOv3_only_correct": int(dino_only[class_id]),
                "PSPNet_only_correct": int(psp_only[class_id]),
                "net_DINOv3_correct_pixels": int(
                    dino_only[class_id] - psp_only[class_id]
                ),
                "DINOv3_errors": int(
                    support[class_id] - both_correct[class_id] - dino_only[class_id]
                ),
                "PSPNet_errors": int(
                    support[class_id] - both_correct[class_id] - psp_only[class_id]
                ),
            }
        )
    write_csv(output_dir / "data/paired-class-transitions.csv", class_rows)
    write_csv(output_dir / "data/paired-scene-transitions.csv", scene_rows)

    transition_rows = []
    for direction, matrix in (
        ("DINOv3 corrects PSPNet", dino_correction_transition),
        ("PSPNet corrects DINOv3", psp_correction_transition),
    ):
        for truth_id, prediction_id in zip(*np.nonzero(matrix), strict=True):
            transition_rows.append(
                {
                    "direction": direction,
                    "true_class": CLASSES[truth_id],
                    "other_model_prediction": CLASSES[prediction_id],
                    "count": int(matrix[truth_id, prediction_id]),
                }
            )
    write_csv(output_dir / "data/paired-correction-confusions.csv", transition_rows)

    comparison = {key: int(value) for key, value in totals.items()}
    attempt_cost = paired_attempt_cost(output_root)
    comparison.update(
        {
            "DINOv3_OA_percent": 100
            * metric_values(confusion_by_model["dinov3"])["overall_accuracy"],
            "PSPNet_OA_percent": 100
            * metric_values(confusion_by_model["pspnet"])["overall_accuracy"],
            "DINOv3_mIoU_percent": 100
            * metric_values(confusion_by_model["dinov3"])["mean_iou"],
            "PSPNet_mIoU_percent": 100
            * metric_values(confusion_by_model["pspnet"])["mean_iou"],
            "net_DINOv3_correct_pixels": int(
                totals["dinov3_only_correct"] - totals["pspnet_only_correct"]
            ),
            "paired_inference_successful_gpu_hours": (
                marker["execution_gpu_seconds"] / 3600
            ),
            "paired_inference_attempt_gpu_hours": (
                attempt_cost["successful_gpu_seconds"]
                + attempt_cost["failed_gpu_seconds"]
            )
            / 3600,
            "paired_inference_failed_bootstrap_gpu_hours": (
                attempt_cost["failed_gpu_seconds"] / 3600
            ),
            "paired_inference_failed_attempts": attempt_cost["failed_attempts"],
            "paired_models": {
                name: {
                    "peak_cuda_allocated_bytes": model["peak_cuda_allocated_bytes"],
                    "peak_cuda_reserved_bytes": model["peak_cuda_reserved_bytes"],
                }
                for name, model in paired["models"].items()
            },
            "reference": "14 historical benchmark scenes with eroded labels",
            "cache_hash": paired["cache_hash"],
            "inference_source_sha256": paired["source_sha256"],
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
            "training_performed": False,
            "model_confusion_matrices_reproduced": True,
        }
    )
    (output_dir / "data/paired-prediction-summary.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )
    cases = select_cases(candidates)
    render_cases(cases, cache, paired_root, output_dir)
    write_csv(output_dir / "data/paired-case-studies.csv", cases)
    return comparison


def select_cases(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    used: set[tuple[str, int, int]] = set()
    selected = []
    for group_id, group_title, score_key in CASE_GROUPS:
        ranked = sorted(
            candidates,
            key=lambda row: (
                -row[score_key],
                -row["prediction_disagreement"],
                row["scene"],
                row["row"],
                row["col"],
            ),
        )
        group_count = 0
        for candidate in ranked:
            key = (candidate["scene"], candidate["row"], candidate["col"])
            if key in used or candidate[score_key] == 0:
                continue
            used.add(key)
            selected.append(
                {
                    **candidate,
                    "case_group": group_id,
                    "case_group_title": group_title,
                    "selection_score": candidate[score_key],
                }
            )
            group_count += 1
            if group_count == 4:
                break
        if group_count < 4:
            raise ValueError(f"Only selected {group_count} cases for {group_id}")
    return selected


def _draw_case_row(
    axes: np.ndarray,
    case: dict[str, Any],
    access: Any,
    paired_root: Path,
    scene_records: dict[str, dict[str, dict[str, Any]]],
    class_cmap: ListedColormap,
    outcome_cmap: ListedColormap,
) -> None:
    scene = case["scene"]
    row, col = case["row"], case["col"]
    height, width = case["height"], case["width"]
    arrays = access._arrays(scene)
    image = np.asarray(arrays["optical"][:3, row : row + height, col : col + width])
    image = np.moveaxis(image, 0, -1)
    truth = np.asarray(arrays["labels"][row : row + height, col : col + width])
    valid = np.asarray(arrays["valid"][row : row + height, col : col + width]).astype(
        bool
    )
    dino_path = paired_root / scene_records["dinov3"][scene]["prediction_path"]
    psp_path = paired_root / scene_records["pspnet"][scene]["prediction_path"]
    with np.load(dino_path, allow_pickle=False) as archive:
        dino = archive["prediction"][row : row + height, col : col + width]
    with np.load(psp_path, allow_pickle=False) as archive:
        psp = archive["prediction"][row : row + height, col : col + width]
    scored = valid & (truth < len(CLASSES))
    dino_correct = (dino == truth) & scored
    psp_correct = (psp == truth) & scored
    outcome = np.full(truth.shape, 1, dtype=np.uint8)
    outcome[dino_correct & psp_correct] = 0
    outcome[(~dino_correct) & (~psp_correct) & (dino != psp)] = 2
    outcome[dino_correct & ~psp_correct] = 3
    outcome[~dino_correct & psp_correct] = 4
    panels = (
        (image, None),
        (np.ma.masked_where(~scored, truth), class_cmap),
        (np.ma.masked_where(~scored, dino), class_cmap),
        (np.ma.masked_where(~scored, psp), class_cmap),
        (np.ma.masked_where(~scored, outcome), outcome_cmap),
    )
    for column, (panel, cmap) in enumerate(panels):
        if cmap is None:
            axes[column].imshow(panel)
        else:
            axes[column].imshow(
                panel,
                cmap=cmap,
                vmin=0,
                vmax=(4 if column == 4 else 5),
                interpolation="nearest",
            )
        axes[column].set_xticks([])
        axes[column].set_yticks([])
        if axes[column].get_subplotspec().rowspan.start == 0:
            axes[column].set_title(
                ("RGB", "Reference", "DINOv3", "PSPNet", "Paired outcome")[column]
            )
    axes[0].set_ylabel(f"{scene} · r{row}, c{col}\n{case['case_group']}", fontsize=8)


def render_cases(
    cases: list[dict[str, Any]], cache: CacheBundle, paired_root: Path, output_dir: Path
) -> None:
    class_cmap = palette_for(cache)
    outcome_cmap = ListedColormap(
        ("#009E73", "#999999", "#E69F00", "#0072B2", "#D55E00")
    )
    outcome_cmap.set_bad("#dedede")
    outcome_labels = (
        "Both correct",
        "Both wrong, same prediction",
        "Both wrong, different prediction",
        "DINOv3 only correct",
        "PSPNet only correct",
    )
    output_figures = output_dir / "figures"
    output_figures.mkdir(parents=True, exist_ok=True)
    manifest = read_json(paired_root / "paired-predictions.json")
    scene_records = {
        model: {row["scene_id"]: row for row in manifest["models"][model]["scenes"]}
        for model in ("dinov3", "pspnet")
    }
    class_transition_rows = []
    with cache.access("test", auxiliary=False) as access:
        for case in cases:
            scene = case["scene"]
            row, col = case["row"], case["col"]
            height, width = case["height"], case["width"]
            arrays = access._arrays(scene)
            truth = np.asarray(
                arrays["labels"][row : row + height, col : col + width]
            )
            valid = np.asarray(
                arrays["valid"][row : row + height, col : col + width]
            ).astype(bool) & (truth < len(CLASSES))
            predictions = {}
            for model in ("dinov3", "pspnet"):
                prediction_path = (
                    paired_root
                    / scene_records[model][scene]["prediction_path"]
                )
                with np.load(prediction_path, allow_pickle=False) as archive:
                    predictions[model] = archive["prediction"][
                        row : row + height, col : col + width
                    ]
            truth_values = truth[valid].astype(np.int64, copy=False)
            dino_values = predictions["dinov3"][valid].astype(
                np.int64, copy=False
            )
            psp_values = predictions["pspnet"][valid].astype(
                np.int64, copy=False
            )
            joint_counts = np.bincount(
                (truth_values * len(CLASSES) + dino_values) * len(CLASSES)
                + psp_values,
                minlength=len(CLASSES) ** 3,
            ).reshape(len(CLASSES), len(CLASSES), len(CLASSES))
            for truth_id, dino_id, psp_id in zip(
                *np.nonzero(joint_counts), strict=True
            ):
                class_transition_rows.append(
                    {
                        "case_group": case["case_group"],
                        "scene": scene,
                        "row": row,
                        "col": col,
                        "true_class": CLASSES[truth_id],
                        "dinov3_prediction": CLASSES[dino_id],
                        "pspnet_prediction": CLASSES[psp_id],
                        "pixels": int(joint_counts[truth_id, dino_id, psp_id]),
                    }
                )
    write_csv(output_dir / "data/paired-case-confusions.csv", class_transition_rows)

    class_handles = [Patch(color=class_cmap(i), label=CLASSES[i]) for i in range(6)]
    outcome_handles = [
        Patch(color=outcome_cmap(i), label=label)
        for i, label in enumerate(outcome_labels)
    ]
    with cache.access("test", auxiliary=False) as access:
        overview_cases = []
        for group_id, _, _ in CASE_GROUPS:
            group_cases = [row for row in cases if row["case_group"] == group_id]
            overview_cases.append(group_cases[0])
            with plt.style.context(eda_plots.STYLE):
                fig, axes = plt.subplots(4, 5, figsize=(17, 11), layout="constrained")
                for row_index, case in enumerate(group_cases):
                    _draw_case_row(
                        axes[row_index],
                        case,
                        access,
                        paired_root,
                        scene_records,
                        class_cmap,
                        outcome_cmap,
                    )
                fig.legend(
                    handles=class_handles,
                    loc="outside lower center",
                    ncol=6,
                    title="Class colors",
                )
                fig.legend(
                    handles=outcome_handles,
                    loc="outside upper center",
                    ncol=3,
                    title="Paired pixel correctness",
                )
                fig.savefig(output_figures / f"paired-cases-{group_id}.png", dpi=180)
                plt.close(fig)

        with plt.style.context(eda_plots.STYLE):
            fig, axes = plt.subplots(5, 5, figsize=(17, 13), layout="constrained")
            for row_index, case in enumerate(overview_cases):
                _draw_case_row(
                    axes[row_index],
                    case,
                    access,
                    paired_root,
                    scene_records,
                    class_cmap,
                    outcome_cmap,
                )
            fig.legend(
                handles=class_handles,
                loc="outside lower center",
                ncol=6,
                title="Class colors",
            )
            fig.legend(
                handles=outcome_handles,
                loc="outside upper center",
                ncol=3,
                title="Paired pixel correctness",
            )
            fig.savefig(output_figures / "paired-cases-overview.png", dpi=180)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle-output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "/Users/macbook/Data/deepnet-for-semantic-labeling-photogrammetry/runs"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/research"))
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(args.kaggle_output, args.cache, args.runs_root, args.output_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
