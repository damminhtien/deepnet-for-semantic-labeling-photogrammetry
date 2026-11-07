"""Restyle saved DeepNet figures without loading rasters or running models."""

import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

from notebooks_src import eda_plots

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Impervious", "Building", "Low vegetation", "Tree", "Car", "Clutter"]
COLORS = ("#0072B2", "#E69F00", "#009E73")
BANDS = ("ir", "red", "green")
LEGACY_COLORS = ("gray", "blue", "cyan", "green", "yellow", "red")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed_file(directory, index, filename):
    path = directory / filename
    records = json.loads((directory / index).read_text())["files"]
    if not any(r["path"] == filename and r["sha256"] == sha256(path) for r in records):
        raise ValueError(f"File does not match its audit index: {path}")
    return path


def save(fig, path):
    fig.savefig(path, dpi=200)
    plt.close(fig)


@plt.style.context(eda_plots.STYLE)
def selection_figure(run):
    review = run / "review-selection-abc-20261106"
    index = "ABC-review-index.json"
    analysis = json.loads(indexed_file(review, index, "ABC-analysis.json").read_text())
    classes = pd.read_csv(indexed_file(review, index, "ABC-classes.csv"))
    if len(classes) != 6:
        raise ValueError("Expected six class IoUs per arm")
    config = ROOT / "configs/training/potsdam-dinov3-rgbir-v2.json"
    reference = np.array(
        json.loads(config.read_text())["reference_validation_confusion"]
    )
    floor = 100 * np.trace(reference) / reference.sum()

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.1), layout="constrained")
    names = ("A: RGB, baseline aug.", "B: RGB + auxiliary", "C: RGB, enhanced aug.")
    for arm, name, color, marker in zip("ABC", names, COLORS, "os^", strict=True):
        source = f"study/{arm}/manifest.json"
        path = run / source
        if sha256(path) != analysis["input_files"][source]:
            raise ValueError(f"Manifest does not match the audit: {path}")
        manifest = json.loads(path.read_text())
        points = [row for row in manifest["history"] if "full_validation" in row]
        if [row["pass"] for row in points] != [4, 8] or any(
            row["full_validation"]["pixels_scored"] != 199_409_714 for row in points
        ):
            raise ValueError(f"Unexpected validation schedule or support: {path}")
        if not np.allclose(
            points[-1]["full_validation"]["per_class_iou"],
            classes[f"{arm}_iou"],
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(f"Class IoUs disagree with the audit table: {path}")
        left.plot(
            [4, 8],
            [100 * row["full_validation"]["overall_accuracy"] for row in points],
            color=color,
            marker=marker,
            label=name,
        )
    left.axhline(floor, color="0.35", ls="--", label="Historical C OA floor")
    left.set(
        xticks=[4, 8],
        xlim=(3.7, 8.3),
        xlabel="Completed coverage passes",
        ylabel="Full validation OA (%)",
    )
    left.set_title("Full-scene validation", loc="left")
    left.legend(loc="lower right")

    y = np.arange(6)
    for offset, arm, color in ((-0.19, "B", COLORS[1]), (0.19, "C", COLORS[2])):
        values = 100 * (classes[f"{arm}_iou"] - classes["A_iou"])
        bars = right.barh(
            y + offset, values, height=0.34, color=color, label=f"{arm} − A"
        )
        right.bar_label(
            bars, labels=[f"{v:+.3f}" for v in values], padding=2, fontsize=8
        )
    right.axvline(0, color="0.25", lw=0.8)
    right.set_yticks(y, CLASSES)
    right.invert_yaxis()
    right.set(xlabel="IoU difference (percentage points)", xlim=(-1.48, 0.35))
    right.set_title("Per-class IoU at pass 8", loc="left")
    right.legend(loc="upper left")
    fig.suptitle("Potsdam DINOv3 selection · seed 42")
    return fig


@plt.style.context(eda_plots.STYLE)
def benchmark_figure(run):
    review = run / "review-final-20261106"
    index = "analysis-inventory.json"
    classes = pd.read_csv(indexed_file(review, index, "per-class.csv"))
    scenes = pd.read_csv(indexed_file(review, index, "per-scene.csv"))
    if (
        classes["class_id"].tolist() != list(range(6))
        or len(scenes) != 14
        or not scenes["scene_id"].is_unique
        or scenes["pixels_scored"].sum() != 467_862_855
        or classes["truth_pixels"].sum() != 467_862_855
    ):
        raise ValueError("Unexpected historical benchmark coverage")

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    x = np.arange(6)
    for i, name in enumerate(("SAM", "DINOv2 refit", "DINOv3 RGB refit")):
        left.bar(
            x + (i - 1) * 0.25,
            classes[f"{name} IoU percent"],
            width=0.24,
            label=name,
            color=COLORS[i],
        )
    left.set_xticks(x, CLASSES, rotation=30, ha="right")
    left.set(ylabel="Class IoU (%)", ylim=(0, 103))
    left.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)

    deltas = scenes["DINOv3 minus DINOv2 OA pp"]
    if not np.allclose(
        deltas,
        scenes["DINOv3 RGB refit OA percent"] - scenes["DINOv2 refit OA percent"],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("Scene OA deltas disagree with the saved scores")
    right.bar(
        np.arange(len(scenes)),
        deltas,
        color=np.where(deltas >= 0, COLORS[0], "#D55E00"),
    )
    right.axhline(0, color="0.25", lw=0.8)
    right.set_xticks(
        np.arange(len(scenes)), scenes["scene_id"], rotation=55, ha="right"
    )
    right.set(ylabel="OA difference (percentage points)")
    right.set_title("DINOv3 RGB refit − DINOv2 refit", loc="left")
    fig.suptitle("Potsdam historical benchmark · 467,862,855 scored pixels")
    return fig


@plt.style.context(eda_plots.STYLE)
def model_comparison_figure(pspnet_run, dinov3_run):
    """Compare audited Potsdam model scores without re-running inference."""
    dino_review = dinov3_run / "review-final-20261106"
    dino_analysis = json.loads(
        indexed_file(
            dino_review, "analysis-inventory.json", "analysis.json"
        ).read_text()
    )
    dino_classes = pd.read_csv(
        indexed_file(dino_review, "analysis-inventory.json", "per-class.csv")
    )
    dino_scenes = pd.read_csv(
        indexed_file(dino_review, "analysis-inventory.json", "per-scene.csv"),
        dtype={"scene_id": str},
    )
    if (
        dino_analysis.get("status") != "verified-final-benchmark"
        or dino_analysis.get("evaluation_count") != 1
        or dino_analysis.get("pixels_scored") != 467_862_855
        or dino_classes["class_id"].tolist() != list(range(6))
        or dino_classes["truth_pixels"].sum() != 467_862_855
        or len(dino_scenes) != 14
        or not dino_scenes["scene_id"].is_unique
        or dino_scenes["pixels_scored"].sum() != 467_862_855
    ):
        raise ValueError("Unexpected audited DINOv3 benchmark coverage")

    psp_output = pspnet_run / "benchmark/outputs"
    psp_metrics_path = indexed_file(
        psp_output, "output-index.json", "study/benchmark/metrics.json"
    )
    psp_metrics = json.loads(psp_metrics_path.read_text())
    psp_attempt = json.loads(
        indexed_file(
            psp_output, "output-index.json", "study/benchmark/attempt.json"
        ).read_text()
    )
    psp_refit = json.loads((pspnet_run / "refit/audit.json").read_text())
    psp_audit = json.loads((pspnet_run / "benchmark/audit.json").read_text())
    comparison = json.loads((pspnet_run / "comparison.json").read_text())

    dino_heldout = dinov3_run / "study/heldout"
    dino_metrics_path = dino_heldout / "metrics.json"
    dino_metrics = json.loads(dino_metrics_path.read_text())
    dino_attempt = json.loads((dino_heldout / "attempt.json").read_text())
    if (
        psp_audit.get("status") != "verified"
        or psp_refit.get("status") != "verified"
        or psp_attempt.get("checkpoint_sha256") != psp_metrics.get("checkpoint_sha256")
        or psp_attempt.get("inference") != psp_metrics.get("inference")
        or psp_attempt.get("cache_hash") != psp_metrics.get("cache_hash")
        or psp_attempt.get("reference_protocol")
        != psp_metrics.get("reference_protocol")
        or psp_attempt.get("heldout_evaluation_count") != 1
        or psp_metrics.get("heldout_evaluation_count") != 1
        or psp_metrics.get("checkpoint_sha256") != psp_refit.get("checkpoint_sha256")
        or dino_attempt.get("evaluation_count") != 1
        or dino_attempt.get("checkpoint_sha256")
        != dino_metrics.get("checkpoint_sha256")
        or dino_attempt.get("protocol_hash") != dino_metrics.get("protocol_hash")
        or dino_attempt.get("inference") != dino_metrics.get("inference")
        or dino_metrics.get("heldout_evaluation_count") != 1
        or dino_metrics.get("checkpoint_sha256")
        != dino_analysis.get("checkpoint_sha256")
        or comparison.get("pspnet_checkpoint_sha256")
        != psp_metrics.get("checkpoint_sha256")
        or comparison.get("dinov3_checkpoint_sha256")
        != dino_metrics.get("checkpoint_sha256")
        or comparison.get("dinov3_metrics_sha256") != sha256(dino_metrics_path)
        or psp_metrics.get("cache_hash") != dino_attempt.get("cache_hash")
        or psp_metrics.get("scene_ids") != dino_metrics.get("scene_ids")
        or psp_metrics.get("window_size") != dino_metrics.get("window_size")
        or psp_metrics.get("stride") != dino_metrics.get("stride")
        or psp_metrics.get("inference") != dino_metrics.get("inference")
        or psp_metrics.get("pixels_scored") != 467_862_855
        or dino_metrics.get("pixels_scored") != 467_862_855
        or len(psp_metrics.get("scene_ids", [])) != 14
        or len(psp_metrics.get("scenes", [])) != 14
        or len(dino_metrics.get("scenes", [])) != 14
        or psp_metrics.get("scene_ids") != dino_scenes["scene_id"].tolist()
        or dino_metrics.get("scene_ids") != dino_scenes["scene_id"].tolist()
    ):
        raise ValueError("Model benchmark audit bindings differ")

    model_names = ("SAM", "DINOv2 refit", "DINOv3 RGB refit", "PSPNet RGB refit")
    model_labels = ("SAM", "DINOv2", "DINOv3", "PSPNet")
    model_colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    colors_by_name = dict(zip(model_names, model_colors, strict=True))
    metrics_by_name = {name: dino_analysis["metrics"][name] for name in model_names[:3]}
    metrics_by_name[model_names[-1]] = psp_metrics
    matrices_by_name = {
        name: np.asarray(metrics["confusion_matrix"])
        for name, metrics in metrics_by_name.items()
    }
    if not np.array_equal(
        matrices_by_name["DINOv3 RGB refit"],
        np.asarray(dino_metrics["confusion_matrix"]),
    ):
        raise ValueError("DINOv3 review matrix differs from its benchmark output")

    for name, metrics in metrics_by_name.items():
        matrix = matrices_by_name[name]
        if matrix.shape != (6, 6):
            raise ValueError(f"Expected a six-class confusion matrix for {name}")
        intersection = np.diag(matrix)
        union = matrix.sum(axis=1) + matrix.sum(axis=0) - intersection
        iou = intersection / union
        if (
            matrix.sum() != 467_862_855
            or not np.allclose(iou, metrics["per_class_iou"], rtol=0, atol=1e-12)
            or not np.isclose(
                np.trace(matrix) / matrix.sum(),
                metrics["overall_accuracy"],
                rtol=0,
                atol=1e-12,
            )
            or not np.isclose(np.mean(iou), metrics["mean_iou"], rtol=0, atol=1e-12)
        ):
            raise ValueError(f"{name} metrics disagree with the saved matrix")

    reference_support = matrices_by_name[model_names[0]].sum(axis=1)
    if any(
        not np.array_equal(matrix.sum(axis=1), reference_support)
        for matrix in matrices_by_name.values()
    ):
        raise ValueError("Model class supports differ")

    for name in model_names[:3]:
        if not np.allclose(
            dino_classes[f"{name} IoU percent"],
            100 * np.asarray(metrics_by_name[name]["per_class_iou"]),
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError(f"{name} class scores disagree with the audited matrix")

    scene_ids = dino_scenes["scene_id"].tolist()
    expected_pixels = dino_scenes["pixels_scored"].astype(int).tolist()
    psp_scenes = psp_metrics["scenes"]
    dino_scenes_output = dino_metrics["scenes"]
    if any(
        psp_scene["scene_id"] != scene_id
        or dino_scene["scene_id"] != scene_id
        or psp_scene["pixels_scored"] != pixels
        or dino_scene["pixels_scored"] != pixels
        for psp_scene, dino_scene, scene_id, pixels in zip(
            psp_scenes,
            dino_scenes_output,
            scene_ids,
            expected_pixels,
            strict=True,
        )
    ):
        raise ValueError("Per-scene model support differs")
    for metrics, scene_results in (
        (psp_metrics, psp_scenes),
        (dino_metrics, dino_scenes_output),
    ):
        scene_matrices = np.sum(
            [np.asarray(scene["confusion_matrix"]) for scene in scene_results], axis=0
        )
        if not np.array_equal(scene_matrices, np.asarray(metrics["confusion_matrix"])):
            raise ValueError("Per-scene matrices do not sum to the aggregate")
    if not np.allclose(
        dino_scenes["DINOv3 RGB refit OA percent"],
        100 * np.asarray([scene["overall_accuracy"] for scene in dino_scenes_output]),
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("DINOv3 per-scene scores disagree with the audit table")

    if (
        not np.isclose(
            100 * (psp_metrics["overall_accuracy"] - dino_metrics["overall_accuracy"]),
            comparison["pspnet_minus_dinov3_oa_pp"],
            rtol=0,
            atol=1e-10,
        )
        or not np.isclose(
            100 * (psp_metrics["mean_iou"] - dino_metrics["mean_iou"]),
            comparison["pspnet_minus_dinov3_miou_pp"],
            rtol=0,
            atol=1e-10,
        )
        or not np.isclose(
            psp_metrics["inference_seconds"],
            comparison["pspnet_inference_seconds"],
            rtol=0,
            atol=1e-10,
        )
        or not np.isclose(
            dino_metrics["inference_seconds"],
            comparison["dinov3_inference_seconds"],
            rtol=0,
            atol=1e-10,
        )
    ):
        raise ValueError("Saved model comparison disagrees with audited metrics")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), layout="constrained")
    score_ax, class_ax, scene_ax, time_ax = axes.flat

    y = np.arange(len(model_names))
    for index, name in enumerate(model_names):
        metrics = metrics_by_name[name]
        oa = 100 * metrics["overall_accuracy"]
        miou = 100 * metrics["mean_iou"]
        score_ax.scatter(
            oa, y[index] - 0.13, marker="o", color=colors_by_name[name], zorder=3
        )
        score_ax.scatter(
            miou, y[index] + 0.13, marker="^", color=colors_by_name[name], zorder=3
        )
        score_ax.annotate(
            f"{oa:.2f}",
            (oa, y[index] - 0.13),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
        score_ax.annotate(
            f"{miou:.2f}",
            (miou, y[index] + 0.13),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    score_ax.axvline(99, color="0.3", linestyle="--")
    score_ax.set(xlim=(70, 101), yticks=y, yticklabels=model_labels, xlabel="Score (%)")
    score_ax.invert_yaxis()
    score_ax.set_title("Overall accuracy and six-class mIoU", loc="left")
    for tick, color in zip(score_ax.get_yticklabels(), model_colors, strict=True):
        tick.set_color(color)
        tick.set_fontweight("bold")
    score_ax.legend(
        handles=(
            plt.Line2D([], [], marker="o", color="0.2", linestyle="", label="OA"),
            plt.Line2D([], [], marker="^", color="0.2", linestyle="", label="mIoU"),
            plt.Line2D([], [], color="0.3", linestyle="--", label="99% OA target"),
        ),
        loc="upper right",
        fontsize=8,
    )

    x = np.arange(6)
    width = 0.18
    for index, name in enumerate(model_names):
        values = 100 * np.asarray(metrics_by_name[name]["per_class_iou"])
        class_ax.bar(
            x + (index - 1.5) * width,
            values,
            width=width,
            label=model_labels[index],
            color=colors_by_name[name],
        )
    class_ax.set_xticks(x, CLASSES, rotation=28, ha="right")
    class_ax.set(ylabel="Class IoU (%)", ylim=(0, 108))
    class_ax.set_title("IoU by class", loc="left")
    class_ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)

    dino3_oa = dino_scenes["DINOv3 RGB refit OA percent"].to_numpy()
    scene_oa = {
        "SAM": dino_scenes["SAM OA percent"].to_numpy(),
        "DINOv2": dino_scenes["DINOv2 refit OA percent"].to_numpy(),
        "PSPNet": 100 * np.asarray([scene["overall_accuracy"] for scene in psp_scenes]),
    }
    for offset, name, model_name in (
        (-0.13, "SAM", "SAM"),
        (0, "DINOv2", "DINOv2 refit"),
        (0.13, "PSPNet", "PSPNet RGB refit"),
    ):
        scene_ax.scatter(
            scene_oa[name] - dino3_oa,
            np.arange(len(scene_ids)) + offset,
            color=colors_by_name[model_name],
            label=f"{name} \N{MINUS SIGN} DINOv3",
            s=22,
        )
    scene_ax.axvline(0, color="0.25", lw=0.8)
    scene_ax.set_yticks(np.arange(len(scene_ids)), scene_ids)
    scene_ax.invert_yaxis()
    scene_ax.set(xlabel="OA difference (percentage points)")
    scene_ax.set_title("Per-scene OA relative to DINOv3", loc="left")
    scene_ax.legend(loc="upper left", fontsize=8)

    times = (dino_metrics["inference_seconds"], psp_metrics["inference_seconds"])
    time_bars = time_ax.bar(
        model_labels[2:],
        times,
        color=(colors_by_name["DINOv3 RGB refit"], colors_by_name["PSPNet RGB refit"]),
    )
    time_ax.bar_label(
        time_bars, labels=[f"{value:.1f} s" for value in times], padding=3
    )
    time_ax.set(ylabel="Full-scene scoring time (seconds)", ylim=(0, max(times) * 1.22))
    time_ax.set_title("Recorded scoring time", loc="left")

    fig.suptitle("Potsdam historical benchmark · 14 scenes · 467,862,855 pixels")
    return fig


def render_eda(source, output):
    def read(name):
        return pd.read_csv(source / f"{name}.csv", dtype={"scene_id": str})

    pooled = read("pooled-label-balance")
    datasets = tuple(pooled["dataset"].unique())
    columns = list(pooled.columns[-6:])
    charts = {
        "asset-coverage": eda_plots.asset_coverage(read("asset-summary")),
        "pooled-class-balance": eda_plots.pooled_class_balance(
            pooled, datasets, columns
        ),
        "per-scene-class-balance": eda_plots.per_scene_class_balance(
            read("per-scene-label-counts"), datasets, columns
        ),
        "irrg-training-validation-histograms": eda_plots.irrg_histograms(
            read("irrg-full-histograms"), datasets
        ),
        "irrg-per-scene-percentiles": eda_plots.irrg_percentiles(
            read("irrg-per-scene-band-statistics"), datasets
        ),
        "potsdam-elevation-value-profiles": eda_plots.elevation_profiles(
            read("auxiliary-per-scene-statistics"), read("auxiliary-uint8-histograms")
        ),
        "augmentation-impact": eda_plots.augmentation_impact(
            read("augmentation-audit"), datasets, BANDS
        ),
    }
    output.mkdir(parents=True)
    for name, fig in charts.items():
        save(fig, output / f"{name}.png")


@plt.style.context(eda_plots.STYLE)
def legacy_pie(counts, scene):
    fig, ax = plt.subplots(figsize=(4.8, 2.65), layout="constrained")
    wedges, _ = ax.pie(
        counts,
        colors=LEGACY_COLORS,
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
    )
    labels = [
        f"{label}  {count / sum(counts):.2%}"
        for label, count in zip(CLASSES, counts, strict=True)
    ]
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(0.96, 0.5))
    ax.set_title(f"Potsdam {scene} · class balance")
    return fig


def render_legacy(source, output):
    """Replace only chart PNG MIME data, using the notebook's saved count rows."""
    nb = nbformat.read(source, as_version=4)
    scene_cell = next(c for c in nb.cells if c.source.startswith("all_images = "))
    scene_ids = ast.literal_eval(scene_cell.source.split("=", 1)[1].strip())
    cell = next(c for c in nb.cells if c.source.startswith("n_pixel_all = "))
    if len(scene_ids) != 38 or len(cell.outputs) != 3 * len(scene_ids):
        raise ValueError("Unexpected legacy scene or output count")
    for i, scene in enumerate(scene_ids):
        image, pie, stream = cell.outputs[3 * i : 3 * i + 3]
        if (
            image.output_type != "display_data"
            or "image/png" not in image.data
            or pie.output_type != "display_data"
            or "image/png" not in pie.data
            or stream.output_type != "stream"
            or stream.name != "stdout"
        ):
            raise ValueError(f"Unexpected notebook outputs for scene {scene}")
        counts = ast.literal_eval(stream.text.strip())
        if len(counts) != 6 or min(counts) < 0 or sum(counts) != 36_000_000:
            raise ValueError(f"Unexpected label counts for scene {scene}")
        fig = legacy_pie(counts, scene)
        buffer = io.BytesIO()
        save(fig, buffer)
        pie.data["image/png"] = base64.b64encode(buffer.getvalue()).decode("ascii")
        pie.data["text/plain"] = f"Potsdam {scene} · class balance"
    nbformat.write(nb, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "group",
        choices=("dinov3", "model-comparison", "eda", "legacy-potsdam"),
    )
    parser.add_argument("source", type=Path, help="Run root, EDA CSV root or notebook")
    parser.add_argument(
        "output", type=Path, help="Output directory, PNG or legacy notebook"
    )
    parser.add_argument(
        "--dinov3-run", type=Path, help="Audited DINOv3 RGBIR campaign run root"
    )
    args = parser.parse_args()
    plt.switch_backend("Agg")
    if args.group == "dinov3":
        args.output.mkdir(parents=True, exist_ok=True)
        for name, fig in (
            ("abc", selection_figure(args.source)),
            ("final", benchmark_figure(args.source)),
        ):
            save(fig, args.output / f"potsdam-dinov3-rgbir-v2-{name}-2026-11-06.png")
    elif args.group == "model-comparison":
        if args.dinov3_run is None:
            parser.error("model-comparison rendering requires --dinov3-run")
        save(model_comparison_figure(args.source, args.dinov3_run), args.output)
    elif args.group == "eda":
        render_eda(args.source, args.output)
    else:
        render_legacy(args.source, args.output)


if __name__ == "__main__":
    main()
