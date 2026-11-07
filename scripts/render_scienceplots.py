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
    parser.add_argument("group", choices=("dinov3", "eda", "legacy-potsdam"))
    parser.add_argument("source", type=Path, help="Run root, EDA CSV root or notebook")
    parser.add_argument("output", type=Path, help="Output directory or legacy notebook")
    args = parser.parse_args()
    plt.switch_backend("Agg")
    if args.group == "dinov3":
        args.output.mkdir(parents=True, exist_ok=True)
        for name, fig in (
            ("abc", selection_figure(args.source)),
            ("final", benchmark_figure(args.source)),
        ):
            save(fig, args.output / f"potsdam-dinov3-rgbir-v2-{name}-2026-11-06.png")
    elif args.group == "eda":
        render_eda(args.source, args.output)
    else:
        render_legacy(args.source, args.output)


if __name__ == "__main__":
    main()
