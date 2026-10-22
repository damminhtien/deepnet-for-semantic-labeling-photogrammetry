# %% [markdown]
# # ISPRS data EDA and augmentation review
#
# This reproducible, cell-oriented analysis reads the DVC-prepared catalogs and
# train/validation quality reports. It never modifies source rasters, fits
# normalization on validation data, or scores the Vaihingen post-selection scenes.
#
# Before running, prepare the DVC inputs and run `dvc repro`. Set
# `DEEPNET_DATA_ROOT` to the materialized `data/raw/inputs` directory and
# `DEEPNET_ARTIFACT_ROOT` to an output directory outside the repository.

# %%
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import albumentations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from IPython.display import display
from matplotlib.colors import ListedColormap
from rasterio.windows import Window as RasterWindow

from deepnet.data.augmentations import AlbumentationsAugmenter
from deepnet.data.catalog import (
    load_augmentation_spec,
    load_domain_spec,
    load_model_input_spec,
    load_preprocessing_spec,
    load_scene_catalog,
    load_split_manifest,
)
from deepnet.data.contracts import SceneRecord, WindowSpec, canonical_hash
from deepnet.data.preprocessing import decode_rgb_labels
from deepnet.data.raster import RasterioSceneReader


def _find_repository_root() -> Path:
    """Find the checkout root from a script or an interactive notebook kernel."""
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "deepnet"
        ).is_dir():
            return candidate
    if "__file__" in globals():
        candidate = Path(__file__).resolve().parents[1]
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("Start the notebook from inside the DeepNet repository")


REPOSITORY_ROOT = _find_repository_root()
DATA_ROOT = (
    Path(os.environ.get("DEEPNET_DATA_ROOT", REPOSITORY_ROOT / "data/raw/inputs"))
    .expanduser()
    .resolve()
)
PREPARED_ROOT = REPOSITORY_ROOT / "data/prepared"
ARTIFACT_ROOT = (
    Path(
        os.environ.get(
            "DEEPNET_ARTIFACT_ROOT",
            Path.home()
            / "Data"
            / REPOSITORY_ROOT.name
            / "eda"
            / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        )
    )
    .expanduser()
    .resolve()
)
CONFIG_ROOT = REPOSITORY_ROOT / "configs"
RANDOM_SEED = 20261022

if not DATA_ROOT.is_dir():
    raise FileNotFoundError(
        f"Data root {DATA_ROOT} does not exist; materialize DVC inputs first"
    )
if ARTIFACT_ROOT == REPOSITORY_ROOT or REPOSITORY_ROOT in ARTIFACT_ROOT.parents:
    raise ValueError("Set DEEPNET_ARTIFACT_ROOT outside the Git repository")
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=False)

# %% [markdown]
# ## Load versioned catalogs and split contracts
#
# Catalog hashes and DVC quality reports provide the data provenance for this
# notebook. Potsdam's 20-scene validation role includes the 14 scenes held out in a
# separate earlier experiment. Vaihingen's official 17-scene cohort stays outside
# this DINOv3 run's model selection.

# %%
DATASET_CONFIGS = {
    "Potsdam": {
        "dataset_id": "isprs-potsdam",
        "split_file": "potsdam-dinov3-v1.json",
        "crop_size": 512,
    },
    "Vaihingen": {
        "dataset_id": "isprs-vaihingen",
        "split_file": "vaihingen-dinov3-v1.json",
        "crop_size": 288,
    },
}

PREPROCESSING = load_preprocessing_spec(
    "isprs-canonical-v2", CONFIG_ROOT / "preprocessing"
)
MODEL_INPUT = load_model_input_spec("isprs-irrg-unit-v1", CONFIG_ROOT / "model-inputs")
DOMAIN = load_domain_spec("isprs-irrg-aerial-v1", CONFIG_ROOT / "domains")
AUGMENTATION = load_augmentation_spec(
    "isprs-irrg-aerial-v2", CONFIG_ROOT / "augmentations"
)
AUGMENTER = AlbumentationsAugmenter(
    DOMAIN,
    MODEL_INPUT,
    AUGMENTATION,
    ignore_label=PREPROCESSING.ignore_label,
)
GEOMETRY_ONLY_AUGMENTER = AlbumentationsAugmenter(
    DOMAIN,
    MODEL_INPUT,
    replace(AUGMENTATION, optical=()),
    ignore_label=PREPROCESSING.ignore_label,
)

CATALOGS: dict[str, dict[str, Any]] = {}
SCENES: dict[str, tuple[SceneRecord, ...]] = {}
SPLITS = {}
QUALITY_REPORTS: dict[str, dict[str, Any]] = {}

for display_name, settings in DATASET_CONFIGS.items():
    dataset_id = settings["dataset_id"]
    prepared_dir = PREPARED_ROOT / display_name.lower()
    catalog_path = prepared_dir / "catalog.json"
    quality_path = prepared_dir / "quality.json"
    split_path = CONFIG_ROOT / "splits" / settings["split_file"]

    if not catalog_path.is_file() or not quality_path.is_file():
        raise FileNotFoundError(
            f"Missing DVC outputs for {dataset_id}; run `dvc repro` first"
        )

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    scenes = load_scene_catalog(catalog_path, expected_dataset_id=dataset_id)
    split = load_split_manifest(split_path)
    quality = json.loads(quality_path.read_text(encoding="utf-8"))

    if catalog.get("hash_status") != "complete":
        raise RuntimeError(f"Catalog hashes are incomplete for {dataset_id}")
    if quality.get("split_hash") != split.content_hash:
        raise RuntimeError(f"Quality report uses a different split for {dataset_id}")
    if quality.get("preprocessing_hash") != PREPROCESSING.content_hash:
        raise RuntimeError(
            f"Quality report uses a different preprocessing spec for {dataset_id}"
        )
    if set(quality.get("included_splits", ())) != {"train", "validation"}:
        raise RuntimeError(f"Quality report scanned unexpected roles for {dataset_id}")
    if set(split.scene_ids) - {scene.scene_id for scene in scenes}:
        raise RuntimeError(f"Catalog is missing split scenes for {dataset_id}")

    for role in ("train", "validation"):
        label_asset = split.label_asset_for(role)
        for scene_id in split.scenes_for(role):
            if label_asset not in next(
                scene.asset_map() for scene in scenes if scene.scene_id == scene_id
            ):
                raise RuntimeError(
                    f"{dataset_id} scene {scene_id} lacks {role} label asset "
                    f"{label_asset}"
                )

    CATALOGS[display_name] = catalog
    SCENES[display_name] = scenes
    SPLITS[display_name] = split
    QUALITY_REPORTS[display_name] = quality

DVC_LOCK_PATH = REPOSITORY_ROOT / "dvc.lock"
input_revision = (
    hashlib.sha256(DVC_LOCK_PATH.read_bytes()).hexdigest()[:16]
    if DVC_LOCK_PATH.is_file()
    else "dvc.lock missing"
)
print(f"Input revision: {input_revision}")
print(f"Output directory: {ARTIFACT_ROOT}")
display(
    pd.DataFrame(
        [
            {
                "preprocessing": PREPROCESSING.preprocessing_id,
                "version": PREPROCESSING.version,
                "label_color_tolerance_rgb": PREPROCESSING.label_color_tolerance,
                "label_palette_classes": len(PREPROCESSING.label_palette),
                "tolerance_policy": (
                    "nearest unique palette color within Euclidean RGB tolerance; "
                    "ties and farther colors fail"
                ),
            }
        ]
    )
)
display(
    pd.DataFrame(
        [
            {
                "dataset": name,
                "scenes": len(SCENES[name]),
                "train": len(SPLITS[name].train_scene_ids),
                "validation": len(SPLITS[name].validation_scene_ids),
                "test": len(SPLITS[name].test_scene_ids),
                "split_hash": SPLITS[name].content_hash,
                "catalog_hash": CATALOGS[name]["catalog_content_hash"],
            }
            for name in DATASET_CONFIGS
        ]
    )
)

# %% [markdown]
# ## Asset inventory, metadata, and alignment
#
# Review band order, shapes, data types, NoData metadata, CRS/affine availability,
# checksums, and source-backed alignment records. The report distinguishes a
# pixel-local pairing from geographic registration. The first DINOv3 view uses IRRG
# only; the unverified Potsdam nDSM is intentionally not a model input.

# %%
asset_rows = []
scene_lookup: dict[str, dict[str, SceneRecord]] = {}
for display_name, scenes in SCENES.items():
    scene_lookup[display_name] = {scene.scene_id: scene for scene in scenes}
    for scene in scenes:
        for asset_name, asset in scene.assets:
            asset_path = DATA_ROOT / asset.uri
            asset_rows.append(
                {
                    "dataset": display_name,
                    "scene_id": scene.scene_id,
                    "asset": asset_name,
                    "bands": ",".join(asset.bands),
                    "width": asset.grid.width,
                    "height": asset.grid.height,
                    "dtype": asset.dtype,
                    "nodata": asset.nodata,
                    "crs": asset.grid.crs,
                    "affine": asset.grid.affine,
                    "georeferenced": asset.grid.georeferenced,
                    "alignment_evidence": asset.alignment_evidence,
                    "sha256_present": asset.sha256 is not None,
                    "bytes": asset_path.stat().st_size,
                }
            )

asset_metadata = pd.DataFrame(asset_rows)
asset_summary = asset_metadata.groupby(["dataset", "asset"], as_index=False).agg(
    scenes=("scene_id", "nunique"),
    bands=("bands", "first"),
    shapes=(
        "width",
        lambda values: ", ".join(
            f"{width}x{height}"
            for width, height in sorted(
                set(
                    zip(values, asset_metadata.loc[values.index, "height"], strict=True)
                )
            )
        ),
    ),
    dtypes=("dtype", lambda values: ", ".join(sorted(set(values)))),
    georeferenced=("georeferenced", "sum"),
    hashed=("sha256_present", "sum"),
    total_gib=("bytes", lambda values: round(values.sum() / 2**30, 3)),
)
display(asset_summary)

alignment_rows = []
for display_name, split in SPLITS.items():
    for role in ("train", "validation"):
        label_asset = split.label_asset_for(role)
        for scene_id in split.scenes_for(role):
            scene = scene_lookup[display_name][scene_id]
            try:
                scene.require_aligned(("irrg", label_asset))
                status = "passed"
                reason = "Both assets align to the scene reference grid"
            except ValueError as error:
                status = "failed"
                reason = str(error)
            image_asset = scene.asset_map()["irrg"]
            reference = scene.reference_grid
            image_grid = image_asset.grid
            label_grid = scene.asset_map()[label_asset].grid
            if image_grid.same_grid(reference) and label_grid.same_grid(reference):
                alignment_basis = "metadata_matches_reference_grid"
            elif (
                image_asset.alignment_evidence
                or scene.asset_map()[label_asset].alignment_evidence
            ):
                alignment_basis = "source_attestation_to_reference_grid"
            else:
                alignment_basis = "unverified"
            alignment_rows.append(
                {
                    "dataset": display_name,
                    "role": role,
                    "scene_id": scene_id,
                    "label_asset": label_asset,
                    "status": status,
                    "alignment_basis": alignment_basis,
                    "reference_grid_id": reference.grid_id,
                    "irrg_grid_id": image_grid.grid_id,
                    "label_grid_id": label_grid.grid_id,
                    "irrg_shape": f"{image_grid.width}x{image_grid.height}",
                    "label_shape": f"{label_grid.width}x{label_grid.height}",
                    "same_shape": (image_grid.width, image_grid.height)
                    == (label_grid.width, label_grid.height),
                    "irrg_georeferenced": image_grid.georeferenced,
                    "label_georeferenced": label_grid.georeferenced,
                    "irrg_alignment_evidence": image_asset.alignment_evidence,
                    "label_alignment_evidence": scene.asset_map()[
                        label_asset
                    ].alignment_evidence,
                    "reason": reason,
                }
            )

alignment = pd.DataFrame(alignment_rows)
if not alignment["status"].eq("passed").all():
    raise RuntimeError("An active IRRG/label scene pair failed alignment checks")
display(alignment.groupby(["dataset", "role", "status"]).size().rename("scenes"))

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
for axis, (display_name, frame) in zip(
    axes, asset_metadata.groupby("dataset"), strict=True
):
    pivot = frame.groupby("asset")["scene_id"].nunique().sort_values()
    axis.barh(pivot.index, pivot.values, color="#4472C4")
    axis.set_title(f"{display_name}: available assets in DVC catalog")
    axis.set_xlabel("scene count")
    axis.grid(axis="x", alpha=0.25)
fig.savefig(ARTIFACT_ROOT / "asset-coverage.png", dpi=160)
plt.show()

# %% [markdown]
# ## Label balance, ignore pixels, and scene variability
#
# Counts come from the full-scene, windowed DVC quality scan. The six semantic
# classes are normalized separately from class 6, which is the ignore/boundary ID.
# Per-scene proportions expose whether pooled class frequency hides spatial
# variation; they should guide train-only sampling, not be used to rebalance
# validation.

# %%
CLASS_NAMES = {
    0: "impervious surface",
    1: "building",
    2: "low vegetation",
    3: "tree",
    4: "car",
    5: "clutter/background",
    6: "ignore/boundary",
}

class_rows = []
for display_name, report in QUALITY_REPORTS.items():
    for role, split_report in report["splits"].items():
        for scene_report in split_report["scenes"]:
            counts = {
                int(key): int(value)
                for key, value in scene_report["class_counts"].items()
            }
            semantic_pixels = sum(counts.get(class_id, 0) for class_id in range(6))
            total_labeled_pixels = sum(counts.values())
            row = {
                "dataset": display_name,
                "role": role,
                "scene_id": scene_report["scene_id"],
                "valid_pixel_fraction": (
                    scene_report["valid_pixels"] / scene_report["total_pixels"]
                    if scene_report["total_pixels"]
                    else 0.0
                ),
                "ignore_fraction": (
                    counts.get(6, 0) / total_labeled_pixels
                    if total_labeled_pixels
                    else 0.0
                ),
            }
            for class_id, class_name in CLASS_NAMES.items():
                row[f"pixels_{class_name}"] = counts.get(class_id, 0)
                row[f"share_{class_name}"] = (
                    counts.get(class_id, 0) / semantic_pixels
                    if class_id < 6 and semantic_pixels
                    else np.nan
                )
            class_rows.append(row)

label_counts = pd.DataFrame(class_rows)
pooled_rows = []
for (dataset, role), frame in label_counts.groupby(["dataset", "role"]):
    totals = {
        class_id: int(frame[f"pixels_{class_name}"].sum())
        for class_id, class_name in CLASS_NAMES.items()
    }
    semantic_total = sum(totals[class_id] for class_id in range(6))
    row = {
        "dataset": dataset,
        "role": role,
        "semantic_pixels": semantic_total,
        "ignore_pixels": totals[6],
        "ignore_fraction": totals[6] / max(semantic_total + totals[6], 1),
    }
    row.update(
        {
            CLASS_NAMES[class_id]: totals[class_id] / max(semantic_total, 1)
            for class_id in range(6)
        }
    )
    pooled_rows.append(row)
pooled_balance = pd.DataFrame(pooled_rows)
display(pooled_balance)

class_columns = [CLASS_NAMES[class_id] for class_id in range(6)]
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
for axis, display_name in zip(axes, DATASET_CONFIGS, strict=True):
    subset = pooled_balance[pooled_balance["dataset"] == display_name].set_index("role")
    subset[class_columns].plot.barh(
        stacked=True,
        ax=axis,
        color=["#8C8C8C", "#2F5597", "#5BC0DE", "#70AD47", "#FFC000", "#C00000"],
    )
    axis.set_title(f"{display_name}: semantic label mix")
    axis.set_xlabel("share of six semantic classes")
    axis.legend(title="class", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.grid(axis="x", alpha=0.2)
fig.savefig(ARTIFACT_ROOT / "pooled-class-balance.png", dpi=160)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
for axis, display_name in zip(axes, DATASET_CONFIGS, strict=True):
    subset = label_counts[label_counts["dataset"] == display_name].copy()
    subset["split_scene"] = subset["role"] + ":" + subset["scene_id"]
    matrix = subset[[f"share_{name}" for name in class_columns]].to_numpy()
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=0.6, cmap="magma")
    axis.set_title(f"{display_name}: per-scene class share")
    axis.set_yticks(np.arange(len(subset)), subset["split_scene"], fontsize=7)
    axis.set_xticks(np.arange(6), class_columns, rotation=40, ha="right")
    axis.set_ylabel("scene role")
    fig.colorbar(image, ax=axis, label="fraction of semantic pixels")
fig.savefig(ARTIFACT_ROOT / "per-scene-class-balance.png", dpi=160)
plt.show()

# %% [markdown]
# ## IRRG value distributions from deterministic training and validation windows
#
# This profile uses a fixed seed and two windows from every active scene in each
# role. It is a scene-stratified sample, not a full-raster histogram. Any fitted
# normalization statistics must still be calculated from training scenes alone by
# `fit-statistics`; validation pixels are never used to fit normalization. Optical
# source values must conform to the declared 8-bit range before any model-view
# scaling or training augmentation.

# %%
SAMPLE_SIZE = 256
WINDOWS_PER_SCENE = 2
rng = np.random.default_rng(RANDOM_SEED)
sampled_values: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
sampled_scene_values: dict[tuple[str, str, str, str], list[np.ndarray]] = defaultdict(
    list
)
window_rows = []

for display_name in DATASET_CONFIGS:
    split = SPLITS[display_name]
    scenes_by_id = scene_lookup[display_name]
    for role in ("train", "validation"):
        scene_ids = sorted(split.scenes_for(role))
        for scene_id in scene_ids:
            scene = scenes_by_id[str(scene_id)]
            asset = scene.asset_map()["irrg"]
            path = DATA_ROOT / asset.uri
            with rasterio.open(path) as source:
                if source.width < SAMPLE_SIZE or source.height < SAMPLE_SIZE:
                    raise ValueError(f"IRRG scene is too small to sample: {scene_id}")
                for _ in range(WINDOWS_PER_SCENE):
                    row = int(rng.integers(0, source.height - SAMPLE_SIZE + 1))
                    col = int(rng.integers(0, source.width - SAMPLE_SIZE + 1))
                    raster_window = RasterWindow(col, row, SAMPLE_SIZE, SAMPLE_SIZE)
                    pixels = source.read(window=raster_window, masked=True)
                    mask = source.read_masks(window=raster_window).all(axis=0) > 0
                    finite = np.isfinite(pixels.data).all(axis=0)
                    valid = mask & finite
                    if np.any(pixels.data[:, valid] < 0) or np.any(
                        pixels.data[:, valid] > 255
                    ):
                        raise ValueError(
                            f"IRRG values fall outside [0, 255] in scene {scene_id}"
                        )
                    for band_index, band_name in enumerate(asset.bands):
                        values = pixels.data[band_index][valid]
                        if values.size:
                            sampled_values[(display_name, role, band_name)].append(
                                values
                            )
                            sampled_scene_values[
                                (display_name, role, str(scene_id), band_name)
                            ].append(values)
                    window_rows.append(
                        {
                            "dataset": display_name,
                            "role": role,
                            "scene_id": str(scene_id),
                            "row": row,
                            "col": col,
                            "size": SAMPLE_SIZE,
                            "valid_fraction": float(valid.mean()),
                        }
                    )

band_rows = []
for (display_name, role, band_name), arrays in sorted(sampled_values.items()):
    values = np.concatenate(arrays)
    quantiles = np.quantile(values, [0.01, 0.05, 0.5, 0.95, 0.99])
    band_rows.append(
        {
            "dataset": display_name,
            "role": role,
            "band": band_name,
            "scenes_sampled": len(
                {
                    scene_id
                    for dataset, sample_role, scene_id, sampled_band in (
                        sampled_scene_values
                    )
                    if (dataset, sample_role, sampled_band)
                    == (display_name, role, band_name)
                }
            ),
            "windows_sampled": len(arrays),
            "samples": int(values.size),
            "min": float(values.min()),
            "p01": float(quantiles[0]),
            "p05": float(quantiles[1]),
            "p50": float(quantiles[2]),
            "p95": float(quantiles[3]),
            "p99": float(quantiles[4]),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "zero_fraction": float(np.mean(values == 0)),
            "max_value_fraction": float(np.mean(values == np.iinfo(values.dtype).max))
            if np.issubdtype(values.dtype, np.integer)
            else float("nan"),
        }
    )
band_statistics = pd.DataFrame(band_rows)
display(band_statistics)

scene_band_rows = []
for (display_name, role, scene_id, band_name), arrays in sorted(
    sampled_scene_values.items()
):
    values = np.concatenate(arrays)
    quantiles = np.quantile(values, [0.05, 0.5, 0.95])
    scene_band_rows.append(
        {
            "dataset": display_name,
            "role": role,
            "scene_id": scene_id,
            "band": band_name,
            "samples": int(values.size),
            "p05": float(quantiles[0]),
            "p50": float(quantiles[1]),
            "p95": float(quantiles[2]),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "zero_fraction": float(np.mean(values == 0)),
        }
    )
scene_band_statistics = pd.DataFrame(scene_band_rows)
display(scene_band_statistics.head(12))

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, constrained_layout=True)
for row_index, display_name in enumerate(DATASET_CONFIGS):
    for column_index, band_name in enumerate(("ir", "red", "green")):
        axis = axes[row_index, column_index]
        for role, color in (("train", "#4472C4"), ("validation", "#ED7D31")):
            arrays = sampled_values.get((display_name, role, band_name), [])
            if arrays:
                values = np.concatenate(arrays)
                axis.hist(
                    values,
                    bins=np.arange(-0.5, 256.5, 1),
                    density=True,
                    histtype="step",
                    linewidth=1.2,
                    label=role,
                    color=color,
                )
        axis.set_title(f"{display_name} · {band_name}")
        axis.set_xlabel("source digital number")
        axis.set_ylabel("density")
        axis.grid(alpha=0.2)
        axis.legend()
fig.savefig(ARTIFACT_ROOT / "irrg-training-validation-histograms.png", dpi=160)
plt.show()

# %% [markdown]
# ## Synchronized augmentation examples and checks
#
# A deterministic search over candidate training windows chooses a patch with
# useful class coverage. Augmentation is demonstrated only on train patches; both
# validation roles remain untouched. The same seed is replayed twice; image,
# labels, and validity must match. The plotted IRRG pseudocolor maps red to red,
# green to green, and blue to NIR. Affine-created borders use ignore ID 6 and do
# not become supervised pixels.

# %%
LABEL_COLORS = (
    np.asarray(
        [
            (255, 255, 255),
            (0, 0, 255),
            (0, 255, 255),
            (0, 255, 0),
            (255, 255, 0),
            (255, 0, 0),
            (0, 0, 0),
        ],
        dtype=np.float32,
    )
    / 255.0
)
LABEL_CMAP = ListedColormap(LABEL_COLORS)


def _candidate_starts(length: int, size: int, count: int = 3) -> tuple[int, ...]:
    """Return deterministic interior crop starts for representative patch search."""
    max_start = length - size
    if max_start < 0:
        raise ValueError(f"Scene dimension {length} is smaller than crop {size}")
    return tuple(sorted({int(value) for value in np.linspace(0, max_start, count)}))


def _read_patch(
    scene: SceneRecord,
    label_asset: str,
    crop_size: int,
    row: int,
    col: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Read aligned source bands and encoded labels from one scene window."""
    window = WindowSpec(row, col, crop_size, crop_size)
    with RasterioSceneReader(scene, DATA_ROOT) as reader:
        optical = reader.read_window("irrg", window)
        labels_rgb, valid = reader.read_window_with_mask(
            label_asset,
            window,
            resampling=PREPROCESSING.categorical_resampling,
        )
    labels = decode_rgb_labels(labels_rgb, PREPROCESSING, valid_mask=valid)
    channels = {
        band_name: optical[..., index]
        for index, band_name in enumerate(scene.asset_map()["irrg"].bands)
    }
    return channels, labels, valid


def _select_patch(
    display_name: str,
    role: str,
) -> tuple[SceneRecord, int, int, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Choose a reproducible crop with diverse valid classes and rare objects."""
    split = SPLITS[display_name]
    crop_size = DATASET_CONFIGS[display_name]["crop_size"]
    label_asset = split.label_asset_for(role)
    candidates = []
    candidate_scene_ids = sorted(split.scenes_for(role))[:3]
    for scene_id in candidate_scene_ids:
        scene = scene_lookup[display_name][scene_id]
        scene.require_aligned(("irrg", label_asset))
        height, width = scene.reference_grid.height, scene.reference_grid.width
        for row in _candidate_starts(height, crop_size):
            for col in _candidate_starts(width, crop_size):
                channels, labels, valid = _read_patch(
                    scene, label_asset, crop_size, row, col
                )
                semantic = labels[valid & (labels != PREPROCESSING.ignore_label)]
                unique, counts = np.unique(semantic, return_counts=True)
                class_count = int(unique.size)
                car_pixels = (
                    int(counts[unique == 4].sum()) if np.any(unique == 4) else 0
                )
                clutter_pixels = (
                    int(counts[unique == 5].sum()) if np.any(unique == 5) else 0
                )
                score = (
                    class_count,
                    int(car_pixels > 0),
                    int(clutter_pixels > 0),
                    float(valid.mean()),
                    -row,
                    -col,
                )
                candidates.append((score, scene, row, col, channels, labels, valid))
    if not candidates:
        raise RuntimeError(f"No representative windows found for {display_name}/{role}")
    _, scene, row, col, channels, labels, valid = max(
        candidates, key=lambda item: item[0]
    )
    return scene, row, col, channels, labels, valid


def _semantic_counts(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Count the six semantic classes, excluding ignore and invalid pixels."""
    semantic = labels[valid & (labels >= 0) & (labels < 6)]
    return np.bincount(semantic.astype(np.int64), minlength=6)


def _channel_summary(values: np.ndarray) -> dict[str, float]:
    """Summarize one valid optical band in its raw 8-bit source range."""
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {name: float("nan") for name in ("mean", "std", "p05", "p50", "p95")}
    p05, p50, p95 = np.percentile(values, [5, 50, 95])
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
    }


# %% [markdown]
# ## Quantify augmentation across training scenes
#
# One deterministic training crop is sampled from each train scene, then replayed
# with three independent augmentation seeds. The geometry-only policy shares the
# exact geometric transform seed with the full policy. Labels and validity masks
# must therefore match pixel-for-pixel; the image difference between those outputs
# measures optical augmentation after the same resampling. These samples are an
# audit of the configured policy, not an estimate of validation performance.

# %%
AUDIT_REPEATS_PER_SCENE = 3
audit_rng = np.random.default_rng(RANDOM_SEED + 50_000)
augmentation_audit_rows = []

for dataset_index, display_name in enumerate(DATASET_CONFIGS):
    split = SPLITS[display_name]
    crop_size = DATASET_CONFIGS[display_name]["crop_size"]
    label_asset = split.label_asset_for("train")
    scenes_by_id = scene_lookup[display_name]

    for scene_index, scene_id in enumerate(sorted(split.train_scene_ids)):
        scene = scenes_by_id[str(scene_id)]
        scene.require_aligned(("irrg", label_asset))
        height, width = scene.reference_grid.height, scene.reference_grid.width
        row = int(audit_rng.integers(0, height - crop_size + 1))
        col = int(audit_rng.integers(0, width - crop_size + 1))
        channels, labels, valid = _read_patch(scene, label_asset, crop_size, row, col)
        source_labels = labels.copy()
        source_labels[~valid] = PREPROCESSING.ignore_label
        source_counts = _semantic_counts(source_labels, valid)
        source_semantic_total = max(int(source_counts.sum()), 1)

        for repeat_index in range(AUDIT_REPEATS_PER_SCENE):
            seed = int(
                np.random.SeedSequence(
                    [RANDOM_SEED, 100_000, dataset_index, scene_index, repeat_index]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            geometric, geometric_labels, geometric_valid = GEOMETRY_ONLY_AUGMENTER(
                channels,
                source_labels,
                valid,
                seed=seed,
            )
            augmented, augmented_labels, augmented_valid = AUGMENTER(
                channels,
                source_labels,
                valid,
                seed=seed,
            )
            np.testing.assert_array_equal(geometric_labels, augmented_labels)
            np.testing.assert_array_equal(geometric_valid, augmented_valid)
            if np.any(augmented_labels[~augmented_valid] != PREPROCESSING.ignore_label):
                raise AssertionError(
                    "Invalid augmentation borders are not ignore-labeled"
                )

            geometric_counts = _semantic_counts(geometric_labels, geometric_valid)
            augmented_counts = _semantic_counts(augmented_labels, augmented_valid)
            geometric_total = max(int(geometric_counts.sum()), 1)
            augmented_total = max(int(augmented_counts.sum()), 1)
            row_record: dict[str, Any] = {
                "dataset": display_name,
                "role": "train",
                "scene_id": str(scene_id),
                "row": row,
                "col": col,
                "height": crop_size,
                "width": crop_size,
                "seed": seed,
                "repeat_index": repeat_index,
                "source_valid_fraction": float(valid.mean()),
                "geometry_valid_fraction": float(geometric_valid.mean()),
                "source_ignore_fraction": float(np.mean(source_labels == 6)),
                "geometry_ignore_fraction": float(np.mean(geometric_labels == 6)),
                "source_class_count": int(np.count_nonzero(source_counts)),
                "geometry_class_count": int(np.count_nonzero(geometric_counts)),
                "source_classes_retained": int(
                    np.count_nonzero((source_counts > 0) & (geometric_counts > 0))
                ),
                "all_source_classes_retained": bool(
                    np.all(geometric_counts[source_counts > 0] > 0)
                ),
                "geometry_class_share_abs_delta_mean": float(
                    np.abs(
                        geometric_counts / geometric_total
                        - source_counts / source_semantic_total
                    ).mean()
                ),
            }
            radiometric_differences = []

            for class_id in range(6):
                source_share = source_counts[class_id] / source_semantic_total
                geometry_share = geometric_counts[class_id] / geometric_total
                row_record[f"class_{class_id}_pixels_source"] = int(
                    source_counts[class_id]
                )
                row_record[f"class_{class_id}_pixels_geometry"] = int(
                    geometric_counts[class_id]
                )
                row_record[f"class_{class_id}_share_source"] = float(source_share)
                row_record[f"class_{class_id}_share_geometry"] = float(geometry_share)
                row_record[f"class_{class_id}_share_delta"] = float(
                    geometry_share - source_share
                )

            for band_name in AUGMENTER.optical_channels:
                before = np.asarray(channels[band_name])[valid]
                after_geometry = np.asarray(geometric[band_name])[geometric_valid]
                after_augmentation = np.asarray(augmented[band_name])[augmented_valid]
                geometry_summary = _channel_summary(after_geometry)
                augmented_summary = _channel_summary(after_augmentation)
                optical_difference = (
                    np.asarray(augmented[band_name], dtype=np.float32)
                    - np.asarray(geometric[band_name], dtype=np.float32)
                )[geometric_valid]
                radiometric_differences.append(optical_difference)
                source_summary = _channel_summary(before)
                row_record[f"band_{band_name}_mean_source"] = source_summary["mean"]
                row_record[f"band_{band_name}_mean_geometry"] = geometry_summary["mean"]
                row_record[f"band_{band_name}_mean_augmented"] = augmented_summary[
                    "mean"
                ]
                row_record[f"band_{band_name}_mean_radiometric_delta"] = (
                    augmented_summary["mean"] - geometry_summary["mean"]
                )
                row_record[f"band_{band_name}_mean_abs_radiometric_delta"] = float(
                    np.abs(optical_difference).mean()
                )
                row_record[f"band_{band_name}_p95_abs_radiometric_delta"] = float(
                    np.percentile(np.abs(optical_difference), 95)
                )
                row_record[f"band_{band_name}_unchanged_fraction"] = float(
                    np.mean(optical_difference == 0)
                )
                row_record[f"band_{band_name}_lower_endpoint_fraction"] = float(
                    np.mean(after_augmentation == 0)
                )
                row_record[f"band_{band_name}_upper_endpoint_fraction"] = float(
                    np.mean(after_augmentation == 255)
                )

                if not np.isfinite(augmented[band_name]).all():
                    raise AssertionError(
                        f"Augmentation produced non-finite {band_name} values"
                    )
                if np.any(augmented[band_name] < 0) or np.any(
                    augmented[band_name] > 255
                ):
                    raise AssertionError(
                        f"Augmentation moved {band_name} outside [0, 255]"
                    )

            all_radiometric_differences = np.concatenate(radiometric_differences)
            row_record["optical_pixels_changed_fraction"] = float(
                np.mean(all_radiometric_differences != 0)
            )
            row_record["optical_transform_changed_image"] = bool(
                np.any(all_radiometric_differences != 0)
            )
            augmentation_audit_rows.append(row_record)

augmentation_audit = pd.DataFrame(augmentation_audit_rows)
delta_columns = [f"class_{class_id}_share_delta" for class_id in range(6)]
augmentation_audit["class_share_abs_delta_mean"] = (
    augmentation_audit[delta_columns].abs().mean(axis=1)
)

augmentation_summary_rows = []
for display_name, frame in augmentation_audit.groupby("dataset", sort=False):
    row_summary = {
        "dataset": display_name,
        "train_scenes": int(frame["scene_id"].nunique()),
        "augmentation_samples": int(len(frame)),
        "source_valid_fraction_mean": float(frame["source_valid_fraction"].mean()),
        "geometry_valid_fraction_p05": float(
            frame["geometry_valid_fraction"].quantile(0.05)
        ),
        "geometry_valid_fraction_median": float(
            frame["geometry_valid_fraction"].median()
        ),
        "all_source_classes_retained_fraction": float(
            frame["all_source_classes_retained"].mean()
        ),
        "optical_transform_changed_image_fraction": float(
            frame["optical_transform_changed_image"].mean()
        ),
        "optical_pixels_changed_fraction_mean": float(
            frame["optical_pixels_changed_fraction"].mean()
        ),
        "class_share_abs_delta_mean": float(frame["class_share_abs_delta_mean"].mean()),
        "class_share_abs_delta_p95": float(
            frame["class_share_abs_delta_mean"].quantile(0.95)
        ),
    }
    for band_name in AUGMENTER.optical_channels:
        delta_column = f"band_{band_name}_mean_abs_radiometric_delta"
        changed = frame[delta_column] > 1e-8
        row_summary[f"band_{band_name}_mean_radiometric_delta"] = float(
            frame[f"band_{band_name}_mean_radiometric_delta"].mean()
        )
        row_summary[f"band_{band_name}_radiometric_change_rate"] = float(changed.mean())
        row_summary[
            f"band_{band_name}_mean_abs_radiometric_delta_when_changed_median"
        ] = float(frame.loc[changed, delta_column].median() if changed.any() else 0.0)
        row_summary[f"band_{band_name}_p95_abs_radiometric_delta_all"] = float(
            frame[f"band_{band_name}_p95_abs_radiometric_delta"].quantile(0.95)
        )
    augmentation_summary_rows.append(row_summary)

augmentation_effect_summary = pd.DataFrame(augmentation_summary_rows)
display(augmentation_effect_summary)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
datasets = list(DATASET_CONFIGS)
valid_data = [
    augmentation_audit.loc[
        augmentation_audit["dataset"] == name, "geometry_valid_fraction"
    ]
    for name in datasets
]
axes[0].boxplot(valid_data, tick_labels=datasets, showfliers=False)
axes[0].set_title("Valid pixels after geometry")
axes[0].set_ylabel("fraction of crop")
axes[0].grid(axis="y", alpha=0.2)

for dataset_index, display_name in enumerate(datasets):
    subset = augmentation_audit[augmentation_audit["dataset"] == display_name]
    values = subset["class_share_abs_delta_mean"] * 100
    axes[1].scatter(
        np.full(len(values), dataset_index) + np.linspace(-0.12, 0.12, len(values)),
        values,
        s=12,
        alpha=0.55,
        label=display_name,
    )
axes[1].set_xticks(np.arange(len(datasets)), datasets)
axes[1].set_title("Mean absolute class-share shift")
axes[1].set_ylabel("percentage points per class")
axes[1].grid(axis="y", alpha=0.2)

band_labels = []
band_values = []
for display_name in datasets:
    subset = augmentation_audit[augmentation_audit["dataset"] == display_name]
    for band_name in AUGMENTER.optical_channels:
        band_labels.append(f"{display_name}\n{band_name}")
        delta_column = f"band_{band_name}_mean_abs_radiometric_delta"
        changed = subset[delta_column] > 1e-8
        band_values.append(subset.loc[changed, delta_column].median())
axes[2].bar(np.arange(len(band_values)), band_values, color="#70AD47")
axes[2].set_xticks(np.arange(len(band_values)), band_labels, fontsize=8)
axes[2].set_title("Optical change when active")
axes[2].set_ylabel("median absolute DN change")
axes[2].grid(axis="y", alpha=0.2)
fig.savefig(ARTIFACT_ROOT / "augmentation-impact.png", dpi=160)
plt.show()


def _display_image(
    channels: dict[str, np.ndarray],
    limits: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Build a stable-stretch RGB pseudocolor image from the IRRG band mapping."""
    output = []
    for band_name in ("red", "green", "ir"):
        low, high = limits[band_name]
        values = np.asarray(channels[band_name], dtype=np.float32)
        output.append(np.clip((values - low) / max(high - low, 1e-6), 0, 1))
    return np.stack(output, axis=-1)


augmentation_rows = []
augmentation_examples = []
role_order = (
    ("Potsdam", "train"),
    ("Vaihingen", "train"),
)

for example_index, (display_name, role) in enumerate(role_order):
    scene, row, col, channels, labels, valid = _select_patch(display_name, role)
    source_labels = labels.copy()
    source_labels[~valid] = PREPROCESSING.ignore_label
    seed = RANDOM_SEED + example_index
    augmented, augmented_labels, augmented_valid = AUGMENTER(
        channels,
        source_labels,
        valid,
        seed=seed,
    )
    replay, replay_labels, replay_valid = AUGMENTER(
        channels,
        source_labels,
        valid,
        seed=seed,
    )
    for band_name in channels:
        np.testing.assert_array_equal(augmented[band_name], replay[band_name])
    np.testing.assert_array_equal(augmented_labels, replay_labels)
    np.testing.assert_array_equal(augmented_valid, replay_valid)
    if np.any(augmented_labels > PREPROCESSING.ignore_label):
        raise AssertionError("Augmentation produced an undeclared label ID")
    if np.any(augmented_labels[~augmented_valid] != PREPROCESSING.ignore_label):
        raise AssertionError("Invalid augmentation borders are not ignore-labeled")
    for band_name, values in augmented.items():
        if not np.isfinite(values).all():
            raise AssertionError(f"Augmentation produced non-finite {band_name} values")
        if np.any(values < 0) or np.any(values > 255):
            raise AssertionError(
                f"Augmentation moved {band_name} outside the declared [0, 255] range"
            )

    limits = {
        band_name: tuple(
            float(value) for value in np.percentile(channels[band_name][valid], [2, 98])
        )
        for band_name in ("red", "green", "ir")
    }
    pre_semantic = source_labels[valid & (source_labels < 6)]
    post_semantic = augmented_labels[augmented_valid & (augmented_labels < 6)]
    augmentation_examples.append(
        {
            "dataset": display_name,
            "role": role,
            "scene_id": scene.scene_id,
            "row": row,
            "col": col,
            "seed": seed,
            "channels": channels,
            "labels": source_labels,
            "valid": valid,
            "augmented": augmented,
            "augmented_labels": augmented_labels,
            "augmented_valid": augmented_valid,
            "limits": limits,
        }
    )
    augmentation_rows.append(
        {
            "dataset": display_name,
            "role": role,
            "scene_id": scene.scene_id,
            "row": row,
            "col": col,
            "height": labels.shape[0],
            "width": labels.shape[1],
            "seed": seed,
            "classes_before": int(np.unique(pre_semantic).size),
            "classes_after": int(np.unique(post_semantic).size),
            "valid_fraction_before": float(valid.mean()),
            "valid_fraction_after": float(augmented_valid.mean()),
            "ignore_fraction_before": float(np.mean(source_labels == 6)),
            "ignore_fraction_after": float(np.mean(augmented_labels == 6)),
            "replay_equal": True,
        }
    )

augmentation_example_manifest = pd.DataFrame(augmentation_rows)
display(augmentation_example_manifest)

fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
for row_index, example in enumerate(augmentation_examples):
    axes[row_index, 0].imshow(_display_image(example["channels"], example["limits"]))
    axes[row_index, 0].set_title(
        f"{example['dataset']} {example['role']} {example['scene_id']}\nIRRG source"
    )
    axes[row_index, 1].imshow(example["labels"], cmap=LABEL_CMAP, vmin=0, vmax=6)
    axes[row_index, 1].set_title("source labels")
    axes[row_index, 2].imshow(_display_image(example["augmented"], example["limits"]))
    axes[row_index, 2].set_title("augmented IRRG")
    axes[row_index, 3].imshow(
        example["augmented_labels"], cmap=LABEL_CMAP, vmin=0, vmax=6
    )
    axes[row_index, 3].set_title("synchronized labels")
    for axis in axes[row_index]:
        axis.set_axis_off()
fig.savefig(ARTIFACT_ROOT / "augmentation-examples.png", dpi=160)
plt.show()

# %% [markdown]
# ## Save tables, figures, and provenance
#
# These artifacts are outside the Git checkout. The provenance records DVC lock,
# catalog, split, quality-report, analysis-source, and augmentation hashes, plus the
# exact runtime versions and sampling coordinates used for the audit.

# %%
asset_metadata.to_csv(ARTIFACT_ROOT / "asset-metadata.csv", index=False)
asset_summary.to_csv(ARTIFACT_ROOT / "asset-summary.csv", index=False)
alignment.to_csv(ARTIFACT_ROOT / "alignment-checks.csv", index=False)
label_counts.to_csv(ARTIFACT_ROOT / "per-scene-label-counts.csv", index=False)
pooled_balance.to_csv(ARTIFACT_ROOT / "pooled-label-balance.csv", index=False)
band_statistics.to_csv(ARTIFACT_ROOT / "irrg-band-statistics.csv", index=False)
scene_band_statistics.to_csv(
    ARTIFACT_ROOT / "irrg-per-scene-band-statistics.csv", index=False
)
pd.DataFrame(window_rows).to_csv(
    ARTIFACT_ROOT / "sampled-window-manifest.csv", index=False
)
augmentation_audit.to_csv(ARTIFACT_ROOT / "augmentation-audit.csv", index=False)
augmentation_effect_summary.to_csv(
    ARTIFACT_ROOT / "augmentation-summary.csv", index=False
)
augmentation_example_manifest.to_csv(
    ARTIFACT_ROOT / "augmentation-examples-manifest.csv", index=False
)


def _sha256(path: Path) -> str:
    """Return SHA-256 for a small pipeline manifest file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


try:
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
except (OSError, subprocess.CalledProcessError):
    git_revision = None

dvc_lock_path = REPOSITORY_ROOT / "dvc.lock"
provenance = {
    "schema_version": 2,
    "created_at_utc": datetime.now(UTC).isoformat(),
    "git_revision": git_revision,
    "analysis_source_sha256": _sha256(Path(__file__).resolve())
    if "__file__" in globals()
    else _sha256(REPOSITORY_ROOT / "notebooks_src/data_eda_augmentation.py"),
    "dvc_lock_sha256": _sha256(dvc_lock_path) if dvc_lock_path.is_file() else None,
    "analysis_design": {
        "label_counts": "full-scene counts from DVC quality reports",
        "irrg_profile": {
            "windows_per_scene": WINDOWS_PER_SCENE,
            "window_size": SAMPLE_SIZE,
            "roles": ["train", "validation"],
            "scenes_sampled_per_role": "all active split scenes",
            "seed": RANDOM_SEED,
        },
        "augmentation_audit": {
            "roles": ["train"],
            "windows_per_scene": 1,
            "seeds_per_window": AUDIT_REPEATS_PER_SCENE,
            "geometry_comparison": (
                "same seeded geometry with and without optical transforms"
            ),
        },
    },
    "runtime": {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "rasterio": rasterio.__version__,
        "albumentations": albumentations.__version__,
        "matplotlib": plt.matplotlib.__version__,
    },
    "augmentation": {
        "id": AUGMENTATION.augmentation_id,
        "version": AUGMENTATION.version,
        "content_hash": canonical_hash(AUGMENTATION.to_dict()),
        "seed_base": RANDOM_SEED,
    },
    "datasets": {
        display_name: {
            "dataset_id": DATASET_CONFIGS[display_name]["dataset_id"],
            "catalog_content_hash": CATALOGS[display_name]["catalog_content_hash"],
            "catalog_file_sha256": _sha256(
                PREPARED_ROOT / display_name.lower() / "catalog.json"
            ),
            "split_id": SPLITS[display_name].split_id,
            "split_hash": SPLITS[display_name].content_hash,
            "quality_file_sha256": _sha256(
                PREPARED_ROOT / display_name.lower() / "quality.json"
            ),
            "quality_splits_scanned": QUALITY_REPORTS[display_name]["included_splits"],
        }
        for display_name in DATASET_CONFIGS
    },
    "artifacts": sorted(
        {
            *(path.name for path in ARTIFACT_ROOT.iterdir() if path.is_file()),
            "eda-provenance.json",
        }
    ),
}
(ARTIFACT_ROOT / "eda-provenance.json").write_text(
    json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
display(provenance)
print(f"EDA outputs saved to {ARTIFACT_ROOT}")
