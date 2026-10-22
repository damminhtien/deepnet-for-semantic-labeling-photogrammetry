# %% [markdown]
# # ISPRS data EDA and augmentation review
#
# This reproducible, cell-oriented analysis reads the DVC-prepared catalogs and
# train/validation quality reports. It scans full-scene IRRG in raster blocks and
# optionally audits Potsdam DSM/nDSM files from a configured auxiliary root. It
# never modifies source rasters, fits normalization on validation data, or reads
# the Vaihingen post-selection scenes.
#
# Before running, prepare the DVC inputs and run `dvc repro`. Set
# `DEEPNET_DATA_ROOT` to the materialized DVC input directory. Set
# `DEEPNET_AUX_DATA_ROOT` to the directory containing the Potsdam DSM/nDSM source
# folders when those assets are available. Set `DEEPNET_ARTIFACT_ROOT` outside Git.

# %%
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
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

from deepnet.data.augmentation_audit import (
    semantic_boundary_mask,
    window_starts,
    window_sum_grid,
)
from deepnet.data.augmentations import AlbumentationsAugmenter
from deepnet.data.catalog import (
    load_augmentation_spec,
    load_domain_spec,
    load_model_input_spec,
    load_preprocessing_spec,
    load_scene_catalog,
    load_split_manifest,
)
from deepnet.data.contracts import GridSpec, SceneRecord, WindowSpec, canonical_hash
from deepnet.data.preprocessing import decode_rgb_labels
from deepnet.data.raster import RasterioSceneReader, sha256_file


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
print(f"Notebook Python interpreter: {sys.executable}")
DATA_ROOT = (
    Path(os.environ.get("DEEPNET_DATA_ROOT", REPOSITORY_ROOT / "data/raw/inputs"))
    .expanduser()
    .resolve()
)
AUX_DATA_ROOT = (
    Path(os.environ.get("DEEPNET_AUX_DATA_ROOT", DATA_ROOT)).expanduser().resolve()
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
# notebook. Potsdam follows the benchmark-aligned 18/6/14 split. Its 14
# challenge-heldout scenes are excluded from this analysis and stay closed until
# selection. The 17 official Vaihingen scenes remain closed until the DINOv3 config
# and selected checkpoint are frozen. A versioned evaluation
# protocol checks both split hashes before analysis begins.

# %%
DATASET_CONFIGS = {
    "Potsdam": {
        "dataset_id": "isprs-potsdam",
        "split_file": "potsdam-dinov3-v2.json",
        "crop_size": 512,
    },
    "Vaihingen": {
        "dataset_id": "isprs-vaihingen",
        "split_file": "vaihingen-dinov3-v1.json",
        "crop_size": 288,
    },
}
EVALUATION_PROTOCOL_PATH = CONFIG_ROOT / "evaluations" / "isprs-dinov3-v2.json"
EVALUATION_PROTOCOL = json.loads(EVALUATION_PROTOCOL_PATH.read_text(encoding="utf-8"))
TRAINING_CONFIG_PATH = CONFIG_ROOT / "training" / "isprs-dinov3-joint-v2.json"
TRAINING_CONFIG = json.loads(TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
TRAINING_CONFIG_SHA256 = hashlib.sha256(TRAINING_CONFIG_PATH.read_bytes()).hexdigest()

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
if EVALUATION_PROTOCOL["status"] != "locked":
    raise RuntimeError("DINOv3 evaluation protocol must be locked before EDA")
if EVALUATION_PROTOCOL["training_config"]["sha256"] != TRAINING_CONFIG_SHA256:
    raise RuntimeError("Training config differs from the configuration locked for EDA")
if TRAINING_CONFIG["evaluation_protocol_id"] != (
    f"{EVALUATION_PROTOCOL['protocol_id']}-v{EVALUATION_PROTOCOL['version']}"
):
    raise RuntimeError("Training config references a different evaluation protocol")
if (
    TRAINING_CONFIG["execution"]["selection_metric"]
    != (EVALUATION_PROTOCOL["model_selection"]["metric"])
):
    raise RuntimeError("Training selection metric differs from the locked protocol")
if (
    TRAINING_CONFIG["execution"]["selection_aggregation"]
    != (EVALUATION_PROTOCOL["model_selection"]["aggregation"])
):
    raise RuntimeError(
        "Training selection aggregation differs from the locked protocol"
    )
if PREPROCESSING.ignore_label != EVALUATION_PROTOCOL["metrics"]["ignore_label"]:
    raise RuntimeError("The evaluation ignore label differs from preprocessing")
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
    protocol_split = EVALUATION_PROTOCOL["split_manifests"][dataset_id]

    if split.split_id != protocol_split["split_id"]:
        raise RuntimeError(
            f"Evaluation protocol uses a different split for {dataset_id}"
        )
    if split.version != protocol_split["version"]:
        raise RuntimeError(
            f"Evaluation protocol uses a different split version for {dataset_id}"
        )
    if split.content_hash != protocol_split["content_hash"]:
        raise RuntimeError(
            f"Split hash differs from the locked protocol for {dataset_id}"
        )

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

if set(EVALUATION_PROTOCOL["model_selection"]["datasets"]) != {
    settings["dataset_id"] for settings in DATASET_CONFIGS.values()
}:
    raise RuntimeError("Evaluation protocol and training datasets do not match")
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
# pixel-local pairing from geographic registration. The configured DINOv3 view uses
# IRRG only; the optional Potsdam DSM/nDSM audit below does not add those assets to
# the model input.

# %%
asset_rows = []
scene_lookup: dict[str, dict[str, SceneRecord]] = {}
for display_name, scenes in SCENES.items():
    scene_lookup[display_name] = {scene.scene_id: scene for scene in scenes}
    split = SPLITS[display_name]
    analysis_scene_ids = set(split.train_scene_ids) | set(split.validation_scene_ids)
    for scene_id in sorted(analysis_scene_ids):
        scene = scene_lookup[display_name][str(scene_id)]
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
# ## Full-scene IRRG value distributions
#
# Every pixel in every train and validation IRRG raster is scanned using native
# raster blocks. Histograms and quantiles are exact for the declared uint8 bands.
# The 17 Vaihingen post-selection scenes are excluded. These diagnostics compare
# distributions; fitted normalization statistics still come from train only.


# %%
def _histogram_summary(
    histogram: np.ndarray,
    *,
    total_pixels: int,
) -> dict[str, float | int]:
    """Summarize a complete uint8 histogram using its empirical distribution."""
    counts = np.asarray(histogram, dtype=np.int64)
    valid_pixels = int(counts.sum())
    if valid_pixels == 0:
        return {
            "valid_pixels": 0,
            "invalid_fraction": 1.0,
            "min": float("nan"),
            "p01": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "zero_fraction": float("nan"),
            "max_value_fraction": float("nan"),
        }

    digital_numbers = np.arange(counts.size, dtype=np.float64)
    cumulative = np.cumsum(counts, dtype=np.int64)

    def quantile(value: float) -> int:
        rank = max(1, int(np.ceil(value * valid_pixels)))
        return int(np.searchsorted(cumulative, rank, side="left"))

    mean = float(np.dot(digital_numbers, counts) / valid_pixels)
    variance = float(np.dot(np.square(digital_numbers - mean), counts) / valid_pixels)
    nonzero_bins = np.flatnonzero(counts)
    return {
        "valid_pixels": valid_pixels,
        "invalid_fraction": float(1 - valid_pixels / max(total_pixels, 1)),
        "min": int(nonzero_bins[0]),
        "p01": quantile(0.01),
        "p05": quantile(0.05),
        "p50": quantile(0.50),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": int(nonzero_bins[-1]),
        "mean": mean,
        "std": float(np.sqrt(max(variance, 0.0))),
        "zero_fraction": float(counts[0] / valid_pixels),
        "max_value_fraction": float(counts[-1] / valid_pixels),
    }


irrg_scene_histograms: dict[tuple[str, str, str], np.ndarray] = {}
irrg_scene_rows = []
for display_name in DATASET_CONFIGS:
    split = SPLITS[display_name]
    for role in ("train", "validation"):
        for scene_id in sorted(split.scenes_for(role)):
            scene = scene_lookup[display_name][str(scene_id)]
            asset = scene.asset_map()["irrg"]
            path = DATA_ROOT / asset.uri
            with rasterio.open(path) as source:
                if any(dtype != "uint8" for dtype in source.dtypes):
                    raise ValueError(
                        f"IRRG must be uint8 for exact histograms: {scene.scene_id}"
                    )
                if source.count != len(asset.bands):
                    raise ValueError(
                        f"IRRG band count differs from catalog: {scene.scene_id}"
                    )

                histograms = np.zeros((source.count, 256), dtype=np.int64)
                for _, raster_window in source.block_windows(1):
                    pixels = source.read(window=raster_window)
                    masks = source.read_masks(window=raster_window) > 0
                    for band_index in range(source.count):
                        values = pixels[band_index]
                        valid = masks[band_index] & np.isfinite(values)
                        valid_values = values[valid]
                        if valid_values.size and (
                            valid_values.min() < 0 or valid_values.max() > 255
                        ):
                            raise ValueError(
                                f"IRRG values fall outside [0, 255] in {scene.scene_id}"
                            )
                        histograms[band_index] += np.bincount(
                            valid_values.astype(np.uint8), minlength=256
                        )

                total_pixels = source.width * source.height
                for band_index, band_name in enumerate(asset.bands):
                    histogram = histograms[band_index]
                    irrg_scene_histograms[
                        (display_name, role, str(scene_id), band_name)
                    ] = histogram
                    irrg_scene_rows.append(
                        {
                            "dataset": display_name,
                            "role": role,
                            "scene_id": str(scene_id),
                            "band": band_name,
                            "total_pixels": total_pixels,
                            **_histogram_summary(histogram, total_pixels=total_pixels),
                        }
                    )

scene_band_statistics = pd.DataFrame(irrg_scene_rows)
band_rows = []
irrg_histogram_rows = []
for display_name in DATASET_CONFIGS:
    split = SPLITS[display_name]
    for role in ("train", "validation"):
        scene_ids = sorted(split.scenes_for(role))
        for band_name in ("ir", "red", "green"):
            scene_histograms = [
                irrg_scene_histograms[(display_name, role, str(scene_id), band_name)]
                for scene_id in scene_ids
            ]
            pooled_histogram = np.sum(scene_histograms, axis=0, dtype=np.int64)
            band_rows.append(
                {
                    "dataset": display_name,
                    "role": role,
                    "band": band_name,
                    "scenes_scanned": len(scene_ids),
                    "total_pixels": int(
                        scene_band_statistics.loc[
                            (scene_band_statistics["dataset"] == display_name)
                            & (scene_band_statistics["role"] == role)
                            & (scene_band_statistics["band"] == band_name),
                            "total_pixels",
                        ].sum()
                    ),
                    **_histogram_summary(
                        pooled_histogram,
                        total_pixels=int(
                            scene_band_statistics.loc[
                                (scene_band_statistics["dataset"] == display_name)
                                & (scene_band_statistics["role"] == role)
                                & (scene_band_statistics["band"] == band_name),
                                "total_pixels",
                            ].sum()
                        ),
                    ),
                }
            )
            for scene_id, histogram in zip(scene_ids, scene_histograms, strict=True):
                irrg_histogram_rows.extend(
                    {
                        "dataset": display_name,
                        "role": role,
                        "scene_id": str(scene_id),
                        "band": band_name,
                        "digital_number": digital_number,
                        "pixel_count": int(count),
                    }
                    for digital_number, count in enumerate(histogram)
                )

band_statistics = pd.DataFrame(band_rows)
irrg_full_histograms = pd.DataFrame(irrg_histogram_rows)
display(band_statistics)
display(
    scene_band_statistics.sort_values(
        ["dataset", "role", "band", "p50"], ascending=[True, True, True, False]
    )
    .groupby(["dataset", "role", "band"], as_index=False)
    .head(1)
)

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, constrained_layout=True)
for row_index, display_name in enumerate(DATASET_CONFIGS):
    for column_index, band_name in enumerate(("ir", "red", "green")):
        axis = axes[row_index, column_index]
        for role, color in (("train", "#4472C4"), ("validation", "#ED7D31")):
            split = SPLITS[display_name]
            histogram = np.sum(
                [
                    irrg_scene_histograms[
                        (display_name, role, str(scene_id), band_name)
                    ]
                    for scene_id in sorted(split.scenes_for(role))
                ],
                axis=0,
                dtype=np.int64,
            )
            axis.plot(
                np.arange(histogram.size),
                histogram / max(int(histogram.sum()), 1),
                label=role,
                color=color,
            )
        axis.set_title(f"{display_name} · {band_name} · all pixels")
        axis.set_xlabel("source digital number")
        axis.set_ylabel("pixel fraction")
        axis.grid(alpha=0.2)
        axis.legend()
fig.savefig(ARTIFACT_ROOT / "irrg-training-validation-histograms.png", dpi=160)
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
for row_index, display_name in enumerate(DATASET_CONFIGS):
    for column_index, band_name in enumerate(("ir", "red", "green")):
        axis = axes[row_index, column_index]
        subset = scene_band_statistics[
            (scene_band_statistics["dataset"] == display_name)
            & (scene_band_statistics["band"] == band_name)
        ]
        for role, marker, color in (
            ("train", "o", "#4472C4"),
            ("validation", "s", "#ED7D31"),
        ):
            role_rows = subset[subset["role"] == role].reset_index(drop=True)
            positions = np.arange(len(role_rows)) + (0 if role == "train" else 0.15)
            axis.errorbar(
                positions,
                role_rows["p50"],
                yerr=np.vstack(
                    [
                        role_rows["p50"] - role_rows["p05"],
                        role_rows["p95"] - role_rows["p50"],
                    ]
                ),
                fmt=marker,
                markersize=3,
                capsize=1,
                color=color,
                label=role,
            )
        axis.set_title(f"{display_name} · {band_name} · scene percentiles")
        axis.set_ylabel("digital number (p05–p95; center=p50)")
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
fig.savefig(ARTIFACT_ROOT / "irrg-per-scene-percentiles.png", dpi=160)
plt.show()

# %% [markdown]
# ## Potsdam DSM and nDSM source audit
#
# These optional assets are read from `DEEPNET_AUX_DATA_ROOT`. The current DVC
# snapshot does not contain them, so each source file is SHA-256 recorded in the
# output manifest. Only train and validation scenes are scanned. DSM alignment is
# accepted only when CRS and affine metadata match the IRRG reference grid. Equal
# nDSM/JPEG dimensions do not establish registration; nDSM remains excluded from
# the model input until independent coregistration evidence is supplied.

# %%
POTSDAM_SOURCE_CONFIG = json.loads(
    (CONFIG_ROOT / "datasets" / "isprs-potsdam.json").read_text(encoding="utf-8")
)
POTSDAM_MULTIMODAL_CONFIG = json.loads(
    (CONFIG_ROOT / "domains" / "isprs-potsdam-multimodal-v1.json").read_text(
        encoding="utf-8"
    )
)
AUXILIARY_BANDS = {band["name"]: band for band in POTSDAM_MULTIMODAL_CONFIG["bands"]}
AUXILIARY_ASSETS = ("dsm", "ndsm")
auxiliary_metadata_rows = []
auxiliary_alignment_rows = []
auxiliary_statistics_rows = []
auxiliary_histogram_rows = []


def _auxiliary_path(asset_name: str, scene_id: str) -> tuple[str, Path]:
    """Resolve a configured Potsdam auxiliary asset for one scene ID."""
    row, column = (int(value) for value in scene_id.split("_", maxsplit=1))
    relative_uri = POTSDAM_SOURCE_CONFIG["assets"][asset_name]["path"].format(
        scene_id=scene_id,
        row=row,
        column=column,
        area=scene_id,
    )
    return relative_uri, AUX_DATA_ROOT / relative_uri


def _grid_from_raster(source: Any, scene: SceneRecord, asset_name: str) -> GridSpec:
    """Describe raster metadata without treating an identity transform as evidence."""
    crs = source.crs.to_string() if source.crs else None
    affine_values = tuple(float(value) for value in tuple(source.transform)[:6])
    identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    affine = (
        affine_values
        if crs is not None
        or not np.allclose(affine_values, identity, rtol=0.0, atol=1e-15)
        else None
    )
    grid_id = (
        scene.reference_grid.grid_id
        if crs is not None
        else f"{asset_name}:{scene.scene_id}:unregistered"
    )
    return GridSpec(
        grid_id=grid_id,
        width=source.width,
        height=source.height,
        crs=crs,
        affine=affine,
    )


def _merge_moments(
    count: int,
    mean: float,
    sum_squared_deviations: float,
    values: np.ndarray,
) -> tuple[int, float, float]:
    """Merge one finite raster block into stable population moments."""
    block = np.asarray(values, dtype=np.float64)
    block_count = int(block.size)
    if block_count == 0:
        return count, mean, sum_squared_deviations
    block_mean = float(block.mean())
    block_m2 = float(np.square(block - block_mean).sum())
    combined_count = count + block_count
    delta = block_mean - mean
    combined_mean = mean + delta * block_count / combined_count
    combined_m2 = (
        sum_squared_deviations
        + block_m2
        + delta * delta * count * block_count / combined_count
    )
    return combined_count, combined_mean, combined_m2


def _scan_auxiliary_raster(
    path: Path,
    *,
    asset_name: str,
    scene: SceneRecord,
    role: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], np.ndarray | None]:
    """Scan one DSM/nDSM in native blocks and return metadata and full-scene stats."""
    configured_asset = POTSDAM_SOURCE_CONFIG["assets"][asset_name]
    band_config = AUXILIARY_BANDS[asset_name]
    file_hash = sha256_file(path)

    with rasterio.open(path) as source:
        raster_grid = _grid_from_raster(source, scene, asset_name)
        reference_grid = scene.reference_grid
        same_shape = (source.width, source.height) == (
            reference_grid.width,
            reference_grid.height,
        )
        if raster_grid.same_grid(reference_grid):
            alignment_status = "verified_metadata_grid_match"
        elif raster_grid.georeferenced and reference_grid.georeferenced:
            alignment_status = "georeferenced_grid_mismatch"
        elif same_shape and configured_asset.get("alignment_evidence"):
            alignment_status = "source_attested_pixel_local"
        elif same_shape:
            alignment_status = "same_shape_only_unverified"
        else:
            alignment_status = "unverified_shape_mismatch"

        metadata = {
            "dataset": "Potsdam",
            "role": role,
            "scene_id": scene.scene_id,
            "asset": asset_name,
            "relative_uri": str(path.relative_to(AUX_DATA_ROOT)),
            "driver": source.driver,
            "width": source.width,
            "height": source.height,
            "band_count": source.count,
            "dtypes": ",".join(source.dtypes),
            "configured_representation": band_config.get("representation"),
            "configured_unit": band_config.get("unit"),
            "file_units": ",".join(unit or "" for unit in source.units),
            "scales": ",".join(str(value) for value in source.scales),
            "offsets": ",".join(str(value) for value in source.offsets),
            "nodata": source.nodata,
            "crs": raster_grid.crs,
            "affine": raster_grid.affine,
            "reference_crs": reference_grid.crs,
            "reference_affine": reference_grid.affine,
            "same_shape_as_irrg": same_shape,
            "dvc_input_snapshot_contains_file": (
                DATA_ROOT / path.relative_to(AUX_DATA_ROOT)
            ).is_file(),
            "file_size_bytes": path.stat().st_size,
            "sha256": file_hash,
        }
        alignment = {
            "dataset": "Potsdam",
            "role": role,
            "scene_id": scene.scene_id,
            "asset": asset_name,
            "status": alignment_status,
            "same_shape_as_irrg": same_shape,
            "asset_georeferenced": raster_grid.georeferenced,
            "reference_georeferenced": reference_grid.georeferenced,
            "alignment_evidence": configured_asset.get("alignment_evidence"),
        }
        if source.count != 1:
            metadata["scan_status"] = "skipped_unexpected_band_count"
            return metadata, alignment, {}, None

        integer_histogram = (
            np.zeros(256, dtype=np.int64) if source.dtypes[0] == "uint8" else None
        )
        sample_rng = np.random.default_rng(
            int.from_bytes(
                hashlib.sha256(
                    f"{RANDOM_SEED}:{scene.scene_id}:{asset_name}".encode()
                ).digest()[:8],
                byteorder="big",
            )
        )
        sample_values = []
        valid_count = 0
        invalid_count = 0
        zero_count = 0
        minimum = float("inf")
        maximum = float("-inf")
        mean = 0.0
        sum_squared_deviations = 0.0
        total_pixels = source.width * source.height
        quantile_sample_limit = 100_000

        for _, raster_window in source.block_windows(1):
            block = source.read(1, window=raster_window)
            block_valid = (
                source.read_masks(1, window=raster_window) > 0
            ) & np.isfinite(block)
            values = block[block_valid]
            invalid_count += int(block.size - values.size)
            if values.size == 0:
                continue
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))
            zero_count += int(np.count_nonzero(values == 0))
            valid_count, mean, sum_squared_deviations = _merge_moments(
                valid_count,
                mean,
                sum_squared_deviations,
                values,
            )
            if integer_histogram is not None:
                integer_histogram += np.bincount(values.astype(np.uint8), minlength=256)
            else:
                block_fraction = block.size / total_pixels
                sample_count = min(
                    values.size,
                    max(1, int(round(quantile_sample_limit * block_fraction))),
                )
                if sample_count < values.size:
                    sample_indices = sample_rng.choice(
                        values.size, size=sample_count, replace=False
                    )
                    sample_values.append(values[sample_indices])
                else:
                    sample_values.append(values)

        if valid_count == 0:
            summary = {"scan_status": "no_valid_pixels"}
            histogram_rows = None
        elif integer_histogram is not None:
            summary = {
                "scan_status": "complete",
                "quantile_method": "exact_empirical_uint8_histogram",
                "total_pixels": total_pixels,
                "valid_pixels": valid_count,
                "invalid_fraction": invalid_count / max(total_pixels, 1),
                **_histogram_summary(integer_histogram, total_pixels=total_pixels),
            }
            histogram_rows = integer_histogram
        else:
            quantile_values = np.concatenate(sample_values)
            quantiles = np.quantile(quantile_values, [0.01, 0.05, 0.5, 0.95, 0.99])
            summary = {
                "scan_status": "complete",
                "quantile_method": "deterministic_block_stratified_sample",
                "quantile_sample_count": int(quantile_values.size),
                "total_pixels": total_pixels,
                "valid_pixels": valid_count,
                "invalid_fraction": invalid_count / max(total_pixels, 1),
                "min": minimum,
                "p01": float(quantiles[0]),
                "p05": float(quantiles[1]),
                "p50": float(quantiles[2]),
                "p95": float(quantiles[3]),
                "p99": float(quantiles[4]),
                "max": maximum,
                "mean": mean,
                "std": float(np.sqrt(sum_squared_deviations / valid_count)),
                "zero_fraction": zero_count / valid_count,
                "max_value_fraction": float("nan"),
            }
            histogram_rows = None

        metadata["scan_status"] = summary.get("scan_status")
        return metadata, alignment, summary, histogram_rows


potsdam_split = SPLITS["Potsdam"]
for role in ("train", "validation"):
    for scene_id in sorted(potsdam_split.scenes_for(role)):
        scene = scene_lookup["Potsdam"][str(scene_id)]
        for asset_name in AUXILIARY_ASSETS:
            relative_uri, asset_path = _auxiliary_path(asset_name, str(scene_id))
            if not asset_path.is_file():
                auxiliary_metadata_rows.append(
                    {
                        "dataset": "Potsdam",
                        "role": role,
                        "scene_id": str(scene_id),
                        "asset": asset_name,
                        "relative_uri": relative_uri,
                        "scan_status": "file_missing",
                        "dvc_input_snapshot_contains_file": False,
                    }
                )
                auxiliary_alignment_rows.append(
                    {
                        "dataset": "Potsdam",
                        "role": role,
                        "scene_id": str(scene_id),
                        "asset": asset_name,
                        "status": "file_missing",
                    }
                )
                continue

            metadata, alignment_record, summary, histogram = _scan_auxiliary_raster(
                asset_path,
                asset_name=asset_name,
                scene=scene,
                role=role,
            )
            auxiliary_metadata_rows.append(metadata)
            auxiliary_alignment_rows.append(alignment_record)
            if summary:
                auxiliary_statistics_rows.append(
                    {
                        "dataset": "Potsdam",
                        "role": role,
                        "scene_id": str(scene_id),
                        "asset": asset_name,
                        **summary,
                    }
                )
            if histogram is not None:
                auxiliary_histogram_rows.extend(
                    {
                        "dataset": "Potsdam",
                        "role": role,
                        "scene_id": str(scene_id),
                        "asset": asset_name,
                        "digital_number": digital_number,
                        "pixel_count": int(count),
                    }
                    for digital_number, count in enumerate(histogram)
                )

auxiliary_metadata = pd.DataFrame(auxiliary_metadata_rows)
auxiliary_alignment = pd.DataFrame(auxiliary_alignment_rows)
auxiliary_statistics = pd.DataFrame(auxiliary_statistics_rows)
auxiliary_histograms = pd.DataFrame(auxiliary_histogram_rows)
if not auxiliary_alignment.empty:
    display(
        auxiliary_alignment.groupby(["asset", "status"], dropna=False)
        .size()
        .rename("scene_count")
        .reset_index()
    )
if not auxiliary_statistics.empty:
    display(
        auxiliary_statistics.groupby(["asset", "role"], as_index=False).agg(
            scenes=("scene_id", "nunique"),
            valid_pixels=("valid_pixels", "sum"),
            value_min=("min", "min"),
            value_p05=("p05", "min"),
            value_p50=("p50", "median"),
            value_p95=("p95", "max"),
            value_max=("max", "max"),
            mean_of_scene_means=("mean", "mean"),
        )
    )
else:
    display("No configured Potsdam DSM/nDSM files were found under the auxiliary root.")

if not auxiliary_statistics.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    dsm_rows = auxiliary_statistics[auxiliary_statistics["asset"] == "dsm"]
    for role, marker, color in (
        ("train", "o", "#4472C4"),
        ("validation", "s", "#ED7D31"),
    ):
        role_rows = dsm_rows[dsm_rows["role"] == role].reset_index(drop=True)
        positions = np.arange(len(role_rows))
        axes[0].errorbar(
            positions,
            role_rows["p50"],
            yerr=np.vstack(
                [
                    role_rows["p50"] - role_rows["p05"],
                    role_rows["p95"] - role_rows["p50"],
                ]
            ),
            fmt=marker,
            markersize=3,
            capsize=1,
            color=color,
            label=role,
        )
    axes[0].set_title("Potsdam DSM · per-scene p05–p95")
    axes[0].set_ylabel("source value (configured as metres)")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend()

    for role, color in (("train", "#4472C4"), ("validation", "#ED7D31")):
        role_histogram = (
            auxiliary_histograms[
                (auxiliary_histograms["asset"] == "ndsm")
                & (auxiliary_histograms["role"] == role)
            ]
            .groupby("digital_number")["pixel_count"]
            .sum()
        )
        if not role_histogram.empty:
            axes[1].plot(
                role_histogram.index,
                role_histogram / role_histogram.sum(),
                label=role,
                color=color,
            )
    axes[1].set_title("Potsdam nDSM JPEG · encoded DN distribution")
    axes[1].set_xlabel("encoded digital number")
    axes[1].set_ylabel("pixel fraction")
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    fig.savefig(ARTIFACT_ROOT / "potsdam-elevation-value-profiles.png", dpi=160)
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
# ## Rare classes, semantic edges, and padding audit
#
# The audit scans overlapping candidate windows across every training scene and
# reports crop-level rare-class coverage and semantic boundary density. A class is
# rare when it occupies at most 6% of train semantic pixels or occurs in at most
# half of train scenes. The held-out test split is never read. Boundary metrics use
# 4-neighbor semantic class transitions; same-class object boundaries are not
# observable because the labels have no instance IDs. Affine padding is measured
# separately from source
# NoData, label ignore pixels, and model patch-stride padding. A class is considered
# rare when it occupies at most 6% of train semantic pixels or occurs in at most half
# of train scenes. Candidate windows overlap at half the configured crop size.

# %%
MODEL_CONFIG_PATH = CONFIG_ROOT / "models" / f"{TRAINING_CONFIG['model_id']}-v1.json"
MODEL_CONFIG = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
MODEL_PATCH_SIZE = int(MODEL_CONFIG["backbone"]["patch_size"])
RARE_CLASS_SHARE_THRESHOLD = 0.06
RARE_CLASS_SCENE_PRESENCE_THRESHOLD = 0.5
AUDIT_GRID_STRIDE_DIVISOR = 2
RARE_BOUNDARY_SEEDS_PER_CROP = 8
RARE_CLASS_IDS: dict[str, tuple[int, ...]] = {}
window_scan_rows: list[dict[str, Any]] = []
focus_summary_rows: list[dict[str, Any]] = []


def _read_full_training_labels(
    scene: SceneRecord,
    label_asset: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one training label raster and its validity mask without imagery."""
    grid = scene.reference_grid
    window = WindowSpec(0, 0, grid.height, grid.width)
    with RasterioSceneReader(scene, DATA_ROOT) as reader:
        labels_rgb, valid = reader.read_window_with_mask(
            label_asset,
            window,
            resampling=PREPROCESSING.categorical_resampling,
        )
    labels = decode_rgb_labels(labels_rgb, PREPROCESSING, valid_mask=valid)
    labels[~valid] = PREPROCESSING.ignore_label
    return labels, valid


def _window_key(row: pd.Series) -> tuple[str, str, int, int]:
    """Return a stable identifier for one dataset scene crop."""
    return (
        str(row["dataset"]),
        str(row["scene_id"]),
        int(row["row"]),
        int(row["col"]),
    )


for dataset_name in DATASET_CONFIGS:
    split = SPLITS[dataset_name]
    crop_size = DATASET_CONFIGS[dataset_name]["crop_size"]
    label_asset = split.label_asset_for("train")
    train_scenes = sorted(split.train_scene_ids)
    scene_counts = label_counts.loc[
        (label_counts["dataset"] == dataset_name) & (label_counts["role"] == "train")
    ]
    semantic_total = int(
        sum(
            scene_counts[f"pixels_{CLASS_NAMES[class_id]}"].sum()
            for class_id in range(6)
        )
    )
    rare_classes = []
    for class_id in range(6):
        class_name = CLASS_NAMES[class_id]
        class_column = f"pixels_{class_name}"
        class_pixels = int(scene_counts[class_column].sum())
        scene_presence = int((scene_counts[class_column] > 0).sum())
        class_share = class_pixels / max(semantic_total, 1)
        presence_fraction = scene_presence / max(len(train_scenes), 1)
        is_rare = (
            class_share <= RARE_CLASS_SHARE_THRESHOLD
            or presence_fraction <= RARE_CLASS_SCENE_PRESENCE_THRESHOLD
        )
        if is_rare:
            rare_classes.append(class_id)
    RARE_CLASS_IDS[dataset_name] = tuple(rare_classes)

    crop_stride = max(1, crop_size // AUDIT_GRID_STRIDE_DIVISOR)
    for scene_id in train_scenes:
        scene = scene_lookup[dataset_name][str(scene_id)]
        scene.require_aligned((label_asset,))
        labels, valid = _read_full_training_labels(scene, label_asset)
        height, width = labels.shape
        rows = window_starts(height, crop_size, crop_stride)
        columns = window_starts(width, crop_size, crop_stride)
        row_grid, column_grid = np.meshgrid(rows, columns, indexing="ij")
        flat_rows = row_grid.ravel()
        flat_columns = column_grid.ravel()
        window_area = crop_size * crop_size

        valid_counts = window_sum_grid(
            valid, rows, columns, (crop_size, crop_size)
        ).ravel()
        ignore_counts = window_sum_grid(
            valid & (labels == PREPROCESSING.ignore_label),
            rows,
            columns,
            (crop_size, crop_size),
        ).ravel()
        semantic_boundaries = semantic_boundary_mask(
            labels,
            valid,
            num_classes=PREPROCESSING.ignore_label,
        )
        boundary_counts = window_sum_grid(
            semantic_boundaries,
            rows,
            columns,
            (crop_size, crop_size),
        ).ravel()
        class_counts = {
            class_id: window_sum_grid(
                valid & (labels == class_id),
                rows,
                columns,
                (crop_size, crop_size),
            ).ravel()
            for class_id in range(PREPROCESSING.ignore_label)
        }

        for window_index, (row, col) in enumerate(
            zip(flat_rows, flat_columns, strict=True)
        ):
            semantic_pixels = int(
                sum(class_counts[class_id][window_index] for class_id in range(6))
            )
            record: dict[str, Any] = {
                "dataset": dataset_name,
                "role": "train",
                "scene_id": str(scene_id),
                "row": int(row),
                "col": int(col),
                "height": crop_size,
                "width": crop_size,
                "candidate_stride": crop_stride,
                "valid_pixels": int(valid_counts[window_index]),
                "source_invalid_pixels": window_area - int(valid_counts[window_index]),
                "source_valid_fraction": float(valid_counts[window_index])
                / window_area,
                "label_ignore_pixels": int(ignore_counts[window_index]),
                "label_ignore_fraction": float(ignore_counts[window_index])
                / window_area,
                "semantic_pixels": semantic_pixels,
                "semantic_boundary_pixels": int(boundary_counts[window_index]),
                "semantic_boundary_fraction": float(boundary_counts[window_index])
                / max(semantic_pixels, 1),
                "model_patch_size": MODEL_PATCH_SIZE,
                "model_padding_bottom": (-crop_size) % MODEL_PATCH_SIZE,
                "model_padding_right": (-crop_size) % MODEL_PATCH_SIZE,
            }
            for class_id in range(6):
                count = int(class_counts[class_id][window_index])
                record[f"class_{class_id}_pixels"] = count
                record[f"class_{class_id}_share"] = count / max(semantic_pixels, 1)
            window_scan_rows.append(record)

        print(
            f"Scanned training label windows: {dataset_name} scene {scene_id} "
            f"({len(rows) * len(columns)} candidate crops)"
        )

window_scan = pd.DataFrame(window_scan_rows)

for dataset_name in DATASET_CONFIGS:
    candidates = window_scan.loc[window_scan["dataset"] == dataset_name]
    train_scenes = label_counts.loc[
        (label_counts["dataset"] == dataset_name) & (label_counts["role"] == "train")
    ]
    scene_count = int(train_scenes["scene_id"].nunique())
    dataset_scene_counts = label_counts.loc[
        (label_counts["dataset"] == dataset_name) & (label_counts["role"] == "train")
    ]
    dataset_semantic_total = int(
        sum(
            dataset_scene_counts[f"pixels_{CLASS_NAMES[class_id]}"].sum()
            for class_id in range(6)
        )
    )
    for class_id in range(6):
        class_column = f"pixels_{CLASS_NAMES[class_id]}"
        train_pixels = int(dataset_scene_counts[class_column].sum())
        scenes_with_class = int((dataset_scene_counts[class_column] > 0).sum())
        global_share = train_pixels / max(dataset_semantic_total, 1)
        presence_fraction = scenes_with_class / max(scene_count, 1)
        is_rare = class_id in RARE_CLASS_IDS[dataset_name]
        class_share = candidates[f"class_{class_id}_share"]
        present_crop_shares = class_share.loc[class_share > 0]
        class_stats = {
            "focus_type": "rare_class" if is_rare else "class_coverage",
            "dataset": dataset_name,
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "rare_class": is_rare,
            "train_scenes": scene_count,
            "train_scenes_with_class": scenes_with_class,
            "train_scene_presence_fraction": presence_fraction,
            "train_class_pixels": train_pixels,
            "global_train_share": global_share,
            "candidate_windows": int(len(candidates)),
            "candidate_windows_with_class": int((class_share > 0).sum()),
            "candidate_window_presence_fraction": float((class_share > 0).mean()),
            "crop_class_share_p50": float(class_share.quantile(0.50)),
            "crop_class_share_p95": float(class_share.quantile(0.95)),
            "crop_class_share_max": float(class_share.max()),
            "crop_class_share_p95_when_present": (
                float(present_crop_shares.quantile(0.95))
                if not present_crop_shares.empty
                else 0.0
            ),
            "crop_boundary_fraction_p95": float(
                candidates["semantic_boundary_fraction"].quantile(0.95)
            ),
            "model_patch_size": MODEL_PATCH_SIZE,
            "model_window_padding_bottom": (-DATASET_CONFIGS[dataset_name]["crop_size"])
            % MODEL_PATCH_SIZE,
            "model_window_padding_right": (-DATASET_CONFIGS[dataset_name]["crop_size"])
            % MODEL_PATCH_SIZE,
        }
        focus_summary_rows.append(class_stats)

    focus_summary_rows.append(
        {
            "focus_type": "semantic_boundary",
            "dataset": dataset_name,
            "class_id": np.nan,
            "class_name": "4-neighbor class transition",
            "rare_class": False,
            "train_scenes": scene_count,
            "train_scenes_with_class": np.nan,
            "train_scene_presence_fraction": np.nan,
            "train_class_pixels": np.nan,
            "global_train_share": np.nan,
            "candidate_windows": int(len(candidates)),
            "candidate_windows_with_class": int(
                (candidates["semantic_boundary_pixels"] > 0).sum()
            ),
            "candidate_window_presence_fraction": float(
                (candidates["semantic_boundary_pixels"] > 0).mean()
            ),
            "crop_class_share_p50": float(
                candidates["semantic_boundary_fraction"].quantile(0.50)
            ),
            "crop_class_share_p95": float(
                candidates["semantic_boundary_fraction"].quantile(0.95)
            ),
            "crop_class_share_max": float(
                candidates["semantic_boundary_fraction"].max()
            ),
            "crop_class_share_p95_when_present": float(
                candidates.loc[
                    candidates["semantic_boundary_pixels"] > 0,
                    "semantic_boundary_fraction",
                ].quantile(0.95)
            ),
            "crop_boundary_fraction_p95": float(
                candidates["semantic_boundary_fraction"].quantile(0.95)
            ),
            "model_patch_size": MODEL_PATCH_SIZE,
            "model_window_padding_bottom": (-DATASET_CONFIGS[dataset_name]["crop_size"])
            % MODEL_PATCH_SIZE,
            "model_window_padding_right": (-DATASET_CONFIGS[dataset_name]["crop_size"])
            % MODEL_PATCH_SIZE,
        }
    )

augmentation_focus_summary = pd.DataFrame(focus_summary_rows)
display(
    augmentation_focus_summary.loc[
        (augmentation_focus_summary["rare_class"])
        | (augmentation_focus_summary["focus_type"] == "semantic_boundary")
    ]
)

# %% [markdown]
# ## Select dense, sparse, and boundary-rich train crops
#
# For each rare class, keep one crop at the highest observed density and one near
# the 10th percentile of positive crop densities. Add a distinct crop with the most
# semantic class transitions. This exposes both class-rich and fragile low-pixel
# examples while keeping crop choice deterministic and train-only.

# %%
selected_focus_examples: dict[tuple[str, str, int, int], dict[str, Any]] = {}


def _add_focus_example(row: pd.Series, focus: str) -> None:
    """Keep a crop once and attach all deterministic reasons for selecting it."""
    key = _window_key(row)
    selected = selected_focus_examples.setdefault(
        key,
        {
            "dataset": str(row["dataset"]),
            "scene_id": str(row["scene_id"]),
            "row": int(row["row"]),
            "col": int(row["col"]),
            "height": int(row["height"]),
            "width": int(row["width"]),
            "focus_categories": [],
            "source_window_boundary_fraction": float(row["semantic_boundary_fraction"]),
            **{
                f"window_class_{class_id}_pixels": int(row[f"class_{class_id}_pixels"])
                for class_id in range(6)
            },
            **{
                f"window_class_{class_id}_share": float(row[f"class_{class_id}_share"])
                for class_id in range(6)
            },
        },
    )
    if focus not in selected["focus_categories"]:
        selected["focus_categories"].append(focus)


for dataset_name in DATASET_CONFIGS:
    candidates = window_scan.loc[window_scan["dataset"] == dataset_name]
    used_keys: set[tuple[str, str, int, int]] = set()
    for class_id in RARE_CLASS_IDS[dataset_name]:
        class_column = f"class_{class_id}_pixels"
        present = candidates.loc[candidates[class_column] > 0]
        if present.empty:
            raise RuntimeError(
                f"Rare class {CLASS_NAMES[class_id]} has no candidate crop in "
                f"{dataset_name} training scenes"
            )
        class_share_column = f"class_{class_id}_share"
        ranked_high = present.sort_values(
            [class_share_column, "semantic_boundary_fraction", class_column],
            ascending=False,
        )
        high_density_row = ranked_high.iloc[0]
        _add_focus_example(
            high_density_row,
            f"rare_class_{class_id}_{CLASS_NAMES[class_id]}_high_density",
        )
        used_keys.add(_window_key(high_density_row))

        lower_quantile = float(present[class_share_column].quantile(0.10))
        ranked_low = present.assign(
            _density_distance=(present[class_share_column] - lower_quantile).abs()
        ).sort_values(
            ["_density_distance", class_column, "semantic_boundary_fraction"],
            ascending=[True, True, False],
        )
        for _, low_density_row in ranked_low.iterrows():
            if _window_key(low_density_row) not in used_keys:
                _add_focus_example(
                    low_density_row,
                    f"rare_class_{class_id}_{CLASS_NAMES[class_id]}_p10_density",
                )
                used_keys.add(_window_key(low_density_row))
                break

    ranked_boundaries = candidates.sort_values(
        ["semantic_boundary_fraction", "semantic_boundary_pixels"],
        ascending=False,
    )
    boundary_row = None
    for _, candidate in ranked_boundaries.iterrows():
        key = _window_key(candidate)
        if key not in used_keys:
            boundary_row = candidate
            break
    if boundary_row is None:
        boundary_row = ranked_boundaries.iloc[0]
    _add_focus_example(boundary_row, "boundary_rich_crop")

rare_boundary_manifest = pd.DataFrame(selected_focus_examples.values())

# %% [markdown]
# ## Replay rare-class and boundary crops
#
# Each selected training crop is replayed with eight deterministic seeds. The audit
# checks class-pixel retention, class-boundary density, affine support masks, label
# ignore fill, radiometric changes inside padding, and model patch-stride padding.
# The visualization shows one affine-padded example per selected crop when available.

# %%
rare_boundary_audit_rows: list[dict[str, Any]] = []
padding_audit_rows: list[dict[str, Any]] = []
rare_boundary_visuals: list[dict[str, Any]] = []

for dataset_name in DATASET_CONFIGS:
    scene_counts = label_counts.loc[
        (label_counts["dataset"] == dataset_name) & (label_counts["role"] == "train")
    ]
    dataset_examples = rare_boundary_manifest.loc[
        rare_boundary_manifest["dataset"] == dataset_name
    ].reset_index(drop=True)
    for example_index, example in dataset_examples.iterrows():
        scene = scene_lookup[dataset_name][str(example["scene_id"])]
        label_asset = SPLITS[dataset_name].label_asset_for("train")
        channels, source_labels, source_valid = _read_patch(
            scene,
            label_asset,
            int(example["height"]),
            int(example["row"]),
            int(example["col"]),
        )
        source_labels = source_labels.copy()
        source_labels[~source_valid] = PREPROCESSING.ignore_label
        source_boundaries = semantic_boundary_mask(
            source_labels,
            source_valid,
            num_classes=PREPROCESSING.ignore_label,
        )
        source_counts = _semantic_counts(source_labels, source_valid)
        source_edge_counts = np.bincount(
            source_labels[source_boundaries], minlength=PREPROCESSING.ignore_label
        )
        source_area = max(int(source_counts.sum()), 1)
        source_ignore_fraction = float(
            np.mean(source_labels == PREPROCESSING.ignore_label)
        )
        target_class_ids = RARE_CLASS_IDS[dataset_name]
        padding_mask_for_visual = None
        visual_channels = None
        visual_labels = None
        visual_valid = None
        visual_seed = None
        last_visual: dict[str, Any] | None = None

        for repeat_index in range(RARE_BOUNDARY_SEEDS_PER_CROP):
            seed = int(
                np.random.SeedSequence(
                    [
                        RANDOM_SEED,
                        300_000,
                        list(DATASET_CONFIGS).index(dataset_name),
                        example_index,
                        repeat_index,
                    ]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            geometric, geometric_labels, geometric_valid = GEOMETRY_ONLY_AUGMENTER(
                channels,
                source_labels,
                source_valid,
                seed=seed,
            )
            augmented, augmented_labels, augmented_valid = AUGMENTER(
                channels,
                source_labels,
                source_valid,
                seed=seed,
            )
            support_labels = np.zeros_like(source_labels)
            _, _, geometric_support = GEOMETRY_ONLY_AUGMENTER(
                channels,
                support_labels,
                np.ones_like(source_valid, dtype=bool),
                seed=seed,
            )
            geometric_support = np.asarray(geometric_support, dtype=bool)
            padding_mask = ~geometric_support
            np.testing.assert_array_equal(geometric_labels, augmented_labels)
            np.testing.assert_array_equal(geometric_valid, augmented_valid)
            if np.any(augmented_valid & padding_mask):
                raise AssertionError("Affine-generated padding is marked valid")
            if np.any(augmented_labels[padding_mask] != PREPROCESSING.ignore_label):
                raise AssertionError("Affine-generated padding is not ignore-labeled")
            if np.any(augmented_labels[~augmented_valid] != PREPROCESSING.ignore_label):
                raise AssertionError(
                    "Invalid augmentation pixels are not ignore-labeled"
                )

            for band_name, values in augmented.items():
                if not np.isfinite(values).all():
                    raise AssertionError(
                        f"Augmentation produced non-finite {band_name} values"
                    )
                if np.any(values < 0) or np.any(values > 255):
                    raise AssertionError(
                        f"Augmentation moved {band_name} outside [0, 255]"
                    )

            geometry_boundaries = semantic_boundary_mask(
                geometric_labels,
                geometric_valid,
                num_classes=PREPROCESSING.ignore_label,
            )
            geometry_counts = _semantic_counts(geometric_labels, geometric_valid)
            geometry_edge_counts = np.bincount(
                geometric_labels[geometry_boundaries],
                minlength=PREPROCESSING.ignore_label,
            )
            semantic_geometry_total = max(int(geometry_counts.sum()), 1)
            padding_pixels = int(padding_mask.sum())
            if padding_pixels:
                geometric_padding = np.stack(
                    [geometric[name] for name in AUGMENTER.optical_channels], axis=-1
                )[padding_mask]
                augmented_padding = np.stack(
                    [augmented[name] for name in AUGMENTER.optical_channels], axis=-1
                )[padding_mask]
                padding_geometry_nonzero_fraction = float(
                    np.mean(np.abs(geometric_padding) > 1e-6)
                )
                padding_augmented_nonzero_fraction = float(
                    np.mean(np.abs(augmented_padding) > 1e-6)
                )
                padding_radiometry_change_fraction = float(
                    np.mean(np.abs(augmented_padding - geometric_padding) > 1e-6)
                )
                padding_radiometry_mean_abs_delta = float(
                    np.mean(np.abs(augmented_padding - geometric_padding))
                )
            else:
                padding_geometry_nonzero_fraction = np.nan
                padding_augmented_nonzero_fraction = np.nan
                padding_radiometry_change_fraction = np.nan
                padding_radiometry_mean_abs_delta = np.nan

            model_padding_bottom = (-int(example["height"])) % MODEL_PATCH_SIZE
            model_padding_right = (-int(example["width"])) % MODEL_PATCH_SIZE
            padding_audit_rows.append(
                {
                    "dataset": dataset_name,
                    "role": "train",
                    "scene_id": str(example["scene_id"]),
                    "example_id": f"{dataset_name}:{example_index}",
                    "row": int(example["row"]),
                    "col": int(example["col"]),
                    "focus_categories": ";".join(example["focus_categories"]),
                    "seed": seed,
                    "repeat_index": repeat_index,
                    "model_patch_size": MODEL_PATCH_SIZE,
                    "model_padding_bottom": model_padding_bottom,
                    "model_padding_right": model_padding_right,
                    "model_patch_padding_fraction": 1.0
                    - (int(example["height"]) - model_padding_bottom)
                    * (int(example["width"]) - model_padding_right)
                    / (int(example["height"]) * int(example["width"])),
                    "affine_padding_pixels": padding_pixels,
                    "affine_padding_fraction": padding_pixels
                    / (int(example["height"]) * int(example["width"])),
                    "source_invalid_fraction_within_support": float(
                        np.mean(~augmented_valid[geometric_support])
                    ),
                    "label_ignore_fraction_within_support": float(
                        np.mean(
                            augmented_labels[geometric_support]
                            == PREPROCESSING.ignore_label
                        )
                    ),
                    "padding_label_is_ignore": bool(
                        np.all(
                            augmented_labels[padding_mask] == PREPROCESSING.ignore_label
                        )
                    ),
                    "padding_mask_is_invalid": bool(
                        not np.any(augmented_valid[padding_mask])
                    ),
                    "padding_optical_nonzero_fraction_after_geometry": (
                        padding_geometry_nonzero_fraction
                    ),
                    "padding_optical_nonzero_fraction_after_radiometry": (
                        padding_augmented_nonzero_fraction
                    ),
                    "padding_optical_radiometry_change_fraction": (
                        padding_radiometry_change_fraction
                    ),
                    "padding_optical_radiometry_mean_abs_delta": (
                        padding_radiometry_mean_abs_delta
                    ),
                    "geometry_valid_fraction": float(geometric_valid.mean()),
                    "source_ignore_fraction": source_ignore_fraction,
                }
            )

            for class_id in target_class_ids:
                source_pixels = int(source_counts[class_id])
                augmented_pixels = int(geometry_counts[class_id])
                source_class_edges = int(source_edge_counts[class_id])
                augmented_class_edges = int(geometry_edge_counts[class_id])
                class_present = source_pixels > 0
                row_record = {
                    "dataset": dataset_name,
                    "role": "train",
                    "example_id": f"{dataset_name}:{example_index}",
                    "scene_id": str(example["scene_id"]),
                    "row": int(example["row"]),
                    "col": int(example["col"]),
                    "focus_categories": ";".join(example["focus_categories"]),
                    "seed": seed,
                    "repeat_index": repeat_index,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "rare_class": True,
                    "source_class_present": class_present,
                    "source_class_pixels": source_pixels,
                    "augmented_class_pixels": augmented_pixels,
                    "class_pixel_retention_ratio": augmented_pixels / source_pixels
                    if class_present
                    else np.nan,
                    "class_disappeared_after_geometry": bool(
                        source_pixels > 0 and augmented_pixels == 0
                    )
                    if class_present
                    else np.nan,
                    "source_class_boundary_pixels": source_class_edges,
                    "augmented_class_boundary_pixels": augmented_class_edges,
                    "source_class_boundary_fraction": source_class_edges / source_pixels
                    if class_present
                    else np.nan,
                    "augmented_class_boundary_fraction": augmented_class_edges
                    / augmented_pixels
                    if augmented_pixels > 0
                    else np.nan,
                    "source_semantic_boundary_fraction": int(source_boundaries.sum())
                    / source_area,
                    "augmented_semantic_boundary_fraction": int(
                        geometry_boundaries.sum()
                    )
                    / semantic_geometry_total,
                    "affine_padding_fraction": padding_pixels
                    / (int(example["height"]) * int(example["width"])),
                    "geometry_valid_fraction": float(geometric_valid.mean()),
                }
                rare_boundary_audit_rows.append(row_record)

            last_visual = {
                "dataset": dataset_name,
                "scene_id": str(example["scene_id"]),
                "row": int(example["row"]),
                "col": int(example["col"]),
                "focus_categories": ";".join(example["focus_categories"]),
                "channels": channels,
                "limits": {
                    band_name: tuple(
                        float(value)
                        for value in np.percentile(
                            channels[band_name][source_valid], [2, 98]
                        )
                    )
                    for band_name in ("red", "green", "ir")
                },
                "source_labels": source_labels,
                "source_valid": source_valid,
                "augmented_channels": augmented,
                "augmented_labels": augmented_labels,
                "augmented_valid": augmented_valid,
                "padding_mask": padding_mask,
                "geometry_boundaries": geometry_boundaries,
                "seed": seed,
            }
            if visual_seed is None or (
                padding_pixels > 0
                and padding_mask_for_visual is not None
                and not padding_mask_for_visual.any()
            ):
                padding_mask_for_visual = padding_mask
                visual_channels = augmented
                visual_labels = augmented_labels
                visual_valid = augmented_valid
                visual_seed = seed

        if last_visual is not None:
            if visual_seed is not None:
                last_visual["augmented_channels"] = visual_channels
                last_visual["augmented_labels"] = visual_labels
                last_visual["augmented_valid"] = visual_valid
                last_visual["padding_mask"] = padding_mask_for_visual
                last_visual["geometry_boundaries"] = semantic_boundary_mask(
                    visual_labels,
                    visual_valid,
                    num_classes=PREPROCESSING.ignore_label,
                )
                last_visual["seed"] = visual_seed
            rare_boundary_visuals.append(last_visual)

rare_boundary_audit = pd.DataFrame(rare_boundary_audit_rows)
padding_audit = pd.DataFrame(padding_audit_rows)
if rare_boundary_audit.empty or padding_audit.empty:
    raise RuntimeError("The rare-class and padding augmentation audit was empty")

present_rare_boundary_audit = rare_boundary_audit.loc[
    rare_boundary_audit["source_class_present"]
].copy()
rare_boundary_augmentation_summary = present_rare_boundary_audit.groupby(
    ["dataset", "class_id", "class_name"], as_index=False
).agg(
    selected_class_crops=("example_id", "nunique"),
    crop_seed_trials=("seed", "size"),
    source_class_pixels_mean=("source_class_pixels", "mean"),
    class_pixel_retention_median=("class_pixel_retention_ratio", "median"),
    class_pixel_retention_p05=(
        "class_pixel_retention_ratio",
        lambda values: values.quantile(0.05),
    ),
    class_disappearance_rate=("class_disappeared_after_geometry", "mean"),
    source_class_boundary_fraction_mean=("source_class_boundary_fraction", "mean"),
    augmented_class_boundary_fraction_mean=(
        "augmented_class_boundary_fraction",
        "mean",
    ),
    source_semantic_boundary_fraction_mean=(
        "source_semantic_boundary_fraction",
        "mean",
    ),
    augmented_semantic_boundary_fraction_mean=(
        "augmented_semantic_boundary_fraction",
        "mean",
    ),
    affine_padding_fraction_mean=("affine_padding_fraction", "mean"),
    affine_padding_fraction_p95=(
        "affine_padding_fraction",
        lambda values: values.quantile(0.95),
    ),
)
padding_summary = padding_audit.groupby("dataset", as_index=False).agg(
    selected_crops=("example_id", "nunique"),
    seed_trials=("seed", "size"),
    affine_padding_trial_fraction=(
        "affine_padding_fraction",
        lambda values: float((values > 0).mean()),
    ),
    affine_padding_fraction_mean=("affine_padding_fraction", "mean"),
    affine_padding_fraction_p95=(
        "affine_padding_fraction",
        lambda values: values.quantile(0.95),
    ),
    affine_padding_fraction_max=("affine_padding_fraction", "max"),
    padding_label_ignore_pass_rate=("padding_label_is_ignore", "mean"),
    padding_mask_invalid_pass_rate=("padding_mask_is_invalid", "mean"),
    padding_nonzero_after_geometry_mean=(
        "padding_optical_nonzero_fraction_after_geometry",
        "mean",
    ),
    padding_nonzero_after_radiometry_mean=(
        "padding_optical_nonzero_fraction_after_radiometry",
        "mean",
    ),
    padding_radiometry_change_trial_rate=(
        "padding_optical_radiometry_change_fraction",
        lambda values: float((values > 0).mean()),
    ),
    padding_radiometry_mean_abs_delta=(
        "padding_optical_radiometry_mean_abs_delta",
        "mean",
    ),
    model_padding_fraction_max=("model_patch_padding_fraction", "max"),
)
display(rare_boundary_augmentation_summary)
display(padding_summary)

EDGE_CMAP = ListedColormap(["#00FFFF"])
PADDING_CMAP = ListedColormap([(0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 1.0, 0.62)])
fig, axes = plt.subplots(
    len(rare_boundary_visuals),
    4,
    figsize=(16, max(4, 3.6 * len(rare_boundary_visuals))),
    squeeze=False,
    constrained_layout=True,
)
for row_index, example in enumerate(rare_boundary_visuals):
    axes[row_index, 0].imshow(_display_image(example["channels"], example["limits"]))
    axes[row_index, 0].set_title("source IRRG")
    axes[row_index, 1].imshow(example["source_labels"], cmap=LABEL_CMAP, vmin=0, vmax=6)
    source_edges = semantic_boundary_mask(
        example["source_labels"],
        example["source_valid"],
        num_classes=PREPROCESSING.ignore_label,
    )
    axes[row_index, 1].imshow(
        np.ma.masked_where(~source_edges, source_edges),
        cmap=EDGE_CMAP,
        alpha=0.9,
    )
    axes[row_index, 1].set_title("source labels + semantic edges")
    axes[row_index, 2].imshow(
        _display_image(example["augmented_channels"], example["limits"])
    )
    axes[row_index, 2].set_title(f"augmented IRRG (seed {example['seed']})")
    axes[row_index, 3].imshow(
        example["augmented_labels"], cmap=LABEL_CMAP, vmin=0, vmax=6
    )
    axes[row_index, 3].imshow(
        np.ma.masked_where(
            ~example["geometry_boundaries"], example["geometry_boundaries"]
        ),
        cmap=EDGE_CMAP,
        alpha=0.9,
    )
    axes[row_index, 3].imshow(
        example["padding_mask"].astype(np.uint8),
        cmap=PADDING_CMAP,
        vmin=0,
        vmax=1,
    )
    axes[row_index, 3].set_title("augmented labels + edges + padding")
    focus_text = example["focus_categories"].replace(";", ", ")
    axes[row_index, 0].set_ylabel(
        f"{example['dataset']} {example['scene_id']}\n"
        f"({example['row']}, {example['col']})\n{focus_text}",
        fontsize=8,
    )
    for axis in axes[row_index]:
        axis.set_axis_off()
fig.suptitle("Cyan: semantic class edges. Magenta: affine-generated padding.")
fig.savefig(ARTIFACT_ROOT / "augmentation-rare-boundary-examples.png", dpi=160)
plt.show()

# %% [markdown]
# ## Save tables, figures, and provenance
#
# These artifacts are outside the Git checkout. Provenance records DVC, catalog,
# split, protocol, quality-report, analysis-source, and auxiliary-file hashes, plus
# runtime versions and full-scene scan policies.

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
irrg_full_histograms.to_csv(ARTIFACT_ROOT / "irrg-full-histograms.csv", index=False)
auxiliary_metadata.to_csv(ARTIFACT_ROOT / "auxiliary-asset-metadata.csv", index=False)
auxiliary_alignment.to_csv(
    ARTIFACT_ROOT / "auxiliary-alignment-checks.csv", index=False
)
auxiliary_statistics.to_csv(
    ARTIFACT_ROOT / "auxiliary-per-scene-statistics.csv", index=False
)
auxiliary_histograms.to_csv(
    ARTIFACT_ROOT / "auxiliary-uint8-histograms.csv", index=False
)
augmentation_audit.to_csv(ARTIFACT_ROOT / "augmentation-audit.csv", index=False)
augmentation_effect_summary.to_csv(
    ARTIFACT_ROOT / "augmentation-summary.csv", index=False
)
augmentation_example_manifest.to_csv(
    ARTIFACT_ROOT / "augmentation-examples-manifest.csv", index=False
)
window_scan.to_csv(ARTIFACT_ROOT / "augmentation-training-window-scan.csv", index=False)
augmentation_focus_summary.to_csv(
    ARTIFACT_ROOT / "augmentation-focus-summary.csv", index=False
)
rare_boundary_manifest.to_csv(
    ARTIFACT_ROOT / "augmentation-rare-boundary-manifest.csv", index=False
)
rare_boundary_audit.to_csv(
    ARTIFACT_ROOT / "augmentation-rare-boundary-audit.csv", index=False
)
rare_boundary_augmentation_summary.to_csv(
    ARTIFACT_ROOT / "augmentation-rare-boundary-summary.csv", index=False
)
padding_audit.to_csv(ARTIFACT_ROOT / "augmentation-padding-audit.csv", index=False)
padding_summary.to_csv(ARTIFACT_ROOT / "augmentation-padding-summary.csv", index=False)


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
    "schema_version": 3,
    "created_at_utc": datetime.now(UTC).isoformat(),
    "git_revision": git_revision,
    "analysis_source_sha256": _sha256(Path(__file__).resolve())
    if "__file__" in globals()
    else _sha256(REPOSITORY_ROOT / "notebooks_src/data_eda_augmentation.py"),
    "dvc_lock_sha256": _sha256(dvc_lock_path) if dvc_lock_path.is_file() else None,
    "evaluation_protocol": {
        "protocol_id": EVALUATION_PROTOCOL["protocol_id"],
        "version": EVALUATION_PROTOCOL["version"],
        "status": EVALUATION_PROTOCOL["status"],
        "file_sha256": _sha256(EVALUATION_PROTOCOL_PATH),
    },
    "training_config": {
        "path": str(TRAINING_CONFIG_PATH.relative_to(REPOSITORY_ROOT)),
        "file_sha256": TRAINING_CONFIG_SHA256,
    },
    "analysis_design": {
        "label_counts": "full-scene counts from DVC quality reports",
        "irrg_profile": {
            "coverage": "all pixels, read in native raster blocks",
            "roles": ["train", "validation"],
            "quantiles": "exact inverse empirical CDF from uint8 histograms",
            "test_scenes_read": False,
        },
        "auxiliary_asset_audit": {
            "dataset": "Potsdam",
            "assets": list(AUXILIARY_ASSETS),
            "roles": ["train", "validation"],
            "coverage": "all available pixels, read in native raster blocks",
            "float_quantiles": "deterministic block-stratified sample",
            "uint8_quantiles": "exact inverse empirical CDF from histograms",
            "source_root_environment_variable": "DEEPNET_AUX_DATA_ROOT",
            "test_scenes_read": False,
            "files_present_in_dvc_input_snapshot": int(
                auxiliary_metadata.get(
                    "dvc_input_snapshot_contains_file", pd.Series(dtype=bool)
                )
                .fillna(False)
                .sum()
            ),
        },
        "augmentation_audit": {
            "roles": ["train"],
            "windows_per_scene": 1,
            "seeds_per_window": AUDIT_REPEATS_PER_SCENE,
            "geometry_comparison": (
                "same seeded geometry with and without optical transforms"
            ),
        },
        "rare_boundary_padding_audit": {
            "roles": ["train"],
            "test_scenes_read": False,
            "candidate_windows": int(len(window_scan)),
            "candidate_stride_fraction_of_window": 1 / AUDIT_GRID_STRIDE_DIVISOR,
            "crop_selection": (
                "highest-density and 10th-percentile positive crop per rare class, "
                "plus a distinct boundary-rich crop"
            ),
            "rare_class_rule": {
                "global_train_share_at_most": RARE_CLASS_SHARE_THRESHOLD,
                "or_train_scene_presence_at_most": (
                    RARE_CLASS_SCENE_PRESENCE_THRESHOLD
                ),
                "class_ids_by_dataset": {
                    name: list(class_ids) for name, class_ids in RARE_CLASS_IDS.items()
                },
            },
            "boundary_definition": (
                "both sides of valid 4-neighbor transitions between semantic classes"
            ),
            "same_class_instance_edges_available": False,
            "selected_crop_seed_trials": RARE_BOUNDARY_SEEDS_PER_CROP,
            "padding_sources_reported_separately": [
                "affine-generated image borders",
                "source invalid mask",
                "label ignore pixels",
                "model patch-stride padding",
            ],
            "model_patch_size": MODEL_PATCH_SIZE,
        },
    },
    "runtime": {
        "python_executable": sys.executable,
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
