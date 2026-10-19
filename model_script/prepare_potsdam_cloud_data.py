"""Analyze training inputs and cache aligned Potsdam class-ID labels."""

import argparse
import json
import os
import random
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from skimage import io

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semantic_labeling.dataset import ISPRS_dataset
from semantic_labeling.labels import (
    convert_from_color,
    convert_from_color_near_palette,
    convert_to_color,
    load_label,
)
from semantic_labeling.preprocessing import assemble_input, normalize_input

VALID_IDS = ("3_12", "4_10", "4_11", "4_12", "5_11", "6_12")
CHANNEL_NAMES = ("IR", "R", "G", "DSM", "nDSM")


def scene_ids(directory, suffix):
    prefix = "top_potsdam_"
    return {
        path.name[len(prefix) : -len(suffix)]
        for path in directory.glob(f"{prefix}*{suffix}")
    }


def dsm_scene_id(scene_id):
    first, second = scene_id.split("_")
    return f"{first}_{int(second):02d}"


def input_paths(data_root):
    potsdam_root = data_root / "potsdamvaihingen"
    return {
        "top": potsdam_root
        / "3_ortho_irrg/3_Ortho_IRRG/top_potsdam_{}_IRRG.tif",
        "dsm": potsdam_root / "1_dsm/1_DSM/dsm_potsdam_0{}.tif",
        "ndsm": potsdam_root
        / "1_dsm_normalisation/1_DSM_normalisation/"
        "dsm_potsdam_0{}_normalized_lastools.jpg",
        "participant": potsdam_root
        / "5_labels_for_participants/5_Labels_for_participants/"
        "top_potsdam_{}_label.tif",
        "no_boundary": data_root
        / "label-all-potsdam-vaihingen/5_labels_all_noboundary/"
        "top_potsdam_{}_label_noBoundary.tif",
    }


def ensure_single_channel(array, name):
    if array.ndim == 2:
        return array[..., np.newaxis]
    if array.ndim == 3 and array.shape[-1] == 1:
        return array
    raise ValueError(f"{name} must have one channel, got {array.shape}")


def describe_channels(samples):
    result = {}
    for index, name in enumerate(CHANNEL_NAMES):
        values = np.concatenate(samples[index]).astype(np.float64, copy=False)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f"No finite samples found for {name}")
        result[name] = {
            "sample_count": int(finite.size),
            "nonfinite_count": int(values.size - finite.size),
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "percentiles": {
                str(percentile): float(np.percentile(finite, percentile))
                for percentile in (1, 25, 50, 75, 99)
            },
        }
    return result


def cache_label(source_path, destination_path, *, allow_near_palette=False, include_counts=False):
    source_image = io.imread(source_path)
    try:
        labels = convert_from_color(source_image)
        decode_audit = {
            "method": "exact_rgb_palette",
            "maximum_channel_error": 0,
            "remapped_pixels": 0,
            "max_channel_error_limit": 0,
        }
    except ValueError:
        if not allow_near_palette:
            raise
        labels, decode_audit = convert_from_color_near_palette(
            source_image, max_channel_error=64
        )
        decode_audit["fallback_reason"] = (
            "source colors do not exactly match the canonical RGB palette"
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_suffix(".npz.tmp")
    with temporary_path.open("wb") as output_file:
        np.savez_compressed(output_file, labels=labels)
    os.replace(temporary_path, destination_path)
    record = {
        "scene_id": destination_path.stem,
        "shape": list(labels.shape),
        "compressed_bytes": destination_path.stat().st_size,
        "decode_audit": decode_audit,
    }
    if include_counts:
        record["pixel_counts"] = np.bincount(labels.ravel(), minlength=7).tolist()
    return record


def find_augmentation_seeds():
    seeds = {}
    for seed in range(10_000):
        random.seed(seed)
        mode = (random.random() < 0.5, random.random() < 0.5)
        seeds.setdefault(mode, seed)
        if len(seeds) == 4:
            return seeds
    raise RuntimeError("Could not find a seed for every flip and mirror combination")


def verify_augmentations(seeds):
    coordinates = np.arange(64, dtype=np.uint8).reshape(8, 8)
    channels = np.stack((coordinates, coordinates + 1, coordinates + 2))
    for (vertical_flip, horizontal_mirror), seed in seeds.items():
        random.seed(seed)
        image_result, label_result = ISPRS_dataset.data_augmentation(
            channels, coordinates
        )
        expected_image = channels
        expected_labels = coordinates
        if vertical_flip:
            expected_image = expected_image[:, ::-1, :]
            expected_labels = expected_labels[::-1, :]
        if horizontal_mirror:
            expected_image = expected_image[:, :, ::-1]
            expected_labels = expected_labels[:, ::-1]
        if not np.array_equal(image_result, expected_image):
            raise AssertionError("Image transform does not match requested augmentation")
        if not np.array_equal(label_result, expected_labels):
            raise AssertionError("Label transform is not aligned with the image")
    return {
        "vertical_flip_probability": 0.5,
        "horizontal_mirror_probability": 0.5,
        "possible_geometric_transforms": 4,
        "synchronization_check": "passed",
    }


def save_augmentation_preview(paths, train_id, labels, seed_map, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene_dsm_id = dsm_scene_id(train_id)
    y = x = 2872
    size = 256
    top = io.imread(paths["top"].as_posix().format(train_id))
    dsm = io.imread(paths["dsm"].as_posix().format(scene_dsm_id))
    ndsm = io.imread(paths["ndsm"].as_posix().format(scene_dsm_id))
    top_patch = top[y : y + size, x : x + size, :]
    dsm_patch = (
        dsm[y : y + size, x : x + size]
        if dsm.ndim == 2
        else dsm[y : y + size, x : x + size, :]
    )
    ndsm_patch = (
        ndsm[y : y + size, x : x + size]
        if ndsm.ndim == 2
        else ndsm[y : y + size, x : x + size, :]
    )
    image_patch = normalize_input(
        assemble_input(top_patch, dsm_patch, ndsm_patch)
    ).transpose(2, 0, 1)
    label_patch = labels[y : y + size, x : x + size]

    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for column, ((vertical_flip, horizontal_mirror), seed) in enumerate(
        sorted(seed_map.items())
    ):
        random.seed(seed)
        transformed, transformed_labels = ISPRS_dataset.data_augmentation(
            image_patch, label_patch
        )
        rgb = transformed[[1, 2, 0]].transpose(1, 2, 0)
        axes[0, column].imshow(np.clip(rgb, 0, 1))
        axes[1, column].imshow(convert_to_color(transformed_labels))
        mode = "original"
        if vertical_flip:
            mode = "vertical flip"
        if horizontal_mirror:
            mode = "horizontal mirror" if mode == "original" else "flip + mirror"
        axes[0, column].set_title(mode)
        axes[1, column].set_title("aligned class map")
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    figure.savefig(output_path, dpi=120)
    plt.close(figure)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=16)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.sample_stride <= 0:
        parser.error("--sample-stride must be positive")
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    if output_dir.with_name(output_dir.name + ".tmp").exists():
        raise FileExistsError(
            f"A previous temporary output exists: {output_dir.name}.tmp"
        )

    paths = input_paths(data_root)
    participant_ids = scene_ids(paths["participant"].parent, "_label.tif")
    all_ids = scene_ids(paths["no_boundary"].parent, "_label_noBoundary.tif")
    train_ids = sorted(participant_ids - set(VALID_IDS))
    test_ids = sorted(all_ids - participant_ids)
    if (len(participant_ids), len(all_ids), len(train_ids), len(test_ids)) != (
        24,
        38,
        18,
        14,
    ):
        raise ValueError(
            "Unexpected Potsdam split counts: "
            f"participants={len(participant_ids)}, all={len(all_ids)}, "
            f"train={len(train_ids)}, test={len(test_ids)}"
        )
    if not set(VALID_IDS) <= participant_ids:
        raise ValueError("The expected six validation scenes are not all present")

    staging_dir = output_dir.with_name(output_dir.name + ".tmp")
    staging_dir.mkdir(parents=True)
    try:
        channel_samples = [[] for _ in CHANNEL_NAMES]
        input_tiles = []
        for scene_id in train_ids:
            dsm_id = dsm_scene_id(scene_id)
            sources = (
                paths["top"].as_posix().format(scene_id),
                paths["dsm"].as_posix().format(dsm_id),
                paths["ndsm"].as_posix().format(dsm_id),
            )
            top, dsm, ndsm = (io.imread(source) for source in sources)
            dsm = ensure_single_channel(dsm, "DSM")
            ndsm = ensure_single_channel(ndsm, "nDSM")
            if top.ndim != 3 or top.shape[-1] != 3:
                raise ValueError(f"Expected three TOP channels for {scene_id}, got {top.shape}")
            if top.shape[:2] != dsm.shape[:2] or top.shape[:2] != ndsm.shape[:2]:
                raise ValueError(
                    f"Misaligned inputs for {scene_id}: "
                    f"TOP={top.shape[:2]}, DSM={dsm.shape[:2]}, nDSM={ndsm.shape[:2]}"
                )
            sampled = assemble_input(
                top[:: args.sample_stride, :: args.sample_stride],
                dsm[:: args.sample_stride, :: args.sample_stride],
                ndsm[:: args.sample_stride, :: args.sample_stride],
            ).astype(np.float32, copy=False)
            sampled /= 255.0
            for index in range(len(CHANNEL_NAMES)):
                channel_samples[index].append(sampled[..., index].reshape(-1).copy())
            input_tiles.append(
                {
                    "scene_id": scene_id,
                    "shape": list(top.shape[:2]),
                    "top_dtype": str(top.dtype),
                    "dsm_dtype": str(dsm.dtype),
                    "ndsm_dtype": str(ndsm.dtype),
                }
            )

        counts_by_class = np.zeros(7, dtype=np.int64)
        participant_cache = []
        for scene_id in train_ids:
            source = Path(paths["participant"].as_posix().format(scene_id))
            destination = staging_dir / "labels/participant" / f"{scene_id}.npz"
            label_record = cache_label(
                source, destination, include_counts=True
            )
            counts_by_class += np.asarray(label_record["pixel_counts"], dtype=np.int64)
            participant_cache.append(label_record)

        evaluation_ids = sorted(set(VALID_IDS) | set(test_ids))
        evaluation_cache = []
        for scene_id in evaluation_ids:
            source = Path(paths["no_boundary"].as_posix().format(scene_id))
            destination = staging_dir / "labels/no_boundary" / f"{scene_id}.npz"
            evaluation_cache.append(
                cache_label(
                    source,
                    destination,
                    allow_near_palette=(scene_id == "4_12"),
                )
            )

        seed_map = find_augmentation_seeds()
        augmentation = verify_augmentations(seed_map)
        preview_path = staging_dir / "augmentation_preview.png"
        preview_labels = load_label(
            staging_dir / "labels/participant" / f"{train_ids[0]}.npz"
        )
        save_augmentation_preview(
            paths, train_ids[0], preview_labels, seed_map, preview_path
        )

        class_names = (
            "impervious surfaces",
            "buildings",
            "low vegetation",
            "trees",
            "cars",
            "clutter/background",
            "ignored boundary",
        )
        total_train_pixels = int(counts_by_class.sum())
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(UTC).isoformat(),
            "data_root": str(data_root),
            "sample_stride": args.sample_stride,
            "normalization": "all five channels divided by 255.0; preserved from project",
            "input_channels": CHANNEL_NAMES,
            "input_statistics": describe_channels(channel_samples),
            "input_tiles_checked": input_tiles,
            "split": {
                "train_ids": train_ids,
                "validation_ids": list(VALID_IDS),
                "held_out_test_ids": test_ids,
            },
            "train_label_pixel_counts": {
                name: int(counts_by_class[index])
                for index, name in enumerate(class_names)
            },
            "train_label_pixel_fractions": {
                name: float(counts_by_class[index] / total_train_pixels)
                for index, name in enumerate(class_names)
            },
            "prepared_labels": {
                "format": "compressed npz with uint8 labels array",
                "participant_train": participant_cache,
                "no_boundary_validation_and_test": evaluation_cache,
            },
            "augmentation": augmentation,
            "augmentation_preview": "augmentation_preview.png",
        }
        (staging_dir / "preprocessing_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging_dir, output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    print(json.dumps({
        "output_dir": str(output_dir),
        "train_scenes": len(train_ids),
        "validation_scenes": len(VALID_IDS),
        "held_out_test_scenes": len(test_ids),
        "label_cache_bytes": sum(
            record["compressed_bytes"]
            for record in participant_cache + evaluation_cache
        ),
        "augmentation": augmentation,
        "manifest": str(output_dir / "preprocessing_manifest.json"),
        "preview": str(output_dir / "augmentation_preview.png"),
    }, indent=2))


if __name__ == "__main__":
    main()
