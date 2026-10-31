"""Replay selected train-only crops before promoting an augmentation policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from deepnet.data.augmentations import AlbumentationsAugmenter
from deepnet.data.bundles import load_bundle
from deepnet.data.catalog import load_augmentation_spec, load_domain_spec
from deepnet.data.contracts import WindowSpec
from deepnet.data.raster import RasterioSceneReader
from deepnet.data.sampling import sample_rng


def audit(config: dict, bundle_root: Path, examples: Path) -> dict:
    domain = load_domain_spec(config["domain_id"])
    policy = load_augmentation_spec(config["augmentation_id"])
    with examples.open(newline="") as source:
        selected = list(csv.DictReader(source))
    records = []
    provenance = {}
    for dataset in config["datasets"]:
        name = dataset["dataset_id"].removeprefix("isprs-")
        bundle = load_bundle(bundle_root / dataset["bundle_path"])
        provenance[name] = bundle.manifest["bundle_sha256"]
        access = bundle.scene_access("train")
        augmenters = [
            AlbumentationsAugmenter(
                domain,
                bundle.model_input,
                spec,
                ignore_label=bundle.preprocessing.ignore_label,
            )
            for spec in (replace(policy, optical=()), policy)
        ]
        crops = [row for row in selected if row["dataset"].lower() == name]
        if not crops:
            raise ValueError(f"No selected training crops for {name}")
        for index, crop in enumerate(crops):
            scene_id = crop["scene_id"]
            if scene_id not in bundle.split.train_scene_ids:
                raise ValueError("Augmentation replay includes a non-training scene")
            height, width = dataset["train_window_size"]
            window = WindowSpec(int(crop["row"]), int(crop["col"]), height, width)
            labels, valid = access.read_labels(scene_id, window)
            channels = {}
            with RasterioSceneReader(access.scenes[scene_id], bundle.root) as reader:
                for channel, asset, band in bundle.model_input.channel_sources:
                    image, mask = reader.read_window_with_mask(asset, window)
                    band_index = (
                        access.scenes[scene_id].asset_map()[asset].bands.index(band)
                    )
                    channels[channel] = image[..., band_index]
                    valid &= mask & np.isfinite(channels[channel])
            for repeat in range(8):
                seed = sample_rng(config["seed"], index, repeat).getrandbits(32)
                geometric = augmenters[0](channels, labels, valid, seed=seed)
                augmented = augmenters[1](channels, labels, valid, seed=seed)
                np.testing.assert_array_equal(geometric[1], augmented[1])
                np.testing.assert_array_equal(geometric[2], augmented[2])
                invalid = ~augmented[2]
                for channel, image in augmented[0].items():
                    if not np.isfinite(image).all():
                        raise ValueError("Non-finite augmented image values")
                    np.testing.assert_array_equal(
                        image[invalid], geometric[0][channel][invalid]
                    )
                assert (
                    augmented[1][invalid] == bundle.preprocessing.ignore_label
                ).all()
                records.append(
                    {
                        "dataset": name,
                        "scene_id": scene_id,
                        "row": window.row,
                        "col": window.col,
                        "seed": seed,
                        "invalid_pixels": int(invalid.sum()),
                        "invalid_fill_changed": False,
                        "rare_class_counts": {
                            str(class_id): {
                                "before": int(((labels == class_id) & valid).sum()),
                                "after": int(
                                    ((augmented[1] == class_id) & augmented[2]).sum()
                                ),
                            }
                            for class_id in (4, 5)
                        },
                    }
                )
        access.close()
    return {
        "augmentation_id": policy.augmentation_id,
        "augmentation_hash": policy.content_hash,
        "examples_sha256": hashlib.sha256(examples.read_bytes()).hexdigest(),
        "bundle_sha256": provenance,
        "trial_count": len(records),
        "trials_with_invalid_pixels": sum(row["invalid_pixels"] > 0 for row in records),
        "rare_class_disappearance_count": sum(
            item["before"] > 0 and item["after"] == 0
            for row in records
            for item in row["rare_class_counts"].values()
        ),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.config.read_text()), args.bundle_root, args.examples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "records"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
