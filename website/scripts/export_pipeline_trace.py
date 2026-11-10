"""Export one verified refit training draw and a bounded real model trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deepnet.data.contracts import WindowSpec, canonical_hash  # noqa: E402
from deepnet.data.raster import sha256_file  # noqa: E402
from deepnet.potsdam_rgbir.augmentation import augment  # noqa: E402
from deepnet.potsdam_rgbir.data import CacheBundle  # noqa: E402
from deepnet.potsdam_rgbir.model import load_model  # noqa: E402
from deepnet.potsdam_rgbir.phases import check_refit_cache  # noqa: E402
from deepnet.potsdam_rgbir.sampling import draw_seed, schedule, training_index  # noqa: E402
from deepnet.training_utils import load_checkpoint  # noqa: E402

DATASET_URL = (
    "https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/"
    "2d-sem-label-potsdam.aspx"
)
CLASS_COLORS = ("#9facad", "#4e8ab9", "#8cb587", "#316953", "#e6af66", "#ba726e")
IGNORE_COLOR = "#d79b62"
ASSET_SPECS = {
    "source_rgb": (
        "source-rgb.png",
        "Optical source RGB for the selected training window.",
    ),
    "reference": (
        "reference-labels.png",
        "Participant labels for the selected training window.",
    ),
    "validity": ("source-validity.png", "Source validity mask; white is valid."),
    "sampling_context": (
        "sampling-context.png",
        "RGB context with the actual 512-pixel training window outlined.",
    ),
    "augmented_rgb": (
        "augmented-rgb.png",
        "Optical image after the exact seeded training augmentation.",
    ),
    "augmented_reference": (
        "augmented-labels.png",
        "Categorical labels after the exact seeded training augmentation.",
    ),
    "augmented_validity": (
        "augmented-validity.png",
        "Validity mask after the exact seeded training augmentation.",
    ),
    "encoder_input": (
        "encoder-input.png",
        "Learned adapter output decoded for display after normalization.",
    ),
    "prediction": (
        "training-prediction.png",
        "Argmax classes from one forward pass of the bound refit checkpoint.",
    ),
    "errors": (
        "training-sample-errors.png",
        "Mismatched scored pixels in this refit training example.",
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def hex_rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value[index : index + 2], 16) for index in (1, 3, 5))


def save_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def rgb_image(chw: np.ndarray) -> Image.Image:
    if chw.ndim != 3 or chw.shape[0] != 3:
        raise ValueError(f"Expected three RGB channels, got {chw.shape}")
    pixels = np.rint(np.clip(chw.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def class_image(labels: np.ndarray, valid: np.ndarray) -> Image.Image:
    pixels = np.empty((*labels.shape, 3), dtype=np.uint8)
    pixels[:] = hex_rgb(IGNORE_COLOR)
    for class_id, color in enumerate(CLASS_COLORS):
        pixels[(labels == class_id) & valid] = hex_rgb(color)
    return Image.fromarray(pixels, mode="RGB")


def validity_image(valid: np.ndarray) -> Image.Image:
    pixels = np.where(valid, 255, 22).astype(np.uint8)
    return Image.fromarray(pixels, mode="L")


def activation_rgb(
    feature: torch.Tensor,
) -> tuple[Image.Image, list[float], list[list[float]]]:
    """Project real patch features to RGB with exact spatial PCA."""
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError(f"Expected one NCHW feature map, got {tuple(feature.shape)}")
    height, width = feature.shape[-2:]
    values = feature[0].detach().to(device="cpu", dtype=torch.float32).numpy()
    samples = values.transpose(1, 2, 0).reshape(height * width, -1).astype(np.float64)
    centered = samples - samples.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:3]
    components = eigenvectors[:, order].T
    scores = centered @ components.T
    for component in range(3):
        pivot = int(np.argmax(np.abs(components[component])))
        if components[component, pivot] < 0:
            scores[:, component] *= -1
    limits = np.percentile(scores, (2, 98), axis=0)
    scaled = np.zeros_like(scores, dtype=np.uint8)
    for component, (low, high) in enumerate(limits.T):
        if high > low:
            scaled[:, component] = np.rint(
                np.clip((scores[:, component] - low) / (high - low), 0, 1) * 255
            ).astype(np.uint8)
    native = Image.fromarray(scaled.reshape(height, width, 3), mode="RGB")
    image = native.resize((512, 512), resample=Image.Resampling.NEAREST)
    total = float(np.maximum(eigenvalues[order], 0).sum())
    variance = [
        float(max(eigenvalues[index], 0) / total) if total else 0.0 for index in order
    ]
    percentile_limits = [[float(low), float(high)] for low, high in limits.T]
    return image, variance, percentile_limits


def asset_record(
    output: Path,
    key: str,
    image: Image.Image,
    note: str,
    *,
    filename: str | None = None,
    alt: str | None = None,
) -> dict[str, Any]:
    default_filename, default_alt = ASSET_SPECS[key]
    filename = filename or default_filename
    alt = alt or default_alt
    path = output / filename
    save_image(path, image)
    return {
        "file": filename,
        "mime_type": "image/png",
        "width_px": image.width,
        "height_px": image.height,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "alt": alt,
        "visualization": note,
    }


def sample_confusion(
    labels: np.ndarray, prediction: np.ndarray, valid: np.ndarray
) -> dict[str, Any]:
    scored = valid & (labels >= 0) & (labels < 6)
    indices = labels[scored] * 6 + prediction[scored]
    matrix = np.bincount(indices, minlength=36).reshape(6, 6)
    errors = scored & (labels != prediction)
    pairs = [
        {
            "reference_class_id": truth,
            "predicted_class_id": predicted,
            "pixels": int(matrix[truth, predicted]),
        }
        for truth in range(6)
        for predicted in range(6)
        if truth != predicted and matrix[truth, predicted]
    ]
    pairs.sort(
        key=lambda row: (
            -row["pixels"],
            row["reference_class_id"],
            row["predicted_class_id"],
        )
    )
    return {
        "scope": "single_refit_training_draw_only",
        "scored_pixels": int(scored.sum()),
        "error_pixels": int(errors.sum()),
        "confusion_matrix_reference_rows_prediction_columns": matrix.tolist(),
        "largest_error_pairs": pairs[:8],
        "ignored_or_invalid_pixels_excluded": int(labels.size - scored.sum()),
    }


def export(args: argparse.Namespace) -> dict[str, Any]:
    study_id = "potsdam-dinov3-rgbir-v2"
    run_root = args.runs_root.expanduser().resolve() / study_id
    study_root = run_root / "study"
    protocol = read_json(study_root / "protocol.json")
    lock = read_json(study_root / "selection-lock.json")
    refit_binding = read_json(study_root / "refit-binding.json")
    refit_manifest = read_json(study_root / "refit/manifest.json")
    checkpoint_path = study_root / "refit/final.pt"
    cache_path = (
        args.cache.expanduser().resolve() if args.cache else run_root / "refit-cache"
    )
    pretrained_path = (
        args.pretrained.expanduser().resolve()
        if args.pretrained
        else run_root / "kaggle/input-refit/pretrained-B"
    )
    output = args.output.expanduser().resolve()

    cache = CacheBundle.load(cache_path, verify=True)
    if cache.manifest["cache_hash"] != refit_manifest["cache_hash"]:
        raise ValueError("Refit cache differs from the final checkpoint manifest")
    check_refit_cache(cache, protocol)
    if (
        protocol["protocol_hash"] != refit_manifest["protocol_hash"]
        or protocol["protocol_hash"] != lock["protocol_hash"]
        or refit_binding["protocol_hash"] != protocol["protocol_hash"]
        or refit_binding["refit_cache_hash"] != cache.manifest["cache_hash"]
        or refit_binding["selection_lock_hash"] != canonical_hash(lock)
    ):
        raise ValueError("Refit protocol, cache, and selection-lock binding mismatch")
    recipe = refit_manifest["recipe"]
    if recipe != lock["recipe"] or recipe != {
        "backbone": "B",
        "auxiliary": False,
        "enhanced": False,
    }:
        raise ValueError("This exporter expects the locked RGB DINOv3-B baseline refit")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != refit_manifest["final_checkpoint_sha256"]:
        raise ValueError("Final checkpoint digest differs from its refit manifest")
    if refit_manifest["updates"] != 1152 or refit_manifest["completed_passes"] != 1:
        raise ValueError("Refit checkpoint is not the completed 1,152-update model")

    device = torch.device(args.device)
    if (
        device.type != "cpu"
        and not torch.cuda.is_available()
        and not torch.backends.mps.is_available()
    ):
        raise ValueError(f"Requested device is unavailable: {device}")
    torch.set_num_threads(args.threads)
    model, model_binding = load_model(
        pretrained_path, recipe["backbone"], recipe["auxiliary"], device
    )
    if model_binding != refit_manifest["model_binding"]:
        raise ValueError(
            "Pinned model/pretrained binding differs from the refit manifest"
        )
    binding_fields = {
        "protocol_hash": protocol["protocol_hash"],
        "cache_hash": cache.manifest["cache_hash"],
        "arm": "refit",
        "recipe": recipe,
        "seed": refit_manifest["seed"],
        "model": model_binding,
        "selection_lock": lock,
    }
    checkpoint_binding = canonical_hash(binding_fields)
    if checkpoint_binding != refit_manifest["binding"]:
        raise ValueError(
            "Refit checkpoint binding cannot be reproduced from its manifest"
        )
    saved = load_checkpoint(checkpoint_path, checkpoint_binding)
    progress = saved["progress"]
    if (
        progress["updates"] != refit_manifest["updates"]
        or progress["completed_passes"] != refit_manifest["completed_passes"]
    ):
        raise ValueError("Checkpoint progress differs from its refit manifest")
    model.load_state_dict(saved["model"], strict=True)
    model.eval()

    evidence = read_json(ROOT / "website/src/generated/evidence.json")["benchmark"]
    class_names = evidence["class_names"]
    if len(class_names) != 6:
        raise ValueError("Expected the verified six-class Potsdam ontology")
    augmentation_recipe = protocol["augmentation"]["baseline"]
    seed = int(refit_manifest["seed"])
    pass_number = 0

    with cache.access("train", auxiliary=False) as access:
        if set(access.scene_shapes) != set(refit_manifest["train_scene_ids"]):
            raise ValueError(
                "Training cache scene identities differ from the refit manifest"
            )
        index = training_index(access)
        draws = schedule(
            index,
            access,
            seed,
            pass_number,
            jitter=protocol["sampling"]["center_jitter"],
        )
        expected_draws = 4 * 1152
        if len(draws) != expected_draws:
            raise ValueError(
                f"Expected {expected_draws} scheduled draws, found {len(draws)}"
            )
        if not 0 <= args.draw_ordinal < len(draws):
            raise ValueError(f"Draw ordinal must be in [0, {len(draws) - 1}]")
        draw = draws[args.draw_ordinal]
        source = access.read_sample(draw.scene, draw.window)
        draw_rng_seed = draw_seed(seed, pass_number, args.draw_ordinal)
        augmented = augment(
            access,
            draw,
            seed=draw_rng_seed,
            recipe=dict(augmentation_recipe),
        )
        scene_height, scene_width = access.scene_shapes[draw.scene]
        context_row = max(0, min(scene_height - 1024, draw.window.row - 256))
        context_col = max(0, min(scene_width - 1024, draw.window.col - 256))
        context_window = WindowSpec(context_row, context_col, 1024, 1024, "constant")
        context = access.read_sample(draw.scene, context_window)

    image_tensor = torch.from_numpy(augmented.image[None]).to(
        device=device, dtype=torch.float32
    )
    height, width = augmented.labels.shape
    with torch.inference_mode():
        encoder_input, spatial_features = model.input_adapter(image_tensor)
        feature_maps = model.backbone.forward_features(encoder_input)
        logits = model.decoder(feature_maps, spatial_features, (height, width))
    if logits.shape != (1, 6, height, width) or not torch.isfinite(logits).all():
        raise ValueError(f"Unexpected dense logits: {tuple(logits.shape)}")
    prediction = logits[0].argmax(dim=0).to(device="cpu").numpy().astype(np.uint8)
    sample_result = sample_confusion(augmented.labels, prediction, augmented.valid_mask)
    if sample_result["scored_pixels"] != sum(
        sum(row)
        for row in sample_result["confusion_matrix_reference_rows_prediction_columns"]
    ):
        raise ValueError("Training sample confusion counts do not reconcile")

    output.mkdir(parents=True, exist_ok=True)
    assets: dict[str, dict[str, Any]] = {}
    assets["source_rgb"] = asset_record(
        output,
        "source_rgb",
        rgb_image(source.image),
        "Cached optical bands divided by 255; RGB display.",
    )
    assets["reference"] = asset_record(
        output,
        "reference",
        class_image(source.labels, source.valid_mask),
        (
            "Participant labels; IDs 0-5 use the Atlas class palette and "
            "ID 6/invalid support is amber."
        ),
    )
    assets["validity"] = asset_record(
        output,
        "validity",
        validity_image(source.valid_mask),
        "White is source-valid; black is invalid source support.",
    )

    context_image = rgb_image(context.image)
    drawing = ImageDraw.Draw(context_image)
    left = draw.window.col - context_window.col
    top = draw.window.row - context_window.row
    drawing.rectangle((left, top, left + 511, top + 511), outline="#f3bf86", width=5)
    assets["sampling_context"] = asset_record(
        output,
        "sampling_context",
        context_image,
        (
            "1024 by 1024 source context; amber outline marks the exact "
            "sampled 512 by 512 window."
        ),
    )
    assets["augmented_rgb"] = asset_record(
        output,
        "augmented_rgb",
        rgb_image(augmented.image),
        (
            "Output of deepnet.potsdam_rgbir.augmentation.augment using the "
            "recorded deterministic draw seed and baseline policy."
        ),
    )
    assets["augmented_reference"] = asset_record(
        output,
        "augmented_reference",
        class_image(augmented.labels, augmented.valid_mask),
        (
            "Categorical labels transformed by the same training operation; "
            "ignored/invalid support remains ID 6."
        ),
    )
    assets["augmented_validity"] = asset_record(
        output,
        "augmented_validity",
        validity_image(augmented.valid_mask),
        (
            "Validity mask transformed by the same training operation using "
            "nearest-neighbor semantics."
        ),
    )

    mean = model.input_adapter.backbone_mean[0, :, 0, 0].detach().cpu().numpy()
    std = model.input_adapter.backbone_std[0, :, 0, 0].detach().cpu().numpy()
    decoded_encoder = (
        encoder_input[0].detach().cpu().numpy() * std[:, None, None]
        + mean[:, None, None]
    )
    assets["encoder_input"] = asset_record(
        output,
        "encoder_input",
        rgb_image(decoded_encoder),
        (
            "Post-adapter backbone tensor mapped back through configured "
            "mean/std and clipped to [0,1] for display."
        ),
    )

    activation_records = []
    for block, feature in zip(model.backbone.feature_blocks, feature_maps, strict=True):
        if tuple(feature.shape) != (1, 768, 32, 32):
            raise ValueError(
                f"Unexpected block {block} feature shape: {tuple(feature.shape)}"
            )
        pca_image, variance, limits = activation_rgb(feature)
        key = f"feature_block_{block}"
        filename = f"feature-pca-block-{block:02}.png"
        record = asset_record(
            output,
            "source_rgb",
            pca_image,
            (
                "PCA projection of real normalized DINOv3 patch features; "
                "nearest-neighbor enlarged from the native patch grid."
            ),
            filename=filename,
            alt=(
                "PCA projection of the real normalized DINOv3 feature map "
                f"at block {block}."
            ),
        )
        record["feature_block"] = int(block)
        assets[key] = record
        activation_records.append(
            {
                "block": int(block),
                "tensor_shape": list(feature.shape),
                "feature_channels": 768,
                "patch_grid": [32, 32],
                "prefix_tokens_removed": 5,
                "normalization": (
                    "pretrained DINOv3 LayerNorm applied by TransformerFeatureBackbone"
                ),
                "projection": (
                    "PCA over 768 channels at each of 1,024 spatial positions; "
                    "components mapped to RGB and scaled independently to their "
                    "2nd-98th percentiles"
                ),
                "component_explained_variance_ratio": variance,
                "component_percentile_limits": limits,
                "asset_key": key,
            }
        )

    assets["prediction"] = asset_record(
        output,
        "prediction",
        class_image(prediction, augmented.valid_mask),
        (
            "Argmax of dense logits from the final refit checkpoint on this "
            "same training draw."
        ),
    )
    error_pixels = (
        augmented.valid_mask & (augmented.labels < 6) & (prediction != augmented.labels)
    )
    error_rgb = np.full((*prediction.shape, 3), (20, 45, 61), dtype=np.uint8)
    error_rgb[~augmented.valid_mask | (augmented.labels == 6)] = hex_rgb(IGNORE_COLOR)
    error_rgb[error_pixels] = (245, 157, 98)
    assets["errors"] = asset_record(
        output,
        "errors",
        Image.fromarray(error_rgb, mode="RGB"),
        (
            "Only misclassified scored pixels are amber; correct pixels are "
            "navy; ignored or invalid pixels are muted amber."
        ),
    )

    sample_id = (
        f"{draw.scene}-{draw.window.row}-{draw.window.col}-refit-pass0-"
        f"draw{args.draw_ordinal:04}"
    )
    target_class_id = draw.rare_class
    stages = [
        {
            "id": "source",
            "title": "Raw optical source",
            "summary": (
                "A real RGB crop read from the verified refit cache. The stored "
                "optical values are scaled from 8-bit digital numbers to [0, 1]."
            ),
            "assets": ["source_rgb"],
            "implementation": "deepnet/potsdam_rgbir/data.py",
        },
        {
            "id": "preprocessing",
            "title": "Reference and validity",
            "summary": (
                "The participant reference labels and source-validity mask stay "
                "aligned. Invalid support is set to ignore ID 6; the training "
                "loss also ignores that class ID."
            ),
            "assets": ["reference", "validity"],
            "implementation": "deepnet/potsdam_rgbir/data.py",
        },
        {
            "id": "sampling",
            "title": "Scheduled training window",
            "summary": (
                "The outlined window is the exact 512 × 512 draw at these scene "
                "pixel coordinates. It was selected by the refit coverage and "
                "class-aware schedule."
            ),
            "assets": ["sampling_context"],
            "implementation": "deepnet/potsdam_rgbir/sampling.py",
        },
        {
            "id": "augmentation",
            "title": "Seeded augmentation",
            "summary": (
                "All three aligned outputs come from the Python training "
                "augment() function with the recorded draw seed and the "
                "locked baseline recipe."
            ),
            "assets": ["augmented_rgb", "augmented_reference", "augmented_validity"],
            "implementation": "deepnet/potsdam_rgbir/augmentation.py",
        },
        {
            "id": "model-input",
            "title": "Backbone input",
            "summary": (
                "The trained residual-identity RGB adapter and ImageNet "
                "mean/std normalization produce the actual DINOv3 tensor. "
                "The PNG reverses normalization only for display and is "
                "clipped to [0, 1]."
            ),
            "assets": ["encoder_input"],
            "implementation": "deepnet/models/adapters.py",
        },
        {
            "id": "activations",
            "title": "Recorded encoder features",
            "summary": (
                "These four images are deterministic PCA renderings of feature "
                "maps captured from the real checkpoint at blocks 3, 6, 9, "
                "and 12. They are not attention maps."
            ),
            "assets": [
                "feature_block_3",
                "feature_block_6",
                "feature_block_9",
                "feature_block_12",
            ],
            "implementation": "deepnet/models/backbones.py",
        },
        {
            "id": "prediction",
            "title": "Dense prediction",
            "summary": (
                "The six-class argmax comes from one identity forward pass of "
                "the bound final refit checkpoint on the same augmented "
                "training draw."
            ),
            "assets": ["prediction"],
            "implementation": "deepnet/models/segmenter.py",
        },
        {
            "id": "evaluation",
            "title": "Training-sample errors",
            "summary": (
                "This crop-only confusion count shows how valid "
                "reference/prediction pairs are accumulated. It is an "
                "in-sample illustration, not independent benchmark evidence."
            ),
            "assets": ["errors"],
            "implementation": "deepnet/data/evaluation.py",
        },
    ]
    trace = {
        "schema_version": 1,
        "dataset_id": "isprs-potsdam",
        "dataset_source_url": DATASET_URL,
        "publication": {
            "project_owner_authorized": True,
            "rights_note": (
                "The project owner confirmed authorization to publish this "
                "curated sample and its derived views. The bundle contains "
                "rendered PNG previews, not a copy of the full source archive "
                "or cached arrays."
            ),
        },
        "sample": {
            "sample_id": sample_id,
            "scene_id": draw.scene,
            "split_role": "final_refit_training",
            "reference_role": "participant labels",
            "used_for_model_training": True,
            "window": draw.window.to_dict(),
            "training_pass_zero_based": pass_number,
            "training_draw_ordinal_zero_based": args.draw_ordinal,
            "training_schedule": (
                "coverage grid plus class-prioritized draws; deterministic shuffle"
            ),
            "rare_class_target_id": target_class_id,
            "rare_class_target_name": class_names[target_class_id]
            if target_class_id is not None
            else None,
            "draw_seed": draw_rng_seed,
            "input_channels": ["red", "green", "blue"],
            "class_names": class_names,
            "class_colors": list(CLASS_COLORS),
            "ignore_id": 6,
            "source_valid_pixels": int(source.valid_mask.sum()),
            "augmented_valid_pixels": int(augmented.valid_mask.sum()),
        },
        "training_lineage": {
            "study_id": study_id,
            "protocol_hash": protocol["protocol_hash"],
            "refit_cache_hash": cache.manifest["cache_hash"],
            "refit_training_index_hash": canonical_hash(index),
            "training_index_base_windows": len(index["base"]),
            "scheduled_draw_count": len(draws),
            "recipe": recipe,
            "augmentation_policy": augmentation_recipe,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_binding": checkpoint_binding,
            "selection_lock_hash": canonical_hash(lock),
        },
        "model": {
            "model_id": "dinov3-vitb16-rgbir-v2",
            "backbone": "DINOv3 ViT-B/16 with LoRA",
            "feature_blocks": list(model.backbone.feature_blocks),
            "feature_channels": 768,
            "patch_size": 16,
            "pretrained_revision": "5931719e67bbdb9737e363e781fb0c67687896bc",
            "model_spec_hash": model_binding["model_spec_hash"],
            "pretrained_hash": model_binding["pretrained_hash"],
            "checkpoint_sha256": checkpoint_sha256,
            "forward": (
                "single RGB crop; identity inference; no TTA; inference mode; "
                "no training"
            ),
            "encoder_normalization": {
                "adapter_projection": "trained residual-identity 1x1 RGB projection",
                "mean": [float(value) for value in mean],
                "std": [float(value) for value in std],
            },
            "activations": activation_records,
        },
        "sample_confusion": sample_result,
        "stages": stages,
        "assets": assets,
    }
    public_manifest = output / "manifest.json"
    public_manifest.write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    generated = ROOT / "website/src/generated/pipeline-trace.json"
    generated.write_text(
        json.dumps(trace, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help="Directory containing potsdam-dinov3-rgbir-v2",
    )
    parser.add_argument("--cache", type=Path, help="Override the run's refit-cache")
    parser.add_argument(
        "--pretrained", type=Path, help="Override the pinned DINOv3-B snapshot"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "website/public/evidence/pipeline-trace"
    )
    parser.add_argument(
        "--draw-ordinal",
        type=int,
        default=0,
        help="Zero-based draw from the first locked refit pass",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    result = export(args)
    sample = result["sample"]
    print(
        f"Exported {len(result['assets'])} assets for {sample['sample_id']} "
        f"({sample['scene_id']} row={sample['window']['row']} "
        f"col={sample['window']['col']})"
    )
    print(f"Checkpoint SHA-256: {result['model']['checkpoint_sha256']}")


if __name__ == "__main__":
    main()
