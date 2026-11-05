"""Pinned offline DINOv3 loading and real-patch backend checks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from dataclasses import replace
from pathlib import Path

import torch

from deepnet.data.contracts import WindowSpec, canonical_hash
from deepnet.data.raster import sha256_file
from deepnet.models.segmenter import MultimodalSegmenter
from deepnet.models.specs import load_model_spec
from deepnet.training_utils import verify_pretrained, write_json

from .data import CacheBundle, model_view

MODEL_SPECS = {
    "B": "dinov3-vitb16-rgbir-v2",
    "L": "dinov3-vitl16-rgbir-sat-v2",
}
WEIGHT_HASHES = {
    "B": "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b",
    "L": "4e6356d992c1301b5e7c275f465e47296c5c4ad17052e262b29fc21e82ccc698",
}
CONFIG_HASHES = {
    "B": "3c9cc418f4622fd6d5587fd142b6f3cba0ba6a69f67ced907d8b7f26118451ec",
    "L": "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5",
}


def prepare_weights(path: Path, size: str, *, download: bool = False) -> dict:
    """Write a manifest only after pinned bytes have passed source checks."""
    spec = load_model_spec(MODEL_SPECS[size]).backbone
    if download:
        from huggingface_hub import HfApi, hf_hub_download

        info = HfApi().model_info(
            spec.model_id, revision=spec.revision, files_metadata=True
        )
        if info.sha != spec.revision:
            raise ValueError("Pretrained revision differs from the pinned commit")
        path.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "model.safetensors", "LICENSE.md"):
            hf_hub_download(spec.model_id, name, revision=spec.revision, local_dir=path)
        config_record = next(f for f in info.siblings if f.rfilename == "config.json")
        config_bytes = (path / "config.json").read_bytes()
        git_blob = hashlib.sha1(
            f"blob {len(config_bytes)}\0".encode() + config_bytes
        ).hexdigest()
        if git_blob != config_record.blob_id:
            raise ValueError("Downloaded config differs from the pinned Git blob")
    if sha256_file(path / "model.safetensors") != WEIGHT_HASHES[size]:
        raise ValueError("Pretrained weight checksum differs from the pinned snapshot")
    if sha256_file(path / "config.json") != CONFIG_HASHES[size]:
        raise ValueError("Pretrained config checksum differs")
    config = json.loads((path / "config.json").read_text())
    if (
        config["hidden_size"] != spec.feature_channels
        or config["num_hidden_layers"] != max(spec.feature_blocks)
        or config["patch_size"] != 16
        or config["num_register_tokens"] != 4
    ):
        raise ValueError("Pretrained architecture does not match the model spec")
    files = [
        {
            "path": name,
            "bytes": (path / name).stat().st_size,
            "sha256": sha256_file(path / name),
        }
        for name in ("config.json", "model.safetensors", "LICENSE.md")
    ]
    manifest = {"model_id": spec.model_id, "revision": spec.revision, "files": files}
    target = path / "pretrained_manifest.json"
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError("Existing pretrained manifest differs")
    write_json(target, manifest)
    return manifest


def load_model(
    pretrained: Path, size: str, auxiliary: bool, device, *, checkpointing=False
):
    spec = load_model_spec(MODEL_SPECS[size])
    spec = replace(
        spec,
        input_adapter=replace(
            spec.input_adapter,
            spectral_index=auxiliary,
            spectral_index_bands=("ir", "red") if auxiliary else (),
        ),
    )
    manifest = verify_pretrained(
        pretrained, spec.backbone.model_id, spec.backbone.revision
    )
    files = {r["path"]: r["sha256"] for r in manifest["files"]}
    if (
        not {"config.json", "model.safetensors"} <= files.keys()
        or files["model.safetensors"] != WEIGHT_HASHES[size]
        or files["config.json"] != CONFIG_HASHES[size]
    ):
        raise ValueError("Pretrained manifest must cover the pinned model bytes")
    domain, view = model_view(auxiliary)
    model = MultimodalSegmenter.from_pretrained(
        domain, view, spec, pretrained_path=pretrained
    ).to(device)
    if checkpointing:
        model.backbone.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model, {
        "model_spec_hash": spec.content_hash,
        "pretrained_hash": canonical_hash(manifest),
    }


def real_smoke(model, sample) -> dict:
    """Require real pretrained normalization, prefix removal, and LoRA gradients."""
    device = next(model.parameters()).device
    image = torch.from_numpy(sample.image[None]).to(device)
    labels = torch.from_numpy(sample.labels[None]).to(device)
    base = model.backbone.model.get_base_model()
    if base.config.model_type != "dinov3_vit" or base.config.num_register_tokens != 4:
        raise ValueError("Smoke must load real pretrained DINOv3 with register tokens")
    model.eval()
    with torch.no_grad():
        encoder, _ = model.input_adapter(image)
        states = base(pixel_values=encoder, output_hidden_states=True).hidden_states
        features = model.backbone.forward_features(encoder)
        h, w = encoder.shape[-2] // 16, encoder.shape[-1] // 16
        for block, feature in zip(model.backbone.feature_blocks, features, strict=True):
            tokens = states[block]
            if tokens.shape[1] != 5 + h * w:
                raise ValueError("Unexpected CLS/register/patch token layout")
            expected = base.norm(tokens[:, 5:]).reshape(1, h, w, -1).permute(0, 3, 1, 2)
            torch.testing.assert_close(feature, expected)
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(image)
    if logits.shape != (1, 6, *sample.labels.shape) or not logits.isfinite().all():
        raise ValueError("Unexpected real DINOv3 segmentation logits")
    loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=6)
    loss.backward()
    parameters = list(model.backbone.named_parameters())
    lora = [p for n, p in parameters if ".lora_B." in n]
    expected_modules = 4 * base.config.num_hidden_layers
    if (
        len(lora) != expected_modules
        or any(
            p.grad is None
            or not p.grad.isfinite().all()
            or not torch.count_nonzero(p.grad)
            for p in lora
        )
        or any(
            p.requires_grad or p.grad is not None
            for n, p in parameters
            if ".lora_" not in n
        )
    ):
        raise ValueError("Pretrained backbone freeze/LoRA gradient contract failed")
    model.zero_grad(set_to_none=True)
    return {
        "lora_b_gradients": len(lora),
        "feature_normalization": "pretrained.norm",
        "register_tokens": 4,
        "logit_shape": list(logits.shape),
        "loss": float(loss.detach()),
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    weights = commands.add_parser("weights")
    weights.add_argument("--download", action="store_true")
    smoke = commands.add_parser("smoke")
    for command in (weights, smoke):
        command.add_argument("--size", choices=MODEL_SPECS, default="B")
        command.add_argument("--pretrained", type=Path, required=True)
    smoke.add_argument("--cache", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "weights":
        prepare_weights(args.pretrained, args.size, download=args.download)
    else:
        bundle = CacheBundle.load(args.cache)
        if "train" not in bundle.manifest["included_splits"]:
            raise ValueError("Smoke requires a training cache")
        with bundle.access("train", auxiliary=True) as access:
            scene = bundle.split.train_scene_ids[0]
            sample = access.read_sample(scene, WindowSpec(1024, 1024, 224, 224))
        model, binding = load_model(args.pretrained, args.size, True, args.device)
        write_json(args.output, {**binding, **real_smoke(model, sample)})


if __name__ == "__main__":
    main()
