"""Bootstrap one read-only Kaggle kernel for paired Potsdam predictions."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

STARTED = time.monotonic()
SETTINGS = {}
PAYLOAD = ""
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_outputs(root: Path) -> None:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "output-index.json":
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    (root / "output-index.json").write_text(
        json.dumps({"files": records}, indent=2) + "\n"
    )


def _setup_source() -> Path:
    source = base64.b85decode(PAYLOAD)
    if hashlib.sha256(source).hexdigest() != SETTINGS["source_sha256"]:
        raise ValueError("Packaged inference source hash differs")
    root = Path("/tmp/potsdam-paired-source")
    with zipfile.ZipFile(io.BytesIO(source)) as archive:
        if any(
            Path(name).is_absolute() or ".." in Path(name).parts
            for name in archive.namelist()
        ):
            raise ValueError("Unsafe source archive path")
        archive.extractall(root)
    installed = {
        name: importlib.metadata.version(name) for name in ("torch", "torchvision")
    }
    constraints = Path("/tmp/potsdam-paired-constraints.txt")
    constraints.write_text(
        "".join(f"{name}=={version}\n" for name, version in installed.items())
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--constraint",
            str(constraints),
            "transformers==5.20.2",
            "peft==0.21.2",
            "rasterio==1.5.2",
            "numpy==2.2.6",
            "opencv-python-headless==4.11.0.86",
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "--yes", "torchao"],
        check=True,
    )
    if any(
        importlib.metadata.version(name) != version
        for name, version in installed.items()
    ):
        raise ValueError("Kaggle accelerator packages changed")
    sys.path.insert(0, str(root))
    return root


def _one(paths, description: str) -> Path:
    found = list(paths)
    if len(found) != 1:
        raise ValueError(f"Expected exactly one {description}, found {len(found)}")
    return found[0]


def _checkpoint(expected: dict) -> tuple[Path, dict]:
    for manifest_path in INPUT.rglob("manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("phase", "refit") == "refit"
            and manifest.get(
                "checkpoint_sha256", manifest.get("final_checkpoint_sha256")
            )
            == expected["checkpoint_sha256"]
        ):
            checkpoint = manifest_path.parent / "final.pt"
            if (
                checkpoint.is_file()
                and _sha256(checkpoint) == expected["checkpoint_sha256"]
            ):
                return checkpoint, manifest
    raise FileNotFoundError(
        f"No attached refit checkpoint matches {expected['checkpoint_sha256']}"
    )


def _pretrained() -> Path:
    matches = []
    for manifest in INPUT.rglob("pretrained_manifest.json"):
        record = json.loads(manifest.read_text())
        if record.get("model_id") == "facebook/dinov3-vitb16-pretrain-lvd1689m":
            matches.append(manifest.parent)
    return _one(matches, "pinned DINOv3-B pretrained snapshot")


def _run() -> dict:
    _setup_source()
    import torch
    from paired_kaggle_inference import (
        _index_outputs as index_paired_outputs,
    )
    from paired_kaggle_inference import (
        run_inference,
    )

    from deepnet.potsdam_rgbir.data import CacheBundle
    from deepnet.potsdam_rgbir.model import load_model
    from deepnet.training_utils import load_checkpoint
    from model_script.pspnet_rgb import RGBPSPNet

    if not torch.cuda.is_available():
        raise RuntimeError("The supplemental inference kernel requires Kaggle CUDA")
    devices = torch.cuda.device_count()
    if devices != SETTINGS["allocation"]:
        raise RuntimeError(
            "Allocated-device count differs: "
            f"expected {SETTINGS['allocation']}, found {devices}"
        )
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False

    cache_manifest = _one(INPUT.rglob("cache_manifest.json"), "held-out RGBIR cache")
    cache = CacheBundle.load(cache_manifest.parent)
    references = SETTINGS["references"]
    if cache.manifest["cache_hash"] != references["protocol"]["cache_hash"]:
        raise ValueError("Attached held-out cache hash differs")

    checkpoints = {name: _checkpoint(references[name]) for name in ("dinov3", "pspnet")}
    pretrained = _pretrained()

    def load_dinov3():
        model, _ = load_model(pretrained, "B", False, device, checkpointing=False)
        checkpoint_path, manifest = checkpoints["dinov3"]
        payload = load_checkpoint(checkpoint_path, manifest["binding"])
        model.load_state_dict(payload["model"], strict=True)
        return model.eval()

    def load_pspnet():
        checkpoint_path, manifest = checkpoints["pspnet"]
        model = RGBPSPNet(weights=None).to(
            device=device, memory_format=torch.channels_last
        )
        payload = load_checkpoint(checkpoint_path, manifest["binding"])
        model.load_state_dict(payload["model"], strict=True)
        return model.eval()

    output = WORK / "paired"
    manifest = run_inference(
        cache=cache,
        model_factories={"dinov3": load_dinov3, "pspnet": load_pspnet},
        references=references,
        output_dir=output,
        device=device,
    )
    manifest.update(
        {
            "allocated_cuda_devices": devices,
            "cuda_device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "torchvision_version": importlib.metadata.version("torchvision"),
            "transformers_version": importlib.metadata.version("transformers"),
            "peft_version": importlib.metadata.version("peft"),
            "source_sha256": SETTINGS["source_sha256"],
            "execution_gpu_seconds": devices * (time.monotonic() - STARTED),
        }
    )
    (output / "paired-predictions.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    index_paired_outputs(output)
    return manifest


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    status = "failed"
    error = None
    marker = {"status": status, "source_sha256": SETTINGS.get("source_sha256")}
    try:
        marker = _run()
        status = "complete"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        marker.update(
            {
                "status": status,
                "error": error,
                "kernel_wall_seconds": time.monotonic() - STARTED,
            }
        )
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "paired-kernel-run.json").write_text(
            json.dumps(marker, indent=2) + "\n"
        )
        _index_outputs(WORK)


if __name__ == "__main__":
    main()
