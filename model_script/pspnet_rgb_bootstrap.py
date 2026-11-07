"""Packaged entry point for one private PSPNet Kaggle phase."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import io
import json
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


def one(paths, kind):
    values = list(paths)
    if len(values) != 1:
        raise ValueError(f"Expected exactly one {kind}, found {len(values)}")
    return values[0]


def setup():
    payload = base64.b85decode(PAYLOAD)
    if hashlib.sha256(payload).hexdigest() != SETTINGS["source_sha256"]:
        raise ValueError("Packaged source hash differs")
    root = Path("/tmp/pspnet-rgb-source")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if any(
            Path(name).is_absolute() or ".." in Path(name).parts
            for name in archive.namelist()
        ):
            raise ValueError("Unsafe packaged source path")
        archive.extractall(root)
    pinned = {
        name: importlib.metadata.version(name) for name in ("torch", "torchvision")
    }
    constraints = root / "accelerator-constraints.txt"
    constraints.write_text("".join(f"{k}=={v}\n" for k, v in pinned.items()))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-c",
            str(constraints),
            "numpy==2.2.6",
            "rasterio==1.5.2",
            "opencv-python-headless==4.11.0.86",
        ],
        check=True,
    )
    if any(importlib.metadata.version(k) != v for k, v in pinned.items()):
        raise RuntimeError("Kaggle accelerator packages changed")
    sys.path.insert(0, str(root))
    return root


def main():
    phase = SETTINGS["phase"]
    out = WORK / "study" / phase
    out.mkdir(parents=True, exist_ok=False)
    status = "failed"
    try:
        root = setup()
        from model_script.pspnet_rgb import PREVIOUS, run

        cache = one(INPUT.rglob("cache_manifest.json"), "role-scoped cache").parent
        markers = list(INPUT.rglob("pspnet-phase.json"))
        previous = None
        if phase in PREVIOUS:
            marker = one(markers, "prerequisite PSPNet output")
            record = json.loads(marker.read_text())
            if record["status"] != "complete" or record["phase"] != PREVIOUS[phase]:
                raise ValueError("Unexpected prerequisite output")
            previous = marker.parent / "study" / PREVIOUS[phase]
        elif markers:
            raise ValueError("Selection must start without a prior model output")
        run(SETTINGS, cache, out, root / "reference", previous, STARTED)
        status = "complete"
    finally:
        files = []
        for path in sorted(out.glob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                files.append(
                    {
                        "path": path.relative_to(WORK).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": digest,
                    }
                )
        (WORK / "output-index.json").write_text(json.dumps({"files": files}) + "\n")
        torch = sys.modules.get("torch")
        devices = (
            torch.cuda.device_count() if torch is not None else SETTINGS["allocation"]
        )
        (WORK / "pspnet-phase.json").write_text(
            json.dumps(
                {
                    "phase": phase,
                    "status": status,
                    "source_sha256": SETTINGS["source_sha256"],
                    "allocated_devices": devices,
                    "execution_gpu_seconds": devices * (time.monotonic() - STARTED),
                    "packages": {
                        name: importlib.metadata.version(name)
                        for name in ("torch", "torchvision")
                    },
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
