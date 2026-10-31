"""Build verified private Kaggle inputs without changing source data."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path

from deepnet.data.bundles import load_bundle
from deepnet.data.raster import sha256_file
from deepnet.models.specs import load_model_spec
from deepnet.training_utils import verify_pretrained, write_json

RUN_ID = "isprs-dinov2-joint-v1-20261031"
SLUGS = {
    "training": "isprs-irrg-train-validation-20261031",
    "test": "isprs-irrg-heldout-20261031",
    "source": "dinov2-joint-source-20261031",
}


def archive(source: Path, destination: Path, prefix: str) -> None:
    with tarfile.open(destination, "w:gz", compresslevel=1) as target:
        for path in sorted(source.rglob("*")):
            if path.is_file() and ".cache" not in path.relative_to(source).parts:
                target.add(path, arcname=f"{prefix}/{path.relative_to(source)}")


def package(data: Path, pretrained: Path, output: Path, owner: str) -> None:
    repository = Path(__file__).resolve().parents[1]
    if output.exists():
        raise FileExistsError("Choose a new staging directory")
    output.mkdir(parents=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    manifests = {}
    for role, slug in SLUGS.items():
        stage = output / role
        stage.mkdir()
        write_json(
            stage / "dataset-metadata.json",
            {
                "id": f"{owner}/{slug}",
                "title": slug.replace("-", " "),
                "licenses": [{"name": "other"}],
            },
        )
        manifest = {"run_id": RUN_ID, "role": role, "source_revision": revision}
        if role != "source":
            bundle_name = (
                "train-validation-bundle"
                if role == "training"
                else "post-selection-test-bundle"
            )
            manifest["bundles"] = {}
            for dataset in ("potsdam", "vaihingen"):
                source = data / dataset / bundle_name
                bundle = load_bundle(source)
                manifest["bundles"][dataset] = bundle.manifest["bundle_sha256"]
                print(f"Verified {dataset}/{bundle_name}; compressing", flush=True)
                archive(source, stage / f"{dataset}.tar.gz", f"{dataset}/{bundle_name}")
                if role == "training":
                    shutil.copy2(
                        data / dataset / "quality.json",
                        stage / f"{dataset}-quality.json",
                    )
        else:
            spec = load_model_spec("dinov2-vitb14-multimodal-v2")
            verify_pretrained(
                pretrained, spec.backbone.model_id, spec.backbone.revision
            )
            archive(pretrained, stage / "pretrained.tar.gz", "pretrained")
            subprocess.run(
                ["uv", "build", "--wheel", "--out-dir", str(stage)],
                cwd=repository,
                check=True,
            )
            for relative in (
                "configs/training/isprs-dinov2-joint-v1.json",
                "configs/evaluations/isprs-dinov2-v1.json",
            ):
                shutil.copy2(repository / relative, stage / Path(relative).name)
        manifest["files"] = [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(stage.iterdir())
            if path.name != "dataset-metadata.json"
        ]
        write_json(stage / "deepnet-input-manifest.json", manifest)
        manifests[role] = sha256_file(stage / "deepnet-input-manifest.json")
    kernel = output / "kernel"
    kernel.mkdir()
    bootstrap = (repository / "model_script/train_dinov2_kaggle.py").read_text()
    bootstrap = bootstrap.replace(
        "EXPECTED_INPUTS = {}", f"EXPECTED_INPUTS = {manifests!r}"
    )
    (kernel / "train_dinov2_kaggle.py").write_text(bootstrap)
    write_json(
        kernel / "kernel-metadata.json",
        {
            "id": f"{owner}/dinov2-joint-irrg-segmentation-20261031",
            "title": "DINOv2 Joint IRRG Segmentation 20261031",
            "code_file": "train_dinov2_kaggle.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [f"{owner}/{slug}" for slug in SLUGS.values()],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        },
    )
    write_json(
        output / "package-manifest.json", {"run_id": RUN_ID, "inputs": manifests}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", default="damminhtien")
    args = parser.parse_args()
    package(args.data_root, args.pretrained, args.output, args.owner)


if __name__ == "__main__":
    main()
