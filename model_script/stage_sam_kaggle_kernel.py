"""Prepare the private Kaggle script that trains SAM ViT-B with LoRA."""

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(
    "/Users/macbook/Data/runs/sam-vitb-lora-kaggle-kernel-20261019"
)
DATASET_ID = "damminhtien/sam-vitb-lora-potsdam-20261019"
KERNEL_ID = "damminhtien/sam-vitb-lora-potsdam-20261019"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing kernel directory: {output_dir}")

    output_dir.mkdir(parents=True)
    package_dir = output_dir / "semantic_labeling"
    package_dir.mkdir()
    try:
        for filename in ("__init__.py", "constants.py", "sam_lora.py"):
            shutil.copy2(PROJECT_ROOT / "semantic_labeling" / filename, package_dir)
        shutil.copy2(
            PROJECT_ROOT / "model_script" / "train_sam_kaggle.py",
            output_dir / "train_sam_kaggle.py",
        )
        metadata = {
            "id": KERNEL_ID,
            "title": "SAM ViT-B LoRA six-class Potsdam segmentation",
            "code_file": "train_sam_kaggle.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [DATASET_ID],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (output_dir / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "kernel_dir": str(output_dir),
                "kernel_id": KERNEL_ID,
                "dataset_id": DATASET_ID,
                "private": True,
                "gpu": True,
                "internet": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
