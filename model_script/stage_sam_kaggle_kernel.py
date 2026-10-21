"""Stage a caller-selected Python script and metadata for Kaggle.

This utility only creates a local directory. It does not upload, publish, or run a
Kaggle kernel.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def stage_kernel(
    script: Path,
    output_dir: Path,
    *,
    kernel_id: str,
    title: str,
    dataset_sources: tuple[str, ...],
    enable_gpu: bool,
    enable_internet: bool,
) -> Path:
    """Copy one script and write explicit Kaggle metadata for review."""
    script = script.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError(script)
    if not kernel_id.strip() or not title.strip():
        raise ValueError("kernel_id and title must not be empty")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite kernel directory: {output_dir}")

    output_dir.mkdir(parents=True)
    try:
        shutil.copy2(script, output_dir / script.name)
        metadata = {
            "id": kernel_id,
            "title": title,
            "code_file": script.name,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true" if enable_gpu else "false",
            "enable_internet": "true" if enable_internet else "false",
            "dataset_sources": list(dataset_sources),
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (output_dir / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--dataset-source", action="append", default=[])
    parser.add_argument("--enable-gpu", action="store_true")
    parser.add_argument("--enable-internet", action="store_true")
    args = parser.parse_args(argv)
    output_dir = stage_kernel(
        args.script,
        args.output_dir,
        kernel_id=args.kernel_id,
        title=args.title,
        dataset_sources=tuple(args.dataset_source),
        enable_gpu=args.enable_gpu,
        enable_internet=args.enable_internet,
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
