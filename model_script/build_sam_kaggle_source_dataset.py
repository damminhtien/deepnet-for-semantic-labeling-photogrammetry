"""Build a private Kaggle-installable wheel for the shared project package.

The resulting wheel contains the data API and default contracts. Attach it as a
Kaggle input and install it with ``pip install --no-deps`` so Kaggle's preinstalled
Torch build is left intact.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_wheel(output_dir: Path) -> Path:
    """Build the project wheel without installing or uploading dependencies."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(output_dir),
            str(PROJECT_ROOT),
        ],
        check=True,
    )
    wheels = tuple(sorted(output_dir.glob("deepnet_satellite_labeling-*.whl")))
    if len(wheels) != 1:
        raise RuntimeError(
            f"Expected one DeepNet wheel in {output_dir}, found {len(wheels)}"
        )
    return wheels[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    wheel = build_wheel(args.output_dir)
    print(wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
