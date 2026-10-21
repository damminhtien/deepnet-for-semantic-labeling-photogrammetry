"""Run common windowed quality checks before exporting a cloud bundle.

This entry point is retained for discoverability. Data preparation, palette
validation, alignment checks, and quality reports are handled by ``deepnet.data``.
Use ``python -m deepnet.data quality-report --help`` for the contract-driven CLI.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the shared quality report with caller-supplied arguments."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from deepnet.data.cli import main as data_main

    arguments = list(sys.argv[1:] if argv is None else argv)
    return data_main(["quality-report", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
