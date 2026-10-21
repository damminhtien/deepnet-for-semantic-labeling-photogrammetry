"""Export a portable Kaggle input bundle through the shared data CLI.

The bundle view and split are supplied by versioned JSON contracts. The script
does not copy model-specific subsets or define a split of its own.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the common model-view exporter with caller-supplied arguments."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from deepnet.data.cli import main as data_main

    arguments = list(sys.argv[1:] if argv is None else argv)
    return data_main(["export-bundle", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
