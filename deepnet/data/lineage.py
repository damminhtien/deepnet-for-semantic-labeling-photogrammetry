"""Run-manifest helpers that bind model runs to immutable data contracts."""

from __future__ import annotations

import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .bundles import BundleData
from .contracts import canonical_hash


def _git_revision() -> tuple[str | None, bool | None]:
    try:
        root = Path(__file__).resolve().parents[2]
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return revision, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None


def build_run_manifest(
    bundle: BundleData,
    *,
    run_id: str,
    seed: int,
    sampler: dict[str, Any],
    configuration: dict[str, Any],
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a provenance header; trainers add checkpoint and metric records."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    revision, worktree_dirty = _git_revision()
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "initialized",
        "created_utc": datetime.now(UTC).isoformat(),
        "dataset_id": bundle.manifest["dataset_id"],
        "bundle_sha256": bundle.manifest["bundle_sha256"],
        "catalog_hash": bundle.manifest["catalog_hash"],
        "split_id": bundle.manifest["split_id"],
        "split_version": bundle.manifest["split_version"],
        "split_hash": bundle.manifest["split_hash"],
        "preprocessing_id": bundle.manifest["preprocessing_id"],
        "preprocessing_version": bundle.manifest["preprocessing_version"],
        "preprocessing_hash": bundle.manifest["preprocessing_hash"],
        "model_input_id": bundle.manifest["model_input_id"],
        "model_input_version": bundle.manifest["model_input_version"],
        "model_input_hash": bundle.manifest["model_input_hash"],
        "seed": seed,
        "sampler": sampler,
        "sampler_hash": canonical_hash(sampler),
        "configuration": configuration,
        "configuration_hash": canonical_hash(configuration),
        "source_revision": revision,
        "source_worktree_dirty": worktree_dirty,
        "runtime": {
            "python": platform.python_version(),
            **(runtime or {}),
        },
    }
