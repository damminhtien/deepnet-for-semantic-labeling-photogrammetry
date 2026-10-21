"""Contract-driven data access for geospatial semantic segmentation."""

from .bundles import BundleData, export_model_bundle, load_bundle, verify_bundle
from .contracts import (
    AlignmentError,
    GridSpec,
    ModelInputSpec,
    NormalizationRule,
    PreprocessingSpec,
    RasterAsset,
    Sample,
    SceneRecord,
    SplitManifest,
    WindowSpec,
    canonical_hash,
)
from .lineage import build_run_manifest

__all__ = (
    "AlignmentError",
    "BundleData",
    "GridSpec",
    "ModelInputSpec",
    "NormalizationRule",
    "PreprocessingSpec",
    "RasterAsset",
    "Sample",
    "SceneRecord",
    "SplitManifest",
    "WindowSpec",
    "canonical_hash",
    "build_run_manifest",
    "export_model_bundle",
    "load_bundle",
    "verify_bundle",
)
