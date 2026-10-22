"""Contract-driven data access for geospatial semantic segmentation."""

from .bundles import BundleData, export_model_bundle, load_bundle, verify_bundle
from .catalog import load_augmentation_spec, load_domain_spec
from .contracts import (
    AlignmentError,
    AugmentationSpec,
    BandSpec,
    DomainSpec,
    GridSpec,
    ModelInputSpec,
    NormalizationRule,
    PreprocessingSpec,
    RasterAsset,
    Sample,
    SceneRecord,
    SplitManifest,
    TransformSpec,
    WindowSpec,
    canonical_hash,
)
from .lineage import build_run_manifest
from .sampling import ClassPixelHistogram

__all__ = (
    "AlignmentError",
    "AugmentationSpec",
    "BandSpec",
    "BundleData",
    "ClassPixelHistogram",
    "GridSpec",
    "DomainSpec",
    "ModelInputSpec",
    "NormalizationRule",
    "PreprocessingSpec",
    "RasterAsset",
    "Sample",
    "SceneRecord",
    "SplitManifest",
    "TransformSpec",
    "WindowSpec",
    "canonical_hash",
    "build_run_manifest",
    "export_model_bundle",
    "load_bundle",
    "load_augmentation_spec",
    "load_domain_spec",
    "verify_bundle",
)
