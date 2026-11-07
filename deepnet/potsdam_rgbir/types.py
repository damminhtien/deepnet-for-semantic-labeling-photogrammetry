"""Persisted shapes shared by the Potsdam RGBIR study phases."""

from __future__ import annotations

from typing import Any, Literal

from typing_extensions import NotRequired, TypedDict

BackboneSize = Literal["B", "L"]
InferencePolicy = Literal["identity", "hflip"]


class Recipe(TypedDict):
    backbone: BackboneSize
    auxiliary: bool
    enhanced: bool


class DeviceBinding(TypedDict):
    accelerator: str
    allocated_devices: int
    packages: dict[str, str]


class ModelBinding(TypedDict):
    model_spec_hash: str
    pretrained_hash: str


class EvaluationMetrics(TypedDict):
    overall_accuracy: float
    mean_iou: float
    per_class_iou: list[float | None]
    per_class_precision: list[float | None]
    per_class_recall: list[float | None]
    per_class_f1: list[float | None]
    pixels_scored: int
    confusion_matrix: list[list[int]]


class Candidate(TypedDict):
    eligible: bool
    arm: str
    step: int
    overall_accuracy: float
    mean_iou: float
    car_iou: float
    clutter_iou: float


class BestCandidate(Candidate):
    checkpoint_sha256: str


class SelectionAsset(TypedDict):
    role: str
    source_rgbir_sha256: str
    source_dsm_sha256: str
    shape: list[int]


class WindowRecord(TypedDict):
    row: int
    col: int
    height: int
    width: int
    padding: str


class PanelRecord(TypedDict):
    scene: str
    window: WindowRecord


class TrainingIndex(TypedDict):
    base: list[list[Any]]
    class_support: list[int]
    rare_tiles: dict[str, dict[str, list[list[int]]]]


class SamplingConfig(TypedDict):
    rare_fraction_per_class: float
    center_jitter: int


class AugmentationPreset(TypedDict):
    affine_probability: float
    rotation: float
    scale: list[float]
    translation: float
    halo: int
    gain_probability: float
    gain_range: list[float]
    gamma_probability: float
    gamma_range: list[float]
    noise_probability: float
    noise_std: float
    dsm_dropout: float


class OptimizerConfig(TypedDict):
    learning_rates: dict[str, float]
    weight_decay: float


class ExecutionConfig(TypedDict):
    mixed_precision: str
    gradient_clip_norm: float
    num_workers: int
    prefetch_factor: int


class LossConfig(TypedDict):
    ignore_index: int
    cross_entropy_weight: float
    soft_dice_weight: float


class InferenceConfig(TypedDict):
    minimum_tta_oa_gain: float
    available_policies: list[InferencePolicy]


class GPUHours(TypedDict):
    preflight: float
    selection: float
    refit: float
    heldout: float
    reserve: float
    total: float


class SmallFitConfig(TypedDict):
    minimum_miou: float
    max_updates: int
    patches: int


class ProtocolConfig(TypedDict):
    schema_version: int
    study_id: str
    selection_split_hash: str
    validation_pixels: int
    heldout_pixels: int
    reference_validation_confusion: list[list[int]]
    reference_checkpoint_sha256: str
    window_size: list[int]
    stride: list[int]
    global_batch: int
    seed: int
    confirmation_seed: int
    models: dict[BackboneSize, str]
    sampling: SamplingConfig
    augmentation: dict[str, AugmentationPreset]
    optimizer: OptimizerConfig
    execution: ExecutionConfig
    loss: LossConfig
    inference: InferenceConfig
    gpu_hours: GPUHours
    small_fit: SmallFitConfig
    oa_goal: float


class ModelTiming(TypedDict):
    update_seconds: float
    update_samples: list[float]
    probe_seconds: float
    scene_seconds: float
    tta_scene_seconds: float
    setup_seconds: float
    batch_size: int
    cached_scene: dict[str, Any]
    cache_io_preprocessing_seconds: dict[str, float]
    tta_scene: dict[str, Any]
    peak_vram_bytes: int
    calibration: NotRequired[dict[str, Any]]
    rasterio_scene: NotRequired[dict[str, Any]]
    rasterio_seconds: NotRequired[float]
    rasterio_io_preprocessing_seconds: NotRequired[dict[str, float]]
    cache_speedup: NotRequired[float]
    nonoverlap_scene: NotRequired[dict[str, Any]]
    nonoverlap_seconds: NotRequired[float]
    overlap_latency_ratio: NotRequired[float]


class Protocol(ProtocolConfig):
    passes: int
    arms: list[str]
    per_arm_gpu_hours: float
    source_hash: str
    runtime: DeviceBinding
    selection_cache_hash: str
    source_bundle_sha256: str
    rgbir_snapshot_hash: str
    dsm_manifest_sha256: str
    relief_scale: float
    feature_recipe: dict[str, Any]
    training_index_hash: str
    train_class_support: list[int]
    input_panel: list[PanelRecord]
    probe_panel: list[PanelRecord]
    full_validation_passes: list[int]
    pretrained_bindings: dict[str, ModelBinding]
    timings: dict[str, ModelTiming]
    selection_assets: dict[str, SelectionAsset]
    selection_file_hashes: dict[str, str]
    protocol_hash: str


PassRecordCore = TypedDict(
    "PassRecordCore",
    {
        "pass": int,
        "updates": int,
        "train_loss": float,
        "train_metrics": EvaluationMetrics,
    },
)


class PassRecord(PassRecordCore, total=False):
    probe: dict[str, Any]
    full_validation: EvaluationMetrics
    backbone_inputs: dict[str, Any]


class PhaseState(TypedDict):
    updates: int
    completed_passes: int
    history: list[PassRecord]
    best: BestCandidate | None
    best_validation: EvaluationMetrics | None
    valid_pixel_presentations: int
    training_seconds: float
    validation_seconds: float
    pass_matrix: list[list[int]]
    pass_loss: float


class CheckpointBinding(TypedDict):
    protocol_hash: str
    cache_hash: str
    arm: str
    recipe: Recipe
    seed: int
    model: ModelBinding
    selection_lock: SelectionLock | None


class TTAValidation(TypedDict):
    seed42: EvaluationMetrics
    seed43: EvaluationMetrics


class SelectionLock(TypedDict):
    protocol_hash: str
    arm: str
    recipe: Recipe
    selection_binding: str
    selection_checkpoint_sha256: str
    confirmation_checkpoint_sha256: str
    validation: EvaluationMetrics
    inference: InferencePolicy
    tta_validation: TTAValidation | None
    refit_updates: int
    lr_multiplier: float
    selection_cache_hash: str


class PhaseManifest(PhaseState):
    status: Literal["complete"]
    arm: str
    seed: int
    recipe: Recipe
    binding: str
    model_binding: ModelBinding
    protocol_hash: str
    cache_hash: str
    train_scene_ids: list[str]
    train_class_support: list[int]
    trainable_parameters: int
    validation_fraction: float
    selection_checkpoint_sha256: NotRequired[str]
    final_checkpoint_sha256: NotRequired[str]
    backbone_inputs: NotRequired[dict[str, Any]]


class Schedule(TypedDict):
    passes: int
    arms: list[str]
    per_arm_gpu_hours: float
