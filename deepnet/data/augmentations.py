"""Domain-aware, synchronized training transforms built with Albumentations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .contracts import (
    AugmentationSpec,
    DomainSpec,
    ModelInputSpec,
    TransformSpec,
)

TransformFactory = Callable[[Any, TransformSpec], Any]
_TRANSFORM_FACTORIES: dict[str, TransformFactory] = {}


def register_transform(name: str, factory: TransformFactory) -> None:
    """Register a transform factory for versioned augmentation configs."""
    if not name.strip():
        raise ValueError("Transform name must not be empty")
    if name in _TRANSFORM_FACTORIES:
        raise ValueError(f"Transform factory {name!r} is already registered")
    _TRANSFORM_FACTORIES[name] = factory


def _build_transforms(
    albumentations: Any, specs: tuple[TransformSpec, ...]
) -> list[Any]:
    transforms = []
    for spec in specs:
        try:
            factory = _TRANSFORM_FACTORIES[spec.name]
        except KeyError as error:
            raise ValueError(
                f"No Albumentations factory is registered for {spec.name!r}"
            ) from error
        transforms.append(factory(albumentations, spec))
    return transforms


def _square_symmetry(albumentations: Any, spec: TransformSpec) -> Any:
    return albumentations.SquareSymmetry(p=spec.probability, **spec.parameter_map())


def _affine(albumentations: Any, spec: TransformSpec) -> Any:
    import cv2

    parameters = spec.parameter_map()
    interpolation_names = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
    }
    border_names = {
        "constant": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT,
        "reflect101": cv2.BORDER_REFLECT_101,
        "replicate": cv2.BORDER_REPLICATE,
    }
    for key in ("interpolation", "mask_interpolation"):
        value = parameters.get(key)
        if isinstance(value, str):
            try:
                parameters[key] = interpolation_names[value]
            except KeyError as error:
                raise ValueError(f"Unsupported {key}: {value!r}") from error
    border_mode = parameters.get("border_mode")
    if isinstance(border_mode, str):
        try:
            parameters["border_mode"] = border_names[border_mode]
        except KeyError as error:
            raise ValueError(f"Unsupported border_mode: {border_mode!r}") from error
    return albumentations.Affine(p=spec.probability, **parameters)


def _random_brightness_contrast(albumentations: Any, spec: TransformSpec) -> Any:
    return albumentations.RandomBrightnessContrast(
        p=spec.probability,
        **spec.parameter_map(),
    )


def _horizontal_flip(albumentations: Any, spec: TransformSpec) -> Any:
    return albumentations.HorizontalFlip(p=spec.probability, **spec.parameter_map())


register_transform("square_symmetry", _square_symmetry)
register_transform("affine", _affine)
register_transform("random_brightness_contrast", _random_brightness_contrast)
register_transform("horizontal_flip", _horizontal_flip)


class AlbumentationsAugmenter:
    """Apply shared geometry and optical-only radiometry to one training sample.

    Inputs are raw, aligned channel arrays. Model-view normalization is applied by
    the data reader after this callable returns. The Compose instances are reseeded
    for each sample so worker scheduling cannot affect augmentation results.
    """

    def __init__(
        self,
        domain: DomainSpec,
        model_input: ModelInputSpec,
        spec: AugmentationSpec,
        *,
        ignore_label: int,
    ) -> None:
        if spec.domain_id != domain.domain_id:
            raise ValueError(
                f"Augmentation domain {spec.domain_id!r} does not match "
                f"{domain.domain_id!r}"
            )
        domain.validate_model_input(model_input)
        self.domain = domain
        self.spec = spec
        self.ignore_label = ignore_label
        self.requires_square_windows = any(
            transform.name == "square_symmetry" for transform in spec.geometry
        )
        band_roles = {band.name: band.role for band in domain.bands}
        self.optical_channels = tuple(
            name for name in model_input.channels if band_roles[name] == "optical"
        )
        self.auxiliary_channels = tuple(
            name for name in model_input.channels if band_roles[name] == "auxiliary"
        )
        if not self.optical_channels:
            raise ValueError("The selected model input has no optical channels")
        self.band_specs = {band.name: band for band in domain.bands}
        if spec.optical:
            ranges = {
                self.band_specs[name].value_range for name in self.optical_channels
            }
            if None in ranges or len(ranges) != 1:
                raise ValueError(
                    "Optical radiometric transforms require one shared, declared "
                    "value_range across selected optical channels"
                )
            self.optical_value_range = next(iter(ranges))
        else:
            self.optical_value_range = None

        try:
            import albumentations as albumentations
        except ImportError as error:
            raise ImportError(
                "Install the pinned vision extra to use Albumentations policies"
            ) from error

        geometry_targets = {
            f"aux_{index}": "image" for index, _ in enumerate(self.auxiliary_channels)
        }
        geometry_targets["valid"] = "mask"
        self.geometry = (
            albumentations.Compose(
                _build_transforms(albumentations, spec.geometry),
                additional_targets=geometry_targets,
                strict=True,
                seed=0,
            )
            if spec.geometry
            else None
        )
        self.optical = (
            albumentations.Compose(
                _build_transforms(albumentations, spec.optical),
                strict=True,
                seed=0,
            )
            if spec.optical
            else None
        )

    def __call__(
        self,
        channels: Mapping[str, np.ndarray],
        labels: np.ndarray,
        valid_mask: np.ndarray,
        *,
        seed: int,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Transform a channel mapping and aligned label/validity arrays."""
        missing = (set(self.optical_channels) | set(self.auxiliary_channels)) - set(
            channels
        )
        if missing:
            raise ValueError("Missing input bands: " + ", ".join(sorted(missing)))
        labels = np.asarray(labels)
        valid = np.asarray(valid_mask, dtype=bool)
        if labels.ndim != 2 or valid.shape != labels.shape:
            raise ValueError("Labels and validity mask must be aligned 2D arrays")
        arrays = {
            name: np.asarray(channels[name])
            for name in (*self.optical_channels, *self.auxiliary_channels)
        }
        if any(array.shape != labels.shape for array in arrays.values()):
            raise ValueError("Every input band must align with the label grid")

        optical = np.stack([arrays[name] for name in self.optical_channels], axis=-1)
        if self.geometry is not None:
            seed_value = int(seed) & 0xFFFFFFFF
            self.geometry.set_random_seed(seed_value)
            geometry_input: dict[str, Any] = {
                "image": optical,
                "mask": labels,
                "valid": valid.astype(np.uint8),
            }
            for index, name in enumerate(self.auxiliary_channels):
                geometry_input[f"aux_{index}"] = arrays[name]
            transformed = self.geometry(**geometry_input)
            optical = transformed["image"]
            labels = transformed["mask"]
            valid = transformed["valid"].astype(bool, copy=False)
            for index, name in enumerate(self.auxiliary_channels):
                arrays[name] = transformed[f"aux_{index}"]

        if self.optical is not None:
            low, high = self.optical_value_range
            optical = np.asarray(optical, dtype=np.float32)
            optical = np.clip((optical - low) / (high - low), 0.0, 1.0)
            self.optical.set_random_seed((int(seed) ^ 0x9E3779B9) & 0xFFFFFFFF)
            optical = self.optical(image=optical)["image"]
            optical = np.clip(optical, 0.0, 1.0) * (high - low) + low

        for index, name in enumerate(self.optical_channels):
            arrays[name] = optical[..., index]
        labels = np.asarray(labels).copy()
        labels[~valid] = self.ignore_label
        return (
            {name: np.ascontiguousarray(array) for name, array in arrays.items()},
            np.ascontiguousarray(labels),
            np.ascontiguousarray(valid),
        )
