"""Framework-neutral boundary between a Torch model and the data pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from deepnet.data.contracts import ModelInputSpec


def _default_logits(output: Any) -> Any:
    """Accept a tensor or the common tuple/list wrapper returned by models."""
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Model returned an empty output sequence")
        return output[0]
    return output


@dataclass(slots=True)
class TorchSegmentationAdapter:
    """Apply one model-input contract to any compatible Torch segmentation model.

    The adapter does not construct a model or select its architecture. Model
    creation and unusual output structures remain injectable at the call site.
    """

    model: Any
    input_spec: ModelInputSpec
    logits_selector: Callable[[Any], Any] = _default_logits

    def prepare_images(
        self,
        images: Any,
        *,
        device: Any,
        non_blocking: bool = False,
    ) -> Any:
        """Validate NCHW inputs against the spec, then move them to the device."""
        import torch

        if images.ndim != 4:
            raise ValueError(
                f"Expected a four-dimensional NCHW batch, got {images.shape}"
            )
        if images.shape[1] != len(self.input_spec.channels):
            raise ValueError(
                f"Input has {images.shape[1]} channels; model contract declares "
                f"{len(self.input_spec.channels)}"
            )
        if self.input_spec.layout != "CHW" or self.input_spec.dtype != "float32":
            raise ValueError("Torch segmentation inputs must use CHW float32 layout")
        tensor = images if isinstance(images, torch.Tensor) else torch.as_tensor(images)
        return tensor.to(
            device=device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        )

    def predict_logits(
        self,
        images: Any,
        *,
        device: Any,
        non_blocking: bool = False,
    ) -> Any:
        """Run the model and validate its batch and class dimensions."""
        inputs = self.prepare_images(
            images,
            device=device,
            non_blocking=non_blocking,
        )
        logits = self.logits_selector(self.model(inputs))
        if logits.ndim != 4:
            raise ValueError(f"Expected NCHW logits, got shape {logits.shape}")
        if logits.shape[0] != inputs.shape[0]:
            raise ValueError("Model output batch dimension differs from its input")
        if logits.shape[1] <= 0:
            raise ValueError("Model returned no output classes")
        return logits
