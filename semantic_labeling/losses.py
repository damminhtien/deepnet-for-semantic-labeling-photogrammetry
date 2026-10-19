"""Loss functions shared by the model notebooks."""

import torch
import torch.nn.functional as F

from .constants import IGNORE_LABEL


def CrossEntropy2d(input, target, weight=None, size_average=True, ignore_index=IGNORE_LABEL):
    """Cross entropy for flat or NCHW logits, ignoring no-boundary pixels."""
    if input.dim() == 2:
        logits = input
        targets = target
    elif input.dim() == 4:
        classes = input.shape[1]
        logits = input.permute(0, 2, 3, 1).reshape(-1, classes)
        targets = target.reshape(-1)
        if targets.numel() != logits.shape[0]:
            raise ValueError("The target map must match the spatial dimensions of the logits")
    else:
        raise ValueError(f"Expected 2D or 4D logits, got {input.dim()} dimensions")

    if not torch.any(targets != ignore_index):
        return input.sum() * 0.0

    reduction = "mean" if size_average else "sum"
    return F.cross_entropy(
        logits,
        targets,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
    )
