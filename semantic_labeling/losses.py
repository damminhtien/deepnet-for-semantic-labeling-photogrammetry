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


def inverse_sqrt_class_weights(
    class_frequencies, *, min_weight=0.5, max_weight=2.0
):
    """Return mean-one inverse-square-root weights bounded to a safe range."""
    frequencies = torch.as_tensor(class_frequencies, dtype=torch.float64)
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError("Class frequencies must be a non-empty one-dimensional value")
    if not torch.isfinite(frequencies).all() or torch.any(frequencies <= 0):
        raise ValueError("Class frequencies must be finite and positive")
    if not 0 < min_weight < 1 < max_weight:
        raise ValueError("Expected 0 < min_weight < 1 < max_weight")

    inverse_sqrt = frequencies.rsqrt()
    low_scale = 0.0
    high_scale = 1.0
    while torch.clamp(
        inverse_sqrt * high_scale, min=min_weight, max=max_weight
    ).mean() < 1:
        high_scale *= 2

    for _ in range(64):
        scale = (low_scale + high_scale) / 2
        mean_weight = torch.clamp(
            inverse_sqrt * scale, min=min_weight, max=max_weight
        ).mean()
        if mean_weight < 1:
            low_scale = scale
        else:
            high_scale = scale

    weights = torch.clamp(
        inverse_sqrt * ((low_scale + high_scale) / 2),
        min=min_weight,
        max=max_weight,
    )
    return weights.to(dtype=torch.float32)


def weighted_ce_macro_dice_loss(
    logits,
    targets,
    class_weights,
    *,
    ignore_index=IGNORE_LABEL,
    ce_weight=0.7,
    dice_weight=0.3,
    epsilon=1e-6,
):
    """Combine weighted cross entropy with macro soft Dice over valid pixels."""
    if logits.ndim != 4 or targets.ndim != 3:
        raise ValueError("Expected NCHW logits and NHW integer targets")
    if logits.shape[0] != targets.shape[0] or logits.shape[-2:] != targets.shape[-2:]:
        raise ValueError("Logits and targets must have matching batch and spatial sizes")
    if ce_weight < 0 or dice_weight < 0 or ce_weight + dice_weight <= 0:
        raise ValueError("Loss weights must be non-negative with a positive sum")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    num_classes = logits.shape[1]
    weights = torch.as_tensor(class_weights, dtype=torch.float32)
    if weights.ndim != 1 or weights.numel() != num_classes:
        raise ValueError(f"Expected {num_classes} class weights")
    if not torch.isfinite(weights).all() or torch.any(weights <= 0):
        raise ValueError("Class weights must be finite and positive")
    weights = weights.to(device=logits.device)

    valid = targets != ignore_index
    if not torch.any(valid):
        return logits.float().sum() * 0.0

    targets = targets.long()
    cross_entropy = F.cross_entropy(
        logits.float(),
        targets,
        weight=weights,
        ignore_index=ignore_index,
    )

    safe_targets = targets.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe_targets, num_classes=num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).to(dtype=torch.float32)
    valid_mask = valid.unsqueeze(1).to(dtype=torch.float32)
    probabilities = logits.float().softmax(dim=1) * valid_mask
    one_hot = one_hot * valid_mask

    reduction_dims = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dim=reduction_dims)
    denominator = probabilities.sum(dim=reduction_dims) + one_hot.sum(
        dim=reduction_dims
    )
    dice = (2 * intersection + epsilon) / (denominator + epsilon)
    present_classes = one_hot.sum(dim=reduction_dims) > 0
    macro_dice_loss = 1 - dice[present_classes].mean()

    return ce_weight * cross_entropy + dice_weight * macro_dice_loss
