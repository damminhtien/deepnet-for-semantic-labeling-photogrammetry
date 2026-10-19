"""Small model initialization utilities shared by model implementations."""

from torch import nn


def initialize_weights(*models):
    """Apply the initialization scheme used by the PSPNet implementation."""
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
