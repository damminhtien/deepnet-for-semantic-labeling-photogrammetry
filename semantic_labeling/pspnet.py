"""PSPNet with a five-channel ResNet-101 backbone for Potsdam tiles."""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet101

from .constants import INPUT_CHANNELS, N_CLASSES
from .model_utils import initialize_weights


def _adaptive_avg_pool2d_mps(features, output_size):
    """Match adaptive average pooling with reductions supported by MPS."""
    input_height, input_width = features.shape[-2:]
    output_height, output_width = output_size
    rows = []

    for index in range(output_height):
        start = index * input_height // output_height
        end = ((index + 1) * input_height + output_height - 1) // output_height
        rows.append(features[:, :, start:end, :].mean(dim=2, keepdim=True))

    pooled_height = torch.cat(rows, dim=2)
    columns = []
    for index in range(output_width):
        start = index * input_width // output_width
        end = ((index + 1) * input_width + output_width - 1) // output_width
        columns.append(
            pooled_height[:, :, :, start:end].mean(dim=3, keepdim=True)
        )

    return torch.cat(columns, dim=3)


class _PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, reduction_channels, bins=(1, 2, 3, 6)):
        super().__init__()
        self.bins = bins
        self.features = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin_size),
                    nn.Conv2d(in_channels, reduction_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(reduction_channels, momentum=0.95),
                    nn.ReLU(inplace=True),
                )
                for bin_size in bins
            ]
        )

    def forward(self, features):
        output_size = features.shape[-2:]
        pooled = [features]
        for bin_size, branch in zip(self.bins, self.features):
            if features.device.type == "mps":
                pooled_features = _adaptive_avg_pool2d_mps(
                    features, (bin_size, bin_size)
                )
                pooled_features = branch[1](pooled_features)
                pooled_features = branch[2](pooled_features)
                pooled_features = branch[3](pooled_features)
            else:
                pooled_features = branch(features)
            pooled.append(
                F.interpolate(
                    pooled_features,
                    size=output_size,
                    mode="bilinear",
                    align_corners=True,
                )
            )
        return torch.cat(pooled, dim=1)


class PSPNet(nn.Module):
    """Potsdam PSPNet; all weights start random for a from-scratch run."""

    def __init__(self, num_classes=N_CLASSES, use_aux=True):
        super().__init__()
        self.use_aux = use_aux
        resnet = resnet101(weights=None)
        resnet.conv1 = nn.Conv2d(
            INPUT_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        for name, module in self.layer3.named_modules():
            if "conv2" in name:
                module.dilation = (2, 2)
                module.padding = (2, 2)
                module.stride = (1, 1)
            elif "downsample.0" in name:
                module.stride = (1, 1)

        for name, module in self.layer4.named_modules():
            if "conv2" in name:
                module.dilation = (4, 4)
                module.padding = (4, 4)
                module.stride = (1, 1)
            elif "downsample.0" in name:
                module.stride = (1, 1)

        self.ppm = _PyramidPoolingModule(2048, 512)
        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1),
        )

        if use_aux:
            self.aux_logits = nn.Conv2d(1024, num_classes, kernel_size=1)
            initialize_weights(self.aux_logits)

        initialize_weights(self.ppm, self.final)

    def forward(self, inputs):
        input_size = inputs.shape[-2:]
        features = self.layer0(inputs)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        auxiliary = self.aux_logits(features) if self.training and self.use_aux else None
        features = self.layer4(features)
        logits = self.final(self.ppm(features))
        logits = F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=True
        )

        if auxiliary is None:
            return logits

        auxiliary = F.interpolate(
            auxiliary, size=input_size, mode="bilinear", align_corners=True
        )
        return logits + auxiliary, auxiliary
