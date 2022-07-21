"""Creation of ArcFace's ResNet-based models."""
import os
import sys

# add "src/" dir to system path
N_LEVELS_UP = 4
sys.path.append(os.pathsep.join([".."] * N_LEVELS_UP))

# pylint: disable=import-error,no-name-in-module
from tools import logger

log = logger.create_logger(__name__)
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

__all__ = ["iresnet18", "iresnet34", "iresnet50", "iresnet100", "iresnet200"]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    """Basic Block for Identity ResNet"""

    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
    ):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(
            in_planes,
            eps=1e-05,
        )
        self.conv1 = conv3x3(in_planes, planes)
        self.bn2 = nn.BatchNorm2d(
            planes,
            eps=1e-05,
        )
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(
            planes,
            eps=1e-05,
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the block."""
        identity = input_tensor
        out = self.bn1(input_tensor)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(input_tensor)
        out += identity
        return out


class IResNet(nn.Module):
    """Base class for Identity ResNet"""

    fc_scale = 7 * 7

    def __init__(
        self,
        block: nn.Module,
        layers: Iterable,
        dropout: float = 0,
        num_features: int = 512,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
        fp16: bool = False,
        architecture: str = "arcface_1dcnn",  # arcface_1dcnn, arcface_2dcnn, arcface_3dcnn
    ):
        super().__init__()
        self.fp16 = fp16
        self.in_planes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes, eps=1e-05)
        self.prelu = nn.PReLU(self.in_planes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.bn2 = nn.BatchNorm2d(
            512 * block.expansion,
            eps=1e-05,
        )
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        # Pareidolia change:
        self.architecture = architecture
        self.build_pipeline()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def build_pipeline(self):
        """Builds the pipeline to forward the model inputs later."""
        backbone = list(
            [
                self.conv1,
                self.bn1,
                self.prelu,
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4,
                self.bn2,
            ]
        )
        if self.architecture in ["arcface_1dcnn", "arcface_2dcnn"]:
            self._pipeline = nn.Sequential(
                *backbone,
                nn.Flatten(start_dim=1),
                nn.Dropout(p=0.5),
                self.fc,
                self.features,
            )
        elif self.architecture in ["arcface_3dcnn"]:
            self._pipeline = nn.Sequential(
                *backbone,
            )

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """Creates a layer of ResNet"""
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(
                    planes * block.expansion,
                    eps=1e-05,
                ),
            )
        layers = []
        layers.append(
            block(
                self.in_planes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
            )
        )
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self._pipeline(input_tensor)
        # log.info(features.shape)
        return features


def _iresnet(
    block: nn.Module, layers: Iterable, architecture: str = "arcface_1dcnn", **kwargs
):
    """Constructs a ResNet-based model."""
    model = IResNet(
        block=block,
        layers=layers,
        architecture=architecture,
        **kwargs,
    )
    return model


def iresnet18(**kwargs):
    """Constructs a ResNet-18 based model."""
    return _iresnet(
        block=IBasicBlock,
        layers=[2, 2, 2, 2],
        **kwargs,
    )


def iresnet34(**kwargs):
    """Constructs a ResNet-34 based model."""
    return _iresnet(
        block=IBasicBlock,
        layers=[3, 4, 6, 3],
        **kwargs,
    )


def iresnet50(**kwargs):
    """Constructs a ResNet-50 based model."""
    return _iresnet(
        block=IBasicBlock,
        layers=[3, 4, 14, 3],
        **kwargs,
    )


def iresnet100(**kwargs):
    """Constructs a ResNet-100 based model."""
    return _iresnet(
        block=IBasicBlock,
        layers=[3, 13, 30, 3],
        **kwargs,
    )


def iresnet200(**kwargs):
    """Constructs a ResNet-200 based model."""
    return _iresnet(
        block=IBasicBlock,
        layers=[6, 26, 60, 6],
        **kwargs,
    )
