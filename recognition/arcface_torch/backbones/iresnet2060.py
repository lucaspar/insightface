"""ArcFace backbone with Resnet 2060."""
from typing import Iterable, Callable

import torch
from torch import nn


def check_torch_version(min_major: int, min_minor: int, min_patch: int):
    """Make sure PyTorch version matches minimum, raises on failure."""
    major, minor, patch = (int(x) for x in torch.__version__.split("."))
    assert (
        (major > min_major)
        or (major == min_major and minor > min_minor)
        or (major == min_major and minor == min_minor and patch >= min_patch)
    ), "PyTorch version must be {}.{}.{} or higher".format(
        min_major, min_minor, min_patch
    )


# Require PyTorch 1.8.1 or higher
check_torch_version(1, 8, 1)

from torch.utils.checkpoint import checkpoint_sequential

__all__ = ["iresnet2060"]


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
    """Basic Block for ResNet."""

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

    def forward(self, input_tensor:torch.Tensor) -> torch.Tensor:
        """Forward pass."""
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
    """ResNet for ArcFace."""

    fc_scale = 7 * 7

    def __init__(
        self,
        block,
        layers,
        dropout=0,
        num_features=512,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        fp16=False,
    ):
        super(IResNet, self).__init__()
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

    def _make_layer(
        self,
        block: nn.Module,
        planes: Iterable,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ):
        """Make a layer with a specific block"""
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

    def checkpoint(self, func: Callable, n_segments: int, input_tensor: torch.Tensor):
        """Checkpoints the output of a layer."""
        if self.training:
            return checkpoint_sequential(func, n_segments, input_tensor)
        else:
            return func(input_tensor)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        with torch.cuda.amp.autocast(self.fp16):
            input_tensor = self.conv1(input_tensor)
            input_tensor = self.bn1(input_tensor)
            input_tensor = self.prelu(input_tensor)
            input_tensor = self.layer1(input_tensor)
            input_tensor = self.checkpoint(
                func=self.layer2, n_segments=20, input_tensor=input_tensor
            )
            input_tensor = self.checkpoint(
                func=self.layer3, n_segments=100, input_tensor=input_tensor
            )
            input_tensor = self.layer4(input_tensor)
            input_tensor = self.bn2(input_tensor)
            input_tensor = torch.flatten(input_tensor, 1)
            input_tensor = self.dropout(input_tensor)
        input_tensor = self.fc(input_tensor.float() if self.fp16 else input_tensor)
        input_tensor = self.features(input_tensor)
        return input_tensor


def iresnet2060(**kwargs):
    """Creates a iResNet-2060 model."""
    model = IResNet(block=IBasicBlock, layers=[3, 128, 1024 - 128, 3], **kwargs)
    return model
