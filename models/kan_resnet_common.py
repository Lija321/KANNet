#Made with assitance of ChatGPT

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

from components.ReluKANOperator2d import ReluKANOperator2d
from components.efficient_kan import KANLinear


# -------------------------
# KAN constructors / modules
# -------------------------

def _default_kan_constructor(in_features: int, out_features: int) -> nn.Module:
    # Vectorized: produce out_features at once (conv-like).
    return KANLinear(in_features, out_features)


class KANPointwise2d(nn.Module):
    """
    1×1 conv replacement using KANLinear without unfold.
    Applies KANLinear across channels independently at each spatial position.

    NOTE: stride must be 1; spatial downsampling is handled outside (AvgPool).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kan_module_constructor: Callable[[int, int], nn.Module] = _default_kan_constructor,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kan = kan_module_constructor(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"KANPointwise2d expected {self.in_channels} channels, got {C}")

        # (B, H, W, C) -> (B*H*W, C)
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
        # (B*H*W, out_channels)
        y = self.kan(x_flat)
        # back to (B, out_channels, H, W)
        y = y.view(B, H, W, self.out_channels).permute(0, 3, 1, 2).contiguous()
        return y


# -------------------------
# Conv factories
# -------------------------

def _conv2d_3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1, groups: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        bias=False,
    )


def _conv2d_1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _kan_conv2d_3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    kan_module_constructor: Callable[[int, int], nn.Module] = _default_kan_constructor,
) -> ReluKANOperator2d:
    return ReluKANOperator2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        kan_module_constructor=kan_module_constructor,
    )


# -------------------------
# Blocks
# -------------------------

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
        conv3x3_factory: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if conv3x3_factory is None:
            conv3x3_factory = _conv2d_3x3

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3_factory(inplanes, planes, stride=stride, groups=groups, dilation=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_factory(planes, planes, stride=1, groups=groups, dilation=1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
        conv3x3_factory: Callable[..., nn.Module] | None = None,
        pointwise_factory: Callable[[int, int, int], nn.Module] | None = None,
    ) -> None:
        """
        pointwise_factory(in_planes, out_planes, stride) -> nn.Module
        Note: When pointwise is KAN, stride should be handled externally (AvgPool + stride=1).
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if conv3x3_factory is None:
            conv3x3_factory = _conv2d_3x3
        if pointwise_factory is None:
            pointwise_factory = lambda in_p, out_p, s: _conv2d_1x1(in_p, out_p, stride=s)

        if groups != 1:
            raise NotImplementedError("groups != 1 not supported with current setup")

        width = int(planes * (base_width / 64.0)) * groups

        # 1×1 reduce
        self.conv1 = pointwise_factory(inplanes, width, 1)
        self.bn1 = norm_layer(width)

        # 3×3 spatial
        self.conv2 = conv3x3_factory(width, width, stride=stride, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)

        # 1×1 expand
        self.conv3 = pointwise_factory(width, planes * self.expansion, 1)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


# -------------------------
# Backbone
# -------------------------

class KANResNetLaterStages(nn.Module):
    def __init__(
        self,
        block: type[BasicBlock] | type[Bottleneck],
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: list[bool] | None = None,
        norm_layer: Callable[..., nn.Module] | None = None,
        width_mult: float = 1.0,
        replace_kan_stages: tuple[int, ...] = (4,),
        kan_1x1_stages: tuple[int, ...] = (4,),
        kan_module_constructor: Callable[[int, int], nn.Module] = _default_kan_constructor,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        if groups != 1:
            raise NotImplementedError("groups != 1 not supported with current ReluKANOperator2d")

        self.groups = groups
        self.base_width = width_per_group
        self.width_mult = width_mult

        self.replace_kan_stages = tuple(sorted(set(replace_kan_stages)))
        self.kan_1x1_stages = tuple(sorted(set(kan_1x1_stages)))
        self.kan_module_constructor = kan_module_constructor

        self.inplanes = max(1, int(64 * width_mult))
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element list")

        stage_planes = [max(1, int(p * width_mult)) for p in (64, 128, 256, 512)]

        # conv1 stays standard
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(stage_idx=1, block=block, planes=stage_planes[0], blocks=layers[0])
        self.layer2 = self._make_layer(
            stage_idx=2,
            block=block,
            planes=stage_planes[1],
            blocks=layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            stage_idx=3,
            block=block,
            planes=stage_planes[2],
            blocks=layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            stage_idx=4,
            block=block,
            planes=stage_planes[3],
            blocks=layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_planes[3] * block.expansion, num_classes)

        # torchvision-like init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    # ---- stage-gated factories ----

    def _stage_conv3x3_factory(self, stage_idx: int) -> Callable[..., nn.Module]:
        if stage_idx in self.replace_kan_stages:
            return lambda in_p, out_p, stride=1, groups=1, dilation=1: _kan_conv2d_3x3(
                in_p,
                out_p,
                stride=stride,
                dilation=dilation,
                groups=groups,
                kan_module_constructor=self.kan_module_constructor,
            )
        return _conv2d_3x3

    def _stage_pointwise_factory(self, stage_idx: int) -> Callable[[int, int, int], nn.Module]:
        """
        Returns a factory: (in_planes, out_planes, stride) -> module

        If stage uses KAN 1×1 and stride != 1, caller should handle stride externally.
        For Bottleneck conv1/conv3, stride is always 1.
        """
        if stage_idx in self.kan_1x1_stages:
            return lambda in_p, out_p, stride: (
                KANPointwise2d(in_p, out_p, kan_module_constructor=self.kan_module_constructor)
                if stride == 1
                else nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride),
                    KANPointwise2d(in_p, out_p, kan_module_constructor=self.kan_module_constructor),
                )
            )
        return lambda in_p, out_p, stride: _conv2d_1x1(in_p, out_p, stride=stride)

    # ---- layer builder ----

    def _make_layer(
        self,
        stage_idx: int,
        block: type[BasicBlock] | type[Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        conv3x3_factory = self._stage_conv3x3_factory(stage_idx)
        pointwise_factory = self._stage_pointwise_factory(stage_idx)

        outplanes = planes * block.expansion

        if stride != 1 or self.inplanes != outplanes:
            # projection for skip connection
            downsample = nn.Sequential(
                pointwise_factory(self.inplanes, outplanes, stride),
                norm_layer(outplanes),
            )

        layers_list: list[nn.Module] = []

        if block is Bottleneck:
            layers_list.append(
                Bottleneck(
                    self.inplanes,
                    planes,
                    stride=stride,
                    downsample=downsample,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=previous_dilation,
                    norm_layer=norm_layer,
                    conv3x3_factory=conv3x3_factory,
                    pointwise_factory=pointwise_factory,
                )
            )
        else:
            # BasicBlock doesn't use internal 1×1s; only the downsample path might.
            layers_list.append(
                BasicBlock(
                    self.inplanes,
                    planes,
                    stride=stride,
                    downsample=downsample,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=previous_dilation,
                    norm_layer=norm_layer,
                    conv3x3_factory=conv3x3_factory,
                )
            )

        self.inplanes = outplanes

        for _ in range(1, blocks):
            if block is Bottleneck:
                layers_list.append(
                    Bottleneck(
                        self.inplanes,
                        planes,
                        stride=1,
                        downsample=None,
                        groups=self.groups,
                        base_width=self.base_width,
                        dilation=self.dilation,
                        norm_layer=norm_layer,
                        conv3x3_factory=conv3x3_factory,
                        pointwise_factory=pointwise_factory,
                    )
                )
            else:
                layers_list.append(
                    BasicBlock(
                        self.inplanes,
                        planes,
                        stride=1,
                        downsample=None,
                        groups=self.groups,
                        base_width=self.base_width,
                        dilation=self.dilation,
                        norm_layer=norm_layer,
                        conv3x3_factory=conv3x3_factory,
                    )
                )

        return nn.Sequential(*layers_list)

    # ---- forward ----

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)