#Made with assitance of ChatGPT

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

from components.ReluKANOperator2d import ReluKANOperator2d
from components.efficient_kan import KANLinear


def _default_kan_constructor(in_features: int, out_features: int) -> nn.Module:
    # Vectorized: one KANLinear outputs all out_channels at once.
    return KANLinear(in_features, out_features)


class TinyKANNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        width: int = 32,
        kan_module_constructor: Callable[[int, int], nn.Module] = _default_kan_constructor,
        apply_kan_at_8x8: bool = True,
    ) -> None:
        super().__init__()
        c1 = width
        c2 = width * 2
        c3 = width * 4

        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),  # 32 -> 16
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),  # 16 -> 8
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )

        if apply_kan_at_8x8:
            self.kan = nn.Sequential(
                ReluKANOperator2d(
                    in_channels=c3,
                    out_channels=c3,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=1,
                    groups=1,
                    kan_module_constructor=kan_module_constructor,
                ),
                nn.BatchNorm2d(c3),
                nn.ReLU(inplace=True),
            )
        else:
            # fallback: standard conv at 8×8
            self.kan = nn.Sequential(
                nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(c3),
                nn.ReLU(inplace=True),
            )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c3, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.kan(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_model(
    num_classes: int,
    pretrained: bool = False,
    width: int = 32,
    apply_kan_at_8x8: bool = True,
) -> nn.Module:
    if pretrained:
        raise ValueError("TinyKANNet does not provide pretrained weights.")
    # keep explicit dependency visible (helps packagers/linters)
    _ = ReluKANOperator2d
    return TinyKANNet(num_classes=num_classes, width=width, apply_kan_at_8x8=apply_kan_at_8x8)