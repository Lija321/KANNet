#Made with assitance of ChatGPT
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class TinyConvNet(nn.Module):
    def __init__(self, num_classes: int, width: int = 32) -> None:
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

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c3, num_classes)

        # init (simple + standard)
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
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

#Made with assitance of ChatGPT
def get_model(
    num_classes: int,
    pretrained: bool = False,
    width: int = 32,
) -> nn.Module:
    if pretrained:
        raise ValueError("TinyConvNet does not provide pretrained weights.")
    return TinyConvNet(num_classes=num_classes, width=width)