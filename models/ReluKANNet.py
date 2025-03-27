import torch
import torch.nn as nn
from components.ReluKANOperator2d import ReluKANOperator2d


class ReluKANBlock(nn.Module):

    def __init__(self, in_channels, out_channels, g, k, stride=1):
        super(ReluKANBlock, self).__init__()
        self.layer = ReluKANOperator2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, g=g, k=k)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.layer(x)
        x = self.bn(x)
        return x


class ReluKANNetB0(nn.Module):

    def __init__(self, 
                 in_channels,
                 num_classes,
                 g=3,
                 k=3,
                 depth_list=[1, 2, 2, 3, 3, 4, 1],
                 base_channels=32,
                 ):
        super(ReluKANNetB0, self).__init__()

        self.depth_list = depth_list
        self.stem = ReluKANBlock(in_channels, base_channels, g, k, stride=1)

        self.stages = nn.ModuleList()
        input_channels = base_channels
        channel_configs = [16, 24, 40, 80, 112, 192, 320]
        for stage, repeats in enumerate(depth_list):
            stage_layers = []
            for i in range(repeats):
                stride = 2 if i == 0 and stage != 0 else 1
                stage_layers.append(ReluKANBlock(input_channels, channel_configs[stage], g, k, stride=stride))
                input_channels = channel_configs[stage]
            self.stages.append(nn.Sequential(*stage_layers))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(input_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
        