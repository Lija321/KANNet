import torch
import torch.nn as nn
import numpy as np


class ReluKANLayer(nn.Module):

    def __init__(self, input_size: int, output_size: int, g: int, k: int,  is_train: bool = False):
        super().__init__()

        self.g, self.k, self.r = g, k, 4*g*g / ((k+1)*(k+1))
        self.input_size, self.output_size = input_size, output_size
        phase_low = np.arange(-k, g) / g
        phase_height = phase_low + (k+1) / g
        self.phase_low = nn.Parameter(torch.Tensor(np.array([phase_low for i in range(input_size)])), requires_grad=is_train)
        self.phase_height = nn.Parameter(torch.Tensor(np.array([phase_height for i in range(input_size)])), requires_grad=is_train)
        self.equal_size_conv = nn.Conv2d(1, output_size, (g+k, input_size))
    
    def forward(self, x):
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.phase_low.size(1))

        x1 = torch.relu(x_expanded - self.phase_low)
        x2 = torch.relu(self.phase_height - x_expanded)
        x = x1 * x2 * self.r
        x = x * x
        x = x.reshape((len(x), 1, self.g + self.k, self.input_size))
        x = self.equal_size_conv(x)
        x = x.reshape((len(x), self.output_size, 1))
        return x