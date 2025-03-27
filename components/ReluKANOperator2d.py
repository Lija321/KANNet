import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from components.ReluKANLayer import ReluKANLayer


# Custom Conv2D module that uses a KAN module (here, the ReluKANLayer) per output channel.
class ReluKANOperator2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 g=None, k=None, kan_module_constructor=None):
        """\
        Parameters:
          - in_channels: Number of channels in the input.
          - out_channels: Number of output channels (each will have its own KAN module).
          - kernel_size: Kernel size (int or tuple).
          - stride, padding, dilation: Convolution parameters.
          - groups: Not used in this basic implementation.
          - bias: Whether to add a learnable bias.
          - g, k: Parameters for ReluKANLayer (if using default constructor).
          - kan_module_constructor: Optional callable that accepts the flattened patch size and returns a KAN module.
        """
        super(ReluKANOperator2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Ensure kernel_size, stride, padding, dilation are tuples.
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        
        # The flattened patch size: in_channels * kernel_height * kernel_width.
        self.patch_size = in_channels * self.kernel_size[0] * self.kernel_size[1]
        
        # Use the provided kan_module_constructor or default to one that uses ReluKANLayer.
        if kan_module_constructor is None:
            if g is None or k is None:
                raise ValueError("Provide g and k parameters for the default ReluKANLayer constructor")
            def default_kan_module_constructor(in_features):
                # Each KAN module converts a flattened patch to a scalar (output_size=1).
                return ReluKANLayer(input_size=in_features, output_size=1, g=g, k=k)
            kan_module_constructor = default_kan_module_constructor

        # Create one KAN module per output channel.
        self.kan_modules = nn.ModuleList(
            [kan_module_constructor(self.patch_size) for _ in range(out_channels)]
        )
        
        # Optional bias per output channel.
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        # x: (B, in_channels, H, W)
        B, C, H, W = x.shape
        
        # Extract sliding patches; shape: (B, patch_size, L) where L is the number of patches.
        patches = F.unfold(x, kernel_size=self.kernel_size, dilation=self.dilation,
                           padding=self.padding, stride=self.stride)
        # Rearrange to (B, L, patch_size)
        patches = patches.transpose(1, 2)
        B, L, patch_size = patches.shape
        
        # Flatten the patches to shape (B*L, patch_size) for processing.
        patches_reshaped = patches.reshape(B * L, patch_size)
        
        outputs = []
        for kan in self.kan_modules:
            # Each KAN module processes the flattened patches.
            # Expected output shape from ReluKANLayer: (B*L, 1, 1)
            out = kan(patches_reshaped)
            # Reshape to (B, L)
            out = out.view(B, L)
            outputs.append(out)
        
        # Stack along a new channel dimension: (B, out_channels, L)
        out_tensor = torch.stack(outputs, dim=1)
        
        # Calculate output spatial dimensions.
        H_out = (H + 2*self.padding[0] - self.dilation[0]*(self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        W_out = (W + 2*self.padding[1] - self.dilation[1]*(self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        
        # Reshape to (B, out_channels, H_out, W_out)
        out_tensor = out_tensor.view(B, self.out_channels, H_out, W_out)
        
        if self.bias is not None:
            out_tensor = out_tensor + self.bias.view(1, -1, 1, 1)
            
        return out_tensor