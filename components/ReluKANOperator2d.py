#The Operator Was made with assitance from ChatGPT

import torch
import torch.nn as nn
import torch.nn.functional as F


from components.efficient_kan import KANLinear


class ReluKANOperator2d(nn.Module):
    """
    Conv2D-like operator implemented as:
      unfold patches -> KANLinear(patch_size -> out_channels) -> reshape

    This is equivalent to having one independent "kernel" per output channel,
    but computed in a single vectorized pass (no Python loop over channels).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        kan_module_constructor=None,
    ):
        super().__init__()

        if groups != 1:
            raise NotImplementedError(
                "groups != 1 is not supported in this operator yet (would require grouped KANs)."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Normalize params to tuples
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

        kH, kW = self.kernel_size
        self.patch_size = in_channels * kH * kW

        # Default: one KANLinear producing all out_channels at once.
        # This is analogous to Conv2d weight shape (out_channels, in_channels, kH, kW).
        if kan_module_constructor is None:
            self.kan = KANLinear(self.patch_size, out_channels)
        else:
            # Expect constructor signature: (in_features, out_features) -> module
            self.kan = kan_module_constructor(self.patch_size, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"

        # Extract sliding patches: (B, patch_size, L)
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
        # (B, L, patch_size)
        patches = patches.transpose(1, 2).contiguous()
        B, L, P = patches.shape
        assert P == self.patch_size

        # Flatten to (B*L, patch_size)
        patches_flat = patches.view(B * L, P)

        # Vectorized KAN: (B*L, out_channels)
        out = self.kan(patches_flat)

        # Reshape back to (B, out_channels, H_out, W_out)
        # Compute output spatial dims (same as conv formula)
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation

        H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1

        out = out.view(B, L, self.out_channels).permute(0, 2, 1).contiguous()
        out = out.view(B, self.out_channels, H_out, W_out)
        return out