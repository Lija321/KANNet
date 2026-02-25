"""KAN-ResNet18 wrapper for Conv2d replacement studies.

This is a Conv2d replacement study using ReluKANOperator2d.
Pretrained workflow:
- Pretrain with `experiments.pretrain_imagenet`.
- Load with `pretrained=True` and `pretrained_path=...` (or env fallback),
  then the classification head is replaced for target `num_classes`.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from components.ReluKANOperator2d import ReluKANOperator2d

from .kan_resnet_common import BasicBlock, KANResNet


_VARIANT_TO_WIDTH = {
    "tiny": 0.50,
    "small": 0.75,
    "base": 1.00,
}


def _resolve_pretrained_path(pretrained_path: str | None) -> str | None:
    if pretrained_path:
        return pretrained_path
    return os.getenv("KAN_RESNET18_PRETRAINED_PATH") or os.getenv("KAN_PRETRAINED_PATH")


def get_model(
    num_classes: int,
    pretrained: bool = False,
    variant: str = "base",
    pretrained_path: str | None = None,
) -> nn.Module:
    if variant not in _VARIANT_TO_WIDTH:
        raise ValueError(f"Unknown variant '{variant}'. Expected one of: {sorted(_VARIANT_TO_WIDTH)}")

    _ = ReluKANOperator2d  # Keep explicit dependency visible in this file.
    model = KANResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1000 if pretrained else num_classes,
        width_mult=_VARIANT_TO_WIDTH[variant],
    )
    if pretrained:
        resolved = _resolve_pretrained_path(pretrained_path)
        if not resolved:
            raise ValueError(
                "pretrained=True requires `pretrained_path` or env var "
                "`KAN_RESNET18_PRETRAINED_PATH` / `KAN_PRETRAINED_PATH`."
            )
        checkpoint = torch.load(resolved, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model
