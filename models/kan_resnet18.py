"""KAN-ResNet18 wrapper for Conv2d replacement studies (stage-gated 3×3 and 1×1).

Pretrained workflow:
- Pretrain with `experiments.pretrain_imagenet`.
- Load with `pretrained=True` and `pretrained_path=...` (or env fallback),
  then replace the classification head for target `num_classes`.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from components.ReluKANOperator2d import ReluKANOperator2d  # explicit dependency
from .kan_resnet_common import BasicBlock, KANResNetLaterStages


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
    replace_kan_stages: tuple[int, ...] = (4,),
    kan_1x1_stages: tuple[int, ...] = (4,),
) -> nn.Module:
    if variant not in _VARIANT_TO_WIDTH:
        raise ValueError(f"Unknown variant '{variant}'. Expected one of: {sorted(_VARIANT_TO_WIDTH)}")

    _ = ReluKANOperator2d  # keep dependency visible

    model = KANResNetLaterStages(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1000 if pretrained else num_classes,
        width_mult=_VARIANT_TO_WIDTH[variant],
        replace_kan_stages=replace_kan_stages,
        kan_1x1_stages=kan_1x1_stages,
    )

    if pretrained:
        resolved = _resolve_pretrained_path(pretrained_path)
        if not resolved:
            raise ValueError(
                "pretrained=True requires `pretrained_path` or env var "
                "`KAN_RESNET18_PRETRAINED_PATH` / `KAN_PRETRAINED_PATH`."
            )
        checkpoint = torch.load(resolved, map_location="cpu")
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=True)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    return model