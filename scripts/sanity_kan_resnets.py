#Made with assitance of ChatGPT
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Ensure local project package resolution even if PYTHONPATH has another `models`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import MODEL_REGISTRY


def main() -> None:
    names = [
        "kan_resnet18_tiny",
        "kan_resnet18_small",
        "kan_resnet18_base",
        "kan_resnet50_tiny",
        "kan_resnet50_small",
        "kan_resnet50_base",
    ]
    x = torch.randn(2, 3, 224, 224)
    for name in names:
        model = MODEL_REGISTRY[name](num_classes=10, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10), f"{name} produced shape {tuple(out.shape)} instead of (2, 10)"
        print(f"[OK] {name}: {tuple(out.shape)}")


if __name__ == "__main__":
    main()
