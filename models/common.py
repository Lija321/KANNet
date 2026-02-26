#Made with assitance of ChatGPT
from __future__ import annotations

from typing import Any

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_preprocess(weights: Any, pretrained: bool = True, img_size: int = 224) -> dict[str, Any]:
    """Build train/eval transforms with ImageNet normalization."""
    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    if pretrained:
        eval_transform = weights.transforms(crop_size=img_size)
    else:
        resize_size = int(img_size * 256 / 224)
        eval_transform = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return {"train": train_transform, "eval": eval_transform}
