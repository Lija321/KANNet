#Made with assitance of ChatGPT
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_dataloaders(
    batch_size: int,
    num_workers: int,
    img_size: int,
    seed: int,
    data_root: str,
    train_aug: str = "standard",
):
    from torchvision import datasets, transforms

    if train_aug != "standard":
        raise ValueError(f"Unsupported train_aug '{train_aug}'. Only 'standard' is supported.")

    root = Path(data_root)
    train_dir = root / "train"
    val_dir = root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"ImageNet-1K folder layout not found under '{data_root}'. "
            "Expected train/ and val/ directories."
        )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transform)

    generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, 1000
