from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset


def split_train_val(
    train_dataset: Dataset,
    val_dataset: Dataset,
    seed: int,
    val_ratio: float = 0.1,
) -> tuple[Subset, Subset]:
    n = len(train_dataset)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n, generator=generator).tolist()
    train_indices = permutation[:n_train]
    val_indices = permutation[n_train:]
    return Subset(train_dataset, train_indices), Subset(val_dataset, val_indices)


def make_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def resolve_data_root(data_root: str, dataset_name: str) -> str:
    return f"{data_root.rstrip('/')}/{dataset_name}"


def with_fixed_random_resized_crop_224(preprocess: Any):
    """Prepend RandomResizedCrop(224) to any incoming preprocess pipeline."""
    from torchvision import transforms

    fixed_crop = transforms.RandomResizedCrop(224)
    if isinstance(preprocess, transforms.Compose):
        return transforms.Compose([fixed_crop, *preprocess.transforms])
    return transforms.Compose([fixed_crop, preprocess])
