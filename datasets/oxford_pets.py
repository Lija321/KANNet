#Made with assitance of ChatGPT
from __future__ import annotations

from .common import (
    make_dataloaders,
    resolve_data_root,
    split_train_val,
    with_fixed_random_resized_crop_224,
)


def get_dataloaders(
    batch_size: int,
    num_workers: int,
    preprocess_train,
    preprocess_eval,
    seed: int,
    data_root: str = "data",
):
    from torchvision.datasets import OxfordIIITPet

    preprocess_train = with_fixed_random_resized_crop_224(preprocess_train)
    preprocess_eval = with_fixed_random_resized_crop_224(preprocess_eval)

    root = resolve_data_root(data_root, "oxford_pets")
    train_full = OxfordIIITPet(root=root, split="trainval", transform=preprocess_train, download=True)
    val_full = OxfordIIITPet(root=root, split="trainval", transform=preprocess_eval, download=True)
    test_dataset = OxfordIIITPet(root=root, split="test", transform=preprocess_eval, download=True)
    train_dataset, val_dataset = split_train_val(train_full, val_full, seed=seed, val_ratio=0.1)
    train_loader, val_loader, test_loader = make_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    return train_loader, val_loader, test_loader, 37
