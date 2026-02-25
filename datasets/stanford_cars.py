from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from .common import (
    make_dataloaders,
    resolve_data_root,
    split_train_val,
    with_fixed_random_resized_crop_224,
)


def _download_stanford_cars_from_hf(root: str) -> None:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    script = r"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
train_dir = root / "train"
test_dir = root / "test"
if train_dir.exists() and test_dir.exists():
    raise SystemExit(0)

from datasets import load_dataset

ds = load_dataset("tanganke/stanford_cars")

def dump_split(split_name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    split = ds[split_name]
    for idx, item in enumerate(split):
        label = int(item["label"])
        cls_dir = out_dir / f"{label:03d}"
        cls_dir.mkdir(parents=True, exist_ok=True)
        img = item["image"]
        img.save(cls_dir / f"{idx:06d}.jpg")

dump_split("train", train_dir)
dump_split("test", test_dir)
"""

    env = dict(os.environ)
    # Avoid importing this repo's local `datasets` package in subprocess.
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(root_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Failed to download Stanford Cars from HuggingFace.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}\n"
            "Install huggingface datasets package if missing: `pip install datasets`."
        )


def get_dataloaders(
    batch_size: int,
    num_workers: int,
    preprocess_train,
    preprocess_eval,
    seed: int,
    data_root: str = "data",
):
    preprocess_train = with_fixed_random_resized_crop_224(preprocess_train)
    preprocess_eval = with_fixed_random_resized_crop_224(preprocess_eval)

    root = resolve_data_root(data_root, "stanford_cars")
    tv_error: Exception | None = None

    # Preferred: torchvision StanfordCars API.
    try:
        from torchvision.datasets import StanfordCars

        train_full = StanfordCars(root=root, split="train", transform=preprocess_train, download=True)
        val_full = StanfordCars(root=root, split="train", transform=preprocess_eval, download=True)
        test_dataset = StanfordCars(root=root, split="test", transform=preprocess_eval, download=True)
        num_classes = 196
    except Exception as e:
        tv_error = e
        # Fallback: local ImageFolder layout at data/stanford_cars/{train,test}/class_x/*.jpg
        from torchvision.datasets import ImageFolder

        train_dir = Path(root) / "train"
        test_dir = Path(root) / "test"
        if not train_dir.exists() or not test_dir.exists():
            _download_stanford_cars_from_hf(root)

        if not train_dir.exists() or not test_dir.exists():
            raise RuntimeError(
                "Stanford Cars unavailable. torchvision StanfordCars failed and no local "
                "ImageFolder fallback found.\n"
                f"Expected folders: '{train_dir}' and '{test_dir}'.\n"
                f"Original torchvision error: {tv_error}"
            ) from e
        train_full = ImageFolder(root=str(train_dir), transform=preprocess_train)
        val_full = ImageFolder(root=str(train_dir), transform=preprocess_eval)
        test_dataset = ImageFolder(root=str(test_dir), transform=preprocess_eval)
        num_classes = len(train_full.classes)

    train_dataset, val_dataset = split_train_val(train_full, val_full, seed=seed, val_ratio=0.1)
    train_loader, val_loader, test_loader = make_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    return train_loader, val_loader, test_loader, num_classes
