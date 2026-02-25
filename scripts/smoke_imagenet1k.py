from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Ensure local project package resolution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.imagenet1k import get_dataloaders
from experiments.pretrain_imagenet import run_pretraining


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test ImageNet-1K dataloader and optional debug pretrain.")
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-debug-pretrain", action="store_true")
    parser.add_argument("--model", type=str, default="kan_resnet18_tiny")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_loader, val_loader, num_classes = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        seed=args.seed,
        data_root=args.data_root,
    )
    x, y = next(iter(train_loader))
    if tuple(x.shape[1:]) != (3, args.img_size, args.img_size):
        raise ValueError(f"Unexpected image shape: {tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise ValueError(f"Unexpected image dtype: {x.dtype}")
    if y.min().item() < 0 or y.max().item() >= 1000:
        raise ValueError(f"Label range out of bounds: min={y.min().item()} max={y.max().item()}")
    print(
        f"[OK] train batch shape={tuple(x.shape)} labels=[{int(y.min().item())},{int(y.max().item())}] "
        f"num_classes={num_classes}"
    )
    x_val, _ = next(iter(val_loader))
    if tuple(x_val.shape[1:]) != (3, args.img_size, args.img_size):
        raise ValueError(f"Unexpected val image shape: {tuple(x_val.shape)}")
    print(f"[OK] val batch shape={tuple(x_val.shape)}")

    if args.run_debug_pretrain:
        debug_dir = Path("outputs/imagenet_pretrain_debug") / args.model / f"seed{args.seed}"
        export_path = Path("weights/imagenet") / f"{args.model}_imagenet1k_debug.pt"
        cfg = {
            "model": args.model,
            "data_root": args.data_root,
            "img_size": args.img_size,
            "epochs": 1,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "lr": 0.1,
            "weight_decay": 1e-4,
            "momentum": 0.9,
            "label_smoothing": 0.0,
            "amp": False,
            "device": "auto",
            "seed": args.seed,
            "save_dir": str(debug_dir),
            "resume": None,
            "export_path": str(export_path),
            "train_aug": "standard",
            "deterministic": True,
            "debug": True,
        }
        run_pretraining(cfg)
        print("[OK] debug pretraining run completed.")


if __name__ == "__main__":
    main()
