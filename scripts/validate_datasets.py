from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path
from typing import Any

import torch

# Ensure local project imports win even if PYTHONPATH has another package with same names.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import DATASET_REGISTRY
from models import MODEL_REGISTRY, PREPROCESS_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate all datasets in DATASET_REGISTRY.")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--max-batches", type=int, default=2)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def default_preprocess(img_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def build_dataset_kwargs(
    dataset_fn: Any,
    dataset_name: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    data_root: str,
) -> dict[str, Any]:
    sig = inspect.signature(dataset_fn)
    params = sig.parameters
    kwargs: dict[str, Any] = {}

    if "batch_size" in params:
        kwargs["batch_size"] = batch_size
    if "num_workers" in params:
        kwargs["num_workers"] = num_workers
    if "img_size" in params:
        kwargs["img_size"] = img_size
    if "seed" in params:
        kwargs["seed"] = seed
    if "data_root" in params:
        kwargs["data_root"] = data_root
    if "root" in params:
        kwargs["root"] = data_root

    if "preprocess_train" in params or "preprocess_eval" in params:
        preprocess_fn = PREPROCESS_REGISTRY.get("resnet18")
        if preprocess_fn is not None:
            preprocess = preprocess_fn(pretrained=False, img_size=img_size)
            train_transform = preprocess["train"]
            eval_transform = preprocess["eval"]
        else:
            train_transform = default_preprocess(img_size)
            eval_transform = default_preprocess(img_size)
        if "preprocess_train" in params:
            kwargs["preprocess_train"] = train_transform
        if "preprocess_eval" in params:
            kwargs["preprocess_eval"] = eval_transform

    return kwargs


def instantiate_model(model_name: str, num_classes: int) -> torch.nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY.keys())}")
    factory = MODEL_REGISTRY[model_name]
    sig = inspect.signature(factory)
    params = sig.parameters
    kwargs: dict[str, Any] = {}

    if "num_classes" in params:
        kwargs["num_classes"] = num_classes
    if "pretrained" in params:
        kwargs["pretrained"] = False

    try:
        return factory(**kwargs)
    except TypeError:
        kwargs.pop("pretrained", None)
        return factory(**kwargs)


def shape_and_label_checks(
    x: torch.Tensor,
    y: torch.Tensor,
    img_size: int,
    num_classes: int,
) -> tuple[bool, bool, str]:
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        return False, False, "batch is not tensor pair"
    if x.ndim != 4 or y.ndim != 1:
        return False, False, f"unexpected rank x={x.shape}, y={y.shape}"
    if x.shape[1:] != (3, img_size, img_size):
        return False, False, f"unexpected x shape {tuple(x.shape)} expected (B,3,{img_size},{img_size})"
    if y.shape[0] != x.shape[0]:
        return False, False, f"batch size mismatch x={x.shape[0]} y={y.shape[0]}"
    if x.dtype != torch.float32:
        return False, False, f"unexpected x dtype {x.dtype}"
    if not torch.isfinite(x).all():
        return False, False, "x contains NaN/Inf"
    y_min = int(y.min().item())
    y_max = int(y.max().item())
    labels_ok = y_min >= 0 and y_max < num_classes
    if not labels_ok:
        return True, False, f"label range invalid min={y_min} max={y_max} num_classes={num_classes}"
    return True, True, "ok"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dataset_names = sorted(DATASET_REGISTRY.keys())
    shape_or_label_failed = False

    for dataset_name in dataset_names:
        shape_status = "n/a"
        label_status = "n/a"
        forward_status = "n/a"
        try:
            dataset_fn = DATASET_REGISTRY[dataset_name]
            kwargs = build_dataset_kwargs(
                dataset_fn=dataset_fn,
                dataset_name=dataset_name,
                img_size=args.img_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
                data_root=args.data_root,
            )
            loaded = dataset_fn(**kwargs)
            if not isinstance(loaded, tuple):
                raise TypeError(f"{dataset_name} returned non-tuple result: {type(loaded)}")
            if len(loaded) == 4:
                train_loader, val_loader, test_loader, num_classes = loaded
            elif len(loaded) == 3:
                train_loader, val_loader, num_classes = loaded
                test_loader = val_loader
            else:
                raise ValueError(f"{dataset_name} returned {len(loaded)} values; expected 3 or 4.")
            train_len = len(train_loader.dataset)
            val_len = len(val_loader.dataset)
            test_len = len(test_loader.dataset)

            batch_iter = iter(train_loader)
            x, y = next(batch_iter)
            shape_ok, labels_ok, reason = shape_and_label_checks(
                x=x,
                y=y,
                img_size=args.img_size,
                num_classes=num_classes,
            )

            if not shape_ok:
                shape_status = "FAIL"
                label_status = "n/a"
                shape_or_label_failed = True
                print(
                    f"{dataset_name} | classes={num_classes} | train={train_len} | "
                    f"val={val_len} | test={test_len} | shape FAIL ({reason}) | labels n/a | forward {forward_status}"
                )
                continue

            shape_status = "OK"
            if not labels_ok:
                label_status = "FAIL"
                shape_or_label_failed = True
                print(
                    f"{dataset_name} | classes={num_classes} | train={train_len} | "
                    f"val={val_len} | test={test_len} | shape OK | labels FAIL ({reason}) | forward {forward_status}"
                )
                continue
            label_status = "OK"

            timings: list[float] = []
            train_iter = iter(train_loader)
            for _ in range(max(1, args.max_batches)):
                t0 = time.perf_counter()
                try:
                    _x, _y = next(train_iter)
                    _ = (_x, _y)
                except StopIteration:
                    break
                timings.append(time.perf_counter() - t0)
            avg_load_sec = sum(timings) / len(timings) if timings else 0.0

            if args.forward:
                try:
                    model = instantiate_model(args.model, num_classes=num_classes).to(device)
                    model.eval()
                    xb = x.to(device)
                    with torch.no_grad():
                        logits = model(xb)
                    if tuple(logits.shape) == (x.shape[0], num_classes):
                        forward_status = "OK"
                    else:
                        forward_status = f"FAIL(shape={tuple(logits.shape)})"
                except Exception as exc:
                    forward_status = f"SKIP({exc})"

            print(
                f"{dataset_name} | classes={num_classes} | train={train_len} | val={val_len} | "
                f"test={test_len} | shape {shape_status} | labels {label_status} | "
                f"forward {forward_status} | avg_load_s={avg_load_sec:.4f}"
            )
        except Exception as exc:
            print(f"[FAILED] {dataset_name} : {exc}")
            continue

    if shape_or_label_failed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
