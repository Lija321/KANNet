from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# Ensure local project imports win even if PYTHONPATH has another `models`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import MODEL_REGISTRY


def parse_kan_targets(value: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    if not value.strip():
        return mapping
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid kan target entry '{item}', expected name=value")
        name, raw_target = item.split("=", 1)
        mapping[name.strip()] = int(raw_target.strip())
    return mapping


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def instantiate_model(factory: Any, num_classes: int) -> nn.Module:
    sig = inspect.signature(factory)
    params = sig.parameters
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    kwargs: dict[str, Any] = {}
    if "num_classes" in params or accepts_var_kwargs:
        kwargs["num_classes"] = num_classes
    if "pretrained" in params or accepts_var_kwargs:
        kwargs["pretrained"] = False

    try:
        return factory(**kwargs)
    except Exception:
        # Retry path for odd callables that may reject one of the kwargs.
        if "pretrained" in kwargs:
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("pretrained", None)
            return factory(**retry_kwargs)
        raise


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def flops_gmacs(model: nn.Module, img_size: int, device: torch.device) -> float | None:
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        total_flops = FlopCountAnalysis(model, x).total()
    return float(total_flops / 1e9)


def measure_throughput(
    model: nn.Module,
    device: torch.device,
    img_size: int,
    warmup_batches: int,
    timed_batches: int,
    amp: bool,
    batch_size: int = 2,
) -> float | None:
    if device.type != "cuda":
        return None
    model.eval()
    enabled_amp = amp
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(warmup_batches):
            with torch.autocast(device_type="cuda", enabled=enabled_amp):
                _ = model(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(timed_batches):
            with torch.autocast(device_type="cuda", enabled=enabled_amp):
                _ = model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return None
    return float((batch_size * timed_batches) / elapsed)


def format_metric(value: float | int | None, float_fmt: str = ".3f") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return format(value, float_fmt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate all models in MODEL_REGISTRY.")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--timed-batches", type=int, default=30)
    parser.add_argument("--check-kan-budgets", action="store_true")
    parser.add_argument(
        "--kan-targets",
        type=str,
        default="kan_resnet18_tiny=3500000,kan_resnet18_small=8000000,kan_resnet50_base=25000000",
    )
    parser.add_argument("--tol", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    names = sorted(MODEL_REGISTRY.keys())
    criterion = nn.CrossEntropyLoss()

    failures: list[str] = []
    results: dict[str, dict[str, Any]] = {}

    print(f"Validating {len(names)} models on device={device}")
    for name in names:
        total_params: int | None = None
        trainable_params: int | None = None
        flops: float | None = None
        throughput: float | None = None
        shape_ok = False
        try:
            model = instantiate_model(MODEL_REGISTRY[name], num_classes=args.num_classes)
            model = model.to(device)
            total_params, trainable_params = count_params(model)

            x = torch.randn(2, 3, args.img_size, args.img_size, device=device)
            y = torch.randint(0, args.num_classes, (2,), device=device)

            model.train(mode=args.backward)
            logits = model(x)
            if tuple(logits.shape) != (2, args.num_classes):
                raise ValueError(f"unexpected logits shape: {tuple(logits.shape)}")
            shape_ok = True

            if args.backward:
                model.zero_grad(set_to_none=True)
                loss = criterion(logits, y)
                loss.backward()

            flops = flops_gmacs(model, img_size=args.img_size, device=device)
            if args.throughput:
                throughput = measure_throughput(
                    model=model,
                    device=device,
                    img_size=args.img_size,
                    warmup_batches=args.warmup_batches,
                    timed_batches=args.timed_batches,
                    amp=args.amp,
                )
        except Exception as exc:
            failures.append(f"{name}: {exc}")

        print(
            f"{name} | total_params={format_metric(total_params)} | "
            f"trainable_params={format_metric(trainable_params)} | "
            f"flops_gmacs={format_metric(flops)} | "
            f"throughput_img_s={format_metric(throughput)} | "
            f"output_shape={'OK' if shape_ok else 'FAIL'}"
        )
        results[name] = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "flops_gmacs": flops,
            "throughput_img_s": throughput,
            "output_shape_ok": shape_ok,
        }

    if failures:
        print("\nModel validation failures:")
        for msg in failures:
            print(f"- {msg}")
        sys.exit(1)

    if args.check_kan_budgets:
        budget_failures: list[str] = []
        targets = parse_kan_targets(args.kan_targets)
        for name, target in targets.items():
            if name not in results:
                budget_failures.append(f"{name}: missing from MODEL_REGISTRY")
                continue
            actual = results[name]["total_params"]
            if actual is None:
                budget_failures.append(f"{name}: no parameter count available")
                continue
            lo = int(target * (1.0 - args.tol))
            hi = int(target * (1.0 + args.tol))
            if not (lo <= actual <= hi):
                budget_failures.append(
                    f"{name}: actual={actual} outside [{lo}, {hi}] around target={target}"
                )

        if budget_failures:
            print("\nKAN budget check failures:")
            for msg in budget_failures:
                print(f"- {msg}")
            sys.exit(2)

    print("\nAll model validations passed.")


if __name__ == "__main__":
    main()
