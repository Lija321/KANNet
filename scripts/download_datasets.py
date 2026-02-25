from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

# Ensure local project imports win even if PYTHONPATH contains same package names.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import DATASET_REGISTRY
from models import PREPROCESS_REGISTRY


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/cache all datasets in DATASET_REGISTRY.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--datasets",
        type=_csv_list,
        default=",".join(sorted(DATASET_REGISTRY.keys())),
        help="Comma-separated dataset list. Default: all registered datasets.",
    )
    return parser.parse_args()


def build_kwargs(
    fn: Any,
    dataset_name: str,
    args: argparse.Namespace,
    preprocess_train: Any,
    preprocess_eval: Any,
) -> dict[str, Any]:
    sig = inspect.signature(fn)
    params = sig.parameters
    kwargs: dict[str, Any] = {}

    if "batch_size" in params:
        kwargs["batch_size"] = args.batch_size
    if "num_workers" in params:
        kwargs["num_workers"] = args.num_workers
    if "img_size" in params:
        kwargs["img_size"] = args.img_size
    if "seed" in params:
        kwargs["seed"] = args.seed
    if "data_root" in params:
        kwargs["data_root"] = args.data_root
    if "root" in params:
        kwargs["root"] = args.data_root
    if "preprocess_train" in params:
        kwargs["preprocess_train"] = preprocess_train
    if "preprocess_eval" in params:
        kwargs["preprocess_eval"] = preprocess_eval
    if "train_aug" in params:
        kwargs["train_aug"] = "standard"

    return kwargs


def main() -> None:
    args = parse_args()
    requested = args.datasets if isinstance(args.datasets, list) else _csv_list(args.datasets)
    preprocess = PREPROCESS_REGISTRY["resnet18"](pretrained=False, img_size=args.img_size)
    preprocess_train = preprocess["train"]
    preprocess_eval = preprocess["eval"]

    for name in requested:
        if name not in DATASET_REGISTRY:
            print(f"[SKIP] {name}: not in DATASET_REGISTRY")
            continue

        if name == "imagenet1k":
            print(
                "[SKIP] imagenet1k: no auto-download. Place data manually as "
                "<data-root>/train/<class>/* and <data-root>/val/<class>/*"
            )
            continue

        fn = DATASET_REGISTRY[name]
        kwargs = build_kwargs(
            fn=fn,
            dataset_name=name,
            args=args,
            preprocess_train=preprocess_train,
            preprocess_eval=preprocess_eval,
        )
        print(f"[DOWNLOAD] {name} -> {args.data_root}")
        try:
            _ = fn(**kwargs)
            print(f"[OK] {name}")
        except Exception as exc:
            print(f"[FAILED] {name}: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
