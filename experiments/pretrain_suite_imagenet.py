#Made with assitance of ChatGPT
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .pretrain_imagenet import run_pretraining


DEFAULT_MODELS = [
    "kan_resnet18_tiny",
    "kan_resnet18_small",
    "kan_resnet18_base",
    "kan_resnet50_tiny",
    "kan_resnet50_small",
    "kan_resnet50_base",
]


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _str2bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ImageNet-1K pretraining suite for KAN models.")
    parser.add_argument("--models", type=_csv_list, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", type=_csv_ints, default="0")
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--amp", type=_str2bool, default=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--train-aug", type=str, default="standard")
    parser.add_argument("--deterministic", type=_str2bool, default=True)
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--no-ips", action="store_true")
    parser.add_argument("--tqdm-update-every", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--export-root", type=str, default="weights/imagenet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = args.models if isinstance(args.models, list) else _csv_list(args.models)
    seeds = args.seeds if isinstance(args.seeds, list) else _csv_ints(args.seeds)

    out_root = Path("outputs/imagenet_pretrain")
    out_root.mkdir(parents=True, exist_ok=True)
    export_root = Path(args.export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int | float]] = []
    for model_name in models:
        for seed in seeds:
            save_dir = out_root / model_name / f"seed{seed}"
            export_path = export_root / f"{model_name}_imagenet1k_seed{seed}.pt"
            cfg = {
                "model": model_name,
                "data_root": args.data_root,
                "img_size": args.img_size,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "momentum": args.momentum,
                "label_smoothing": args.label_smoothing,
                "amp": args.amp,
                "device": args.device,
                "seed": seed,
                "save_dir": str(save_dir),
                "resume": args.resume,
                "export_path": str(export_path),
                "train_aug": args.train_aug,
                "deterministic": args.deterministic,
                "use_tqdm": not args.no_tqdm,
                "show_ips": not args.no_ips,
                "tqdm_update_every": args.tqdm_update_every,
                "debug": args.debug,
            }
            print(f"[RUN] model={model_name} seed={seed}")
            result = run_pretraining(cfg)
            rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "best_top1": result["best_top1"],
                    "best_top5": result["best_top5"],
                    "best_epoch": result["best_epoch"],
                    "total_time_sec": result["total_time_sec"],
                    "export_path": result["export_path"],
                }
            )

    summary_path = out_root / "suite_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "seed",
                "best_top1",
                "best_top5",
                "best_epoch",
                "total_time_sec",
                "export_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote suite summary: {summary_path}")


if __name__ == "__main__":
    main()
