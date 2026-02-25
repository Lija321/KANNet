from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from datasets import DATASET_REGISTRY
from models import MODEL_REGISTRY

from .run_one import run_one


def _str2bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model x dataset x seed experiment suite.")
    parser.add_argument("--models", type=_csv_list, default=",".join(sorted(MODEL_REGISTRY.keys())))
    default_datasets = [name for name in sorted(DATASET_REGISTRY.keys()) if name != "imagenet1k"]
    parser.add_argument("--datasets", type=_csv_list, default=",".join(default_datasets))
    parser.add_argument("--seeds", type=_csv_ints, default="0")
    parser.add_argument("--pretrained", type=_str2bool, default=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", type=_str2bool, default=True)
    parser.add_argument("--deterministic", type=_str2bool, default=True)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-root", type=str, default="outputs/runs")
    parser.add_argument("--save-weight-models", type=_csv_list, default="kannet*")
    parser.add_argument("--weights-root", type=str, default="outputs/weights")
    parser.add_argument("--throughput-warmup", type=int, default=10)
    parser.add_argument("--throughput-batches", type=int, default=30)
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--no-ips", action="store_true")
    parser.add_argument("--tqdm-update-every", type=int, default=10)
    return parser.parse_args()


def _aggregate_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_keys = [
        "test_top1",
        "test_top5",
        "test_roc_auc_macro",
        "param_count",
        "flops_gmacs",
        "throughput_images_per_sec",
        "total_train_time_sec",
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[(row["dataset"], row["model"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (dataset, model), rows in sorted(grouped.items()):
        out: dict[str, Any] = {"dataset": dataset, "model": model, "num_seeds": len(rows)}
        for key in metric_keys:
            values = [float(r[key]) for r in rows if r[key] is not None]
            if not values:
                out[f"{key}_mean"] = None
                out[f"{key}_std"] = None
                out[f"{key}_mean_std"] = "NA"
            else:
                m = mean(values)
                s = pstdev(values) if len(values) > 1 else 0.0
                out[f"{key}_mean"] = m
                out[f"{key}_std"] = s
                out[f"{key}_mean_std"] = f"{m:.6f}±{s:.6f}"
        summary_rows.append(out)
    return summary_rows


def _write_summary_csv(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    if not summary_rows:
        return
    fieldnames = list(summary_rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    args = parse_args()
    models = args.models if isinstance(args.models, list) else _csv_list(args.models)
    datasets = args.datasets if isinstance(args.datasets, list) else _csv_list(args.datasets)
    seeds = args.seeds if isinstance(args.seeds, list) else _csv_ints(args.seeds)

    for model in models:
        if model not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model}'. Available: {sorted(MODEL_REGISTRY.keys())}")
    for dataset in datasets:
        if dataset not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset '{dataset}'. Available: {sorted(DATASET_REGISTRY.keys())}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cfg = {
        "pretrained": args.pretrained,
        "img_size": args.img_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "amp": args.amp,
        "deterministic": args.deterministic,
        "data_root": args.data_root,
        "output_root": str(output_root),
        "save_weight_models": args.save_weight_models,
        "weights_root": args.weights_root,
        "throughput_warmup": args.throughput_warmup,
        "throughput_batches": args.throughput_batches,
        "no_tqdm": args.no_tqdm,
        "no_ips": args.no_ips,
        "tqdm_update_every": args.tqdm_update_every,
    }

    results: list[dict[str, Any]] = []
    for dataset in datasets:
        for model in models:
            for seed in seeds:
                print(f"[RUN] dataset={dataset} model={model} seed={seed}")
                result = run_one(model_name=model, dataset_name=dataset, seed=seed, cfg=cfg)
                results.append(result)

    suite_path = output_root / "suite_results.json"
    with suite_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary_rows = _aggregate_summary(results)
    summary_path = output_root / "summary.csv"
    _write_summary_csv(summary_path, summary_rows)
    print(f"Wrote {suite_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
