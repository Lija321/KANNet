from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from datasets import DATASET_REGISTRY
from models import MODEL_REGISTRY, PREPROCESS_REGISTRY
from training.metrics import flops_gmacs, param_count, throughput_images_per_sec
from training.trainer import Trainer
from training.utils import set_seed


def _str2bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _should_export_weights(model_name: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern.endswith("*") and model_name.startswith(pattern[:-1]):
            return True
        if model_name == pattern:
            return True
    return False


def run_one(model_name: str, dataset_name: str, seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(seed, deterministic=bool(cfg.get("deterministic", True)))
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if dataset_name == "imagenet1k":
        raise ValueError(
            "Use `python -m experiments.pretrain_imagenet` for ImageNet-1K. "
            "`run_one` expects datasets that return train/val/test splits."
        )

    img_size = int(cfg.get("img_size", 224))
    pretrained = bool(cfg.get("pretrained", True))
    preprocess = PREPROCESS_REGISTRY[model_name](pretrained=pretrained, img_size=img_size)
    train_loader, val_loader, test_loader, num_classes = DATASET_REGISTRY[dataset_name](
        batch_size=int(cfg.get("batch_size", 64)),
        num_workers=int(cfg.get("num_workers", 4)),
        preprocess_train=preprocess["train"],
        preprocess_eval=preprocess["eval"],
        seed=seed,
        data_root=str(cfg.get("data_root", "data")),
    )

    run_dir = (
        Path(cfg.get("output_root", "outputs/runs"))
        / dataset_name
        / model_name
        / f"seed{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = MODEL_REGISTRY[model_name](num_classes=num_classes, pretrained=pretrained)
    trainer = Trainer(run_dir=run_dir)
    fit_cfg = {
        "seed": seed,
        "num_classes": num_classes,
        "epochs": int(cfg.get("epochs", 30)),
        "lr": float(cfg.get("lr", 0.01)),
        "momentum": float(cfg.get("momentum", 0.9)),
        "weight_decay": float(cfg.get("weight_decay", 1e-4)),
        "amp": bool(cfg.get("amp", True)),
        "device": str(device),
        "use_tqdm": not bool(cfg.get("no_tqdm", False)),
        "show_ips": not bool(cfg.get("no_ips", False)),
        "tqdm_update_every": int(cfg.get("tqdm_update_every", 10)),
    }
    best_ckpt = trainer.fit(model, train_loader, val_loader, fit_cfg)
    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    save_weight_models = cfg.get("save_weight_models", ["kannet*"])
    if isinstance(save_weight_models, str):
        save_weight_models = _csv_list(save_weight_models)
    exported_weight_path = None
    if _should_export_weights(model_name=model_name, patterns=save_weight_models):
        weights_root = Path(cfg.get("weights_root", "outputs/weights"))
        weights_dir = weights_root / dataset_name / model_name / f"seed{seed}"
        weights_dir.mkdir(parents=True, exist_ok=True)
        exported_weight_path = weights_dir / "model_state_dict.pt"
        torch.save(model.state_dict(), exported_weight_path)

    test_metrics = trainer.evaluate(model, test_loader, fit_cfg)
    result = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "num_classes": num_classes,
        "pretrained": pretrained,
        "img_size": img_size,
        "best_checkpoint": str(best_ckpt),
        "best_val_top1": trainer.best_val_top1,
        "test_top1": test_metrics["top1"],
        "test_top5": test_metrics["top5"],
        "test_roc_auc_macro": test_metrics["roc_auc_macro"],
        "test_loss": test_metrics["loss"],
        "exported_weight_path": str(exported_weight_path) if exported_weight_path else None,
        "param_count": param_count(model),
        "flops_gmacs": flops_gmacs(model, input_size=(1, 3, img_size, img_size)),
        "throughput_images_per_sec": throughput_images_per_sec(
            model=model,
            dataloader=test_loader,
            device=device,
            num_warmup=int(cfg.get("throughput_warmup", 10)),
            num_batches=int(cfg.get("throughput_batches", 30)),
            amp=bool(cfg.get("amp", True)),
        ),
        "total_train_time_sec": trainer.total_train_time_sec,
    }
    with (run_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one image-classification experiment.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY.keys()))
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_REGISTRY.keys()))
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--no-ips", action="store_true")
    parser.add_argument("--tqdm-update-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = vars(args)
    model_name = cfg.pop("model")
    dataset_name = cfg.pop("dataset")
    seed = int(cfg.pop("seed"))
    result = run_one(model_name=model_name, dataset_name=dataset_name, seed=seed, cfg=cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
