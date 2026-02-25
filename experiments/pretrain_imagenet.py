from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from datasets.imagenet1k import get_dataloaders
from models import MODEL_REGISTRY
from training.utils import is_main_process, set_seed


def _str2bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    if _is_dist_initialized():
        return dist.get_rank()
    return 0


def _get_world_size() -> int:
    if _is_dist_initialized():
        return dist.get_world_size()
    return 1


def _is_rank0() -> bool:
    return _get_rank() == 0


def _setup_distributed(requested_device: str) -> tuple[torch.device, bool, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if requested_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda", local_rank if distributed else 0)
        else:
            device = torch.device("cpu")
    elif requested_device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested --device cuda, but CUDA is not available.")
        device = torch.device("cuda", local_rank if distributed else 0)
    else:
        device = torch.device("cpu")

    if distributed:
        backend = "nccl" if device.type == "cuda" else "gloo"
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
    return device, distributed, local_rank


def _cleanup_distributed() -> None:
    if _is_dist_initialized():
        dist.destroy_process_group()


def _reduce_scalar(value: float, device: torch.device) -> float:
    if not _is_dist_initialized():
        return value
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= _get_world_size()
    return float(t.item())


def _accuracy_counts(logits: torch.Tensor, target: torch.Tensor) -> tuple[int, int, int]:
    with torch.no_grad():
        maxk = min(5, logits.shape[1])
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.unsqueeze(0))
        top1 = int(correct[:1].reshape(-1).float().sum().item())
        top5 = int(correct[:maxk].reshape(-1).float().sum().item())
        total = int(target.size(0))
    return top1, top5, total


def _instantiate_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY.keys())}")
    factory = MODEL_REGISTRY[model_name]
    try:
        return factory(num_classes=num_classes, pretrained=False)
    except TypeError:
        return factory(num_classes=num_classes)


def _maybe_wrap_ddp(model: nn.Module, device: torch.device, distributed: bool, local_rank: int) -> nn.Module:
    if not distributed:
        return model
    if device.type == "cuda":
        return DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    return DDP(model, find_unused_parameters=False)


def _model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    if isinstance(model, DDP):
        return model.module.state_dict()
    return model.state_dict()


def _build_imagenet_loaders(cfg: dict[str, Any], distributed: bool, seed: int):
    train_loader, val_loader, num_classes = get_dataloaders(
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        img_size=int(cfg["img_size"]),
        seed=seed,
        data_root=str(cfg["data_root"]),
        train_aug=str(cfg.get("train_aug", "standard")),
    )

    if bool(cfg.get("debug", False)):
        train_n = min(64, len(train_loader.dataset))
        val_n = min(64, len(val_loader.dataset))
        generator = torch.Generator().manual_seed(seed)
        train_idx = torch.randperm(len(train_loader.dataset), generator=generator)[:train_n].tolist()
        val_idx = torch.randperm(len(val_loader.dataset), generator=generator)[:val_n].tolist()
        train_subset = Subset(train_loader.dataset, train_idx)
        val_subset = Subset(val_loader.dataset, val_idx)
        train_loader = DataLoader(
            train_subset,
            batch_size=int(cfg["batch_size"]),
            shuffle=not distributed,
            num_workers=int(cfg["num_workers"]),
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=int(cfg["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["num_workers"]),
            pin_memory=torch.cuda.is_available(),
        )

    if distributed:
        train_sampler = DistributedSampler(train_loader.dataset, shuffle=True)
        val_sampler = DistributedSampler(val_loader.dataset, shuffle=False)
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=int(cfg["batch_size"]),
            sampler=train_sampler,
            num_workers=int(cfg["num_workers"]),
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_loader.dataset,
            batch_size=int(cfg["batch_size"]),
            sampler=val_sampler,
            num_workers=int(cfg["num_workers"]),
            pin_memory=torch.cuda.is_available(),
        )
    return train_loader, val_loader, num_classes


def run_pretraining(cfg: dict[str, Any]) -> dict[str, Any]:
    device, distributed, local_rank = _setup_distributed(str(cfg.get("device", "auto")))
    rank0 = _is_rank0()
    seed = int(cfg.get("seed", 0))
    set_seed(seed=seed, deterministic=bool(cfg.get("deterministic", True)))

    model_name = str(cfg["model"])
    save_dir = Path(cfg["save_dir"])
    export_path = Path(cfg["export_path"])
    if rank0:
        save_dir.mkdir(parents=True, exist_ok=True)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with (save_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    train_loader, val_loader, num_classes = _build_imagenet_loaders(cfg, distributed=distributed, seed=seed)
    model = _instantiate_model(model_name=model_name, num_classes=num_classes).to(device)
    model = _maybe_wrap_ddp(model, device=device, distributed=distributed, local_rank=local_rank)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))
    optimizer = SGD(
        model.parameters(),
        lr=float(cfg.get("lr", 0.1)),
        momentum=float(cfg.get("momentum", 0.9)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(cfg.get("epochs", 90))
    if bool(cfg.get("debug", False)):
        epochs = 1
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    use_tqdm = bool(cfg.get("use_tqdm", True)) and is_main_process()
    show_ips = bool(cfg.get("show_ips", True))
    tqdm_update_every = max(1, int(cfg.get("tqdm_update_every", 10)))
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    history: list[dict[str, Any]] = []
    best_top1 = -1.0
    best_top5 = -1.0
    best_epoch = -1
    run_start = time.perf_counter()

    resume = cfg.get("resume")
    start_epoch = 1
    if resume:
        ckpt = torch.load(str(resume), map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "history" in ckpt and rank0:
            history = list(ckpt["history"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_top1 = float(ckpt.get("best_top1", best_top1))
        best_top5 = float(ckpt.get("best_top5", best_top5))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))

    for epoch in range(start_epoch, epochs + 1):
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        epoch_start = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_seen = 0

        train_iter = train_loader
        if use_tqdm:
            train_iter = tqdm(train_loader, desc=f"train e{epoch}/{epochs}", leave=False)
        for step, (x, y) in enumerate(train_iter, start=1):
            batch_start = time.perf_counter()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = torch.autocast(device_type=device.type, enabled=use_amp) if use_amp else nullcontext()
            with amp_ctx:
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = y.size(0)
            train_loss_sum += float(loss.item() * bs)
            train_seen += bs
            if use_tqdm and (step % tqdm_update_every == 0 or step == len(train_loader)):
                postfix: dict[str, str] = {
                    "loss": f"{float(loss.item()):.3f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
                if show_ips:
                    batch_time = max(time.perf_counter() - batch_start, 1e-8)
                    postfix["ips"] = f"{bs / batch_time:.1f}"
                train_iter.set_postfix(postfix)

        scheduler.step()

        model.eval()
        val_top1_sum = 0
        val_top5_sum = 0
        val_seen = 0
        with torch.no_grad():
            val_iter = val_loader
            if use_tqdm:
                val_iter = tqdm(val_loader, desc="val", leave=False)
            for step, (x, y) in enumerate(val_iter, start=1):
                batch_start = time.perf_counter()
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                amp_ctx = torch.autocast(device_type=device.type, enabled=use_amp) if use_amp else nullcontext()
                with amp_ctx:
                    logits = model(x)
                t1, t5, n = _accuracy_counts(logits, y)
                val_top1_sum += t1
                val_top5_sum += t5
                val_seen += n
                if use_tqdm and (step % tqdm_update_every == 0 or step == len(val_loader)):
                    postfix = {
                        "top1": f"{(val_top1_sum / max(val_seen, 1)):.3f}",
                        "top5": f"{(val_top5_sum / max(val_seen, 1)):.3f}",
                    }
                    if show_ips:
                        batch_time = max(time.perf_counter() - batch_start, 1e-8)
                        postfix["ips"] = f"{n / batch_time:.1f}"
                    val_iter.set_postfix(postfix)

        train_loss = train_loss_sum / max(train_seen, 1)
        val_top1 = val_top1_sum / max(val_seen, 1)
        val_top5 = val_top5_sum / max(val_seen, 1)

        if _is_dist_initialized():
            train_loss = _reduce_scalar(train_loss, device=device)
            val_top1 = _reduce_scalar(val_top1, device=device)
            val_top5 = _reduce_scalar(val_top5, device=device)

        epoch_time = time.perf_counter() - epoch_start
        lr_now = optimizer.param_groups[0]["lr"]
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "epoch_time_sec": epoch_time,
            "lr": lr_now,
        }
        if rank0:
            history.append(epoch_log)
            last_state = {
                "epoch": epoch,
                "model_state_dict": _model_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "history": history,
                "best_top1": best_top1,
                "best_top5": best_top5,
                "best_epoch": best_epoch,
            }
            torch.save(last_state, save_dir / "last.pt")

            if val_top1 > best_top1:
                best_top1 = float(val_top1)
                best_top5 = float(val_top5)
                best_epoch = epoch
                best_state = {
                    "epoch": epoch,
                    "model_state_dict": _model_state_dict(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if use_amp else None,
                    "history": history,
                    "best_top1": best_top1,
                    "best_top5": best_top5,
                    "best_epoch": best_epoch,
                }
                torch.save(best_state, save_dir / "best.pt")

            with (save_dir / "history.json").open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            print(
                f"[Epoch {epoch}/{epochs}] "
                f"train_loss={train_loss:.4f} val_top1={val_top1:.4f} "
                f"val_top5={val_top5:.4f} lr={lr_now:.6f} time={epoch_time:.1f}s"
            )

    total_time_sec = time.perf_counter() - run_start
    result = {
        "model": model_name,
        "seed": seed,
        "best_top1": best_top1,
        "best_top5": best_top5,
        "best_epoch": best_epoch,
        "total_time_sec": total_time_sec,
        "save_dir": str(save_dir),
        "export_path": str(export_path),
    }

    if rank0:
        best_ckpt = torch.load(save_dir / "best.pt", map_location="cpu")
        torch.save(best_ckpt["model_state_dict"], export_path)
        with (save_dir / "result.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(
            f"[DONE] best_epoch={best_epoch} best_top1={best_top1:.4f} "
            f"best_top5={best_top5:.4f} total_time_sec={total_time_sec:.1f}"
        )
        print(f"Exported pretrained state_dict to: {export_path}")

    _cleanup_distributed()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ImageNet-1K pretraining for KAN models.")
    parser.add_argument("--model", required=True, help="Model key from MODEL_REGISTRY.")
    parser.add_argument("--data-root", required=True, type=str, help="ImageNet root containing train/ and val/.")
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--export-path", type=str, default=None)
    parser.add_argument("--train-aug", type=str, default="standard")
    parser.add_argument("--deterministic", type=_str2bool, default=True)
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--no-ips", action="store_true", help="Disable images/sec in tqdm postfix.")
    parser.add_argument("--tqdm-update-every", type=int, default=10)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a tiny subset smoke train (1 epoch, small subset) for validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{args.model}'. Available: {sorted(MODEL_REGISTRY.keys())}")

    save_dir = args.save_dir or f"outputs/imagenet_pretrain/{args.model}/seed{args.seed}"
    export_path = args.export_path or f"weights/imagenet/{args.model}_imagenet1k.pt"

    cfg = {
        "model": args.model,
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
        "seed": args.seed,
        "save_dir": save_dir,
        "resume": args.resume,
        "export_path": export_path,
        "train_aug": args.train_aug,
        "deterministic": args.deterministic,
        "use_tqdm": not args.no_tqdm,
        "show_ips": not args.no_ips,
        "tqdm_update_every": args.tqdm_update_every,
        "debug": args.debug,
    }
    run_pretraining(cfg)


if __name__ == "__main__":
    main()
