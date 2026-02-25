from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .metrics import macro_roc_auc, top1_accuracy, top5_accuracy
from .utils import is_main_process


class Trainer:
    """Unified trainer for multiclass image-classification experiments."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt_path = self.run_dir / "best.pt"
        self.config_path = self.run_dir / "config.json"
        self.history_path = self.run_dir / "history.json"
        self.result_path = self.run_dir / "result.json"
        self.history: list[dict[str, Any]] = []
        self.total_train_time_sec: float = 0.0
        self.best_val_top1: float = -1.0

    def _resolve_device(self, cfg: dict[str, Any]) -> torch.device:
        if "device" in cfg and cfg["device"] is not None:
            return torch.device(cfg["device"])
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_optimizer(self, model: nn.Module, cfg: dict[str, Any]):
        lr = float(cfg.get("lr", 0.01))
        momentum = float(cfg.get("momentum", 0.9))
        weight_decay = float(cfg.get("weight_decay", 1e-4))
        return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    def fit(self, model, train_loader, val_loader, cfg) -> str:
        device = self._resolve_device(cfg)
        epochs = int(cfg.get("epochs", 30))
        amp = bool(cfg.get("amp", True))
        enabled_amp = amp and device.type == "cuda"
        num_classes = int(cfg["num_classes"])
        use_tqdm = bool(cfg.get("use_tqdm", True)) and is_main_process()
        show_ips = bool(cfg.get("show_ips", True))
        tqdm_update_every = max(1, int(cfg.get("tqdm_update_every", 10)))

        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = self._build_optimizer(model, cfg)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.amp.GradScaler(device="cuda", enabled=enabled_amp)
        global_start = time.perf_counter()

        best_epoch = -1
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            running_loss = 0.0
            total = 0

            train_iter = train_loader
            if use_tqdm:
                train_iter = tqdm(train_loader, desc=f"train e{epoch}/{epochs}", leave=False)

            for step, (x, y) in enumerate(train_iter, start=1):
                batch_start = time.perf_counter()
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                autocast_ctx = (
                    torch.autocast(device_type=device.type, enabled=True)
                    if enabled_amp
                    else nullcontext()
                )
                with autocast_ctx:
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_size = y.shape[0]
                total += batch_size
                running_loss += float(loss.item() * batch_size)

                if use_tqdm and (step % tqdm_update_every == 0 or step == len(train_loader)):
                    postfix: dict[str, str] = {
                        "loss": f"{float(loss.item()):.3f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    }
                    if show_ips:
                        batch_time = max(time.perf_counter() - batch_start, 1e-8)
                        postfix["ips"] = f"{batch_size / batch_time:.1f}"
                    train_iter.set_postfix(postfix)

            scheduler.step()
            train_loss = running_loss / max(total, 1)
            val_metrics = self.evaluate(
                model,
                val_loader,
                {**cfg, "device": str(device), "num_classes": num_classes, "eval_name": "val"},
            )
            epoch_time = time.perf_counter() - epoch_start
            epoch_log = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "val_top1": val_metrics["top1"],
                "val_top5": val_metrics["top5"],
                "val_roc_auc_macro": val_metrics["roc_auc_macro"],
                "val_loss": val_metrics["loss"],
                "epoch_time_sec": epoch_time,
            }
            self.history.append(epoch_log)
            if is_main_process():
                print(
                    f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f} "
                    f"val_top1={val_metrics['top1']:.4f} val_top5={val_metrics['top5']} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} time={epoch_time:.1f}s"
                )

            if float(val_metrics["top1"]) > self.best_val_top1:
                self.best_val_top1 = float(val_metrics["top1"])
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_metrics": val_metrics,
                    },
                    self.best_ckpt_path,
                )

        self.total_train_time_sec = time.perf_counter() - global_start
        with self.history_path.open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        if best_epoch < 0:
            raise RuntimeError("Training did not produce a valid checkpoint.")
        return str(self.best_ckpt_path)

    @torch.no_grad()
    def evaluate(self, model, loader, cfg) -> dict[str, Any]:
        device = self._resolve_device(cfg)
        num_classes = int(cfg["num_classes"])
        amp = bool(cfg.get("amp", True))
        enabled_amp = amp and device.type == "cuda"
        use_tqdm = bool(cfg.get("use_tqdm", True)) and is_main_process()
        show_ips = bool(cfg.get("show_ips", True))
        tqdm_update_every = max(1, int(cfg.get("tqdm_update_every", 10)))
        eval_name = str(cfg.get("eval_name", "val"))
        criterion = nn.CrossEntropyLoss()

        model.to(device)
        model.eval()
        running_loss = 0.0
        total = 0
        logits_all: list[torch.Tensor] = []
        labels_all: list[torch.Tensor] = []

        running_correct = 0
        eval_iter = loader
        if use_tqdm:
            eval_iter = tqdm(loader, desc=eval_name, leave=False)
        for step, (x, y) in enumerate(eval_iter, start=1):
            batch_start = time.perf_counter()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            autocast_ctx = (
                torch.autocast(device_type=device.type, enabled=True)
                if enabled_amp
                else nullcontext()
            )
            with autocast_ctx:
                logits = model(x)
                loss = criterion(logits, y)
            batch_size = y.shape[0]
            running_loss += float(loss.item() * batch_size)
            total += batch_size
            preds = logits.argmax(dim=1)
            running_correct += int((preds == y).sum().item())
            logits_all.append(logits.detach().cpu())
            labels_all.append(y.detach().cpu())

            if use_tqdm and (step % tqdm_update_every == 0 or step == len(loader)):
                postfix: dict[str, str] = {
                    "top1": f"{(running_correct / max(total, 1)):.3f}",
                    "loss": f"{(running_loss / max(total, 1)):.3f}",
                }
                if show_ips:
                    batch_time = max(time.perf_counter() - batch_start, 1e-8)
                    postfix["ips"] = f"{batch_size / batch_time:.1f}"
                eval_iter.set_postfix(postfix)

        logits_tensor = torch.cat(logits_all, dim=0)
        labels_tensor = torch.cat(labels_all, dim=0)
        return {
            "loss": running_loss / max(total, 1),
            "top1": top1_accuracy(logits_tensor, labels_tensor),
            "top5": top5_accuracy(logits_tensor, labels_tensor),
            "roc_auc_macro": macro_roc_auc(logits_tensor, labels_tensor, num_classes),
        }
