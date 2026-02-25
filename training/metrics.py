from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Optional

import torch


def top1_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == y).float().mean().item())


def top5_accuracy(logits: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    if logits.shape[1] < 5:
        return None
    top5 = torch.topk(logits, k=5, dim=1).indices
    correct = top5.eq(y.unsqueeze(1)).any(dim=1).float().mean().item()
    return float(correct)


def macro_roc_auc(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> Optional[float]:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise ImportError("scikit-learn is required for ROC-AUC metrics.") from exc

    try:
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        targets = y.cpu().numpy()
        if num_classes == 2:
            return float(roc_auc_score(targets, probs[:, 1]))
        return float(
            roc_auc_score(
                targets,
                probs,
                multi_class="ovr",
                average="macro",
                labels=list(range(num_classes)),
            )
        )
    except ValueError:
        return None


def param_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def flops_gmacs(model: torch.nn.Module, input_size: tuple[int, int, int, int] = (1, 3, 224, 224)) -> Optional[float]:
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as exc:
        raise ImportError("fvcore is required to compute FLOPs.") from exc

    model.eval()
    device = next(model.parameters()).device
    x = torch.randn(*input_size, device=device)
    with torch.no_grad():
        flops = FlopCountAnalysis(model, x).total()
    return float(flops / 1e9)


def throughput_images_per_sec(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    num_warmup: int = 10,
    num_batches: int = 30,
    amp: bool = True,
) -> float:
    model.eval()
    enabled_amp = amp and device.type == "cuda"
    processed = 0
    with torch.no_grad():
        for idx, (x, _) in enumerate(dataloader):
            if idx >= num_warmup:
                break
            x = x.to(device, non_blocking=True)
            autocast_ctx = (
                torch.autocast(device_type=device.type, enabled=True)
                if enabled_amp
                else nullcontext()
            )
            with autocast_ctx:
                _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for idx, (x, _) in enumerate(dataloader):
            if idx >= num_batches:
                break
            x = x.to(device, non_blocking=True)
            autocast_ctx = (
                torch.autocast(device_type=device.type, enabled=True)
                if enabled_amp
                else nullcontext()
            )
            with autocast_ctx:
                _ = model(x)
            processed += x.shape[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    if elapsed <= 0:
        return 0.0
    return float(processed / elapsed)
