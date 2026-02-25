from __future__ import annotations

import random

import numpy as np
import torch
import torch.distributed as dist


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
