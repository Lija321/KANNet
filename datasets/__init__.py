from typing import Callable

from .cifar10 import get_dataloaders as cifar10
from .cifar100 import get_dataloaders as cifar100
from .food101 import get_dataloaders as food101
from .imagenet1k import get_dataloaders as imagenet1k
from .oxford_pets import get_dataloaders as oxford_pets
from .stanford_cars import get_dataloaders as stanford_cars

DATASET_REGISTRY: dict[str, Callable] = {
    "cifar10": cifar10,
    "cifar100": cifar100,
    "food101": food101,
    "stanford_cars": stanford_cars,
    "oxford_pets": oxford_pets,
    "imagenet1k": imagenet1k,
}

__all__ = ["DATASET_REGISTRY"]
