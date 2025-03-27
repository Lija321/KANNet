import torchvision
import torchvision.transforms as transforms
from torch.utils.data import dataset

def cifar10() -> (dataset, dataset, int,int):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root="./data/cifar10/train", train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10(root="./data/cifar10/test", train=False, download=True, transform=transform)
    in_channels = 3
    num_classes = 10
    return train_dataset, val_dataset, in_channels, num_classes

def cifar100() -> (dataset, dataset, int,int):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR100(root="./data/cifar100/train", train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR100(root="./data/cifar100/test", train=False, download=True, transform=transform)
    in_channels = 3
    num_classes = 100
    return train_dataset, val_dataset, in_channels, num_classes

def stanford_cars() -> (dataset, dataset, int,int):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.ImageFolder(root="./data/stanford_cars/train", transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root="./data/stanford_cars/test", transform=transform)
    in_channels = 3
    num_classes = 196
    return train_dataset, val_dataset, in_channels, num_classes

def food101() -> (dataset, dataset, int,int):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.ImageFolder(root="./data/food101/train", transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root="./data/food101/test", transform=transform)
    in_channels = 3
    num_classes = 101
    return train_dataset, val_dataset, in_channels, num_classes

def oxford_iiit_pet() -> (dataset, dataset, int,int):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.ImageFolder(root="./data/oxford_iiit_pet/train", transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root="./data/oxford_iiit_pet/test", transform=transform)
    in_channels = 3
    num_classes = 37
    return train_dataset, val_dataset, in_channels, num_classes