import argparse
import utils.databases as db
import json

import torch
import torch.nn as nn
import torch.optim as optim
import timm

from models.ReluKANNet import ReluKANNetB0

from tqdm import tqdm



# ---------------------
# Argument Parsing Helpers
# ---------------------
def add_general_args(parser):
    """Add general training arguments."""
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--output", type=str, default="results.json", help="File to save results.")


def add_model_args(parser):
    """Add model-specific arguments."""
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "relukannet"
        ],
        choices =[
            "relukannet",
            "mobilenetv2_100",
            "efficientnet_lite0",
            "resnet18",
            "resnet34",
            #"resnet50",  #Will take a lot of time to train; Pretrained weights will be used for testing
            "vit_tiny_patch16_224",
            #"vit_base_patch16_224", #Will take a lot of time to train; Pretrained weights will be used for testing
            #"convnext_base" #Will take a lot of time to train; Pretrained weights will be used for testing
        ],
        help="List of model architectures to train (e.g., custom baseline)."
    )
    parser.add_argument("--g", type=int, default=2, help="g parameter for CustomConvNet.")
    parser.add_argument("--k", type=int, default=1, help="k parameter for CustomConvNet.")


def add_dataset_args(parser):
    """Add dataset-related arguments."""
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "stanford_cars", "food101", "oxford_iiit_pet"],
        help="Dataset to train on(choose from cifar10, cifar100, stanford_cars, food101, oxford_iiit_pet)."
    )


def parse_args():
    """Create and return the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train multiple model architectures and save the training/validation history."
    )
    add_general_args(parser)
    add_model_args(parser)
    add_dataset_args(parser)
    return parser.parse_args()


# ---------------------
# Dataset Setup Helper
# ---------------------
def get_datasets(dataset_name, batch_size):
    """Return the training and validation datasets along with input channels and number of classes."""
    datasets = {
        "cifar10": db.cifar10,
        "cifar100": db.cifar100,
        "stanford_cars": db.stanford_cars,
        "food101": db.food101,
        "oxford_iiit_pet": db.oxford_iiit_pet
    }
    if dataset_name in datasets:
        train_dataset, val_dataset, in_channels, num_classes = datasets[dataset_name]()
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, in_channels, num_classes


# ---------------------
# Training & Validation Functions
# ---------------------
def train_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    scaler = torch.cuda.amp.GradScaler()
    for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch}/{total_epochs}", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)


def validate(model, val_loader, criterion, device, epoch, total_epochs):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc=f"Validation Epoch {epoch}/{total_epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
    avg_loss = running_loss / len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    return avg_loss, accuracy


# ---------------------
# Main Training Script
# ---------------------
def main():
    args = parse_args()
    train_loader, val_loader, in_channels, num_classes = get_datasets(args.dataset, args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    results = {}

    # Map model names to their constructors with appropriate in_channels and num_classes.
    model_dict = {
        "relukannet": lambda: ReluKANNetB0(in_channels=in_channels, num_classes=num_classes, g=args.g, k=args.k),
        "mobilenetv2_100": lambda: timm.create_model("mobilenetv2_100", pretrained=False, num_classes=num_classes),
        "efficientnet_lite0": lambda: timm.create_model("efficientnet_lite0", pretrained=False, num_classes=num_classes),
        "resnet18": lambda: timm.create_model("resnet18", pretrained=False, num_classes=num_classes),
        "resnet34": lambda: timm.create_model("resnet34", pretrained=False, num_classes=num_classes),
        #"resnet50": lambda: timm.create_model("resnet50", pretrained=False, num_classes=num_classes),
        "vit_tiny_patch16_224": lambda: timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes),
        #"vit_base_patch16_224": lambda: timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes),
        #"convnext_base": lambda: timm.create_model("convnext_base", pretrained=False, num_classes=num_classes),
    }

    for model_name in args.models:
        if model_name not in model_dict:
            print(f"Model '{model_name}' not recognized. Skipping.")
            continue

        print(f"\nTraining model: {model_name}")
        model = model_dict[model_name]()
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        # To track training progress.
        train_losses = []
        val_losses = []
        val_accuracies = []

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, args.epochs)
            val_loss, val_accuracy = validate(model, val_loader, criterion, device, epoch, args.epochs)
            print(
                f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}"
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)


        results[model_name] = {
            "metrics": {
                "train_loss": train_losses,
                "val_loss": val_losses,
                "val_accuracy": val_accuracies,
            },
            "number_of_parameters": sum(p.numel() for p in model.parameters()),
        }

    # Save the training and validation history to a JSON file.
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nTraining results saved to {args.output}")


if __name__ == "__main__":
    model = ReluKANNetB0(in_channels=3, num_classes=10, g=3, k=3)
    print( "number_of_parameters", sum(p.numel() for p in model.parameters()))
    #main()

