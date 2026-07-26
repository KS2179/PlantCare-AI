"""
Trains a transfer-learning classifier (ResNet50 or EfficientNet-B0) on a
PlantVillage-style dataset laid out as:

    data/
      train/
        Apple___Apple_scab/*.jpg
        Apple___healthy/*.jpg
        Tomato___Late_blight/*.jpg
        ...
      val/
        Apple___Apple_scab/*.jpg
        ...

Download the PlantVillage dataset yourself (it's not bundled here), split
it into train/val folders in this structure, then run e.g.:

    python train.py --data-dir ./data --arch resnet50 --epochs 15

This produces:
    checkpoints/best_model.pt      -- state_dict for the best val-accuracy epoch
    checkpoints/class_names.txt    -- class order matching the model's output logits

The backend API (app/services/inference.py) expects both files, so after
training, the API just needs a restart to pick them up.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.model_factory import build_model  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int):
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, val_tf


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def limit_dataset(dataset, limit, label):
    """Cap images per class for a fast smoke-test run -- groups sample
    indices by class label, then keeps only the first N per class."""
    by_class = {}
    for idx, (_, cls) in enumerate(dataset.samples):
        by_class.setdefault(cls, []).append(idx)
    keep_indices = [i for indices in by_class.values() for i in indices[:limit]]
    limited = Subset(dataset, keep_indices)
    print(f"Limited {label} set to {limit} images/class -> {len(limited)} total images.")
    return limited


def main():
    parser = argparse.ArgumentParser(description="Train the plant disease classifier.")
    parser.add_argument("--data-dir", type=str, required=True, help="Folder containing train/ and val/ subfolders")
    parser.add_argument("--arch", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--freeze-backbone", action="store_true", help="Only train the final classifier layer")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 is safest/fastest on Windows)")
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Cap images per class (train set) for a fast test run, e.g. 50. "
        "Validation set is automatically capped to a smaller number too, "
        "so smoke-test runs don't stall on a full validation pass.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_dir, val_dir = data_dir / "train", data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit(
            f"Expected {train_dir} and {val_dir} to exist. See the module docstring "
            "for the expected folder layout."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tf, val_tf = get_transforms(args.image_size)
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    # ImageFolder sorts class names alphabetically -- this MUST match
    # what inference.py reads from class_names.txt, so we write it
    # straight from train_ds.classes rather than reconstructing it.
    class_names = train_ds.classes
    with open(output_dir / "class_names.txt", "w") as f:
        f.write("\n".join(class_names))
    print(f"Found {len(class_names)} classes.")

    if args.limit_per_class:
        train_ds = limit_dataset(train_ds, args.limit_per_class, "training")
        # Val set uses a smaller cap since it's just for monitoring progress
        # during a smoke test, not for a real accuracy measurement.
        val_limit = max(5, args.limit_per_class // 3)
        val_ds = limit_dataset(val_ds, val_limit, "validation")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = build_model(args.arch, num_classes=len(class_names), pretrained=True)

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False
        print("Backbone frozen -- training classifier head only.")

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_acc)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  -> New best val_acc={val_acc:.4f}, saved checkpoint.")

    print(f"Training complete. Best val_acc={best_val_acc:.4f}")
    print(f"Checkpoint: {output_dir / 'best_model.pt'}")
    print(f"Class names: {output_dir / 'class_names.txt'}")


if __name__ == "__main__":
    main()