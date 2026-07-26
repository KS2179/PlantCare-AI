"""
Evaluates a trained checkpoint on a held-out folder and prints per-class
precision/recall/F1 plus overall accuracy. Useful for the "how did you
validate it" question in an interview.

Usage:
    python evaluate.py --data-dir ./data/val --checkpoint ./checkpoints/best_model.pt \
        --class-names ./checkpoints/class_names.txt --arch resnet50
"""
import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.model_factory import build_model  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--class-names", type=str, required=True)
    parser.add_argument("--arch", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    with open(args.class_names) as f:
        class_names = [line.strip() for line in f if line.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    dataset = datasets.ImageFolder(args.data_dir, transform=tf)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = build_model(args.arch, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    print("Confusion matrix shape:", confusion_matrix(all_labels, all_preds).shape)


if __name__ == "__main__":
    main()
