"""
Downloads the PlantVillage dataset via kagglehub and splits it into the
train/val folder structure that train.py expects:

    backend/train/data/
      train/
        Apple___Apple_scab/*.jpg
        ...
      val/
        Apple___Apple_scab/*.jpg
        ...

Usage (run from the backend/ folder, with your venv activated):
    pip install kagglehub
    python train/download_dataset.py

You'll need a Kaggle account -- the first run will prompt you to
authenticate (either paste an API token, or it opens a browser login).
"""
import argparse
import random
import shutil
from pathlib import Path

import kagglehub


def find_class_folders(root: Path) -> list[Path]:
    """PlantVillage ships with a nested folder layout that varies by
    mirror -- this walks the download to find the actual leaf-image
    class folders (each containing .jpg/.png files directly)."""
    class_dirs = []
    for path in root.rglob("*"):
        if path.is_dir():
            images = list(path.glob("*.jpg")) + list(path.glob("*.JPG")) + list(path.glob("*.png"))
            if images:
                class_dirs.append(path)
    return class_dirs


def split_dataset(source_root: Path, output_root: Path, val_ratio: float, seed: int):
    random.seed(seed)
    class_dirs = find_class_folders(source_root)

    if not class_dirs:
        raise SystemExit(f"No class folders with images found under {source_root}. Dataset layout may differ.")

    train_root = output_root / "train"
    val_root = output_root / "val"

    print(f"Found {len(class_dirs)} class folders. Splitting {val_ratio:.0%} into val...")

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = (
            list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.JPG"))
            + list(class_dir.glob("*.png"))
        )
        random.shuffle(images)

        split_idx = int(len(images) * (1 - val_ratio))
        train_imgs, val_imgs = images[:split_idx], images[split_idx:]

        (train_root / class_name).mkdir(parents=True, exist_ok=True)
        (val_root / class_name).mkdir(parents=True, exist_ok=True)

        for img in train_imgs:
            shutil.copy2(img, train_root / class_name / img.name)
        for img in val_imgs:
            shutil.copy2(img, val_root / class_name / img.name)

        print(f"  {class_name}: {len(train_imgs)} train / {len(val_imgs)} val")

    print(f"\nDone. Dataset ready at: {output_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="train/data", help="Where to write train/ and val/ folders")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Downloading PlantVillage via kagglehub (this may take a few minutes)...")
    download_path = kagglehub.dataset_download("emmarex/plantdisease")
    print(f"Downloaded to: {download_path}")

    split_dataset(Path(download_path), Path(args.output_dir), args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
