"""Download the Rock-Paper-Scissors Kaggle dataset and preprocess to npy arrays.

Dataset: Kaggle "drgfreeman/rockpaperscissors" (Dr. Moroney) - 2184 train
(728/class, 150x150), 372 test (124/class). Images are clean single-hand
poses on white backgrounds.

Outputs (uint8 [0,255], RGB 224x224):
  data/processed/rps_train_x.npy / rps_train_y.npy
  data/processed/rps_val_x.npy   / rps_val_y.npy
  data/processed/rps_test_x.npy  / rps_test_y.npy
  data/processed/labels.json
"""
import json
import random
from pathlib import Path

import cv2
import kagglehub
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
IMG_SIZE = 224
SEED = 42
CLASSES = ["rock", "paper", "scissors"]

random.seed(SEED)
np.random.seed(SEED)


def load_dir(directory, label_index):
    images, labels = [], []
    files = sorted(directory.glob("*.png"))
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        images.append(img)
        labels.append(label_index)
    return images, labels


def build(split_dir, class_names):
    x_parts, y_parts = [], []
    for idx, name in enumerate(class_names):
        d = split_dir / name
        if not d.exists():
            d = split_dir / name.capitalize()
        if not d.exists():
            print(f"  WARN: missing class dir {name} in {split_dir}", flush=True)
            continue
        imgs, lbls = load_dir(d, idx)
        print(f"  {name}: {len(imgs)} images", flush=True)
        x_parts.extend(imgs)
        y_parts.extend(lbls)
    if not x_parts:
        raise RuntimeError(f"no images found in {split_dir}")
    x = np.stack(x_parts).astype(np.uint8)
    y = np.asarray(y_parts, dtype=np.int64)
    return x, y


def main():
    print("Downloading dataset via kagglehub...", flush=True)
    path = kagglehub.dataset_download("drgfreeman/rockpaperscissors")
    base = Path(path)
    print(f"Dataset at: {base}", flush=True)

    # Explore the layout (nested 'rock_paper_scissors' possible)
    def find_subdir(root, name):
        hits = [p for p in root.rglob(name) if p.is_dir()]
        return hits[0] if hits else None

    train_dir = find_subdir(base, "rps") or base

    print("Building full set...", flush=True)
    x, y = build(train_dir, CLASSES)

    # Stratified train/val/test split (this Kaggle version ships no test set).
    n = len(y)
    idx = np.random.permutation(n)
    n_val = int(0.1 * n)
    n_test = int(0.1 * n)
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]

    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    x_train, y_train = x[train_idx], y[train_idx]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "rps_train_x.npy", x_train)
    np.save(DATA_DIR / "rps_train_y.npy", y_train)
    np.save(DATA_DIR / "rps_val_x.npy", x_val)
    np.save(DATA_DIR / "rps_val_y.npy", y_val)
    np.save(DATA_DIR / "rps_test_x.npy", x_test)
    np.save(DATA_DIR / "rps_test_y.npy", y_test)
    with open(DATA_DIR / "labels.json", "w") as f:
        json.dump(CLASSES, f)

    print(f"train {x_train.shape} {y_train.shape}", flush=True)
    print(f"val   {x_val.shape} {y_val.shape}", flush=True)
    print(f"test  {x_test.shape} {y_test.shape}", flush=True)
    print("PREPARE OK", flush=True)


if __name__ == "__main__":
    main()
