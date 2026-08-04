"""
preprocess.py

Preprocessing pipeline for Sketch Matcher (multi-dataset version).

Merges sketch datasets into Sketchy's canonical categories:
  - Sketchy          (canonical: sketches + photos, 125 cats)  [REQUIRED]
  - TU-Berlin        (sketches only, mapped by name)
  - QuickDraw        (28x28 numpy bitmaps, mapped by name)
  - ImageNet-Sketch  (sketches only, mapped by name)

Processing steps for EACH sketch image:
  1. Load grayscale
  2. Crop whitespace (bounding box of dark content + padding)
  3. Binarize (Otsu) -> black ink on white paper
  4. Resize to 224x224 with aspect ratio + white padding
  5. Convert to 3-channel, normalize to [0, 1]

Photos (Sketchy only): crop + resize + 3-channel + normalize (no binarize).

Output (data/processed/):
  - sketches.npy       (N, 224, 224, 3) float32 [0,1]
  - photos.npy         (M, 224, 224, 3) float32 [0,1]
  - sketch_labels.npy  (N,)  canonical category index per sketch
  - photo_labels.npy   (M,)  canonical category index per photo
  - category_names.json {index: canonical_name}
  - metadata.json      splits + counts per dataset

Usage:
    python src/preprocess.py
"""

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    from config import (
        RAW_DIR, PROCESSED_DIR, DATASETS,
        IMG_SIZE, IMG_CHANNELS, BINARY_THRESHOLD, PADDING,
        TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, LOG_LEVEL,
        USE_QUICKDRAW, USE_TUBERLIN, USE_IMAGENETSKETCH,
        QUICKDRAW_MAX_PER_CATEGORY, TUBERLIN_MAX_PER_CATEGORY,
        IMAGENETSKETCH_MAX_PER_CATEGORY,
    )
except ModuleNotFoundError:  # Colab: imported as src.preprocess
    from src.config import (
        RAW_DIR, PROCESSED_DIR, DATASETS,
        IMG_SIZE, IMG_CHANNELS, BINARY_THRESHOLD, PADDING,
        TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, LOG_LEVEL,
        USE_QUICKDRAW, USE_TUBERLIN, USE_IMAGENETSKETCH,
        QUICKDRAW_MAX_PER_CATEGORY, TUBERLIN_MAX_PER_CATEGORY,
        IMAGENETSKETCH_MAX_PER_CATEGORY,
    )

from class_mapping import build_canonical_map, map_to_canonical, normalize_name

import logging

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# =============================================================================
# IMAGE TRANSFORMS  (must match pi_deploy/camera.py EXACTLY)
# =============================================================================
def crop_to_content(img: np.ndarray, padding: int = PADDING) -> np.ndarray:
    """Crop whitespace around the drawn content (dark = content)."""
    mask = img < 250
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return img

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1

    y_min = max(0, y_min - padding)
    x_min = max(0, x_min - padding)
    y_max = min(img.shape[0], y_max + padding)
    x_max = min(img.shape[1], x_max + padding)

    return img[y_min:y_max, x_min:x_max]


def binarize(img: np.ndarray) -> np.ndarray:
    """Otsu binarize -> black ink on white paper."""
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def resize_with_padding(img: np.ndarray, target_size: int = IMG_SIZE) -> np.ndarray:
    """Aspect-ratio resize onto a white canvas of target_size x target_size."""
    h, w = img.shape
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size), 255, dtype=np.uint8)

    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def process_sketch(img_uint8: np.ndarray) -> np.ndarray:
    """Full sketch pipeline: crop -> binarize -> resize -> 3ch uint8."""
    img = crop_to_content(img_uint8)
    img = binarize(img)
    img = resize_with_padding(img)
    return grayscale_to_rgb(img)


def process_photo(img_uint8: np.ndarray) -> np.ndarray:
    """Photo pipeline: crop -> resize -> 3ch uint8 (no binarize)."""
    img = crop_to_content(img_uint8)
    img = resize_with_padding(img)
    return grayscale_to_rgb(img)


def grayscale_to_rgb(img: np.ndarray) -> np.ndarray:
    return np.stack([img] * 3, axis=-1)


# =============================================================================
# SKETCHY (canonical source: sketches + photos)
# =============================================================================
def load_sketchy(canonical_cats):
    sketchy_dir = DATASETS["sketchy"]["raw_dir"]
    sketch_dir = sketchy_dir / "sketch"
    photo_dir = sketchy_dir / "photo"

    if not sketch_dir.exists() or not photo_dir.exists():
        log.error(f"Sketchy dataset not found at {sketchy_dir}")
        log.error("Please run: python src/download_data.py")
        sys.exit(1)

    categories = sorted([d.name for d in sketch_dir.iterdir() if d.is_dir()])
    log.info(f"Sketchy: found {len(categories)} category dirs")

    # canonical index: only categories that have BOTH sketch and photo dirs
    category_to_idx = {}
    for cat in categories:
        if cat in canonical_cats and (photo_dir / cat).is_dir():
            category_to_idx[cat] = canonical_cats.index(cat)

    sketches, photos, sketch_labels, photo_labels = [], [], [], []
    counts = {"sketchy": 0}

    for cat, idx in tqdm(category_to_idx.items(), desc="Sketchy"):
        cat_sketch_dir = sketch_dir / cat
        for sf in sorted(cat_sketch_dir.glob("*.png")):
            img = cv2.imread(str(sf), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            sketches.append(process_sketch(img))
            sketch_labels.append(idx)
            counts["sketchy"] += 1

        cat_photo_dir = photo_dir / cat
        for pf in sorted(cat_photo_dir.glob("*.jpg")) + sorted(cat_photo_dir.glob("*.png")):
            img = cv2.imread(str(pf), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            photos.append(process_photo(img))
            photo_labels.append(idx)

    log.info(f"Sketchy: {len(sketches)} sketches, {len(photos)} photos, "
             f"{len(category_to_idx)} mapped categories")
    return (sketches, photos, sketch_labels, photo_labels, category_to_idx, counts)


# =============================================================================
# TU-BERLIN (sketches only)
# =============================================================================
def load_tuberlin(canonical_map, canonical_index):
    root = DATASETS["tuberlin"]["raw_dir"]
    if not root.exists():
        log.info("TU-Berlin not downloaded - skipping (USE_TUBERLIN=True needs it).")
        return [], [], {}

    sketches, labels = [], []
    counts = {"tuberlin": 0, "tuberlin_mapped": 0, "tuberlin_unmapped": 0}

    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        canonical = map_to_canonical(cat_dir.name, canonical_map)
        if canonical is None or canonical not in canonical_index:
            counts["tuberlin_unmapped"] += 1
            continue
        idx = canonical_index[canonical]
        files = [f for f in cat_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]
        if TUBERLIN_MAX_PER_CATEGORY > 0:
            files = files[:TUBERLIN_MAX_PER_CATEGORY]
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            sketches.append(process_sketch(img))
            labels.append(idx)
            counts["tuberlin"] += 1
        counts["tuberlin_mapped"] += 1

    log.info(f"TU-Berlin: {counts['tuberlin']} sketches "
             f"({counts['tuberlin_mapped']} mapped cats, "
             f"{counts['tuberlin_unmapped']} unmapped skipped)")
    return sketches, labels, counts


# =============================================================================
# QUICKDRAW (28x28 numpy bitmaps)
# =============================================================================
def load_quickdraw(canonical_map, canonical_index):
    root = DATASETS["quickdraw"]["raw_dir"]
    if not root.exists():
        log.info("QuickDraw not downloaded - skipping.")
        return [], [], {}

    sketches, labels = [], []
    counts = {"quickdraw": 0, "quickdraw_mapped": 0, "quickdraw_unmapped": 0}

    for npy_path in sorted(root.glob("*.npy")):
        canonical = map_to_canonical(npy_path.stem, canonical_map)
        if canonical is None or canonical not in canonical_index:
            counts["quickdraw_unmapped"] += 1
            continue
        idx = canonical_index[canonical]
        data = np.load(str(npy_path))
        if data.ndim != 2 or data.shape[1] != 784:
            continue
        data = data[:QUICKDRAW_MAX_PER_CATEGORY]
        for flat in data:
            img = (255 - flat).astype(np.uint8).reshape(28, 28)
            sketches.append(process_sketch(img))
            labels.append(idx)
        counts["quickdraw"] += len(data)
        counts["quickdraw_mapped"] += 1

    log.info(f"QuickDraw: {counts['quickdraw']} sketches "
             f"({counts['quickdraw_mapped']} mapped cats, "
             f"{counts['quickdraw_unmapped']} unmapped skipped)")
    return sketches, labels, counts


# =============================================================================
# IMAGENET-SKETCH (sketches only; folders may be synset IDs)
# =============================================================================
def load_imagenetsketch(canonical_map, canonical_index):
    root = DATASETS["imagenetsketch"]["raw_dir"]
    if not root.exists():
        log.info("ImageNet-Sketch not downloaded - skipping.")
        return [], [], {}

    sketches, labels = [], []
    counts = {"imagenetsketch": 0, "imagenetsketch_mapped": 0,
              "imagenetsketch_unmapped": 0}

    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        canonical = map_to_canonical(cat_dir.name, canonical_map)
        if canonical is None or canonical not in canonical_index:
            counts["imagenetsketch_unmapped"] += 1
            continue
        idx = canonical_index[canonical]
        files = [f for f in cat_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]
        if IMAGENETSKETCH_MAX_PER_CATEGORY > 0:
            files = files[:IMAGENETSKETCH_MAX_PER_CATEGORY]
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            sketches.append(process_sketch(img))
            labels.append(idx)
            counts["imagenetsketch"] += 1
        counts["imagenetsketch_mapped"] += 1

    log.info(f"ImageNet-Sketch: {counts['imagenetsketch']} sketches "
             f"({counts['imagenetsketch_mapped']} mapped cats, "
             f"{counts['imagenetsketch_unmapped']} unmapped skipped)")
    return sketches, labels, counts


# =============================================================================
# SPLITS + SAVE
# =============================================================================
def train_val_test_split(sketches, photos, sketch_labels, photo_labels, category_to_idx):
    categories = list(category_to_idx.keys())
    random.shuffle(categories)

    n_categories = len(categories)
    n_train = max(1, int(n_categories * TRAIN_SPLIT))
    n_val = max(1, int(n_categories * VAL_SPLIT))

    train_cats = categories[:n_train]
    val_cats = categories[n_train:n_train + n_val]
    test_cats = categories[n_train + n_val:]

    log.info(f"Split: {len(train_cats)} train, {len(val_cats)} val, "
             f"{len(test_cats)} test categories")

    def filter_by_category(data_array, label_array, cat_set):
        cat_indices = [category_to_idx[c] for c in cat_set]
        mask = np.isin(label_array, cat_indices)
        return data_array[mask], label_array[mask]

    train_sk, train_sk_lb = filter_by_category(sketches, sketch_labels, train_cats)
    val_sk, val_sk_lb = filter_by_category(sketches, sketch_labels, val_cats)
    test_sk, test_sk_lb = filter_by_category(sketches, sketch_labels, test_cats)

    train_ph, train_ph_lb = filter_by_category(photos, photo_labels, train_cats)
    val_ph, val_ph_lb = filter_by_category(photos, photo_labels, val_cats)
    test_ph, test_ph_lb = filter_by_category(photos, photo_labels, test_cats)

    return (train_sk, train_ph, train_sk_lb, train_ph_lb,
            val_sk, val_ph, val_sk_lb, val_ph_lb,
            test_sk, test_ph, test_sk_lb, test_ph_lb,
            train_cats, val_cats, test_cats)


def save_processed_data(sketches, photos, sketch_labels, photo_labels,
                        category_to_idx, train_cats, val_cats, test_cats, counts):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    np.save(PROCESSED_DIR / "sketches.npy", sketches.astype(np.float32) / 255.0)
    np.save(PROCESSED_DIR / "photos.npy", photos.astype(np.float32) / 255.0)
    np.save(PROCESSED_DIR / "sketch_labels.npy", np.array(sketch_labels, dtype=np.int32))
    np.save(PROCESSED_DIR / "photo_labels.npy", np.array(photo_labels, dtype=np.int32))

    with open(PROCESSED_DIR / "category_names.json", "w") as f:
        json.dump({str(v): k for k, v in category_to_idx.items()}, f, indent=2)

    metadata = {
        "total_sketches": len(sketches),
        "total_photos": len(photos),
        "categories": len(category_to_idx),
        "counts_per_source": counts,
        # categories stored as CANONICAL INDICES (not names) so that
        # data_loader.get_split_masks / evaluate can np.isin() directly.
        "train_categories": [category_to_idx[c] for c in train_cats],
        "val_categories": [category_to_idx[c] for c in val_cats],
        "test_categories": [category_to_idx[c] for c in test_cats],
        "input_shape": [IMG_SIZE, IMG_SIZE, IMG_CHANNELS],
        "preprocessing": "crop+otsu+resize224+norm",
    }
    with open(PROCESSED_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Saved processed data to {PROCESSED_DIR}")
    log.info(f"  sketches.npy      shape: {sketches.shape}")
    log.info(f"  photos.npy        shape: {photos.shape}")
    log.info(f"  sketch_labels.npy shape: {sketch_labels.shape}")
    log.info(f"  photo_labels.npy  shape: {photo_labels.shape}")


def main():
    log.info("=" * 60)
    log.info("Sketch Matcher - Data Preprocessing (multi-dataset)")
    log.info("=" * 60)

    # 0) Determine canonical categories (Sketchy category dirs)
    sketchy_dir = DATASETS["sketchy"]["raw_dir"]
    canonical_cats = sorted([d.name for d in (sketchy_dir / "sketch").iterdir()]
                            if (sketchy_dir / "sketch").exists() else [])
    if not canonical_cats:
        log.error(f"Sketchy not found at {sketchy_dir}. Run: python src/download_data.py")
        sys.exit(1)
    canonical_map = build_canonical_map(canonical_cats)
    canonical_index = {name: i for i, name in enumerate(canonical_cats)}
    log.info(f"Canonical categories (Sketchy): {len(canonical_cats)}")

    # 1) Sketchy (required, provides sketches + ALL photos)
    log.info("\n[1/4] Loading Sketchy...")
    sketches, photos, sketch_labels, photo_labels, category_to_idx, counts = \
        load_sketchy(canonical_cats)

    # 2) Extra sketch datasets (mapped to canonical categories)
    counts.setdefault("sketchy", len(sketches))
    if USE_TUBERLIN:
        log.info("\n[2/4] Loading TU-Berlin...")
        s, l, c = load_tuberlin(canonical_map, canonical_index)
        sketches += s; sketch_labels += l; counts.update(c)
    if USE_QUICKDRAW:
        log.info("\n[3/4] Loading QuickDraw...")
        s, l, c = load_quickdraw(canonical_map, canonical_index)
        sketches += s; sketch_labels += l; counts.update(c)
    if USE_IMAGENETSKETCH:
        log.info("\n[4/4] Loading ImageNet-Sketch...")
        s, l, c = load_imagenetsketch(canonical_map, canonical_index)
        sketches += s; sketch_labels += l; counts.update(c)

    if len(sketches) == 0 or len(photos) == 0:
        log.error("No data loaded. Aborting.")
        sys.exit(1)

    sketches = np.array(sketches, dtype=np.uint8)
    photos = np.array(photos, dtype=np.uint8)
    sketch_labels = np.array(sketch_labels, dtype=np.int32)
    photo_labels = np.array(photo_labels, dtype=np.int32)

    log.info(f"\nTotal: {len(sketches)} sketches, {len(photos)} photos")

    # 3) Split by category
    log.info("\nSplitting data (by category)...")
    (train_sk, train_ph, train_sk_lb, train_ph_lb,
     val_sk, val_ph, val_sk_lb, val_ph_lb,
     test_sk, test_ph, test_sk_lb, test_ph_lb,
     train_cats, val_cats, test_cats) = train_val_test_split(
        sketches, photos, sketch_labels, photo_labels, category_to_idx
    )

    # 4) Save (normalized to [0,1])
    log.info("\nSaving...")
    save_processed_data(sketches, photos, sketch_labels, photo_labels,
                        category_to_idx, train_cats, val_cats, test_cats, counts)

    log.info("\n" + "=" * 60)
    log.info("Preprocessing complete!")
    log.info(f"  Train categories: {len(train_cats)}")
    log.info(f"  Val categories:   {len(val_cats)}")
    log.info(f"  Test categories:  {len(test_cats)}")
    log.info("Next step: python src/train.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
