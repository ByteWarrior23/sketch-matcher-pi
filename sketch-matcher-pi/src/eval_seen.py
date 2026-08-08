"""
eval_seen.py

Diagnostics for the trained embedding model.

Part A: Top-K retrieval accuracy on SEEN (training) categories.
  - Loads best_model.keras, loads processed uint8 data.
  - Samples up to N_SAMPLES sketches per category from a few train
    categories, matches each against that category's photos, reports
    top-1/3/5. This tells us whether the model learned anything at all
    on categories it actually saw during training.

Part B: Single-image matching (internet sketch).
  - Preprocesses ONE image exactly like pi_deploy/camera.py (gray ->
    crop -> Otsu binarize -> resize-to-224 white-padded -> 3ch -> /255),
    embeds it, and matches against the FULL photo gallery
    (models/photo_embeddings.npy, all 125 categories).
  - Prints top-10 categories with cosine similarity.

Usage:
    python src/eval_seen.py                 # Part A (seen cats)
    python src/eval_seen.py /path/to/sketch.jpg   # Part B (single image)
    python src/eval_seen.py /path/to/sketch.jpg --n 3   # also run Part A
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    BEST_MODEL_PATH, PROCESSED_DIR, LOGS_DIR, LOG_LEVEL,
    IMG_SIZE, PADDING, TOP_K,
)
from data_loader import load_processed_data, predict_normalized
import model  # registers custom Keras layers

import logging

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

N_TRAIN_CATS = 6          # how many train categories to test
N_SAMPLES = 300           # max sketches/photo-rows sampled per category


def _norm(batch):
    if batch.dtype == np.uint8:
        return batch.astype(np.float32) / 255.0
    return batch


def part_a(model, category_names, metadata):
    train_cats = [int(c) for c in metadata["train_categories"]]
    rng = np.random.RandomState(0)
    chosen = list(rng.choice(train_cats, size=min(N_TRAIN_CATS, len(train_cats)), replace=False))
    log.info("Part A: SEEN (train) categories -> %s", [category_names[c] for c in chosen])

    sketches, photos, sketch_labels, photo_labels, _, _ = load_processed_data()

    # keep row indices so photo columns can be mapped to categories later
    sk_rows = []
    ph_rows = []
    for cat in chosen:
        sk_idx = np.flatnonzero(sketch_labels == cat)
        ph_idx = np.flatnonzero(photo_labels == cat)
        rng.shuffle(sk_idx)
        rng.shuffle(ph_idx)
        sk_rows.append(sk_idx[:N_SAMPLES])
        ph_rows.append(ph_idx[:N_SAMPLES])

    sk_rows = np.concatenate(sk_rows)
    ph_rows = np.concatenate(ph_rows)

    t0 = time.time()
    sk_embs = predict_normalized(model, _norm(sketches[sk_rows]), batch_size=64, verbose=0)
    ph_embs = predict_normalized(model, _norm(photos[ph_rows]), batch_size=64, verbose=0)
    dt = time.time() - t0
    log.info("  embedded %d sketches + %d photos in %.1fs", len(sk_rows), len(ph_rows), dt)

    s = sk_embs[:, None, :] * ph_embs[None, :, :]
    all_sim = s.sum(axis=-1)  # (N_sk, N_ph_total) cosine (both L2-normalized)
    all_sk_lbl = sketch_labels[sk_rows]
    col_cats = photo_labels[ph_rows]

    n = len(all_sk_lbl)
    for k in TOP_K:
        top = np.argsort(-all_sim, axis=1)[:, :k]
        correct = sum(
            all_sk_lbl[i] in col_cats[top[i]]
            for i in range(n)
        )
        log.info("  SEEN Top-%d accuracy: %.2f%% (%d/%d)", k, correct / n * 100, correct, n)

    # per-category breakdown (top-1)
    log.info("  SEEN per-category Top-1:")
    for ci, cat in enumerate(chosen):
        mask = all_sk_lbl == cat
        sub = all_sim[mask]
        col_mask = col_cats == cat
        n_sub = len(sub)
        if n_sub == 0:
            continue
        top1 = np.argmax(sub, axis=1)
        correct = int((col_cats[top1] == cat).sum())
        log.info("    %-18s top-1 %.2f%% (%d/%d)", category_names[cat], correct / n_sub * 100, correct, n_sub)

    return all_sim, all_sk_lbl, col_cats


def preprocess_sketch_file(path):
    """Mirror pi_deploy/camera.py.preprocess for an image on disk."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = gray < 250
    coords = np.argwhere(mask)
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
        y_min = max(0, y_min - PADDING)
        x_min = max(0, x_min - PADDING)
        y_max = min(gray.shape[0], y_max + PADDING)
        x_max = min(gray.shape[1], x_max + PADDING)
        gray = gray[y_min:y_max, x_min:x_max]

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = binary.shape
    scale = min(IMG_SIZE / h, IMG_SIZE / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
    y_off = (IMG_SIZE - new_h) // 2
    x_off = (IMG_SIZE - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    return (rgb.astype(np.float32) / 255.0)[None, ...]


def part_b(model, image_paths):
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    emb_path = Path(__file__).resolve().parent.parent / "models" / "photo_embeddings.npy"
    ph_path = Path(__file__).resolve().parent.parent / "models" / "photo_labels.npy"
    lbl_path = Path(__file__).resolve().parent.parent / "models" / "labels.json"

    ph_embs = np.load(emb_path)
    ph_embs /= (np.linalg.norm(ph_embs, axis=1, keepdims=True) + 1e-8)
    ph_labels = np.load(ph_path)
    labels = {int(k): v for k, v in json.load(open(lbl_path)).items()}

    for image_path in image_paths:
        log.info("=" * 60)
        log.info("Part B: matching %s", image_path)
        img = preprocess_sketch_file(image_path)
        emb = predict_normalized(model, img, batch_size=1, verbose=0)[0]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        sims = ph_embs @ emb  # (N_photos,)
        log.info("  matched against %d photos across %d categories", len(sims), len(labels))

        n_cats = int(ph_labels.max()) + 1
        cat_max = np.full(n_cats, -np.inf, dtype=np.float32)
        np.maximum.at(cat_max, ph_labels, sims)
        valid = np.isfinite(cat_max)
        order = np.argsort(cat_max[valid])[::-1]

        log.info("  Top-10 matching categories:")
        for i, ci in enumerate(order[:10]):
            name = labels.get(int(ci), f"cat_{int(ci)}")
            log.info("    %2d. %-20s sim=%.4f", i + 1, name, cat_max[ci])

        # nearest single photos
        top_ph = np.argsort(sims)[::-1][:5]
        log.info("  Nearest photos (row, category, sim):")
        for row in top_ph:
            log.info("    row %-6d cat=%-15s sim=%.4f", row, labels.get(int(ph_labels[row]), "?"), sims[row])


def main():
    log.info("=" * 60)
    log.info("Sketch Matcher - SEEN-category diagnostics")
    log.info("=" * 60)

    if not BEST_MODEL_PATH.exists():
        log.error("Model not found: %s", BEST_MODEL_PATH)
        sys.exit(1)

    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    model = tf.keras.models.load_model(BEST_MODEL_PATH, compile=False)
    log.info("Model loaded: %s", BEST_MODEL_PATH)

    with open(PROCESSED_DIR / "category_names.json") as f:
        category_names = {int(k): v for k, v in json.load(f).items()}
    with open(PROCESSED_DIR / "metadata.json") as f:
        metadata = json.load(f)

    args = sys.argv[1:]
    img_args = [a for a in args if a != "--n"]
    run_seen = "--n" in args

    if img_args:
        part_b(model, img_args)
        if run_seen:
            part_a(model, category_names, metadata)
    else:
        part_a(model, category_names, metadata)
if __name__ == "__main__":
    main()
