"""verify_teacher.py — GO/NO-GO gate on the freshly trained TEACHER.

The teacher is only trusted if it is ACTUALLY well-separated on the real
125-category task the user performs (drawing TRAINED categories). Measurements:

  A1) SEEN-category in-distribution: a sample of the 100 TRAIN categories'
      sketches (exact training preprocessing) searched against the full
      125-cat / 49,430-photo DB. This is the population real users draw.
  A2) ZERO-SHOT (unseen val+test cats): informational only — NOT a gate.
  B)  DEPLOYED-PIPELINE: real exported sketches (SEEN cats) through the exact
      pi_deploy CameraCapture preprocessing + category-dedup search (this
      harness caught the old teacher at 32.5% and student int8 at 22.5%).

Gate (PASS to proceed to student distillation):
    seen-cat in-distribution top-1 >= 60%
    deployed-pipeline top-1 >= 45%
    teacher photo cross-cat cosine < 0.5

Usage (GPU, cluster):
    python hpc/verify_teacher.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import model  # noqa: E402
from config import TEACHER_MODEL_PATH, PROCESSED_DIR  # noqa: E402
from data_loader import (load_processed_data, get_split_masks,  # noqa: E402
                         predict_normalized)

IMG_SIZE = 224
PADDING = 10
TEST_DIR = Path(__file__).resolve().parent / "sketch_test_export"


def preprocess_sketch(path):
    """Exact mirror of pi_deploy/camera.py CameraCapture.preprocess."""
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = np.argwhere(gray < 250)
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
    return rgb.astype(np.float32) / 255.0


def topk_retrieval(query_embs, query_labels, db_embs, db_labels,
                   labels, k=5):
    """Per-query dedup-by-category top-k accuracy (matcher.match logic)."""
    n_cats = int(db_labels.max()) + 1
    hits = np.zeros(k, dtype=int)
    sims = db_embs @ query_embs.T  # (n_db, n_q)
    for j in range(query_embs.shape[0]):
        cat_max = np.full(n_cats, -np.inf, dtype=np.float32)
        np.maximum.at(cat_max, db_labels, sims[:, j])
        order = np.argsort(cat_max)[::-1]
        expected = labels[query_labels[j]]
        for kk in range(k):
            if expected in [labels[int(o)] for o in order[:kk + 1]]:
                hits[kk] += 1
    return hits / query_embs.shape[0]


def distinctness(embs, labels):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(embs), 2000, replace=False)
    lab = labels[idx]
    sim = embs[idx] @ embs[idx].T
    same_mask = (lab[:, None] == lab[None, :])
    cross_mask = (lab[:, None] != lab[None, :])
    np.fill_diagonal(same_mask, False)
    return float(sim[same_mask].mean()), float(sim[cross_mask].mean())


def main():
    teacher = tf.keras.models.load_model(TEACHER_MODEL_PATH, compile=False)
    print("teacher loaded:", TEACHER_MODEL_PATH, flush=True)

    labels = json.loads((PROCESSED_DIR / "category_names.json").read_text())
    labels = {int(k): v for k, v in labels.items()}

    (sketches, photos, sk_labels, ph_labels, _, metadata) = load_processed_data()
    m = get_split_masks(sk_labels, ph_labels, metadata, labels)

    print("embedding all photos...", flush=True)
    ph_embs = predict_normalized(teacher, photos, batch_size=128, verbose=1)
    ph_embs = ph_embs / np.linalg.norm(ph_embs, axis=1, keepdims=True)
    print(f"photos embedded: {ph_embs.shape}", flush=True)

    same, cross = distinctness(ph_embs, ph_labels)
    print(f"A) photo distinctness: same={same:.3f} cross={cross:.3f}", flush=True)

    def _acc(name, idx, k=5):
        embs = predict_normalized(teacher, sketches[idx], batch_size=128, verbose=0)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        acc = topk_retrieval(embs, sk_labels[idx], ph_embs, ph_labels, labels, k=k)
        print(f"{name} 125-cat top-1/3/5: {acc[0]*100:.1f}% / "
              f"{acc[1]*100:.1f}% / {acc[2]*100:.1f}% (n={len(idx)})", flush=True)
        return acc

    # A1) SEEN categories (the 100 train cats) — the population the user draws.
    # This is the true in-distribution retrieval quality.
    tr_idx = np.where(m["tr_sk"])[0]
    rng = np.random.default_rng(0)
    tr_idx = rng.choice(tr_idx, size=min(12000, len(tr_idx)), replace=False)
    a1 = _acc("A1) SEEN-cat in-distribution", tr_idx)

    # A2) UNSEEN categories (val + test cats) — zero-shot, informational only
    # (real users draw trained categories, so this is not the target).
    zs_idx = np.where(m["va_sk"] | m["te_sk"])[0]
    a2 = _acc("A2) ZERO-SHOT (unseen cats)", zs_idx)

    b = np.zeros(5)
    n = 0
    for cat_dir in sorted(TEST_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        expected = cat_dir.name.split("_", 1)[1]
        pngs = sorted(cat_dir.glob("*.png"))
        for p in pngs:
            pre = preprocess_sketch(p)
            emb = teacher.predict(pre[None], verbose=0)[0]
            emb = emb / np.linalg.norm(emb)
            cat_max = np.full(len(labels), -np.inf, dtype=np.float32)
            np.maximum.at(cat_max, ph_labels, ph_embs @ emb)
            order = np.argsort(cat_max)[::-1]
            for kk in range(5):
                if expected in [labels[int(o)] for o in order[:kk + 1]]:
                    b[kk] += 1
            n += 1
    b /= max(n, 1)
    print(f"B) DEPLOYED-PIPELINE (seen cats) top-1/3/5: "
          f"{b[0]*100:.1f}% / {b[1]*100:.1f}% / {b[2]*100:.1f}% (n={n})", flush=True)

    gate1 = a1[0] >= 0.60   # seen-cat in-distribution (the population users draw)
    gate2 = b[0] >= 0.45    # real deployed pipeline on seen cats
    gate3 = cross < 0.5     # photos well-separated (no cone collapse)
    print("\n==== TEACHER GATE ====", flush=True)
    print(f"  seen-cat in-distribution top-1 {a1[0]*100:.1f}%  >= 60%  -> {gate1}", flush=True)
    print(f"  deployed-pipeline top-1 {b[0]*100:.1f}%  >= 45%  -> {gate2}", flush=True)
    print(f"  photo cross-cat cos {cross:.3f}  < 0.5  -> {gate3}", flush=True)
    print(f"  [info] zero-shot top-1 {a2[0]*100:.1f}% (unseen cats — not a gate)", flush=True)
    if gate1 and gate2 and gate3:
        print("  VERDICT: PASS — teacher good enough; proceed to re-embed + student.", flush=True)
    else:
        print("  VERDICT: FAIL — do NOT trust this teacher yet; retrain/improve first.", flush=True)


if __name__ == "__main__":
    main()
