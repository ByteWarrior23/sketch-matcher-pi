"""test_teacher_125.py — TEACHER evaluated on the REAL 125-category deployed task.

Same preprocessing (CameraCapture.preprocess: grayscale, crop<250 w/ PADDING=10,
Otsu binarize, white-pad resize), same category-dedup search (matcher.match),
same 49,430-photo DB as the deployed student bundle.

This directly answers "is the TEACHER itself good at 125 cats?" — with the
student distillation removed from the equation. Reference points:
  - deployed student (int8 TFLite): 22.5% top-1
  - student (Keras float):          28.7% top-1
Teacher is good iff it lands well above those.

Usage (GPU, cluster):
    python hpc/test_teacher_125.py
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

import model  # noqa: E402  (registers custom layers for load_model)
from config import TEACHER_MODEL_PATH, PROCESSED_DIR  # noqa: E402
from data_loader import predict_normalized  # noqa: E402

IMG_SIZE = 224
PADDING = 10
TEST_DIR = Path(__file__).resolve().parent / "sketch_test_export"


def preprocess_sketch(path):
    """Exact mirror of pi_deploy/camera.py CameraCapture.preprocess."""
    img = cv2.imread(str(path))  # BGR
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
    return rgb.astype(np.float32) / 255.0


def main():
    teacher = tf.keras.models.load_model(TEACHER_MODEL_PATH, compile=False)
    print("teacher loaded:", TEACHER_MODEL_PATH, flush=True)

    labels = json.loads((PROCESSED_DIR / "category_names.json").read_text())
    labels = {int(k): v for k, v in labels.items()}

    photos = np.load(str(PROCESSED_DIR / "photos_u8.npy"), mmap_mode="r")
    print("photos:", photos.shape, flush=True)
    ph_embs = predict_normalized(teacher, photos, batch_size=128, verbose=1)
    ph_embs = ph_embs / np.linalg.norm(ph_embs, axis=1, keepdims=True)
    print("teacher photo embeddings:", ph_embs.shape, flush=True)

    ph_labels = np.load(str(PROCESSED_DIR / "photo_labels_u8.npy"))
    n_cats = int(ph_labels.max()) + 1
    print("photo labels:", ph_labels.shape, "n_cats:", n_cats, flush=True)

    # sanity: teacher photo-embedding distinctness (same vs cross category)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ph_embs), 2000, replace=False)
    lab = ph_labels[idx]
    sim = ph_embs[idx] @ ph_embs[idx].T
    same_mask = lab[:, None] == lab[None, :]
    cross_mask = lab[:, None] != lab[None, :]
    np.fill_diagonal(same_mask, False)
    if same_mask.any():
        print(f"  photo distinctness: same-cat cos={sim[same_mask].mean():.3f} "
              f"cross-cat cos={sim[cross_mask].mean():.3f} "
              f"(healthy: cross < 0.9, same > cross)", flush=True)

    results = []
    for cat_dir in sorted(TEST_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        expected = cat_dir.name.split("_", 1)[1]
        pngs = sorted(cat_dir.glob("*.png"))
        cat_hits = [0, 0, 0]
        cat_confs = []
        for p in pngs:
            pre = preprocess_sketch(p)
            emb = teacher.predict(pre[None], verbose=0)[0]
            emb = emb / np.linalg.norm(emb)
            sims = ph_embs @ emb
            cat_max = np.full(n_cats, -np.inf, dtype=np.float32)
            np.maximum.at(cat_max, ph_labels, sims)
            order = np.argsort(cat_max)[::-1]
            names = [labels[int(o)] for o in order]
            cat_confs.append(float(cat_max[order[0]]))
            for k in range(3):
                if expected in names[:k + 1]:
                    cat_hits[k] += 1
        accs = [h / len(pngs) for h in cat_hits]
        results.append((expected, len(pngs), accs, np.mean(cat_confs)))
        print(f"  {expected:<14} n={len(pngs):2d}  "
              f"top1={accs[0]:.2f} top3={accs[1]:.2f} top5={accs[2]:.2f}  "
              f"top1conf={np.mean(cat_confs):.3f}", flush=True)

    tot = sum(r[1] for r in results)
    agg = [sum(r[2][k] * r[1] for r in results) / tot for k in range(3)]
    print("\n==== TEACHER on 125-cat deployed task ====", flush=True)
    print(f"  top-1 = {agg[0]*100:.1f}%  top-3 = {agg[1]*100:.1f}%  "
          f"top-5 = {agg[2]*100:.1f}%  (n={tot})", flush=True)
    print("  vs deployed student (int8) 22.5% / 28.7% (Keras float)", flush=True)


if __name__ == "__main__":
    main()
