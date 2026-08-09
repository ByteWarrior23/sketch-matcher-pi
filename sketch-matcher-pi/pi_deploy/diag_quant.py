"""
diag_quant.py — float Keras student vs int8 TFLite: how much does quantization
hurt? Computes both embeddings for the same real sketches and compares, plus
top-1 retrieval for each against the deployed photo DB.

Usage: python pi_deploy/diag_quant.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # src must win over stale root data_loader.py

import tensorflow as tf  # noqa: E402

from camera import CameraCapture  # noqa: E402
from matcher import SketchMatcher  # noqa: E402

MODEL_DIR = Path(__file__).parent / "model_data"
TEST_DIR = ROOT / "hpc" / "sketch_test_export"
KERAS_PATH = ROOT / "models" / "best_model.keras"

import model  # noqa: E402  (registers custom layers)


def main():
    matcher = SketchMatcher(MODEL_DIR)
    keras = tf.keras.models.load_model(str(KERAS_PATH), compile=False, safe_mode=False)
    print(f"Keras model: in={keras.input_shape} out={keras.output_shape}")

    photo_emb = matcher.photo_embeddings
    photo_lbl = matcher.photo_labels
    photo_norm = photo_emb / (np.linalg.norm(photo_emb, axis=1, keepdims=True) + 1e-8)
    labels = matcher.labels

    cam = CameraCapture()
    rows = []
    for cdir in sorted(TEST_DIR.iterdir()):
        if not cdir.is_dir():
            continue
        cat = cdir.name.split("_", 1)[1]
        for p in sorted(cdir.glob("*.png")):
            img = cv2.imread(str(p))
            pre = cam.preprocess(img)  # (1,224,224,3) float [0,1]
            e_tflite = matcher.embed(pre)
            e_keras = keras.predict(pre, verbose=0)[0]
            n = np.linalg.norm(e_keras)
            e_keras = e_keras / n if n > 0 else e_keras
            rows.append((cat, p.name, e_tflite, e_keras))

    print(f"\nEvaluated {len(rows)} real sketches. Cosine(tflite, keras) per sketch:")
    cos_vals = []
    hits_t, hits_k = 0, 0
    per_cat = {}
    for cat, name, e_t, e_k in rows:
        c = float(np.dot(e_t, e_k))
        cos_vals.append(c)
        sim_t = photo_norm @ e_t
        sim_k = photo_norm @ e_k
        n_cats = int(photo_lbl.max()) + 1
        cat_max_t = np.full(n_cats, -np.inf)
        cat_max_k = np.full(n_cats, -np.inf)
        np.maximum.at(cat_max_t, photo_lbl, sim_t)
        np.maximum.at(cat_max_k, photo_lbl, sim_k)
        top_t = labels[int(np.argmax(cat_max_t))]
        top_k = labels[int(np.argmax(cat_max_k))]
        hits_t += top_t == cat
        hits_k += top_k == cat
        per_cat.setdefault(cat, [0, 0])
        per_cat[cat][0] += top_k == cat
        per_cat[cat][1] += 1

    print(f"  cosine mean={np.mean(cos_vals):.4f} min={min(cos_vals):.4f} "
          f"p05={np.percentile(cos_vals,5):.4f}")
    print(f"  top-1 acc  TFLite(int8): {hits_t}/{len(rows)} = {hits_t/len(rows)*100:.1f}%")
    print(f"  top-1 acc  Keras(float): {hits_k}/{len(rows)} = {hits_k/len(rows)*100:.1f}%")
    print("  Keras top-1 by category:", {k: f"{v[0]}/{v[1]}" for k, v in per_cat.items()})


if __name__ == "__main__":
    main()
