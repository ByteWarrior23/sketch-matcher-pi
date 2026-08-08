"""check_collapse.py — verify whether the embedding model has collapsed.

Prints:
  - pairwise cosine similarity between photo embeddings of DIFFERENT categories
    (a healthy model gives ~0.2-0.7; a collapsed model gives ~1.0)
  - same-category pairwise similarity
  - embedding of a blank white image vs the cat/apple sketches
"""
import numpy as np
import tensorflow as tf
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import BEST_MODEL_PATH, PROCESSED_DIR, IMG_SIZE, PADDING
from data_loader import load_processed_data, predict_normalized
import model

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)

model_ = tf.keras.models.load_model(BEST_MODEL_PATH, compile=False)
print("model loaded")

# ---- 1. photo embedding collapse check ----
emb = np.load("models/photo_embeddings.npy")
emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
phl = np.load("models/photo_labels.npy")
rng = np.random.RandomState(0)

n_cat = int(phl.max()) + 1
cat_rows = {c: np.flatnonzero(phl == c) for c in range(n_cat)}

cross = []
same = []
for _ in range(200):
    a, b = rng.choice(n_cat, 2, replace=False)
    if len(cat_rows[a]) == 0 or len(cat_rows[b]) == 0:
        continue
    i = rng.choice(cat_rows[a])
    j = rng.choice(cat_rows[b])
    cross.append(float(emb[i] @ emb[j]))
for _ in range(200):
    c = rng.choice([c for c in range(n_cat) if len(cat_rows[c]) > 1])
    i, j = rng.choice(cat_rows[c], 2, replace=False)
    same.append(float(emb[i] @ emb[j]))

print("PHOTO CROSS-CATEGORY cos sim  mean=%.4f std=%.4f  (healthy: ~0.2-0.7)" % (np.mean(cross), np.std(cross)))
print("PHOTO SAME-CATEGORY cos sim    mean=%.4f std=%.4f  (healthy: >0.7)" % (np.mean(same), np.std(same)))

# ---- 2. input -> embedding sensitivity ----
def emb_of(path):
    img = cv2.imread(path) if path else np.full((IMG_SIZE, IMG_SIZE, 3), 255, np.uint8)
    if path:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray < 250
        coords = np.argwhere(mask)
        if len(coords):
            y1, x1 = coords.min(axis=0); y2, x2 = coords.max(axis=0) + 1
            gray = gray[max(0,y1-PADDING):min(gray.shape[0],y2+PADDING),
                        max(0,x1-PADDING):min(gray.shape[1],x2+PADDING)]
        _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        h, w = b.shape
        s = min(IMG_SIZE/h, IMG_SIZE/w)
        nh, nw = int(h*s), int(w*s)
        r = cv2.resize(b, (nw, nh), interpolation=cv2.INTER_AREA)
        c = np.full((IMG_SIZE, IMG_SIZE), 255, np.uint8)
        y0 = (IMG_SIZE-nh)//2; x0 = (IMG_SIZE-nw)//2
        c[y0:y0+nh, x0:x0+nw] = r
        rgb = cv2.cvtColor(c, cv2.COLOR_GRAY2RGB)
    else:
        rgb = img
    x = (rgb.astype(np.float32)/255.0)[None, ...]
    e = predict_normalized(model_, x, batch_size=1, verbose=0)[0]
    return e / (np.linalg.norm(e) + 1e-8)

blank = emb_of(None)
cat = emb_of("hpc/cat_sketch.png")
apple = emb_of("hpc/apple_sketch.png")

print("\nBLANK (white) embedding norm-ish:", np.linalg.norm(emb_of(None)))
print("blank vs cat    cos = %.4f" % float(blank @ cat))
print("blank vs apple  cos = %.4f" % float(blank @ apple))
print("cat   vs apple  cos = %.4f" % float(cat @ apple))
