"""check_teacher.py — is the TEACHER embedding collapsed too?"""
import numpy as np
import tensorflow as tf
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import BEST_MODEL_PATH, PROCESSED_DIR, IMG_SIZE, PADDING, TEACHER_MODEL_PATH
from data_loader import predict_normalized
import model

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)

print("loading teacher:", TEACHER_MODEL_PATH)
t = tf.keras.models.load_model(TEACHER_MODEL_PATH, compile=False)
print("teacher loaded")

# teacher stored embeddings (distillation targets), if present
for f in ["teacher_sketch_embeddings.npy", "teacher_photo_embeddings.npy",
          "sketch_teacher_emb.npy", "photo_teacher_emb.npy"]:
    p = Path("models") / f
    if p.exists():
        e = np.load(p)
        e_n = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
        # sample pairwise cross-similarity
        rng = np.random.RandomState(0)
        idx = rng.choice(len(e_n), 200, replace=False)
        sub = e_n[idx]
        pair = (sub * sub).sum(axis=1)
        cross = (sub @ sub.T)
        off = cross[~np.eye(len(sub), dtype=bool)]
        print(f"{f}: shape={e.shape} mean_cos(vs self after norm)={pair.mean():.4f} "
              f"mean_offdiag_cos={off.mean():.4f} std={off.std():.4f}")

# live teacher forward pass: blank vs cat
def emb_of(model_, path):
    img = np.full((IMG_SIZE, IMG_SIZE, 3), 255, np.uint8) if not path else None
    if path:
        import cv2
        img = cv2.imread(path)
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

blank = emb_of(t, None)
cat = emb_of(t, "hpc/cat_sketch.png")
apple = emb_of(t, "hpc/apple_sketch.png")
print("\nTEACHER blank vs cat  cos = %.4f" % float(blank @ cat))
print("TEACHER blank vs apple cos = %.4f" % float(blank @ apple))
print("TEACHER cat vs apple   cos = %.4f" % float(cat @ apple))
