"""probe_convnext.py — find the input range that un-collapses ConvNeXtTiny.

The sanity check showed ConvNeXtTiny still emits near-constant embeddings
(cross-cat cos ~0.997) when fed [0,1], contradicting the earlier "Normalization,
[0,1] fine" conclusion. Probe several input ranges on real photos and report
feature distinctness (cross-category cosine) per range.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)

from data_loader import load_processed_data

base = tf.keras.applications.ConvNeXtTiny(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg")
print("layers[:2]:", [(l.name, l.__class__.__name__) for l in base.layers[:2]])
for l in base.layers[:2]:
    cfg = getattr(l, "get_config", lambda: {})()
    if cfg.get("mean") is not None:
        print("  Normalization mean:", cfg["mean"][:3], "var:", cfg["variance"][:3])

sketches, photos, sk_l, ph_l, names, _ = load_processed_data()
rng = np.random.RandomState(0)
cats = sorted(set(sk_l) & set(ph_l))
chosen = list(rng.choice(cats, size=6, replace=False))
imgs = np.concatenate([photos[np.flatnonzero(ph_l == c)[0]][None] for c in chosen], axis=0)
print("img dtype/range:", imgs.dtype, imgs.min(), imgs.max())

transforms = {
    "01_asis": lambda x: x,
    "01_to_0255": lambda x: x * 255.0,
    "01_to_11": lambda x: x * 2.0 - 1.0,
    "01_to_meannorm255": lambda x: (x * 255.0 - 0.485 * 255.0) / (0.229 * 255.0),
    "01_to_meannorm1": lambda x: (x - 0.485) / 0.229,
}
for name, fn in transforms.items():
    x = np.stack([fn(i) for i in imgs]).astype(np.float32)
    e = base.predict(x, verbose=0)
    e /= (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
    cross = []
    for i in range(len(chosen)):
        for j in range(len(chosen)):
            if i != j:
                cross.append(float(e[i] @ e[j]))
    print(f"{name:18s} feat_norm={np.linalg.norm(base.predict(x[:1], verbose=0)):.3f} "
          f"cross_cos={np.mean(cross):.4f} std={np.std(cross):.4f}")
