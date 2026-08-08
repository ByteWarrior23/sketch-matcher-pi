import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
import sys, json
import numpy as np
import tensorflow as tf
sys.path.insert(0, "/Data4/ee_24126016/sketch_matcher/src")
from config import PROCESSED_DIR
from data_loader import load_processed_data, create_data_generators, predict_normalized
import model as M

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)

# 1) generator batch content
_, _, _, _ = create_data_generators(batch_size=16, pairs_per_epoch=160, arcface=False)
gen, val, test, names = create_data_generators(batch_size=16, pairs_per_epoch=160, arcface=False)
X, y = gen[0]
sk, ph = X
print("=== generator batch ===")
print("inputs:", [t.shape for t in X])
print("labels:", np.ravel(y)[:16], "unique:", np.unique(y))
print("sk pixel min/max:", sk.min(), sk.max(), "ph pixel min/max:", ph.min(), ph.max())
print("sk distinct (unique flat values sample):", np.unique(sk[:2])[:5])
d = np.abs(sk[0].astype(np.float32) - sk[1].astype(np.float32)).sum()
print("sum|sk0-sk1| =", d, "(0 would mean identical images)")

# 2) generator with arcface -> check category label distribution
gen2, _, _, _ = create_data_generators(batch_size=16, pairs_per_epoch=160, arcface=True)
X2, y2 = gen2[0]
print("arcface inputs:", [t.shape for t in X2])
print("arcface y tuple len:", len(y2))
if len(y2) >= 3:
    sk_cats, ph_cats = y2[1], y2[2]
    print("sk_cats unique:", np.unique(sk_cats)[:10], "ph_cats unique:", np.unique(ph_cats)[:10])
    print("sk_cats range:", sk_cats.min(), sk_cats.max())
    # check that positive pairs have same cat and negatives differ
    lbls = np.ravel(y2[0])
    for i in range(min(16, len(lbls))):
        print(f"  pair {i}: label={lbls[i]} sk_cat={sk_cats[i]} ph_cat={ph_cats[i]}")

# 3) probe trained model intermediate layers on DIFFERENT images
print("\n=== trained model probes ===")
net = tf.keras.models.load_model("/Data4/ee_24126016/sketch_matcher/models/best_model.keras", compile=False)
print("model layers:", [l.name + ":" + l.__class__.__name__ for l in net.layers])

sketch_labels = np.load(PROCESSED_DIR / "sketch_labels_u8.npy")
sk_mmap = np.load(PROCESSED_DIR / "sketches_u8.npy", mmap_mode="r")
# pick 3 different categories, 1 sketch each
cats = [0, 40, 80]
rows = [int(np.flatnonzero(sketch_labels == c)[0]) for c in cats]
imgs = np.stack([sk_mmap[r].astype(np.float32) / 255.0 for r in rows])
print("input images (should differ):", [np.abs(imgs[0] - imgs[i]).sum() for i in range(3)])

probe = tf.keras.Model(inputs=net.input, outputs=[net.get_layer("embedding").output, net.output])
pre_norm, emb = probe(imgs, training=False)
print("pre-L2Norm embedding norms:", np.linalg.norm(pre_norm, axis=1))
print("pre-L2Norm cos (img0,img1):", float(np.dot(pre_norm[0], pre_norm[1]) / (np.linalg.norm(pre_norm[0]) * np.linalg.norm(pre_norm[1]) + 1e-8)))
print("final emb cos (img0,img1):", float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-8)))
