import json
import numpy as np
from pathlib import Path

PROCESSED = Path("/Data4/ee_24126016/sketch_matcher/data/processed")

skl = np.load(PROCESSED / "sketch_labels_u8.npy")
phl = np.load(PROCESSED / "photo_labels_u8.npy")

print("sketch_labels: shape", skl.shape, "min", skl.min(), "max", skl.max())
print("photo_labels:  shape", phl.shape, "min", phl.min(), "max", phl.max())
u_sk = np.unique(skl)
u_ph = np.unique(phl)
print("unique sketch cats:", len(u_sk))
print("unique photo cats: ", len(u_ph))
print("sketch cats 0..124 present:", np.isin(np.arange(125), u_sk).all())
print("photo cats  0..124 present:", np.isin(np.arange(125), u_ph).all())
print("sketch label histogram (first 10):")
print(np.bincount(skl.astype(np.int64), minlength=130)[:12])

meta = json.load(open(PROCESSED / "metadata.json"))
print("train_categories:", len(meta.get("train_categories", [])))
print("val_categories:", len(meta.get("val_categories", [])))
print("test_categories:", len(meta.get("test_categories", [])))
print("train cats sample:", meta.get("train_categories", [])[:5])
