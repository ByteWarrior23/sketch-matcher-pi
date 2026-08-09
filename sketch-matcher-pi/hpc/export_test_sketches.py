"""export_test_sketches.py — export a sample of REAL Sketchy training sketches as
PNGs so the deployed TFLite matcher can be tested offline on the laptop.

Writes: DATA_ROOT/sketch_test_export/<idx>_<name>/<i>.png  (10 sketches per cat)
Usage: python hpc/export_test_sketches.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
from config import PROCESSED_DIR  # noqa: E402

CATS = ["flower", "banana", "apple", "star", "clock", "hotdog",
        "knife", "airplane", "cup", "car_(sedan)", "sun", "moon"]
N_PER_CAT = 10
SEED = 123

with open(PROCESSED_DIR / "category_names.json") as f:
    names = {int(k): v for k, v in json.load(f).items()}
name2idx = {v: k for k, v in names.items()}

sk = np.load(PROCESSED_DIR / "sketches_u8.npy", mmap_mode="r")
labels = np.load(PROCESSED_DIR / "sketch_labels_u8.npy").astype(int)

out = ROOT / "sketch_test_export"
out.mkdir(exist_ok=True)

rng = np.random.RandomState(SEED)
for cname in CATS:
    idx = name2idx.get(cname)
    if idx is None:
        print(f"SKIP (not in 125): {cname}")
        continue
    rows = np.flatnonzero(labels == idx)
    rows = rng.choice(rows, size=min(N_PER_CAT, len(rows)), replace=False)
    cdir = out / f"{idx}_{cname}"
    cdir.mkdir(exist_ok=True)
    for j, r in enumerate(rows):
        img = np.asarray(sk[r])  # (224,224,3) uint8
        cv2.imwrite(str(cdir / f"{j}.png"), img)
    print(f"{cname} (cat {idx}): exported {len(rows)} sketches to {cdir}")

print(f"\nDONE -> {out}")
