"""smoke_test.py - end-to-end pipeline smoke test on a tiny synthetic dataset.

Proves the full chain works BEFORE real HPC training:
  tiny Sketchy dataset -> preprocess -> train (1 epoch) -> export TFLite
  -> match a sketch against the photo embeddings.

Usage:
    python smoke_test.py                          # synthetic flower/tree
    python smoke_test.py path/to/sketch.jpg       # your own sketch as query

WARNING: overwrites data/processed/* and models/*. Not for real training.
"""

import os
import sys
import random
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "pi_deploy"))
sys.path.insert(0, str(BASE / "src"))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["SKETCH_BATCH1"] = "8"
os.environ["SKETCH_BATCH2"] = "8"
os.environ["SKETCH_BATCH3"] = "8"

import numpy as np
import cv2

# ---- override config BEFORE importing the pipeline modules ----
import config
config.ENABLE_DISTILLATION = False       # skip teacher: just MobileNetV2
config.STAGE1_EPOCHS = 1
config.STAGE2_EPOCHS = 1
config.STAGE3_EPOCHS = 1
config.PAIRS_PER_EPOCH = 64
config.EARLY_STOPPING_PATIENCE = 1
config.USE_TUBERLIN = False
config.USE_QUICKDRAW = False
config.USE_IMAGENETSKETCH = False
config.TRAIN_SPLIT = 0.34    # 3 cats -> 1 train / 1 val / 1 test
config.VAL_SPLIT = 0.33

RAW = config.RAW_DIR / "sketchy"
CATS = ["flower", "tree", "sun"]


# =============================================================================
# Synthetic dataset
# =============================================================================
def make_flower_sketch():
    img = np.full((224, 224), 255, np.uint8)
    cx, cy = 112, 95
    for i in range(8):
        a = 2 * np.pi * i / 8
        cv2.circle(img, (int(cx + 26 * np.cos(a)), int(cy + 26 * np.sin(a))), 17, 0, 2)
    cv2.circle(img, (cx, cy), 13, 0, 2)
    cv2.line(img, (cx, cy + 20), (cx, 200), 0, 3)
    cv2.line(img, (cx, 155), (cx - 32, 138), 0, 3)
    cv2.line(img, (cx, 155), (cx + 32, 138), 0, 3)
    return img


def make_tree_sketch():
    img = np.full((224, 224), 255, np.uint8)
    pts = np.array([[112, 35], [55, 155], [169, 155]], np.int32)
    cv2.fillPoly(img, [pts], 0)
    cv2.rectangle(img, (95, 155), (129, 205), 0, -1)
    return img


def make_flower_photo():
    img = np.full((224, 224, 3), 210, np.uint8)
    cx, cy = 112, 100
    for i in range(8):
        a = 2 * np.pi * i / 8
        r = random.randint(20, 28)
        color = random.choice([(220, 40, 200), (40, 220, 240), (60, 60, 240)])
        cv2.circle(img, (int(cx + 40 * np.cos(a)), int(cy + 40 * np.sin(a))), r, color, -1)
    cv2.circle(img, (cx, cy), 20, (60, 200, 255), -1)
    cv2.line(img, (cx, cy + 35), (cx, 205), (60, 120, 40), 6)
    return img


def make_tree_photo():
    img = np.full((224, 224, 3), 230, np.uint8)
    canopy = random.choice([(40, 160, 60), (60, 180, 80), (30, 130, 50)])
    trunk = random.choice([(70, 60, 50), (90, 75, 60)])
    cv2.rectangle(img, (95, 155), (129, 205), trunk, -1)
    pts = np.array([[112, 35], [55, 160], [169, 160]], np.int32)
    cv2.fillPoly(img, [pts], canopy)
    return img


def make_sun_sketch():
    img = np.full((224, 224), 255, np.uint8)
    cx, cy = 112, 112
    cv2.circle(img, (cx, cy), 45, 0, 3)
    for i in range(12):
        a = 2 * np.pi * i / 12
        cv2.line(img,
                 (int(cx + 60 * np.cos(a)), int(cy + 60 * np.sin(a))),
                 (int(cx + 95 * np.cos(a)), int(cy + 95 * np.sin(a))), 0, 3)
    return img


def make_sun_photo():
    img = np.full((224, 224, 3), 200, np.uint8)
    cv2.circle(img, (112, 112), 60, (40, 220, 255), -1)
    for i in range(12):
        a = 2 * np.pi * i / 12
        cv2.line(img,
                 (int(112 + 70 * np.cos(a)), int(112 + 70 * np.sin(a))),
                 (int(112 + 100 * np.cos(a)), int(112 + 100 * np.sin(a))),
                 (40, 220, 255), 8)
    return img


def build_dataset():
    makers = {
        "flower": (make_flower_sketch, make_flower_photo),
        "tree": (make_tree_sketch, make_tree_photo),
        "sun": (make_sun_sketch, make_sun_photo),
    }
    for cat in CATS:
        sk_dir = RAW / "sketch" / cat
        ph_dir = RAW / "photo" / cat
        sk_dir.mkdir(parents=True, exist_ok=True)
        ph_dir.mkdir(parents=True, exist_ok=True)
        sk_maker, ph_maker = makers[cat]
        for i in range(3):
            cv2.imwrite(str(sk_dir / f"s{i}.png"), sk_maker())
            cv2.imwrite(str(ph_dir / f"p{i}.jpg"), ph_maker())
    print(f"Synthetic dataset: {RAW}")
    for cat in CATS:
        print(f"  {cat}: 3 sketches + 3 photos")


# =============================================================================
# Pipeline
# =============================================================================
def run_preprocess():
    import preprocess
    preprocess.main()


def run_train():
    from model import create_model
    import train

    siamese, embedding_net, backbone = create_model(
        backbone_name=config.BACKBONE, loss_type=config.LOSS_TYPE,
        include_embeddings=False, embedding_dim=config.EMBEDDING_DIM)
    train.run_three_stages(siamese, embedding_net, backbone, "smoke",
                           augment=False)
    siamese.save(config.FINAL_MODEL_PATH)
    embedding_net.save(config.BEST_MODEL_PATH)
    print(f"Saved: {config.BEST_MODEL_PATH}")


def run_export():
    import export_tflite
    export_tflite.main()


def run_match_test(query_path):
    from matcher import SketchMatcher
    from preprocess import process_sketch

    model_dir = BASE / "pi_deploy" / "model_data"
    matcher = SketchMatcher(model_dir)

    q = cv2.imread(str(query_path), cv2.IMREAD_GRAYSCALE)
    if q is None:
        print(f"Cannot read query image: {query_path}")
        return 1
    qp = process_sketch(q).astype(np.float32) / 255.0

    print(f"\nQuery sketch: {query_path}")
    results, inf_ms, search_ms, rejected = matcher.match(np.expand_dims(qp, 0))
    print(f"  Inference: {inf_ms:.1f}ms | Search: {search_ms:.1f}ms")
    print(f"  Rejected (NOT FOUND): {rejected}")
    for i, (cat, conf) in enumerate(results):
        print(f"  {i + 1}. {cat}: {conf:.1f}%")
    return 0


def main():
    print("=" * 60)
    print("SMOKE TEST - tiny end-to-end run (not real training)")
    print("=" * 60)

    build_dataset()

    print("\n[1/4] PREPROCESS")
    run_preprocess()

    print("\n[2/4] TRAIN (1 epoch x 3 stages, MobileNetV2, CPU)")
    run_train()

    print("\n[3/4] EXPORT TFLite")
    run_export()

    print("\n[4/4] MATCH TEST")
    query = sys.argv[1] if len(sys.argv) > 1 else str(RAW / "sketch" / "flower" / "s0.png")
    return run_match_test(query)


if __name__ == "__main__":
    sys.exit(main())
