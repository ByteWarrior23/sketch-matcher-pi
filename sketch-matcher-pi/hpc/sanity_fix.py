"""sanity_fix.py — verify the ROOT-CAUSE fixes before the full retrain.

Part A  (input-range fix, root cause 1): build embedding nets for BOTH the
        teacher (ConvNeXtTiny) and student (MobileNetV3Large) with the new
        BACKBONE_INPUT_MAP. Embed real photos of a few categories + a blank
        canvas. A healthy untrained ImageNet backbone must give
        cross-category cos << 1.0 and same-category cos >> cross. The old bug
        collapsed everything to cos ~= 1.0.

Part B  (loss fix, root cause 2): mini-train the TEACHER with contrastive
        loss (ArcFace OFF) for 1 epoch at a tiny pair budget and confirm the
        loss actually DECREASES. The old circle+arcface setup froze
        bit-identical at ~0.31/0.05.

Runs on CPU (cpuq). Threads capped for the node watchdog.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # src wins over stale root copy

import numpy as np
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)

import model  # registers layers + BACKBONE_INPUT_MAP
from data_loader import load_processed_data, create_data_generators, predict_normalized


def embed_check(backbone_name):
    print("\n" + "=" * 70)
    print(f"Part A: embedding distinctness check  backbone={backbone_name}")
    print("=" * 70)
    net, _ = model.build_embedding_network(backbone_name=backbone_name)
    print(f"  built embedding net OK")

    sketches, photos, sketch_labels, photo_labels, names, _ = load_processed_data()

    rng = np.random.RandomState(0)
    cats = sorted(set(sketch_labels) & set(photo_labels))
    chosen = list(rng.choice(cats, size=min(6, len(cats)), replace=False))
    rows_sk = [np.flatnonzero(sketch_labels == c)[0] for c in chosen]
    rows_ph = [np.flatnonzero(photo_labels == c)[0] for c in chosen]

    imgs = np.concatenate([photos[r][None] for r in rows_ph], axis=0)
    t0 = time.time()
    emb = predict_normalized(net, imgs, batch_size=8, verbose=0)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    print(f"  embedded {len(imgs)} photos in {time.time()-t0:.1f}s")

    same, cross = [], []
    for i in range(len(chosen)):
        for j in range(len(chosen)):
            c = float(emb[i] @ emb[j])
            if i == j:
                same.append(c)
            else:
                cross.append(c)
    print(f"  SAME-category cos: mean={np.mean(same):.4f}  (expect >> cross)")
    print(f"  CROSS-category cos: mean={np.mean(cross):.4f} std={np.std(cross):.4f}  (expect << 1.0, old bug gave ~1.0)")

    blank = np.full((1, 224, 224, 3), 1.0, dtype=np.float32)
    eb = predict_normalized(net, blank, batch_size=1, verbose=0)[0]
    eb /= (np.linalg.norm(eb) + 1e-8)
    print(f"  blank vs photo cos: mean={np.mean([float(eb @ e) for e in emb]):.4f}  (expect << 1.0)")

    ok = np.mean(cross) < 0.9 and np.mean(same) > np.mean(cross) + 0.05
    print(f"  Part A {'PASSED' if ok else 'FAILED'}")
    return ok


def mini_train_teacher():
    print("\n" + "=" * 70)
    print("Part B: teacher mini-train (contrastive, ArcFace OFF)")
    print("=" * 70)
    from config import EMBEDDING_DIM, LOSS_TYPE, USE_ARC_FACE
    print(f"  LOSS_TYPE={LOSS_TYPE}  USE_ARC_FACE={USE_ARC_FACE}")

    siamese, net, base = model.create_model(
        backbone_name="convnexttiny", loss_type=LOSS_TYPE,
        include_embeddings=False, embedding_dim=EMBEDDING_DIM,
        use_arcface=USE_ARC_FACE)
    print("  built teacher siamese OK")

    train_gen, val_gen, _, _ = create_data_generators(
        batch_size=16, pairs_per_epoch=256, augment=False,
        arcface=USE_ARC_FACE)
    print("  built generators OK")

    t0 = time.time()
    h = siamese.fit(train_gen, epochs=2, validation_data=val_gen, verbose=1)
    dt = time.time() - t0
    losses = h.history["loss"]
    print(f"  train loss: {losses}")
    decreasing = losses[-1] < losses[0] * 0.98
    print(f"  Part B {'PASSED' if decreasing else 'FAILED'}  (2 epochs in {dt:.0f}s)")
    return decreasing


def main():
    a = embed_check("convnexttiny")
    b = embed_check("mobilenetv3large")
    c = mini_train_teacher()
    print("\n" + "=" * 70)
    print(f"RESULT:  PartA_teacher={a}  PartA_student={b}  PartB_loss={c}")
    ok = a and b and c
    print("OVERALL: " + ("PASSED - safe to launch full retrain" if ok else "FAILED - do NOT retrain yet"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
