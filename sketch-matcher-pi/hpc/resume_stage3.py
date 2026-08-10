"""resume_stage3.py — continue Stage 3 after the main GPU job is walltime-killed
mid-Stage-3 (or to extend training past one 48h window). Works for BOTH the
teacher (plain contrastive siamese) and the student (distill siamese).

Loads the latest <phase>_stage3/best_epoch.keras into a freshly-built siamese,
VERIFIES the weights actually transferred, then continues Stage 3 with the same
early-stopping / reduce-LR schedule, writing checkpoints to
<phase>_stage3_resume/ and the final best_model.keras / final_model.keras.

Usage (GPU, cluster):
    python hpc/resume_stage3.py --phase teacher|student [--ckpt PATH] [--epochs N]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # src/ wins over stale root data_loader.py

import model  # noqa: E402  (registers custom layers)
import data_loader  # noqa: E402
print("data_loader loaded from:", data_loader.__file__, flush=True)
from config import (  # noqa: E402
    BACKBONE, TEACHER_BACKBONE, CHECKPOINT_DIR, BEST_MODEL_PATH, FINAL_MODEL_PATH,
    TEACHER_MODEL_PATH,
    STAGE3_EPOCHS, STAGE3_BATCH_SIZE, STAGE3_LEARNING_RATE,
)
from model import create_model, unfreeze_all  # noqa: E402
from train import (make_generators, train_stage,  # noqa: E402
                   compute_teacher_embeddings)


def latest_stage3_ckpt(phase):
    cands = sorted(CHECKPOINT_DIR.glob(f"{phase}_stage3*/best_epoch.keras"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"no {phase}_stage3 best_epoch.keras found")
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="student", choices=["teacher", "student"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--epochs", type=int, default=STAGE3_EPOCHS)
    args = ap.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else latest_stage3_ckpt(args.phase)
    print(f"resuming stage3 ({args.phase}) from: {ckpt}  "
          f"(mtime {time.strftime('%H:%M:%S', time.localtime(ckpt.stat().st_mtime))})",
          flush=True)

    backbone = TEACHER_BACKBONE if args.phase == "teacher" else BACKBONE
    distill = (args.phase == "student")
    print(f"building {'distill ' if distill else ''}siamese ({backbone})...", flush=True)
    siamese, embedding_net, backbone_net = create_model(
        backbone_name=backbone, loss_type="contrastive",
        include_embeddings=distill, embedding_dim=model.EMBEDDING_DIM,
        distill=distill)
    print("loading weights from checkpoint...", flush=True)
    siamese.load_weights(str(ckpt))

    # verify the embedding subnet really transferred
    net = siamese.get_layer("embedding_network")
    ref = tf.keras.models.load_model(str(ckpt), compile=False)
    net_ref = ref.get_layer("embedding_network")
    a = net_ref.get_layer("embedding").get_weights()[0]
    b = net.get_layer("embedding").get_weights()[0]
    if not np.allclose(a, b):
        raise SystemExit("weight-mismatch after load_weights — aborting")
    print("weight match: OK (embedding dense kernel identical)", flush=True)
    del ref

    t_sk = t_ph = None
    if distill:
        t_sk = np.load(data_loader.PROCESSED_DIR / "teacher_sketch_embs.npy")
        t_ph = np.load(data_loader.PROCESSED_DIR / "teacher_photo_embs.npy")

    train_gen, val_gen, _, _ = make_generators(
        STAGE3_BATCH_SIZE, True, t_sk, t_ph)

    print(f"continuing Stage 3 for up to {args.epochs} epochs "
          f"(batch {STAGE3_BATCH_SIZE}, LR {STAGE3_LEARNING_RATE})", flush=True)
    train_stage(siamese, embedding_net, backbone_net, train_gen, val_gen,
                epochs=args.epochs, batch_size=STAGE3_BATCH_SIZE,
                learning_rate=STAGE3_LEARNING_RATE,
                stage_name=f"{args.phase}_stage3_resume", freeze_fn=unfreeze_all)

    if args.phase == "teacher":
        # End-of-teacher run: save the embedding net + recompute the cache
        # (matches what train.py's teacher phase does after stage 3).
        net = siamese.get_layer("embedding_network")
        net.save(TEACHER_MODEL_PATH)
        print(f"saved teacher embedding net: {TEACHER_MODEL_PATH}", flush=True)
        compute_teacher_embeddings(net, force=True)
    else:
        siamese.save(FINAL_MODEL_PATH)
        embedding_net.save(BEST_MODEL_PATH)
        print(f"saved: {FINAL_MODEL_PATH}, {BEST_MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
