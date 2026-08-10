"""finalize_teacher.py — produce the artifacts the teacher gate needs.

If the main 48h teacher job is walltime-KILLED mid-stage-3, train.py's
end-of-teacher save NEVER runs (no teacher_model.keras, no embedding cache).
This script recovers from the stage-3 checkpoint:

  1. loads teacher_stage3*/best_epoch.keras  (the contrastive siamese)
  2. extracts the shared embedding_network  -> saves TEACHER_MODEL_PATH
  3. verifies a forward pass (shape + L2 norm == 1)
  4. force-recomputes + caches teacher_sketch_embs.npy / teacher_photo_embs.npy

Run BEFORE verify_teacher.py (GPU, cluster):
    python hpc/finalize_teacher.py
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # src/ wins over stale root data_loader.py

import model  # noqa: E402  (registers custom layers)
from config import TEACHER_MODEL_PATH, CHECKPOINT_DIR  # noqa: E402
from train import compute_teacher_embeddings  # noqa: E402


def main():
    ckpt = sorted(CHECKPOINT_DIR.glob("teacher_stage3*/best_epoch.keras"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpt:
        raise SystemExit("no teacher_stage3*/best_epoch.keras found")
    ckpt = ckpt[0]
    print(f"loading checkpoint: {ckpt}", flush=True)

    siamese = tf.keras.models.load_model(str(ckpt), compile=False)
    net = siamese.get_layer("embedding_network")
    print(f"embedding net extracted; trainable params: {net.count_params()}", flush=True)

    net.save(TEACHER_MODEL_PATH)
    print(f"saved teacher embedding net: {TEACHER_MODEL_PATH}", flush=True)

    x = np.zeros((2, 224, 224, 3), dtype=np.float32)
    e = net.predict(x, verbose=0)
    print(f"sanity: emb shape {e.shape}, L2 norm {np.linalg.norm(e, axis=1)}", flush=True)
    if not np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-3):
        raise SystemExit("embeddings are not L2-normalized — aborting")

    compute_teacher_embeddings(net, force=True)
    print("DONE: teacher_model.keras + embedding cache ready for the gate.", flush=True)


if __name__ == "__main__":
    main()
