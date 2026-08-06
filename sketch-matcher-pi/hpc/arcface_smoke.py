"""Smoke test for the ArcFace training path on a real full H100.

Verifies, in ~1 minute:
  1. A GPU is visible after CUDA_VISIBLE_DEVICES=<GPU UUID> (fix for the
     MIG-index mapping bug).
  2. The XLA_FLAGS Triton-GEMM workaround lets the ArcFace MatMul gradient
     compile (fix for "Autotuner could not compile any configs for HLO ...").
Run via hpc/job_arcface_smoke.pbs.
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))

print("== TF ==", tf.__version__)
devs = tf.config.list_physical_devices("GPU")
print("physical GPU devices:", [d.name for d in devs])
assert devs, "NO GPU VISIBLE - GPU UUID addressing failed"

from model import create_model  # noqa: E402

print("== building siamese w/ ArcFace (convnexttiny) ==")
siamese, emb_net, backbone = create_model(
    backbone_name="convnexttiny", include_embeddings=False, use_arcface=True)
print("inputs:", [i.name for i in siamese.inputs])
print("outputs:", [o.name for o in siamese.outputs])


def fake_gen():
    n = 8
    while True:
        sk = np.random.rand(n, 224, 224, 3).astype(np.float32)
        ph = np.random.rand(n, 224, 224, 3).astype(np.float32)
        cats = np.random.randint(0, 125, size=(n,)).astype(np.int64)
        y = (np.random.rand(n, 1) > 0.5).astype(np.float32)
        yield (sk, ph, cats, cats), (y, cats, cats)


print("== fitting 2 steps (gradient through ArcFace MatMul) ==")
h = siamese.fit(fake_gen(), steps_per_epoch=2, epochs=1, verbose=1)
print("SMOKE OK:", {k: float(v[0]) for k, v in h.history.items()})
