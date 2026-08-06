import os
import time

import tensorflow as tf

print("TF version:", tf.__version__)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))

gpus = tf.config.list_physical_devices("GPU")
print("Visible GPU count:", len(gpus))
if not gpus:
    raise SystemExit("ERROR: no GPU visible to TensorFlow!")

for g in gpus:
    print("GPU:", g.name, "->", g.device_type)

with tf.device("/GPU:0"):
    a = tf.random.normal((4096, 4096))
    b = tf.random.normal((4096, 4096))
    t0 = time.time()
    for _ in range(5):
        c = tf.matmul(a, b)
    dt = time.time() - t0

print(f"5x matmul(4096x4096) on GPU took {dt:.2f}s (~{5 * 2 * 4096 ** 3 / dt / 1e12:.2f} TFLOP/s)")
print("GPU sanity test PASSED")
