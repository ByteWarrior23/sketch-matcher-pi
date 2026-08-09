"""recompute_teacher_embs.py — regenerate teacher sketch+photo embeddings with
the NEW (post-fix) teacher_model.keras, overwriting the stale Aug-7 cache that
was produced by the collapsed model. Must run BEFORE the student phase.
"""
import os
import sys
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf_nthreads = int(os.environ.get("MID_N_THREADS", "4"))
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(tf_nthreads)
tf.config.threading.set_inter_op_parallelism_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # src wins over stale root data_loader.py

import numpy as np
import model  # registers custom layers
from config import PROCESSED_DIR, TEACHER_MODEL_PATH
from data_loader import load_processed_data, predict_normalized

teacher = tf.keras.models.load_model(TEACHER_MODEL_PATH, compile=False)
print(f"teacher loaded: {TEACHER_MODEL_PATH}", flush=True)

sketches, photos, _, _, _, _ = load_processed_data()
print(f"data: sketches={sketches.shape} photos={photos.shape}", flush=True)

t0 = time.time()
t_sk = predict_normalized(teacher, sketches, batch_size=128, verbose=1)
t_ph = predict_normalized(teacher, photos, batch_size=128, verbose=1)
print(f"embedded in {time.time()-t0:.0f}s: {t_sk.shape} {t_ph.shape}", flush=True)

np.save(PROCESSED_DIR / "teacher_sketch_embs.npy", t_sk)
np.save(PROCESSED_DIR / "teacher_photo_embs.npy", t_ph)
print("saved fresh teacher embeddings (overwrote stale cache)", flush=True)
