"""final_health_check.py — post-train health report on the FINAL teacher.

Validates the artifacts that feed the student (cached teacher embeddings) and
the final teacher_model.keras, WITHOUT re-running model inference:
  - teacher model file present (mtime/size)
  - NaN/Inf scan of cached embeddings + loss histories
  - embedding distinctness (collapse guard: cross-cat cos < 0.9)
  - TEST top-1/3/5 retrieval on the 13 held-out categories (the real metric)
  - SEEN top-1
All matrix math on CPU (256-d cached embs), fast. Exit 0 healthy / 1 RED FLAG.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # src wins over stale root data_loader.py

from config import PROCESSED_DIR, TEACHER_MODEL_PATH  # noqa: E402

RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"; RESET = "\033[0m"
flag = lambda m: f"{RED}{m}{RESET}"; ok = lambda m: f"{GREEN}{m}{RESET}"
warn = lambda m: f"{YELLOW}{m}{RESET}"

N_SAMP = int(__import__("os").environ.get("MID_N_SAMP", "80"))
EMB_SK = PROCESSED_DIR / "teacher_sketch_embs.npy"
EMB_PH = PROCESSED_DIR / "teacher_photo_embs.npy"
LBL_SK = PROCESSED_DIR / "sketch_labels_u8.npy"
LBL_PH = PROCESSED_DIR / "photo_labels_u8.npy"


def topk(sk, ph, sk_l, ph_l, chosen):
    rows_sk = np.concatenate([np.flatnonzero(sk_l == c)[:N_SAMP] for c in chosen])
    rows_ph = np.concatenate([np.flatnonzero(ph_l == c)[:N_SAMP] for c in chosen])
    sim = sk[rows_sk] @ ph[rows_ph].T
    labels = sk_l[rows_sk]
    cols = ph_l[rows_ph]
    out = {}
    for k in (1, 3, 5):
        top = np.argsort(-sim, axis=1)[:, :k]
        n = len(labels)
        out[k] = sum(labels[i] in cols[top[i]] for i in range(n)) / n * 100
    return out, len(rows_sk)


def main():
    problems = []

    with open(PROCESSED_DIR / "metadata.json") as f:
        metadata = json.load(f)

    t0 = time.time()
    sk = np.load(EMB_SK)
    ph = np.load(EMB_PH)
    sk_l = np.load(LBL_SK).astype(np.int64).ravel()
    ph_l = np.load(LBL_PH).astype(np.int64).ravel()
    assert sk.shape[0] == len(sk_l) and ph.shape[0] == len(ph_l), "embed/label mismatch"
    print(f"embeddings: sketches={sk.shape} photos={ph.shape} "
          f"(loaded {time.time()-t0:.0f}s)")

    if not np.isfinite(sk).all() or not np.isfinite(ph).all():
        problems.append("NaN/Inf in cached teacher embeddings")
        print(flag("  NaN scan embeddings: RED FLAG"))
    else:
        print(ok("  NaN scan embeddings: clean"))

    norms = np.linalg.norm(sk, axis=1)
    normp = np.linalg.norm(ph, axis=1)
    print(f"  embed norms: sketch min={norms.min():.3f} mean={norms.mean():.3f} | "
          f"photo min={normp.min():.3f} mean={normp.mean():.3f}")
    if norms.min() < 1e-6 or normp.min() < 1e-6:
        problems.append("zero-norm embeddings (collapse)")
        print(flag("  RED FLAG: zero-norm embeddings"))

    if TEACHER_MODEL_PATH.exists():
        m = TEACHER_MODEL_PATH.stat()
        print(f"  teacher_model.keras: {m.st_size/1e6:.0f} MB, "
              f"mtime {time.strftime('%Y-%m-%d %H:%M', time.localtime(m.st_mtime))}")
    else:
        problems.append("teacher_model.keras missing")
        print(flag("  RED FLAG: teacher_model.keras missing"))

    test_cats = sorted(int(c) for c in metadata["test_categories"])
    train_cats = sorted(int(c) for c in metadata["train_categories"])

    t0 = time.time()
    res_t, n_t = topk(sk, ph, sk_l, ph_l, test_cats)
    res_s, n_s = topk(sk, ph, sk_l, ph_l, train_cats[:12])
    print(f"  TEST ({len(test_cats)} held-out cats, {n_t} sketches): "
          f"top-1={ok(f'{res_t[1]:.1f}%')} top-3={res_t[3]:.1f}% top-5={res_t[5]:.1f}%")
    print(f"  SEEN ({len(train_cats[:12])} train cats, {n_s} sketches): "
          f"top-1={ok(f'{res_s[1]:.1f}%')} top-3={res_s[3]:.1f}% top-5={res_s[5]:.1f}%")
    print(f"  retrieval computed in {time.time()-t0:.0f}s")

    if res_t[1] < 1.0:
        problems.append(f"TEST top-1 {res_t[1]:.1f}% near random -> teacher not learning")

    rng = np.random.RandomState(0)
    d_cats = list(rng.choice(train_cats, size=min(6, len(train_cats)), replace=False))
    idx = np.concatenate([np.flatnonzero(ph_l == c)[:1] for c in d_cats])
    ph_d = ph[idx]
    ph_d /= (np.linalg.norm(ph_d, axis=1, keepdims=True) + 1e-8)
    same, cross = [], []
    for i in range(len(ph_d)):
        for j in range(len(ph_d)):
            c = float(ph_d[i] @ ph_d[j])
            (same if i == j else cross).append(c)
    print(f"  distinctness: same={np.mean(same):.3f} cross={np.mean(cross):.3f} "
          f"(healthy: cross<0.9, same>cross)")
    if np.mean(cross) >= 0.9:
        problems.append(f"collapse: cross-cat cos {np.mean(cross):.3f} >= 0.9")

    bad = []
    for csv in sorted((ROOT / "logs").glob("*_history.csv"),
                      key=lambda p: p.stat().st_mtime, reverse=True)[:6]:
        try:
            vals = [float(x) for x in np.loadtxt(csv, delimiter=",", skiprows=1, usecols=(2,))]
        except Exception:
            continue
        if any(not np.isfinite(v) for v in vals):
            bad.append(csv.name)
    if bad:
        problems.append(f"NaN/Inf in histories: {bad}")
        print(flag(f"  NaN scan histories: RED FLAG {bad}"))
    else:
        print(ok("  NaN scan histories: clean"))

    print("\n" + "=" * 60)
    print("FINAL HEALTH REPORT (teacher):", flush=True)
    if problems:
        for p in problems:
            print(flag("  RED FLAG: " + p))
        print(flag("STOP before submitting student."))
        sys.exit(1)
    print(ok("HEALTHY - teacher artifacts good, safe to run student."))
    sys.exit(0)


if __name__ == "__main__":
    main()
