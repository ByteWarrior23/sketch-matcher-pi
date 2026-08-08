"""mid_check.py — mid-training diagnostic gate.

Loads the LATEST checkpoint from the currently-running teacher/student stage
and measures REAL retrieval accuracy on the 13 held-out TEST categories (the
final-eval metric) plus the seen (train) categories. Run this during training
so a regression or silent failure is caught HOURS before the job ends, not
after 40h of GPU.

Usage (CPU-only, on the cluster):
    python hpc/mid_check.py [--stage teacher] [--backbone convnexttiny]

Checks emitted:
  TEST top-1/3/5 accuracy      (the metric that matters — 13 unseen cats)
  SEEN top-1/3/5 accuracy      (did it learn at all)
  embedding distinctness       (collapse guard: same vs cross cos)
  NaN scan of recent CSVs      (exploding gradients)
  score vs last checkpoint     (regression guard between polls)

Exit code 0 = healthy, 1 = RED FLAG (stop & investigate).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

N_THREADS = int(__import__("os").environ.get("MID_N_THREADS", "4"))
tf.config.threading.set_intra_op_parallelism_threads(N_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(1)
N_SAMP = int(__import__("os").environ.get("MID_N_SAMP", "80"))  # samples/cat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # src/ MUST win over stale root data_loader.py
# NOTE: root dir contains a STALE data_loader.py that shadows src/; src must
# stay first on sys.path (insert order above guarantees it).
import data_loader
print("data_loader loaded from:", data_loader.__file__, flush=True)

import model  # registers custom layers
from config import (
    CHECKPOINT_DIR, PROCESSED_DIR, EMBEDDING_DIM, BACKBONE, TEACHER_BACKBONE,
)
from data_loader import load_processed_data, predict_normalized

RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"; RESET = "\033[0m"


def flag(msg):
    return f"{RED}{msg}{RESET}"


def ok(msg):
    return f"{GREEN}{msg}{RESET}"


def warn(msg):
    return f"{YELLOW}{msg}{RESET}"


def latest_checkpoint(phase):
    """Path of the most recently-modified stage checkpoint for a phase."""
    base = CHECKPOINT_DIR / f"{phase}_stage"
    cands = sorted(base.parent.glob(f"{phase}_stage*/best_epoch.keras"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"no checkpoints for phase '{phase}'")
    return cands[0]


def nan_scan():
    """Scan CSV histories for NaN/Inf loss values."""
    bad = []
    for csv in sorted((ROOT / "logs").glob("*_history.csv"),
                      key=lambda p: p.stat().st_mtime, reverse=True)[:6]:
        vals = [float(x) for x in np.loadtxt(csv, delimiter=",", skiprows=1,
                                             usecols=(2,))]  # loss col
        if any(not np.isfinite(v) for v in vals):
            bad.append((csv.name, len(vals)))
    return bad


def eval_retrieval(embed_net, split, category_names, metadata, phase):
    """Top-k sketch->photo retrieval accuracy on a split (train/test/val)."""
    cats = [int(c) for c in metadata[f"{split}_categories"]]
    rng = np.random.RandomState(7)
    n_samp = N_SAMP  # cap per category (CPU embed cost)
    chosen = list(rng.choice(cats, size=min(13, len(cats)), replace=False))

    sketches, photos, sk_l, ph_l, _, _ = load_processed_data()
    sk_rows, ph_rows = [], []
    for cat in chosen:
        si = np.flatnonzero(sk_l == cat); pi = np.flatnonzero(ph_l == cat)
        rng.shuffle(si); rng.shuffle(pi)
        sk_rows.append(si[:n_samp]); ph_rows.append(pi[:n_samp])
    sk_rows = np.concatenate(sk_rows); ph_rows = np.concatenate(ph_rows)

    t0 = time.time()
    sk_embs = predict_normalized(embed_net, sketches[sk_rows].astype(np.float32) / 255.0,
                                 batch_size=64, verbose=0)
    ph_embs = predict_normalized(embed_net, photos[ph_rows].astype(np.float32) / 255.0,
                                 batch_size=64, verbose=0)
    dt = time.time() - t0

    s = sk_embs[:, None, :] * ph_embs[None, :, :]
    sim = s.sum(axis=-1)
    labels = sk_l[sk_rows]
    col_cats = ph_l[ph_rows]

    line = f"  [{phase} {split}] ({len(chosen)} cats, {len(sk_rows)} sketches, {dt:.0f}s)"
    print(line, flush=True)
    for k in (1, 3, 5):
        top = np.argsort(-sim, axis=1)[:, :k]
        n = len(labels)
        correct = sum(labels[i] in col_cats[top[i]] for i in range(n))
        print(f"    top-{k}: {ok(f'{correct/n*100:.1f}%')} ({correct}/{n})", flush=True)
    return sim, labels, col_cats, chosen


def distinctness(embed_net):
    """Collapse guard: same-cat cos vs cross-cat cos (6 photos)."""
    sketches, photos, sk_l, ph_l, _, _ = load_processed_data()
    cats = sorted(set(sk_l) & set(ph_l))
    rng = np.random.RandomState(0)
    chosen = list(rng.choice(cats, size=6, replace=False))
    imgs = np.concatenate([photos[np.flatnonzero(ph_l == c)[0]][None] for c in chosen])
    emb = predict_normalized(embed_net, imgs.astype(np.float32) / 255.0,
                             batch_size=8, verbose=0)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    same, cross = [], []
    for i in range(6):
        for j in range(6):
            c = float(emb[i] @ emb[j])
            (same if i == j else cross).append(c)
    print(f"  distinctness: same={np.mean(same):.3f} cross={np.mean(cross):.3f} "
          f"(healthy: cross < 0.9, same > cross)")
    return np.mean(cross), np.mean(same)


def extract_embedding(siamese):
    """Pull the shared embedding sub-network out of a saved siamese."""
    try:
        net = siamese.get_layer("embedding_network")
        return net
    except Exception:
        pass
    inp = siamese.get_layer("input_sketch").input
    out = siamese.get_layer("l2_normalize").output
    return tf.keras.Model(inp, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="teacher", choices=["teacher", "student"])
    ap.add_argument("--backbone", default=None,
                    help="defaults to TEACHER_BACKBONE for teacher, BACKBONE for student")
    args = ap.parse_args()

    backbone = args.backbone or (TEACHER_BACKBONE if args.stage == "teacher" else BACKBONE)
    ckpt = latest_checkpoint(args.stage)
    print(f"checkpoint: {ckpt}  (mtime {time.strftime('%H:%M:%S', time.localtime(ckpt.stat().st_mtime))})", flush=True)

    with open(PROCESSED_DIR / "category_names.json") as f:
        category_names = {int(k): v for k, v in json.load(f).items()}
    with open(PROCESSED_DIR / "metadata.json") as f:
        metadata = json.load(f)

    siamese = tf.keras.models.load_model(ckpt, compile=False)
    embed_net = extract_embedding(siamese)
    print(f"loaded siamese -> embedding net (backbone={backbone})", flush=True)

    sim_test, lab_test, col_test, chosen_test = eval_retrieval(
        embed_net, "test", category_names, metadata, args.stage)
    sim_train, lab_train, col_train, chosen_train = eval_retrieval(
        embed_net, "train", category_names, metadata, args.stage)

    cross, same = distinctness(embed_net)

    nans = nan_scan()
    if nans:
        print(flag(f"  NaN/Inf found in histories: {nans}"))
    else:
        print(ok("  NaN scan: clean"))

    # verdict
    top1 = 0.0
    n = len(lab_test)
    for i in range(n):
        if lab_test[i] in col_test[np.argsort(-sim_test[i])[:1]]:
            top1 += 1
    top1 = top1 / n * 100

    problems = []
    if cross >= 0.9:
        problems.append("collapse: cross-cat cos >= 0.9")
    if nans:
        problems.append("NaN/Inf in loss history")
    if top1 < 1.0:
        problems.append(f"test top-1 ({top1:.1f}%) is near random (2%?) -> model not learning")

    print("\n" + "=" * 60)
    print(f"MID-CHECK RESULT ({args.stage}): test top-1 = {top1:.1f}%", flush=True)
    if problems:
        for p in problems:
            print(flag("  RED FLAG: " + p))
        print(flag("STOP and investigate before continuing."))
        sys.exit(1)
    else:
        print(ok("HEALTHY - no errors detected, continue training."))
        sys.exit(0)


if __name__ == "__main__":
    main()
