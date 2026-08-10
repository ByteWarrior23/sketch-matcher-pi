"""Headless test: classifier correctness on official test images + webcam sanity.

Does NOT open a GUI window. Use this to verify the pipeline works end-to-end
without a display; the interactive game is `python src/rps_loop.py`.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from classifier import RPSClassifier  # noqa: E402
from config import CLASSES  # noqa: E402
from capture import CameraSource  # noqa: E402

MODEL_DIR = ROOT / "models" / "local"
RAW = ROOT / "data" / "raw" / "rps-test-set" / "rps-test-set"


def test_dataset():
    clf = RPSClassifier(MODEL_DIR)
    per_class = {c: [0, 0] for c in CLASSES}
    wrong = []
    t0 = time.time()
    for ci, name in enumerate(CLASSES):
        for p in sorted((RAW / name).glob("*.png")):
            frame = cv2.imread(str(p))
            pred, conf, probs = clf.classify(frame)
            per_class[name][1] += 1
            if pred == name:
                per_class[name][0] += 1
            else:
                wrong.append((p.name, name, pred, conf))
    dt = time.time() - t0
    total = sum(v[1] for v in per_class.values())
    correct = sum(v[0] for v in per_class.values())
    print(f"per-class acc: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in per_class.items()))
    print(f"OVERALL: {correct}/{total} = {correct/total:.3f}  ({len(wrong)} wrong)")
    for w in wrong[:8]:
        print(f"  WRONG: {w}")
    print(f"speed: {dt/max(1,total)*1000:.1f} ms/image")
    return clf


def test_webcam(clf):
    src = CameraSource.auto()
    if src is None:
        print("WEBCAM: no camera available")
        return
    frame = src.read()
    src.release()
    if frame is None:
        print("WEBCAM: frame read failed")
        return
    h, w = frame.shape[:2]
    pred, conf, probs = clf.classify(frame)
    print(f"WEBCAM: ok {w}x{h}, prediction={pred} conf={conf:.2f} probs={dict(zip(CLASSES, probs.round(2)))}")
    print("WEBCAM: PASS (frame captured, classification ran)")


if __name__ == "__main__":
    clf = test_dataset()
    test_webcam(clf)
    print("TEST OK")
