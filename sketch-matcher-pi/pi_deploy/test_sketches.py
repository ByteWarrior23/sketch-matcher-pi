"""
test_sketches.py — EXTENSIVE offline test of the deployed TFLite matcher.

Part A: 80 real Sketchy training sketches (8 categories x 10) through the FULL
        deployed pipeline (TFLite int8 + 49,430-photo DB). Reports top-1/3
        accuracy per category + overall + the actual predicted names, so we can
        see exactly what the model tells us (not a black box).

Part B: synthetic black-on-white shapes (thick vs thin) drawn in OpenCV, to
        demonstrate how input style affects matching — verifies the webcam
        thin-line/blue-ink collapse hypothesis.

Usage: python pi_deploy/test_sketches.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from camera import CameraCapture  # reuse EXACT webcam preprocessing
from matcher import SketchMatcher

MODEL_DIR = Path(__file__).parent / "model_data"
TEST_DIR = Path(__file__).parent.parent / "hpc" / "sketch_test_export"


def preprocess_png(path):
    img = cv2.imread(str(path))  # BGR
    return CameraCapture().preprocess(img)


def run_single(matcher, pre):
    results, inf_ms, search_ms, rejected = matcher.match(pre)
    top1_name = results[0][0] if results else "NONE"
    top3_names = [r[0] for r in results[:3]]
    return top1_name, top3_names, results, rejected


def part_a(matcher):
    print("=" * 72)
    print("PART A — 80 REAL Sketchy sketches through deployed TFLite matcher")
    print("=" * 72)
    total = correct = 0
    per_cat = {}
    for cdir in sorted(TEST_DIR.iterdir()):
        if not cdir.is_dir():
            continue
        cat_name = cdir.name.split("_", 1)[1]
        n_c = correct_c = 0
        names = []
        for png in sorted(cdir.glob("*.png")):
            pre = preprocess_png(png)
            top1, top3, _, _ = run_single(matcher, pre)
            total += 1; n_c += 1
            if top1 == cat_name:
                correct += 1; correct_c += 1
            names.append(top1)
        acc = correct_c / n_c * 100
        per_cat[cat_name] = (correct_c, n_c, names)
        print(f"  {cat_name:14s} top-1 {acc:5.1f}%  ({correct_c}/{n_c})")
        print(f"      predicts: {names}")
    print(f"\n  OVERALL top-1: {correct}/{total} = {correct/total*100:.1f}%")
    return per_cat


def draw_shape(matcher, thickness, style="circle"):
    canvas = np.full((400, 400), 255, dtype=np.uint8)
    if style == "circle":
        cv2.circle(canvas, (200, 200), 120, 0, thickness)
    elif style == "filled_circle":
        cv2.circle(canvas, (200, 200), 120, 0, -1)
    elif style == "ring":
        cv2.circle(canvas, (200, 200), 120, 0, thickness)
        cv2.circle(canvas, (200, 200), 60, 255, thickness)
    elif style == "banana":
        cv2.ellipse(canvas, (200, 200), (130, 55), 30, 0, 360, 0, thickness)
    elif style == "star":
        pts = np.array([[200, 60], [245, 170], [360, 175], [270, 250],
                        [305, 365], [200, 295], [95, 365], [130, 250],
                        [40, 175], [155, 170]], np.int32)
        cv2.polylines(canvas, [pts], True, 0, thickness)
    elif style == "sun":
        cv2.circle(canvas, (200, 200), 80, 0, thickness)
        for a in range(0, 360, 30):
            import math
            x1 = 200 + int(110 * math.cos(math.radians(a)))
            y1 = 200 + int(110 * math.sin(math.radians(a)))
            x2 = 200 + int(160 * math.cos(math.radians(a)))
            y2 = 200 + int(160 * math.sin(math.radians(a)))
            cv2.line(canvas, (x1, y1), (x2, y2), 0, thickness)
    pre = CameraCapture().preprocess(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))
    top1, top3, results, rejected = run_single(matcher, pre)
    confs = ", ".join(f"{n}:{c:.1f}%" for n, c in results[:3])
    print(f"  {style:14s} th={thickness:2d}: top1={top1:14s}  top3={top3}  confs=[{confs}]")


def part_b(matcher):
    print("\n" + "=" * 72)
    print("PART B — synthetic shapes: thick vs thin strokes (webcam hypothesis)")
    print("=" * 72)
    for style in ["circle", "filled_circle", "ring", "banana", "star", "sun"]:
        draw_shape(matcher, 5, style)   # thick (like Sketchy training)
        draw_shape(matcher, 1, style)   # thin (like a thin webcam pen line)


def main():
    matcher = SketchMatcher(MODEL_DIR)
    per_cat = part_a(matcher)
    part_b(matcher)
    print("\nDone.")


if __name__ == "__main__":
    main()
