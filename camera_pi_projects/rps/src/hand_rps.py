"""MediaPipe-based hand gesture recognizer for Rock-Paper-Scissors.

Uses Google's Hand Landmarker (21 landmarks) + geometric rules:
  - no hand in frame  -> detected=False (never guesses)
  - fingertips extended/cuied  -> rock / paper / scissors
Industry-standard approach (same as MediaPipe's own gesture demos).

Landmark indices: 0=wrist; 1-4=thumb; 5-8=index; 9-12=middle;
13-16=ring; 17-20=pinky (4=MCP..TIP at 8/12/16/20).
"""
from __future__ import annotations

import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision.core.image import Image as MPImage
from mediapipe.tasks.python.vision.core.image import ImageFormat

FINGERS = {
    "index": (5, 8),
    "middle": (9, 12),
    "ring": (13, 16),
    "pinky": (17, 20),
}
THUMB = (1, 4)
WRIST = 0

MODEL = None  # lazy singleton


def _model(path: str | None = None):
    global MODEL
    if MODEL is None:
        path = path or "E:/SoftComputing/camera_pi_projects/rps/pi_deploy/model_data/hand_landmarker.task"
        options = BaseOptions(model_asset_path=path)
        MODEL = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=options,
                num_hands=1,
                min_hand_detection_confidence=0.25,
                min_tracking_confidence=0.25,
            )
        )
    return MODEL


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _finger_extended(pts, mcp_idx, tip_idx, palm_len):
    mcp = pts[mcp_idx]
    tip = pts[tip_idx]
    wrist = pts[WRIST]
    ratio = _dist(tip, wrist) / max(_dist(mcp, wrist), 1e-6)
    return ratio > 1.12


def recognize(img, path: str | None = None):
    """Return dict: {detected, label, conf, hand_score}."""
    if img is None:
        return {"detected": False, "label": None, "conf": 0.0, "hand_score": 0.0}

    frame = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
    if frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = MPImage(image_format=ImageFormat.SRGB, data=frame)
    result = _model(path).detect(mp_img)

    if not result.hand_landmarks or not result.handedness:
        return {"detected": False, "label": None, "conf": 0.0, "hand_score": 0.0}

    hand_score = float(result.handedness[0][0].score)
    lm = result.hand_landmarks[0]
    pts = [(lm[i].x, lm[i].y) for i in range(21)]
    palm = max(_dist(pts[0], pts[5]), _dist(pts[0], pts[17]), 1e-6)

    ext = {n: _finger_extended(pts, mcp, tip, palm) for n, (mcp, tip) in FINGERS.items()}
    n_ext = sum(ext.values())

    # which fingers: scissors needs index+middle up, ring+pinky down
    two = [n for n in FINGERS if ext[n]]
    if n_ext >= 3:
        label, score = "paper", 0.5 + 0.5 * (n_ext - 2) / 2
    elif n_ext == 2 and set(two) == {"index", "middle"}:
        label, score = "scissors", 0.85
    elif n_ext == 2 and set(two) == {"ring", "pinky"}:
        label, score = "scissors", 0.7
    else:
        label, score = "rock", 0.5 + 0.5 * (2 - n_ext) / 2  # fist / ambiguous

    conf = float(round(min(0.99, score * hand_score), 3))
    return {"detected": True, "label": label, "conf": conf, "hand_score": round(hand_score, 3)}
