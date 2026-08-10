"""Rock-Paper-Scissors live loop (laptop or Pi).

  python src/rps_loop.py                       # built-in webcam
  python src/rps_loop.py --url http://IP:8080/video   # Android IP Webcam
  python src/rps_loop.py --strategy random     # fair play; default counter

Player shows rock/paper/scissors to the camera. Pi responds with a move
(by default it counters your most common recent move). Hold the gesture still
for a moment to lock in a round. ESC to quit.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from capture import CameraSource  # noqa: E402
from classifier import RPSClassifier  # noqa: E402
from config import EXPORT_DIR, HOLD_FRAMES, MIN_CONFIDENCE  # noqa: E402
from game import RPSGame  # noqa: E402


def draw_text(img, text, pos, scale=1.0, color=(255, 255, 255), bg=(0, 0, 0)):
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, bg, 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="IP Webcam MJPEG URL")
    ap.add_argument("--strategy", default="counter", choices=["counter", "random"])
    ap.add_argument("--model-dir", default=str(EXPORT_DIR))
    args = ap.parse_args()

    source = CameraSource(args.url if args.url else "0")
    clf = RPSClassifier(args.model_dir)
    game = RPSGame(strategy=args.strategy)

    current = None       # currently registered player move
    pending = None       # candidate move being held
    hold_count = 0
    last_round = None    # (player, pi, outcome)
    last_round_t = 0.0
    fps_t = time.time()
    fps = 0.0

    print("Ready. Show rock / paper / scissors. ESC to quit.", flush=True)
    while True:
        frame = source.read()
        if frame is None:
            print("no frame; reconnecting...", flush=True)
            time.sleep(0.5)
            continue

        now = time.time()
        fps = 1.0 / max(1e-6, now - fps_t)
        fps_t = now

        label, conf, probs = clf.classify(frame)
        over = frame.copy()

        if conf >= MIN_CONFIDENCE:
            if pending == label:
                hold_count += 1
            else:
                pending = label
                hold_count = 1
            if hold_count >= HOLD_FRAMES and label != current:
                current = label
                player, pi, outcome = game.round(label)
                last_round = (player, pi, outcome)
                last_round_t = time.time()
                hold_count = 0
        else:
            pending = None
            hold_count = 0

        # status bar
        draw_text(over, f"YOU: {label} {conf:.0%}   [{pending or '-'} {hold_count}/{HOLD_FRAMES}]",
                  (10, 40), 0.9, (0, 255, 0))
        draw_text(over, f"PI (strategy={game.strategy})   FPS={fps:.0f}",
                  (10, 80), 0.7)

        # last round banner
        if last_round is not None and now - last_round_t < 3.0:
            player, pi, outcome = last_round
            color = (0, 255, 0) if outcome == "player" else (0, 0, 255) if outcome == "pi" else (255, 255, 0)
            draw_text(over, f"YOU: {player.upper()}  vs  PI: {pi.upper()}  ->  {outcome.upper()}", (10, 140), 1.1, color)

        # score
        s = game.score
        draw_text(over, f"SCORE  You {s['player']}  |  Pi {s['pi']}  |  Draw {s['draw']}",
                  (10, 180), 0.8, (255, 255, 0))

        # small probability bars
        bw = int(over.shape[1] * 0.3)
        for i, name in enumerate(clf.labels):
            x0, y0 = 10, 210 + i * 30
            cv2.rectangle(over, (x0, y0), (x0 + bw, y0 + 22), (50, 50, 50), -1)
            cv2.rectangle(over, (x0, y0), (x0 + int(bw * probs[i]), y0 + 22), (0, 200, 0), -1)
            draw_text(over, f"{name} {probs[i]:.0%}", (x0, y0 + 18), 0.5)

        cv2.imshow("RPS vs Pi", over)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == ord("r"):
            game.score = {"player": 0, "pi": 0, "draw": 0}
            print("score reset", flush=True)

    source.release()
    cv2.destroyAllWindows()
    print(f"FINAL SCORE: {game.score}", flush=True)


if __name__ == "__main__":
    main()
