"""Raspberry Pi version of the Rock-Paper-Scissors game with physical output.

- Camera: Android phone via IP Webcam (--url) or Pi camera (default).
- Outputs: green LED (you win), red LED (Pi wins), yellow LED (draw),
  optional servo pointer showing Pi's move (pan to ROCK/PAPER/SCISSORS label),
  optional buzzer on any round end.
- Falls back to on-screen-only if GPIO is unavailable (laptop testing).

  python main_rps.py --url http://192.168.x.x:8080/video
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from capture import CameraSource  # noqa: E402
from classifier import RPSClassifier  # noqa: E402
from config import EXPORT_DIR, HOLD_FRAMES, MIN_CONFIDENCE  # noqa: E402
from game import RPSGame  # noqa: E402


class PiOut:
    """Best-effort GPIO: LEDs + buzzer + servo. No-op if GPIO unavailable."""

    def __init__(self, led_pins=(17, 27, 22), buzzer_pin=23, servo_pin=18):
        self.available = False
        self.pwm = None
        try:
            import RPi.GPIO as GPIO  # noqa: PLC0415
        except ImportError:
            print("GPIO unavailable -> on-screen only", flush=True)
            return
        self.GPIO = GPIO
        GPIO.setmode(GPIO.BCM)
        self.led_pins = led_pins
        for pin in led_pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        if buzzer_pin:
            self.buzzer_pin = buzzer_pin
            GPIO.setup(buzzer_pin, GPIO.OUT)
            GPIO.output(buzzer_pin, GPIO.LOW)
        else:
            self.buzzer_pin = None
        self.available = True
        try:
            if servo_pin:
                GPIO.setup(servo_pin, GPIO.OUT)
                self.pwm = GPIO.PWM(servo_pin, 50)
                self.pwm.start(2.5)
                self._servo_angle(90)
        except Exception as e:  # noqa: BLE001
            print(f"servo init failed: {e}", flush=True)
            self.pwm = None

    def _servo_angle(self, angle):
        if self.pwm is None:
            return
        duty = 2.5 + (angle / 180.0) * 10.0
        self.pwm.ChangeDutyCycle(duty)

    def show(self, outcome, pi_move):
        if not self.available:
            return
        led = {"player": self.led_pins[0], "pi": self.led_pins[1], "draw": self.led_pins[2]}.get(outcome)
        for pin in self.led_pins:
            self.GPIO.output(pin, 0)
        if led:
            self.GPIO.output(led, 1)
        if self.pwm and pi_move:
            self._servo_angle({"rock": 10, "paper": 90, "scissors": 170}[pi_move])
        if self.buzzer_pin:
            self.GPIO.output(self.buzzer_pin, 1)
            time.sleep(0.08)
            self.GPIO.output(self.buzzer_pin, 0)

    def cleanup(self):
        if self.available:
            self.GPIO.cleanup()


def draw_text(img, text, pos, scale=1.0, color=(255, 255, 255), bg=(0, 0, 0)):
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, bg, 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--strategy", default="counter", choices=["counter", "random"])
    ap.add_argument("--model-dir", default=str(EXPORT_DIR))
    ap.add_argument("--no-gpio", action="store_true")
    args = ap.parse_args()

    if args.url:
        source = CameraSource(args.url)
    else:
        try:
            from picamera2 import Picamera2  # noqa: PLC0415
            picam = Picamera2()
            picam.preview_configuration = picam.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"})
            picam.configure("preview")
            picam.start()
            source = None
        except ImportError:
            source = CameraSource("0")

    clf = RPSClassifier(args.model_dir)
    game = RPSGame(strategy=args.strategy)
    out = PiOut() if not args.no_gpio else None

    current, pending, hold_count = None, None, 0
    last_round, last_round_t = None, 0.0
    fps_t, fps = time.time(), 0.0

    def get_frame():
        if source is not None:
            return source.read()
        if picam is not None:  # noqa: F821
            return picam.capture_array()
        return None

    print("Ready. Show rock / paper / scissors. ESC to quit.", flush=True)
    while True:
        now = time.time()
        fps = 1.0 / max(1e-6, now - fps_t)
        fps_t = now
        frame = get_frame()
        if frame is None:
            time.sleep(0.3)
            continue
        if frame.dtype != "uint8":
            frame = frame.astype("uint8")
        label, conf, probs = clf.classify(frame)
        over = frame.copy()

        if conf >= MIN_CONFIDENCE:
            if pending == label:
                hold_count += 1
            else:
                pending, hold_count = label, 1
            if hold_count >= HOLD_FRAMES and label != current:
                current = label
                player, pi, outcome = game.round(label)
                last_round = (player, pi, outcome)
                last_round_t = time.time()
                hold_count = 0
                if out:
                    out.show(outcome, pi)
        else:
            pending, hold_count = None, 0

        draw_text(over, f"YOU: {label} {conf:.0%}", (10, 40), 0.9, (0, 255, 0))
        if last_round is not None and now - last_round_t < 3.0:
            player, pi, outcome = last_round
            color = (0, 255, 0) if outcome == "player" else (0, 0, 255) if outcome == "pi" else (255, 255, 0)
            draw_text(over, f"YOU: {player.upper()} vs PI: {pi.upper()} -> {outcome.upper()}",
                      (10, 100), 1.1, color)
        s = game.score
        draw_text(over, f"SCORE You {s['player']} | Pi {s['pi']} | Draw {s['draw']}  FPS={fps:.0f}",
                  (10, 140), 0.8)

        cv2.imshow("RPS vs Pi (Raspberry Pi)", over)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("r"):
            game.score = {"player": 0, "pi": 0, "draw": 0}

    if source is not None:
        source.release()
    cv2.destroyAllWindows()
    if out:
        out.cleanup()
    print(f"FINAL SCORE: {game.score}", flush=True)


if __name__ == "__main__":
    main()
