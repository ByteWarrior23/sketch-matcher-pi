"""Backend virtual trackpad — webcam + MediaPipe drives the real OS cursor.

Hand detection runs inside this Python server instead of the browser tab.
That is what makes it work on ANY window (Chrome, another website, a game):
when you switch away, Chrome pauses the browser tab's requestAnimationFrame
loop, but the server-side capture loop keeps running and keeps moving the
real cursor via pydirectinput.

Gesture scheme (the cursor only moves while you actively gesture — the
physical trackpad is always supreme otherwise). The thumb is the mode
switch: stretching it OUT freezes the cursor and enters button mode, so the
system never has to guess whether a moving finger is a cursor move or a
click:

  - index finger up (others down) ............ move the cursor like a real
    trackpad (relative: cursor travels as far as the hand sweeps, with
    acceleration, and stops when the hand stops; pushing toward the frame
    edge glides the cursor to the window edge, taskbar included)
  - index + middle spread, slide ............ scroll up / down
  - thumb stretched out .................... freezes the cursor, enters
    button mode
  - curl index down (thumb out) ............. left-click (the press = the
    click; two quick presses = double-click)
  - curl middle down (thumb out) ............. right-click
  - open palm ............................... release — no mouse events
    (with thumb in); with thumb out it is just rest in button mode

The page only shows the MJPEG preview; all vision + input lives here.
"""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"

from input_bridge import get_bridge  # noqa: E402

# ---- constants (mirrors the retired in-browser trackpad) ----
WRIST = 0
TIP_TMB = 4   # thumb tip
TIP_IDX = 8   # index tip
TIP_MID = 12  # middle tip
TIP_RNG = 16  # ring tip
TIP_PNK = 20  # pinky tip
MCP_IDX = 5   # index MCP (thumb-out reference)

MOVE_SEND_MS = 40   # cursor move throttle
ZERO = 1e-4

JITTER = 0.0035     # fingertip delta below this is treated as jitter (no move)
REL_GAIN = 1.5      # screen-fractions cursor moves per full-frame hand sweep
ACCEL_REF = 1.5     # sweep speed (normalized/s) at half of max acceleration
ACCEL_MAX = 0.35    # extra gain added at very fast flicks (max 1.7x total)
EDGE_ZONE = 0.22    # fingertip pushed past this offset from center adds drift
EDGE_GAIN = 1.8     # up to 1.8 screen/s of drift at the frame edge
AUTO_STOP_GRACE = 4.0  # seconds with no page viewer before the camera stops
DETECT_W, DETECT_H = 480, 360  # downscale for detection (faster, less lag)

FINGER_EXT_RATIO = 1.3  # tip must be this much farther than PIP from wrist to count as "up"
SCROLL_DEAD = 0.014
SCROLL_GAIN = 80
SCROLL_SPREAD = 1.15  # index/middle tip spread must exceed this x the knuckle gap

THUMB_OUT_D45 = 0.55   # thumb tip must be > this x palm-height from index MCP
THUMB_OUT_PALM = 0.48  # ...and > this x palm-height from the knuckle center
BTN_DEAD = 0.12        # finger-curl depth that counts as a click press
BTN_REARM = 0.05       # finger-curl depth that counts as released

# MediaPipe hand skeleton connections (0..20 landmark indices)
CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
]
PALM_IDX = [0, 5, 9, 13, 17]


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


class GestureTrackpad:
    """One camera capture loop that turns hand poses into OS mouse input."""

    def __init__(self):
        self._running = False
        self._thread: threading.Thread | None = None
        self._cam: cv2.VideoCapture | None = None
        self._landmarker = None

        self._lock = threading.Lock()
        self._latest: bytes | None = None
        self._error: str | None = None

        # page-viewer tracking: camera auto-stops if no page is watching
        self._preview_count = 0
        self._last_preview = 0.0

        # cursor (normalized 0..1 over the whole desktop)
        self.cx = 0.5
        self.cy = 0.5
        self.sens = 5

        # gesture state
        self.mode = "idle"  # idle | move | scroll | buttons
        self.scroll_streak = 0
        self.leave_streak = 0
        self.thumb_streak = 0
        self.hand_seen = False
        self.last_action = ""

        # button mode (thumb stretched out): index curl = left click, middle
        # curl = right click. Each button fires on the curl press and re-arms
        # when the finger extends again.
        self.btn = {
            "left": {"base": None, "down": False},
            "right": {"base": None, "down": False},
        }

        # thumb-out diagnostics (ratios of palm height; handy for tuning)
        self.thumb = {"out": False, "d45": 0.0, "d_palm": 0.0}

        # two-finger scroll
        self.scr_base: float | None = None
        self.scr_acc = 0.0

        # per-frame pose classifier diagnostics (shown in the page HUD)
        self.pose = {}

        # previous fingertip position for relative (trackpad-style) move
        self.pfx: float | None = None
        self.pfy: float | None = None

        self.last_move_ms = 0.0

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def running(self) -> bool:
        return self._running

    def start(self, sens: int | None = None) -> dict:
        if self._running:
            if sens is not None:
                self.set_sens(sens)
            return {"ok": True, "note": "already running"}
        if sens is not None:
            self.set_sens(sens)
        try:
            self._cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self._cam.isOpened():
                self._cam.release()
                self._cam = None
                return {"ok": False, "error": "Camera unavailable — close other apps using the webcam"}
            self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception as e:
            return {"ok": False, "error": f"camera open failed: {e}"}

        try:
            self._landmarker = self._build_landmarker()
        except Exception as e:
            self._cam.release()
            self._cam = None
            return {"ok": False, "error": f"hand model load failed: {e}"}

        self._reset_state()
        self._error = None
        self._preview_count = 0
        self._last_preview = time.monotonic()
        self._running = True
        self._thread = threading.Thread(target=self._run, name="gesture-trackpad", daemon=True)
        self._thread.start()
        return {"ok": True}

    def _build_landmarker(self):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        return vision.HandLandmarker.create_from_options(options)

    @staticmethod
    def _mp_image(rgb):
        from mediapipe.tasks.python.vision.core import image as mp_image

        return mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb)

    def stop(self) -> dict:
        self._running = False
        if self._thread and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._cleanup()
        self._reset_state()
        return {"ok": True}

    def _cleanup(self) -> None:
        if self._cam is not None:
            self._cam.release()
            self._cam = None
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None
        with self._lock:
            self._latest = None

    def set_sens(self, sens: int) -> dict:
        self.sens = max(1, min(10, int(sens)))
        return {"ok": True, "sens": self.sens}

    def recenter(self) -> dict:
        self.cx = 0.5
        self.cy = 0.5
        get_bridge().mouse_move(0.5, 0.5)
        self.last_action = "center"
        return {"ok": True}

    def _reset_state(self) -> None:
        self.mode = "idle"
        self.scroll_streak = 0
        self.leave_streak = 0
        self.hand_seen = False
        self.btn = {
            "left": {"base": None, "down": False},
            "right": {"base": None, "down": False},
        }
        self.thumb = {"out": False, "d45": 0.0, "d_palm": 0.0}
        self.scr_base = None
        self.scr_acc = 0.0
        self.pfx = None
        self.pfy = None
        self.last_move_ms = 0.0

    # ------------------------------------------------------------------ #
    # status / preview
    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        return {
            "ok": True,
            "running": self._running,
            "mode": self.mode,
            "hand_seen": self.hand_seen,
            "last_action": self.last_action,
            "cursor": {"x": round(self.cx, 3), "y": round(self.cy, 3)},
            "sens": self.sens,
            "thumb": self.thumb,
            "pose": self.pose,
            "error": self._error,
        }

    def preview_frames(self):
        """MJPEG generator: yields one JPEG per ~30ms while running."""
        while self._running:
            with self._lock:
                jpg = self._latest
            if jpg:
                yield jpg
            time.sleep(0.033)

    def on_preview_opened(self) -> None:
        self._preview_count += 1
        self._last_preview = time.monotonic()

    def on_preview_closed(self) -> None:
        if self._preview_count > 0:
            self._preview_count -= 1
        self._last_preview = time.monotonic()

    # ------------------------------------------------------------------ #
    # capture + vision loop
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        try:
            while self._running:
                if self._preview_count <= 0 and time.monotonic() - self._last_preview > AUTO_STOP_GRACE:
                    self.last_action = "auto-stop (page closed)"
                    break
                ok, frame = self._cam.read()
                if not ok:
                    for _ in range(10):
                        time.sleep(0.1)
                        if not self._running:
                            break
                        ok, frame = self._cam.read()
                        if ok:
                            break
                if not ok:
                    break
                frame = cv2.flip(frame, 1)  # mirror so hand-right == cursor-right
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                det = cv2.resize(rgb, (DETECT_W, DETECT_H))
                mp_image = self._mp_image(det)
                result = self._landmarker.detect_for_video(
                    mp_image, int(time.monotonic() * 1000)
                )
                self._on_hand(result)
                self._draw(frame, result)

                enc = cv2.resize(frame, (DETECT_W, DETECT_H))
                encoded = cv2.imencode(".jpg", enc, [cv2.IMWRITE_JPEG_QUALITY, 70])[1]
                with self._lock:
                    self._latest = encoded.tobytes()
        except Exception as e:
            self._error = str(e)
        finally:
            self._cleanup()
            self._running = False

    def _on_hand(self, result) -> None:
        now_ms = int(time.monotonic() * 1000)
        hands = result.hand_landmarks
        scores = result.handedness
        if not hands:
            self.hand_seen = False
            self.mode = "idle"
            self.scroll_streak = 0
            self.leave_streak = 0
            self.thumb_streak = 0
            self.scr_base = None
            self.scr_acc = 0.0
            self.thumb = {"out": False, "d45": 0.0, "d_palm": 0.0}
            self.pose = {}
            return

        self.hand_seen = True
        best = 0
        best_score = 0.0
        for i, hand in enumerate(hands):
            sc = scores[i][0].score if scores and scores[i] else 0.0
            if sc >= 0.6 and sc > best_score:
                best_score = sc
                best = i
        lm = hands[best]

        palm = self._palm_center(lm)
        py = palm[1]
        fx = lm[TIP_IDX].x
        fy = lm[TIP_IDX].y

        # --- pose booleans ---
        scroll_pose = self._is_scroll_pose(lm)
        open_palm = self._is_open_palm(lm)
        thumb_out = self._is_thumb_out(lm)
        point_pose = self._is_point_pose(lm)
        self.pose = {
            "scroll": scroll_pose,
            "point": point_pose,
            "open": open_palm,
            "thumb": thumb_out,
            "thumb_streak": self.thumb_streak,
            "streak": self.scroll_streak,
        }

        if scroll_pose:
            self.scroll_streak = min(self.scroll_streak + 1, 4)
            self.leave_streak = 0
        else:
            self.scroll_streak = 0
            self.leave_streak = min(self.leave_streak + 1, 3)

        # --- button mode: thumb stretched out freezes the cursor and arms
        # the click buttons (index = left, middle = right). The thumb must be
        # held out for 4 consecutive frames so a momentary flicker while
        # signing scroll never hijacks the mode. ---
        if thumb_out:
            self.thumb_streak = min(self.thumb_streak + 1, 8)
        else:
            self.thumb_streak = 0

        if self.thumb_streak >= 4:
            self._enter_buttons()
            self._update_button(lm, now_ms)
            return

        if self.mode == "buttons":
            self.mode = "idle"

        # --- navigation: scroll wins over open palm — index+middle raised
        # with ring/pinky folded is a scroll pose, not a release. ---
        if scroll_pose and self.scroll_streak >= 3:
            self.mode = "scroll"
            self.scr_base = None
            self.scr_acc = 0.0
        elif open_palm:
            self._release()
        else:
            if self.mode == "scroll" and self.leave_streak >= 2:
                self.mode = "move"
            if self.mode != "scroll":
                if point_pose:
                    self.mode = "move"
                else:
                    self.mode = "idle"

        if self.mode == "scroll":
            self._update_scroll(py)
        elif self.mode == "move":
            self._update_cursor(fx, fy, now_ms)

    # ------------------------------------------------------------------ #
    # gesture actions
    # ------------------------------------------------------------------ #
    def _release(self) -> None:
        self.mode = "idle"
        self.scroll_streak = 0
        self.leave_streak = 0
        self.scr_base = None
        self.scr_acc = 0.0
        self.pfx = None
        self.pfy = None

    def _update_cursor(self, fx: float, fy: float, now_ms: int) -> None:
        """Relative move like a real trackpad: the cursor travels the distance
        your fingertip travels (with a speed-based acceleration curve), and it
        stops the moment your hand stops. Clamped to [0,1], so the whole
        screen (taskbar included) is reachable with repeated sweeps."""
        if self.pfx is None:
            self.pfx, self.pfy = fx, fy
            return
        dx = fx - self.pfx
        dy = fy - self.pfy
        self.pfx, self.pfy = fx, fy
        if abs(dx) > 0.2 or abs(dy) > 0.2:
            return  # hand re-acquired / jumped — treat as a fresh baseline
        if abs(dx) < JITTER and abs(dy) < JITTER:
            dx = dy = 0.0
        dt = (now_ms - self.last_move_ms) / 1000.0
        if dt <= 0.0 or dt > 0.5:
            dt = 1.0 / 30.0
        speed = math.hypot(dx, dy) / dt
        accel = 1.0 + min(speed / ACCEL_REF, 2.0) * ACCEL_MAX
        gain = REL_GAIN * accel
        self.cx = max(0.0, min(1.0, self.cx + dx * gain))
        self.cy = max(0.0, min(1.0, self.cy + dy * gain))

        # edge assist: push the fingertip past the center zone toward the frame
        # edge and the cursor glides toward the screen edge (taskbar included).
        # Bring the hand back toward center to stop.
        ox = abs(fx - 0.5)
        oy = abs(fy - 0.5)
        if ox > EDGE_ZONE:
            self.cx = max(0.0, min(1.0, self.cx + math.copysign((ox - EDGE_ZONE) / (0.5 - EDGE_ZONE), fx - 0.5) * EDGE_GAIN * dt))
        if oy > EDGE_ZONE:
            self.cy = max(0.0, min(1.0, self.cy + math.copysign((oy - EDGE_ZONE) / (0.5 - EDGE_ZONE), fy - 0.5) * EDGE_GAIN * dt))

        if now_ms - self.last_move_ms >= MOVE_SEND_MS:
            self.last_move_ms = now_ms
            get_bridge().mouse_move(self.cx, self.cy)
            self.last_action = "move %d,%d" % (round(self.cx * 100), round(self.cy * 100))

    def _update_scroll(self, py: float) -> None:
        if self.scr_base is None:
            self.scr_base = py
            return
        dy = py - self.scr_base
        self.scr_base = py
        if abs(dy) > SCROLL_DEAD:
            self.scr_acc += (dy - math.copysign(SCROLL_DEAD, dy)) * SCROLL_GAIN
        else:
            self.scr_acc *= 0.5
        out = 0
        while abs(self.scr_acc) >= 1:
            s = 1 if self.scr_acc > 0 else -1
            self.scr_acc -= s
            out += s
        if out != 0:
            get_bridge().mouse_scroll(lines=-out)
            self.last_action = "scroll down" if out > 0 else "scroll up"

    def _enter_buttons(self) -> None:
        if self.mode != "buttons":
            self.mode = "buttons"
            self.pfx = None
            self.pfy = None
            self.last_action = "buttons"
        self.scroll_streak = 0
        self.leave_streak = 0
        self.scr_base = None
        self.scr_acc = 0.0

    def _update_button(self, lm, now_ms: int) -> None:
        py = self._palm_center(lm)[1]
        self._update_click("left", lm[TIP_IDX].y - py, now_ms)
        self._update_click("right", lm[TIP_MID].y - py, now_ms)

    def _update_click(self, button: str, rel_y: float, now_ms: int) -> None:
        """Click on finger curl. The click fires the moment the fingertip folds
        down (a press); two quick presses become a double-click via Windows.
        Re-arms only after the finger comes back up, so holding the curl never
        repeats the click."""
        st = self.btn[button]
        if st["base"] is None:
            st["base"] = rel_y
        st["base"] += 0.05 * (rel_y - st["base"])
        dev = rel_y - st["base"]
        if not st["down"] and dev > BTN_DEAD:
            st["down"] = True
            self._fire_click(button)
        elif st["down"] and dev < BTN_REARM:
            st["down"] = False

    def _fire_click(self, button: str = "left", times: int = 1) -> None:
        res = get_bridge().mouse_click(times=times, button=button)
        if res and res.get("ok"):
            self.last_action = "left click" if button == "left" else "right click"

    # ------------------------------------------------------------------ #
    # hand geometry helpers
    # ------------------------------------------------------------------ #
    def _palm_center(self, lm):
        sx = sum(lm[i].x for i in PALM_IDX) / len(PALM_IDX)
        sy = sum(lm[i].y for i in PALM_IDX) / len(PALM_IDX)
        return sx, sy

    def _finger_ext(self, lm, tip, pip) -> bool:
        return _dist(lm[tip], lm[WRIST]) > _dist(lm[pip], lm[WRIST]) * FINGER_EXT_RATIO

    def _palm_span(self, lm) -> float:
        return max(_dist(lm[0], lm[5]), _dist(lm[0], lm[17]), ZERO)

    def _is_scroll_pose(self, lm) -> bool:
        if not self._finger_ext(lm, 8, 6) or not self._finger_ext(lm, 12, 10):
            return False
        if not _dist(lm[8], lm[12]) > SCROLL_SPREAD * max(_dist(lm[5], lm[9]), ZERO):
            return False
        # Only a full open palm blocks scroll: BOTH ring AND pinky must be
        # nearly as raised as the index (finger coupling when signing "two
        # fingers up" is fine — it is not an open palm).
        idx_ext = _dist(lm[8], lm[WRIST])
        if _dist(lm[16], lm[WRIST]) > idx_ext * 0.90 and _dist(lm[20], lm[WRIST]) > idx_ext * 0.90:
            return False
        return True

    def _is_open_palm(self, lm) -> bool:
        return all(self._finger_ext(lm, t, p) for t, p in ((8, 6), (12, 10), (16, 14), (20, 18)))

    def _is_thumb_out(self, lm) -> bool:
        """Thumb stretched away from the fingers. Measured against the palm
        height (wrist -> index/middle MCP) so it scales with distance. Two
        checks: the thumb tip is far from the index MCP (not tucked against
        it) AND far from the knuckle line (not folded across the palm)."""
        palm_h = max(_dist(lm[0], lm[5]), _dist(lm[0], lm[9]), ZERO)
        kx = sum(lm[i].x for i in (5, 9, 13, 17)) / 4
        ky = sum(lm[i].y for i in (5, 9, 13, 17)) / 4
        d45 = _dist(lm[TIP_TMB], lm[MCP_IDX])
        d_palm = math.hypot(lm[TIP_TMB].x - kx, lm[TIP_TMB].y - ky)
        self.thumb = {
            "out": d45 > THUMB_OUT_D45 * palm_h and d_palm > THUMB_OUT_PALM * palm_h,
            "d45": round(d45 / palm_h, 2),
            "d_palm": round(d_palm / palm_h, 2),
        }
        return self.thumb["out"]

    def _is_point_pose(self, lm) -> bool:
        if not self._finger_ext(lm, 8, 6):
            return False
        return not any(self._finger_ext(lm, t, p) for t, p in ((12, 10), (16, 14), (20, 18)))

    # ------------------------------------------------------------------ #
    # preview overlay (drawn on the mirrored frame, so x maps directly)
    # ------------------------------------------------------------------ #
    def _draw(self, frame, result) -> None:
        h, w = frame.shape[:2]
        lm = (result.hand_landmarks[0] if result.hand_landmarks else None)

        if lm:
            for a, b in CONNECTIONS:
                cv2.line(
                    frame,
                    (int(lm[a].x * w), int(lm[a].y * h)),
                    (int(lm[b].x * w), int(lm[b].y * h)),
                    (0, 255, 20), 2,
                )
            px, py = self._palm_center(lm)
            cv2.circle(frame, (int(px * w), int(py * h)), 5, (0, 212, 255), -1)
            left_down = self.btn["left"].get("down", False)
            right_down = self.btn["right"].get("down", False)
            tip_col = (92, 92, 255) if left_down else (196, 46, 255)
            cv2.circle(frame, (int(lm[8].x * w), int(lm[8].y * h)), 7, tip_col, -1)
            if right_down:
                cv2.circle(frame, (int(lm[12].x * w), int(lm[12].y * h)), 7, (92, 92, 255), -1)
            if self.mode == "scroll":
                cv2.circle(frame, (int(lm[12].x * w), int(lm[12].y * h)), 5, (0, 212, 255), -1)

        # status text
        text = "MOVE" if self.mode == "move" else "SCROLL" if self.mode == "scroll" else "BUTTONS" if self.mode == "buttons" else "RELEASED"
        if not lm:
            text = "NO HAND"
        cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 20), 2)
        if lm:
            th = "THUMB OUT" if self.thumb.get("out") else "thumb in"
            cv2.putText(
                frame, f"{th}  d45={self.thumb.get('d45', 0)} palm={self.thumb.get('d_palm', 0)}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1,
            )


_TRACKPAD: GestureTrackpad | None = None


def get_trackpad() -> GestureTrackpad:
    global _TRACKPAD
    if _TRACKPAD is None:
        _TRACKPAD = GestureTrackpad()
    return _TRACKPAD
