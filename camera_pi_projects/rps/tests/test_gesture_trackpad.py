"""Unit tests for the backend gesture trackpad (src/gesture_trackpad.py).

The camera and MediaPipe are never touched: the OS-input bridge is a mock,
the clock is faked, and hands are synthetic landmark arrays.
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gesture_trackpad as gt  # noqa: E402

ORIG_GET_BRIDGE = gt.get_bridge
ORIG_MONOTONIC = gt.time.monotonic


class Lm:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Score:
    def __init__(self, s=0.99):
        self.score = s


class Result:
    def __init__(self, lms):
        self.hand_landmarks = [lms] if lms is not None else []
        self.handedness = [[Score()]] if lms is not None else []


class MockBridge:
    def __init__(self):
        self.moves = []
        self.clicks = []
        self.scrolls = []

    def mouse_move(self, x, y):
        self.moves.append((x, y))
        return {"ok": True}

    def mouse_click(self, times=1, state="tap", button="left"):
        self.clicks.append((button, times))
        return {"ok": True}

    def mouse_scroll(self, lines, ctrl=False):
        self.scrolls.append((lines, ctrl))
        return {"ok": True}


class FakeClock:
    def __init__(self, start=1000.0):
        self.t = start

    def monotonic(self):
        self.t += 33.0 / 1000.0
        return self.t


MCP_ANGLE = {5: -0.7, 9: -0.2, 13: 0.2, 17: 0.7}
FINGER_MCP = [5, 9, 13, 17]


def build_hand(*, index=True, middle=True, ring=True, pinky=True, spread=0.0, cx=0.5, cy=0.7, thumb="in"):
    lms = [Lm(cx, cy) for _ in range(21)]
    ext = {5: index, 9: middle, 13: ring, 17: pinky}
    ang = dict(MCP_ANGLE)
    ang[5] -= spread
    ang[9] += spread
    for mcp in FINGER_MCP:
        a = ang[mcp]
        sa, ca = math.sin(a), math.cos(a)
        lms[mcp] = Lm(cx + 0.08 * sa, cy - 0.08 * ca)
        if ext[mcp]:
            lms[mcp + 1] = Lm(cx + 0.30 * sa, cy - 0.30 * ca)
            lms[mcp + 2] = Lm(cx + 0.42 * sa, cy - 0.42 * ca)
            lms[mcp + 3] = Lm(cx + 0.52 * sa, cy - 0.52 * ca)
        else:
            lms[mcp + 1] = Lm(cx + 0.26 * sa, cy - 0.26 * ca)
            lms[mcp + 2] = Lm(cx + 0.22 * sa, cy - 0.22 * ca)
            lms[mcp + 3] = Lm(cx + 0.16 * sa, cy - 0.16 * ca)
    if thumb == "out":
        lms[4] = Lm(cx + 0.55, cy - 0.05)
    else:
        lms[4] = Lm(lms[5].x, lms[5].y)
    return lms


def point_hand(**kw):
    return build_hand(index=True, middle=False, ring=False, pinky=False, **kw)


def scroll_hand(**kw):
    return build_hand(index=True, middle=True, ring=False, pinky=False, **kw)


def test_pose_classifiers():
    tp = gt.GestureTrackpad()
    assert tp._is_point_pose(point_hand())
    assert not tp._is_point_pose(build_hand(index=True, middle=True))
    assert not tp._is_point_pose(build_hand(index=False))
    assert tp._is_open_palm(build_hand(index=True, middle=True, ring=True, pinky=True))
    assert not tp._is_open_palm(point_hand())
    assert tp._is_scroll_pose(scroll_hand(spread=0.4))
    assert not tp._is_scroll_pose(point_hand())
    assert tp._is_scroll_pose(build_hand(index=True, middle=True, ring=True, pinky=False, spread=0.4))
    assert not tp._is_scroll_pose(build_hand(index=True, middle=True, ring=True, pinky=True, spread=0.4))


def test_mode_transitions():
    bridge = MockBridge()
    gt.get_bridge = lambda: bridge
    tp = gt.GestureTrackpad()
    gt.time.monotonic = FakeClock().monotonic
    try:
        tp._on_hand(Result(None))
        assert tp.mode == "idle" and not tp.hand_seen

        tp._on_hand(Result(point_hand()))
        assert tp.hand_seen and tp.mode == "move"

        scroll = Result(scroll_hand(spread=0.4))
        tp._on_hand(scroll)
        assert tp.scroll_streak == 1
        tp._on_hand(scroll)
        assert tp.scroll_streak == 2
        tp._on_hand(scroll)
        assert tp.mode == "scroll"
        assert bridge.scrolls == []

        tp._on_hand(Result(build_hand(index=True, middle=True, ring=True, pinky=True)))
        assert tp.mode == "idle"
        assert bridge.clicks == [] and bridge.scrolls == []
    finally:
        gt.get_bridge = ORIG_GET_BRIDGE
        gt.time.monotonic = ORIG_MONOTONIC


def test_point_pose_drives_cursor():
    bridge = MockBridge()
    gt.get_bridge = lambda: bridge
    tp = gt.GestureTrackpad()
    gt.time.monotonic = FakeClock().monotonic
    try:
        tp._on_hand(Result(point_hand()))
        assert tp.mode == "move" and bridge.moves == []

        tp._on_hand(Result(point_hand(cx=0.56)))
        assert len(bridge.moves) == 1
        mx, my = bridge.moves[0]
        assert 0.0 <= mx <= 1.0 and 0.0 <= my <= 1.0
        assert mx > 0.5
    finally:
        gt.get_bridge = ORIG_GET_BRIDGE
        gt.time.monotonic = ORIG_MONOTONIC


def test_cursor_move_jitter_clamp():
    bridge = MockBridge()
    gt.get_bridge = lambda: bridge
    tp = gt.GestureTrackpad()
    try:
        tp._update_cursor(0.5, 0.5, 1000)
        assert bridge.moves == []

        tp.last_move_ms = 1000
        tp._update_cursor(0.55, 0.53, 1066)
        assert len(bridge.moves) == 1
        assert bridge.moves[0][0] > 0.5 and bridge.moves[0][1] > 0.5

        before = (tp.cx, tp.cy)
        tp.last_move_ms = 1066
        tp._update_cursor(0.551, 0.531, 1132)
        assert (tp.cx, tp.cy) == before

        tp.cx = 0.99
        tp.last_move_ms = 1132
        tp._update_cursor(0.60, 0.53, 1198)
        assert tp.cx == 1.0
    finally:
        gt.get_bridge = ORIG_GET_BRIDGE


def test_hand_reacquire_resets_baseline():
    bridge = MockBridge()
    gt.get_bridge = lambda: bridge
    tp = gt.GestureTrackpad()
    try:
        tp._update_cursor(0.5, 0.5, 1000)
        tp.last_move_ms = 1000
        tp._update_cursor(0.8, 0.6, 1033)
        assert bridge.moves == []
    finally:
        gt.get_bridge = ORIG_GET_BRIDGE


def test_scroll_down_and_up():
    bridge = MockBridge()
    gt.get_bridge = lambda: bridge
    try:
        down = gt.GestureTrackpad()
        down._update_scroll(0.5)
        assert bridge.scrolls == []
        down._update_scroll(0.55)
        assert bridge.scrolls and bridge.scrolls[-1][0] < 0
        assert down.last_action == "scroll down"

        up = gt.GestureTrackpad()
        up._update_scroll(0.5)
        up._update_scroll(0.45)
        assert bridge.scrolls and bridge.scrolls[-1][0] > 0
        assert up.last_action == "scroll up"
    finally:
        gt.get_bridge = ORIG_GET_BRIDGE


def test_thumb_out_detection():
    tp = gt.GestureTrackpad()
    assert not tp._is_thumb_out(point_hand())
    assert not tp._is_thumb_out(build_hand(index=True, middle=True, ring=True, pinky=True))
    assert tp._is_thumb_out(point_hand(thumb="out"))


def test_button_mode_click_mapping():
    bridge = MockBridge()
    gt.get_bridge = lambda: bridge
    tp = gt.GestureTrackpad()
    try:
        arm = Result(build_hand(index=True, middle=True, thumb="out"))
        for _ in range(4):
            tp._on_hand(arm)
        assert tp.mode == "buttons"
        assert bridge.clicks == [] and bridge.moves == []

        tp._on_hand(Result(build_hand(index=False, thumb="out")))
        assert len(bridge.clicks) == 1 and bridge.clicks[0][0] == "left"

        tp._on_hand(arm)
        assert len(bridge.clicks) == 1

        tp._on_hand(Result(build_hand(index=True, middle=False, thumb="out")))
        assert len(bridge.clicks) == 2 and bridge.clicks[1][0] == "right"
        assert bridge.moves == []
    finally:
        gt.get_bridge = ORIG_GET_BRIDGE


def test_thumb_back_resumes_move():
    bridge = MockBridge()
    gt.get_bridge = lambda: bridge
    tp = gt.GestureTrackpad()
    gt.time.monotonic = FakeClock().monotonic
    try:
        for _ in range(4):
            tp._on_hand(Result(point_hand(thumb="out")))
        assert tp.mode == "buttons"

        tp._on_hand(Result(point_hand()))
        assert tp.mode == "move"
    finally:
        gt.get_bridge = ORIG_GET_BRIDGE
        gt.time.monotonic = ORIG_MONOTONIC


def test_clicks_fire_once_per_curl():
    bridge = MockBridge()
    gt.get_bridge = lambda: bridge
    tp = gt.GestureTrackpad()
    try:
        tp._update_click("left", -0.30, 1000)
        assert bridge.clicks == []

        tp._update_click("left", -0.15, 1033)
        assert len(bridge.clicks) == 1

        tp._update_click("left", -0.15, 1066)
        assert len(bridge.clicks) == 1

        tp._update_click("left", -0.30, 1099)
        assert len(bridge.clicks) == 1

        tp._update_click("left", -0.15, 1132)
        assert len(bridge.clicks) == 2

        tp._update_click("right", -0.30, 1165)
        tp._update_click("right", -0.15, 1198)
        assert len(bridge.clicks) == 3
        assert bridge.clicks[-1][0] == "right"
    finally:
        gt.get_bridge = ORIG_GET_BRIDGE


def test_draw_overlay_smoke():
    import numpy as np

    bridge = MockBridge()
    gt.get_bridge = lambda: bridge
    tp = gt.GestureTrackpad()
    try:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        tp._draw(frame, Result(point_hand(thumb="out")))
        tp._draw(frame, Result(point_hand()))
        tp._draw(frame, Result(None))
    finally:
        gt.get_bridge = ORIG_GET_BRIDGE


def test_config_and_status():
    bridge = MockBridge()
    gt.get_bridge = lambda: bridge
    tp = gt.GestureTrackpad()
    try:
        assert tp.set_sens(0)["sens"] == 1
        assert tp.set_sens(99)["sens"] == 10
        assert tp.set_sens(7)["sens"] == 7

        assert tp.recenter()["ok"]
        assert bridge.moves[-1] == (0.5, 0.5)

        st = tp.status()
        assert st["running"] is False
        assert st["cursor"] == {"x": 0.5, "y": 0.5}
    finally:
        gt.get_bridge = ORIG_GET_BRIDGE


TESTS = [
    test_pose_classifiers,
    test_thumb_out_detection,
    test_mode_transitions,
    test_point_pose_drives_cursor,
    test_cursor_move_jitter_clamp,
    test_hand_reacquire_resets_baseline,
    test_scroll_down_and_up,
    test_button_mode_click_mapping,
    test_thumb_back_resumes_move,
    test_clicks_fire_once_per_curl,
    test_draw_overlay_smoke,
    test_config_and_status,
]


def main():
    fails = []
    for test in TESTS:
        try:
            test()
            print(f"  ok  {test.__name__}")
        except Exception as e:  # noqa: BLE001
            fails.append(f"{test.__name__} -> {e}")
            print(f"FAIL  {test.__name__}: {e}")
    if fails:
        print("\nFAILED:")
        for f in fails:
            print(" ", f)
        sys.exit(1)
    print(f"\nOK: {len(TESTS)} gesture-trackpad tests")


if __name__ == "__main__":
    main()
