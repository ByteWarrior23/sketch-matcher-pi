"""Send game inputs to desktop web games (keyboard bridge)."""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LEVEL_DEVIL_URL = "https://level-devil.org/en/"
SUBWAY_POKI_URL = "https://poki.com/en/g/subway-surfers"

KEY_MAP = {"left": "left", "right": "right", "jump": "space", "up": "space", "down": "down"}

# pydirectinput is not thread-safe, and the web server handles requests on
# concurrent threads. Serialize every input so "down" always lands before its
# matching "up" (out-of-order events made movement stick/mismatch).
_INPUT_LOCK = threading.Lock()


def _screen_size():
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


def _mouse_wheel(amount: int) -> None:
    """Send raw wheel events via SendInput (pydirectinput has no scroll()).

    `amount` is in wheel "clicks": positive = up, negative = down. Sending the
    wheel with Ctrl held reproduces a trackpad pinch-to-zoom (what Windows
    precision touchpads actually emit).
    """
    import ctypes
    from ctypes import wintypes

    MOUSEEVENTF_WHEEL = 0x0800
    WHEEL_DELTA = 120
    INPUT_MOUSE = 0

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        ]

    class INPUT_UNION(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("union", INPUT_UNION)]

    # Use a private user32 handle: pydirectinput installs its own INPUT type
    # as the argtypes on the shared SendInput, which would reject our struct.
    send_input = ctypes.WinDLL("user32").SendInput
    send_input.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    send_input.restype = wintypes.UINT

    step = (WHEEL_DELTA if amount > 0 else -WHEEL_DELTA) & 0xFFFFFFFF
    for _ in range(abs(amount)):
        ev = INPUT()
        ev.type = INPUT_MOUSE
        ev.union.mi.mouseData = step
        ev.union.mi.dwFlags = MOUSEEVENTF_WHEEL
        send_input(1, ctypes.byref(ev), ctypes.sizeof(INPUT))


class InputBridge:
    def __init__(self):
        self._game_proc: subprocess.Popen | None = None
        self._last_launch: dict[str, float] = {}

    def status(self) -> dict:
        return {
            "keyboard": sys.platform == "win32",
            "level_devil_running": self._game_proc is not None and self._game_proc.poll() is None,
            "message": "Play in the game window; keep the camera window visible.",
        }

    def keyboard_key(self, action: str) -> dict:
        key = KEY_MAP.get(action)
        if not key:
            return {"ok": False, "error": f"unknown key {action}"}
        try:
            import pydirectinput  # noqa: WPS433
            with _INPUT_LOCK:
                pydirectinput.PAUSE = 0
                pydirectinput.press(key)
            return {"ok": True, "action": action, "key": key}
        except ImportError:
            return {"ok": False, "error": "pip install pydirectinput"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def keyboard_state(self, game: str, key: str, state: str) -> dict:
        if game == "leveldevil":
            map_key = {"left": "left", "right": "right", "jump": "space", "click": "space"}.get(key)
        elif game == "subway":
            map_key = {"left": "left", "right": "right", "up": "up", "down": "down"}.get(key)
        else:
            return {"ok": False, "error": f"key hold is only for leveldevil/subway"}
        if not map_key:
            return {"ok": False, "error": f"unknown key {key}"}
        if state not in ("down", "up", "tap"):
            return {"ok": False, "error": f"bad state {state}"}
        try:
            import pydirectinput  # noqa: WPS433
            with _INPUT_LOCK:
                pydirectinput.PAUSE = 0
                if state == "down":
                    pydirectinput.keyDown(map_key)
                elif state == "up":
                    pydirectinput.keyUp(map_key)
                else:
                    pydirectinput.keyDown(map_key)
                    time.sleep(0.03)
                    pydirectinput.keyUp(map_key)
            return {"ok": True, "key": map_key, "state": state}
        except ImportError:
            return {"ok": False, "error": "pip install pydirectinput"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def mouse_move(self, x: float, y: float) -> dict:
        """Move the OS cursor to an absolute position. x/y are 0..1 screen.

        Coordinates are clamped to [1, size-1] so the cursor never lands on
        (0, 0), which pydirectinput treats as its fail-safe corner.
        """
        try:
            import pydirectinput  # noqa: WPS433
            w, h = _screen_size()
            px = max(1, min(w - 1, int(x * w)))
            py = max(1, min(h - 1, int(y * h)))
            with _INPUT_LOCK:
                pydirectinput.PAUSE = 0
                pydirectinput.moveTo(px, py)
            return {"ok": True, "x": px, "y": py}
        except ImportError:
            return {"ok": False, "error": "pip install pydirectinput"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def mouse_click(self, times: int = 1, state: str = "tap", button: str = "left") -> dict:
        """Left/right-click. times=1 single, times=2 double; state down/up = press."""
        try:
            import pydirectinput  # noqa: WPS433
            with _INPUT_LOCK:
                pydirectinput.PAUSE = 0
                if state == "down":
                    pydirectinput.mouseDown(button=button)
                elif state == "up":
                    pydirectinput.mouseUp(button=button)
                else:
                    pydirectinput.click(clicks=times or 1, button=button)
            return {"ok": True, "times": times or 1, "button": button}
        except ImportError:
            return {"ok": False, "error": "pip install pydirectinput"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def mouse_scroll(self, lines: int, ctrl: bool = False) -> dict:
        """Wheel scroll. With ctrl held it mimics a trackpad pinch zoom;
        plain (ctrl=False) is a normal two-finger scroll.

        The installed pydirectinput has no scroll(), so the wheel events are
        sent directly via SendInput.
        """
        try:
            import pydirectinput  # noqa: WPS433
            with _INPUT_LOCK:
                pydirectinput.PAUSE = 0
                if ctrl:
                    pydirectinput.keyDown("ctrl")
                    try:
                        _mouse_wheel(int(lines))
                    finally:
                        pydirectinput.keyUp("ctrl")
                else:
                    _mouse_wheel(int(lines))
            return {"ok": True, "lines": int(lines), "ctrl": ctrl}
        except ImportError:
            return {"ok": False, "error": "pip install pydirectinput"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def launch_web(self, url: str, half_screen: bool = False) -> dict:
        now = time.time()
        if self._game_proc and self._game_proc.poll() is None and url in (LEVEL_DEVIL_URL, SUBWAY_POKI_URL):
            return {"ok": True, "note": "already running"}
        browsers = [
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        ]
        exe = next((str(b) for b in browsers if b.exists()), None)
        if not exe:
            return {"ok": False, "error": "Chrome/Edge not found"}
        try:
            args = [exe, f"--app={url}", "--new-window"]
            if half_screen:
                w, h = _screen_size()
                args += [f"--window-position={w // 2},0", f"--window-size={w // 2},{h}"]
            self._game_proc = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self._last_launch[url] = now
            return {"ok": True, "url": url}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def prepare_game(self, game: str) -> dict:
        if game == "subway":
            launch = self.launch_web(SUBWAY_POKI_URL, half_screen=True)
            return {"ok": launch.get("ok", False), "steps": [{"step": "launch_poki", **launch}]}
        if game == "leveldevil":
            launch = self.launch_web(LEVEL_DEVIL_URL)
            return {"ok": launch.get("ok", False), "steps": [{"step": "launch", **launch}]}
        return {"ok": False, "error": f"unknown game {game}"}

    def launch(self, game: str) -> dict:
        if game == "leveldevil":
            return self.launch_web(LEVEL_DEVIL_URL)
        if game == "subway":
            return self.launch_web(SUBWAY_POKI_URL)
        return {"ok": False, "error": f"unknown game {game}"}


_BRIDGE: InputBridge | None = None


def get_bridge() -> InputBridge:
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = InputBridge()
    return _BRIDGE
