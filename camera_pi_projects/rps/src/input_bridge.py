"""Send game inputs to desktop web games (keyboard bridge)."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LEVEL_DEVIL_URL = "https://level-devil.org/en/"
SUBWAY_POKI_URL = "https://poki.com/en/g/subway-surfers"

KEY_MAP = {"left": "left", "right": "right", "jump": "space", "up": "space", "down": "down"}


def _screen_size():
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


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
