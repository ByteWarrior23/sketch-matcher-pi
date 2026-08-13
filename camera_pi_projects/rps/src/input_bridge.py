"""Send game inputs to BlueStacks (ADB) and desktop games (keyboard)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ANDROID_PACKAGES = {
    "subway": ["com.kiloo.subwaysurf", "com.sybo.subwaysurf"],
}

PLAY_STORE = {
    "subway": "com.kiloo.subwaysurf",
}

LEVEL_DEVIL_URL = "https://level-devil.org/en/"
SUBWAY_POKI_URL = "https://poki.com/en/g/subway-surfers"

SWIPE = {
    "left": (540, 1200, 180, 1200),
    "right": (540, 1200, 900, 1200),
    "up": (540, 1200, 540, 500),
    "down": (540, 1200, 540, 1700),
}

KEY_MAP = {"left": "left", "right": "right", "jump": "space", "up": "space", "down": "down"}

ADB_PORTS = [
    "127.0.0.1:5555", "127.0.0.1:5556",
    "127.0.0.1:62001", "127.0.0.1:62025",
    "127.0.0.1:62026", "127.0.0.1:62027",
]


def _conf_ports() -> list[str]:
    try:
        from arcade_setup import adb_ports_from_config  # noqa: WPS433
        return adb_ports_from_config()
    except Exception:
        return []


def _ensure_adb_config() -> dict:
    try:
        from arcade_setup import enable_adb_in_config  # noqa: WPS433
        return enable_adb_in_config()
    except Exception as e:
        return {"ok": False, "error": str(e)}

_last_action: dict[str, float] = {}
COOLDOWN_MS = 260


def _adb_bin() -> str | None:
    bundled = ROOT / "tools" / "platform-tools" / "adb.exe"
    if bundled.exists():
        return str(bundled)
    for name in ("adb", "adb.exe"):
        p = shutil.which(name)
        if p:
            return p
    for c in [
        Path(r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe"),
        Path(r"C:\Program Files\BlueStacks\HD-Adb.exe"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "BlueStacks_nxt" / "HD-Adb.exe",
    ]:
        if c.exists():
            return str(c)
    return None


def _find_bluestacks() -> str | None:
    cands = [
        Path(r"C:\Program Files\BlueStacks_nxt\HD-Player.exe"),
        Path(r"C:\Program Files\BlueStacks_nxt\Bluestacks.exe"),
        Path(r"C:\Program Files\BlueStacks\Bluestacks.exe"),
        Path(os.environ.get("ProgramFiles", "")) / "BlueStacks_nxt" / "HD-Player.exe",
    ]
    for c in cands:
        if c.exists():
            return str(c)
    return None


def _bluestacks_running() -> bool:
    try:
        r = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq HD-Player.exe"],
            capture_output=True, text=True, timeout=5,
        )
        return "HD-Player.exe" in (r.stdout or "")
    except Exception:
        return False


def _run(cmd: list[str], timeout: float = 10.0) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, ((r.stdout or "") + (r.stderr or "")).strip()
    except Exception as e:
        return 1, str(e)


class InputBridge:
    def __init__(self):
        self.adb = _adb_bin()
        self._connected = False
        self._screen = (1080, 1920)
        self._game_proc: subprocess.Popen | None = None
        self._last_launch: dict[str, float] = {}

    def status(self) -> dict:
        adb_ok = bool(self.adb)
        devices = []
        connected = False
        if adb_ok:
            code, out = _run([self.adb, "devices"])
            if code == 0:
                for line in out.splitlines()[1:]:
                    if "\tdevice" in line:
                        devices.append(line.split("\t")[0])
                connected = len(devices) > 0
        self._connected = connected
        if connected:
            self._read_screen_size()
        bs = _find_bluestacks()
        running = _bluestacks_running()
        if not bs:
            msg = "Installing BlueStacks automatically — first run takes a few minutes"
        elif not running:
            msg = "BlueStacks installed — click Start BlueStacks"
        elif connected:
            msg = "Emulator connected — gesture swipes active"
        else:
            msg = "BlueStacks running — enable ADB in Settings, then Connect"
        return {
            "adb": adb_ok,
            "adb_path": self.adb,
            "connected": connected,
            "devices": devices,
            "message": msg,
            "bluestacks_installed": bool(bs),
            "bluestacks_running": running,
            "bluestacks_path": bs,
            "screen": {"w": self._screen[0], "h": self._screen[1]},
            "keyboard": sys.platform == "win32",
            "level_devil_running": self._game_proc is not None and self._game_proc.poll() is None,
        }

    def start_bluestacks(self) -> dict:
        exe = _find_bluestacks()
        if not exe:
            return {
                "ok": False,
                "error": "BlueStacks not installed. Run scripts/install_bluestacks.bat as Administrator.",
            }
        if _bluestacks_running():
            return {"ok": True, "note": "BlueStacks already running"}
        try:
            subprocess.Popen(
                [exe],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(Path(exe).parent),
            )
            return {"ok": True, "path": exe, "note": "BlueStacks starting — wait 30s, enable ADB, then Connect"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def connect_adb(self, retries: int = 3) -> dict:
        if not self.adb:
            return {"ok": False, "error": "ADB tools missing"}
        if not _find_bluestacks() and not _bluestacks_running():
            return {"ok": False, "error": "BlueStacks not installed yet"}
        _ensure_adb_config()
        ports = _conf_ports() + [p for p in ADB_PORTS if p not in _conf_ports()]
        messages = []
        for attempt in range(retries):
            for port in ports:
                code, out = _run([self.adb, "connect", port], timeout=3)
                messages.append(f"{port}: {out or code}")
            st = self.status()
            if st["connected"]:
                st["ok"] = True
                st["connect_log"] = messages
                return st
            time.sleep(2)
        st = self.status()
        st["connect_log"] = messages
        st["ok"] = False
        st["error"] = "Cannot reach emulator — setup is still running or BlueStacks is starting."
        return st

    def _read_screen_size(self):
        if not self.adb or not self._connected:
            return
        code, out = _run([self.adb, "shell", "wm", "size"])
        if code == 0 and "x" in out:
            try:
                part = out.split(":")[-1].strip().split()[0]
                w, h = part.split("x")
                self._screen = (int(w), int(h))
            except Exception:
                pass

    def _scale_swipe(self, x1, y1, x2, y2):
        bw, bh = 1080, 1920
        w, h = self._screen
        return int(x1 * w / bw), int(y1 * h / bh), int(x2 * w / bw), int(y2 * h / bh)

    def adb_swipe(self, direction: str) -> dict:
        if not self.adb:
            return {"ok": False, "error": "ADB missing"}
        if not self._connected:
            self.connect_adb(retries=1)
        if not self._connected:
            return {"ok": False, "error": "Emulator not connected"}
        if direction not in SWIPE:
            return {"ok": False, "error": f"unknown swipe {direction}"}
        x1, y1, x2, y2 = self._scale_swipe(*SWIPE[direction])
        code, out = _run([self.adb, "shell", "input", "swipe",
                          str(x1), str(y1), str(x2), str(y2), "90"])
        return {"ok": code == 0, "action": direction, "detail": out or "sent"}

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
        """Hold/release/tap a key (Level Devil zones + Subway Surfers web swipes)."""
        if game == "leveldevil":
            map_key = {"left": "left", "right": "right", "jump": "space", "click": "space"}.get(key)
        elif game == "subway":
            map_key = {"left": "left", "right": "right", "up": "up", "down": "down"}.get(key)
        else:
            return {"ok": False, "error": f"key hold is only for leveldevil/subway"}
        map_key = {"left": "left", "right": "right", "jump": "space", "click": "space"}.get(key)
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
                pydirectinput.press(map_key)
            return {"ok": True, "key": map_key, "state": state}
        except ImportError:
            return {"ok": False, "error": "pip install pydirectinput"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def launch_android(self, game: str) -> dict:
        if game not in ANDROID_PACKAGES:
            return {"ok": False, "error": f"unknown game {game}"}
        now = time.time()
        if now - self._last_launch.get(game, 0) < 2.0:
            return {"ok": True, "note": "cooldown"}
        if not self._connected:
            self.connect_adb()
        if not self._connected:
            return {"ok": False, "error": "Connect emulator first"}
        for pkg in ANDROID_PACKAGES[game]:
            code, out = _run([self.adb, "shell", "monkey", "-p", pkg,
                              "-c", "android.intent.category.LAUNCHER", "1"])
            if code == 0 and "No activities" not in (out or ""):
                self._last_launch[game] = now
                return {"ok": True, "package": pkg}
        pkg = PLAY_STORE.get(game, "")
        return {
            "ok": False,
            "error": f"Install the game in BlueStacks Play Store first (package: {pkg})",
        }

    def launch_level_devil(self) -> dict:
        return self.launch_web(LEVEL_DEVIL_URL)

    def launch_web(self, url: str) -> dict:
        now = time.time()
        if self._game_proc and self._game_proc.poll() is None and url == LEVEL_DEVIL_URL:
            return {"ok": True, "note": "already running"}
        browsers = [
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        ]
        exe = next((str(b) for b in browsers if b.exists()), None)
        if not exe:
            return {"ok": False, "error": "Chrome/Edge not found"}
        try:
            self._game_proc = subprocess.Popen(
                [exe, f"--app={url}", "--new-window"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self._last_launch[url] = now
            return {"ok": True, "url": url}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def prepare_game(self, game: str) -> dict:
        """Install/start BlueStacks, connect ADB, sideload if needed, launch game."""
        steps = []
        if game == "subway":
            # Subway Surfers runs as a web game (poki.com) — no emulator needed.
            launch = self.launch_web(SUBWAY_POKI_URL)
            steps.append({"step": "launch", **launch})
            return {"ok": launch.get("ok", False), "steps": steps}
        try:
            from arcade_setup import (  # noqa: WPS433
                find_bluestacks,
                full_setup,
                install_game_apk,
            )
            if game in ANDROID_PACKAGES and not find_bluestacks():
                setup = full_setup(games=[game])
                steps.extend(setup.get("steps", []))
                if not setup.get("ok"):
                    return {"ok": False, "steps": steps, "error": setup.get("error")}
            elif game in ANDROID_PACKAGES:
                _ensure_adb_config()
        except Exception as e:
            steps.append({"step": "setup_import", "ok": False, "error": str(e)})

        bs = self.start_bluestacks()
        steps.append({"step": "start_bluestacks", **bs})
        if not bs.get("ok") and "already running" not in bs.get("note", ""):
            return {"ok": False, "steps": steps, "error": bs.get("error")}
        if not _bluestacks_running() and bs.get("ok"):
            time.sleep(25)
        conn = self.connect_adb(retries=8)
        steps.append({"step": "connect", **conn})
        if not conn.get("ok"):
            return {"ok": False, "steps": steps, "error": conn.get("error")}
        if game in ANDROID_PACKAGES:
            try:
                from arcade_setup import install_game_apk  # noqa: WPS433
                apk = install_game_apk(game, self.adb)
                steps.append({"step": "install_apk", **apk})
            except Exception as e:
                steps.append({"step": "install_apk", "ok": False, "error": str(e)})
            launch = self.launch_android(game)
            steps.append({"step": "launch", **launch})
            return {"ok": launch.get("ok", False), "steps": steps, "error": launch.get("error")}
        if game == "leveldevil":
            launch = self.launch_level_devil()
            steps.append({"step": "launch", **launch})
            return {"ok": launch.get("ok", False), "steps": steps}
        return {"ok": False, "error": f"unknown game {game}"}

    def dispatch(self, game: str, action: str) -> dict:
        key = f"{game}:{action}"
        now = time.time() * 1000
        if now - _last_action.get(key, 0) < COOLDOWN_MS:
            return {"ok": True, "skipped": "cooldown"}
        _last_action[key] = now
        if game == "subway":
            swipe = {"jump": "up", "duck": "down", "roll": "down"}.get(action, action)
            if swipe not in SWIPE:
                return {"ok": False, "error": f"bad action {action}"}
            return self.adb_swipe(swipe)
        if game == "leveldevil":
            if action in ("duck", "roll", "down"):
                return {"ok": True, "skipped": "n/a"}
            return self.keyboard_key("jump" if action == "jump" else action)
        return {"ok": False, "error": f"unknown game {game}"}

    def launch(self, game: str) -> dict:
        if game == "leveldevil":
            return self.launch_level_devil()
        if game == "subway":
            return self.launch_web(SUBWAY_POKI_URL)
        if game in ANDROID_PACKAGES:
            return self.launch_android(game)
        return {"ok": False, "error": f"unknown game {game}"}


_BRIDGE: InputBridge | None = None


def get_bridge() -> InputBridge:
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = InputBridge()
    return _BRIDGE
