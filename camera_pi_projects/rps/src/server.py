"""Hand-gesture arcade web server — stdlib HTTP + keyboard input bridge.

Serves the static frontend and the input API. Hand detection runs in the
browser (MediaPipe HandLandmarker); the server presses the keys for the
focused game window.
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gesture_trackpad import get_trackpad  # noqa: E402
from input_bridge import get_bridge  # noqa: E402

WEB_DIR = ROOT / "web"

ROUTES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/subway.html": "games/subway.html",
    "/subway": "games/subway.html",
    "/leveldevil.html": "games/leveldevil.html",
    "/leveldevil": "games/leveldevil.html",
    "/games/leveldevil.html": "games/leveldevil.html",
    "/trackpad.html": "games/trackpad.html",
    "/trackpad": "games/trackpad.html",
    "/games/trackpad.html": "games/trackpad.html",
}

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".wasm": "application/wasm",
    ".task": "application/octet-stream",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".json": "application/json",
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj), "application/json")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ROUTES:
            p = WEB_DIR / ROUTES[path]
            ctype = CONTENT_TYPES.get(p.suffix, "application/octet-stream")
            self._send(200, p.read_bytes(), ctype)
            return
        rel = path.lstrip("/")
        candidate = WEB_DIR / rel
        if "/" in rel and candidate.exists() and candidate.is_file() and path.startswith(("/css/", "/js/", "/vendor/")):
            ctype = CONTENT_TYPES.get(candidate.suffix, "application/octet-stream")
            self._send(200, candidate.read_bytes(), ctype)
            return
        if path == "/hand_model.task":
            p = ROOT / "models" / "hand_landmarker.task"
            if p.exists():
                self._send(200, p.read_bytes(), "application/octet-stream")
                return
        if path == "/health":
            self._json(200, {"ok": True})
            return
        if path == "/api/status":
            self._json(200, get_bridge().status())
            return
        if path == "/api/trackpad/status":
            self._json(200, get_trackpad().status())
            return
        if path == "/api/trackpad/preview":
            self._stream_trackpad_preview()
            return
        self._send(404, "not found", "text/plain")

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8") or "{}")

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/api/key":
            self._post_key()
            return
        if path == "/api/mouse":
            self._post_mouse()
            return
        if path == "/api/prepare":
            self._post_prepare()
            return
        if path == "/api/trackpad/start":
            self._post_trackpad_start()
            return
        if path == "/api/trackpad/stop":
            self._post_trackpad_stop()
            return
        if path == "/api/trackpad/config":
            self._post_trackpad_config()
            return
        if path == "/api/trackpad/recenter":
            self._post_trackpad_recenter()
            return
        self._send(404, "not found", "text/plain")

    def _post_key(self):
        try:
            data = self._read_json()
            game = str(data.get("game", "")).lower()
            key = str(data.get("key", "")).lower()
            state = str(data.get("state", "")).lower()
            result = get_bridge().keyboard_state(game, key, state)
            self._json(200, result)
        except Exception as e:
            self._json(500, {"ok": False, "error": str(e)})

    def _post_mouse(self):
        try:
            data = self._read_json()
            op = str(data.get("op", "")).lower()
            bridge = get_bridge()
            if op == "move":
                result = bridge.mouse_move(
                    float(data.get("x", 0.5)), float(data.get("y", 0.5))
                )
            elif op == "click":
                result = bridge.mouse_click(
                    times=int(data.get("times", 1)),
                    state=str(data.get("state", "tap")).lower(),
                    button=str(data.get("button", "left")).lower(),
                )
            elif op == "scroll":
                result = bridge.mouse_scroll(
                    int(data.get("lines", 0)),
                    ctrl=bool(data.get("ctrl", False)),
                )
            else:
                result = {"ok": False, "error": f"unknown mouse op {op}"}
            self._json(200, result)
        except Exception as e:
            self._json(500, {"ok": False, "error": str(e)})

    def _post_prepare(self):
        try:
            data = self._read_json()
            game = str(data.get("game", "")).lower()
            result = get_bridge().prepare_game(game)
            self._json(200, result)
        except Exception as e:
            self._json(500, {"ok": False, "error": str(e)})

    def _stream_trackpad_preview(self):
        trackpad = get_trackpad()
        if not trackpad.running():
            self._send(409, "trackpad not running", "text/plain")
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        trackpad.on_preview_opened()
        try:
            for jpg in trackpad.preview_frames():
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError, ValueError):
            pass
        finally:
            trackpad.on_preview_closed()

    def _post_trackpad_start(self):
        try:
            data = self._read_json()
            result = get_trackpad().start(sens=int(data.get("sens", 5)))
            self._json(200, result)
        except Exception as e:
            self._json(500, {"ok": False, "error": str(e)})

    def _post_trackpad_stop(self):
        try:
            self._json(200, get_trackpad().stop())
        except Exception as e:
            self._json(500, {"ok": False, "error": str(e)})

    def _post_trackpad_config(self):
        try:
            data = self._read_json()
            result = get_trackpad().set_sens(int(data.get("sens", 5)))
            self._json(200, result)
        except Exception as e:
            self._json(500, {"ok": False, "error": str(e)})

    def _post_trackpad_recenter(self):
        try:
            self._json(200, get_trackpad().recenter())
        except Exception as e:
            self._json(500, {"ok": False, "error": str(e)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1234)
    args = ap.parse_args()

    print(f"Arcade on http://localhost:{args.port}")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
