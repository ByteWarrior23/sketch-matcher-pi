"""Hand-gesture arcade web server — stdlib HTTP + game input bridge.

Serves the static frontend and the input bridge API. Hand detection runs in
the browser (MediaPipe HandLandmarker); the server only receives the final
game action (swipe / key hold / key tap).
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from input_bridge import get_bridge  # noqa: E402

WEB_DIR = ROOT / "web"

ROUTES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/subway.html": "games/subway.html",
    "/subway": "games/subway.html",
    "/leveldevil.html": "games/leveldevil.html",
    "/leveldevil": "games/leveldevil.html",
    "/games/subway.html": "games/subway.html",
    "/games/leveldevil.html": "games/leveldevil.html",
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
        self._send(404, "not found", "text/plain")

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8") or "{}")

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/api/action":
            self._post_action()
            return
        if path == "/api/key":
            self._post_key()
            return
        if path == "/api/launch":
            self._post_launch()
            return
        if path == "/api/connect":
            self._post_connect()
            return
        if path == "/api/bluestacks/start":
            self._post_bluestacks_start()
            return
        if path == "/api/prepare":
            self._post_prepare()
            return
        if path == "/api/setup":
            self._post_setup()
            return
        self._send(404, "not found", "text/plain")

    def _post_action(self):
        try:
            data = self._read_json()
            game = str(data.get("game", "")).lower()
            action = str(data.get("action", "")).lower()
            result = get_bridge().dispatch(game, action)
            self._json(200, result)
        except Exception as e:
            self._json(500, {"ok": False, "error": str(e)})

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

    def _post_launch(self):
        try:
            data = self._read_json()
            game = str(data.get("game", "")).lower()
            result = get_bridge().launch(game)
            self._json(200, result)
        except Exception as e:
            self._json(500, {"ok": False, "error": str(e)})

    def _post_connect(self):
        try:
            result = get_bridge().connect_adb()
            self._json(200, result)
        except Exception as e:
            self._json(500, {"ok": False, "error": str(e)})

    def _post_bluestacks_start(self):
        try:
            result = get_bridge().start_bluestacks()
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

    def _post_setup(self):
        try:
            from arcade_setup import full_setup  # noqa: WPS433
            data = self._read_json() if int(self.headers.get("Content-Length", 0)) else {}
            games = data.get("games") or ["subway"]
            result = full_setup(games=games)
            self._json(200, result)
        except Exception as e:
            self._json(500, {"ok": False, "error": str(e)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    args = ap.parse_args()

    bridge = get_bridge()
    st = bridge.status()
    print(f"adb: {st.get('adb_path', 'missing')} connected={st.get('connected')} — {st.get('message')}")
    print(f"Arcade on http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
