"""RPS Web Game server: serves the frontend + runs the classifier.

Standard-library only (http.server) so it runs anywhere Python 3.9+ is
installed — no Flask/FastAPI required. Start with:

  python web/server.py [--port 8000]

Then open http://localhost:8000  (camera requires localhost or HTTPS).

Endpoints:
  GET  /            index.html
  GET  /<static>    style.css, app.js, favicon
  POST /classify    body = JPEG image bytes -> {"label","conf","probs"}
"""
import argparse
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from classifier import RPSClassifier  # noqa: E402
from hand_rps import recognize as hand_recognize  # noqa: E402
import cv2  # noqa: E402

WEB_DIR = Path(__file__).resolve().parent
ROUTES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/rps.html": "rps.html",
    "/ttt.html": "ttt.html",
    "/rps": "rps.html",
    "/ttt": "ttt.html",
}
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".json": "application/json",
}


def pick_model():
    """Prefer the HPC int8 model, else the local stopgap float32 model."""
    int8 = ROOT / "pi_deploy" / "model_data"
    if (int8 / "rps_model.tflite").exists():
        return int8
    local = ROOT / "models" / "local"
    if (local / "rps_local.tflite").exists():
        return local
    raise FileNotFoundError("no RPS tflite model found")


class Handler(BaseHTTPRequestHandler):
    classifier = None

    def log_message(self, fmt, *args):
        pass  # keep console clean

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ROUTES:
            p = WEB_DIR / ROUTES[path]
            ctype = CONTENT_TYPES.get(p.suffix, "application/octet-stream")
            self._send(200, p.read_bytes(), ctype)
            return
        # static files under /css, /js, /assets
        rel = path.lstrip("/")
        candidate = WEB_DIR / rel
        if "/" in rel and candidate.exists() and candidate.is_file() and path.startswith(("/css/", "/js/", "/assets/")):
            ctype = CONTENT_TYPES.get(candidate.suffix, "application/octet-stream")
            self._send(200, candidate.read_bytes(), ctype)
            return
        if path == "/health":
            self._send(200, b'{"ok":true}', "application/json")
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/classify":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            img = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("bad image")

            # Primary: MediaPipe hand gate + geometric RPS. Never guesses on empty frames.
            r = hand_recognize(img)

            # Fallback: if a hand IS present but geometry is ambiguous (<0.55), ask the CNN.
            label, conf = r["label"], r["conf"]
            probs = []
            if r["detected"] and conf < 0.55:
                c_label, c_conf, c_probs = self.classifier.classify(img)
                if c_conf > conf:
                    label, conf, probs = c_label, c_conf, c_probs

            resp = json.dumps({
                "detected": bool(r["detected"]),
                "label": label,
                "conf": float(conf),
                "probs": [float(p) for p in probs],
            }).encode()
            self._send(200, resp, "application/json")
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model-dir", default=None, help="override model dir")
    args = ap.parse_args()

    Handler.classifier = RPSClassifier(args.model_dir) if args.model_dir else RPSClassifier(pick_model())
    print(f"model: {Handler.classifier.tflite_path}")
    print(f"RPS web game on http://localhost:{args.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
